#!/usr/bin/env python3
"""Enumerate small feasible families and verify allocation certificate arithmetic.

The first design seed of each application fixes its context stream. The audit
compares exhaustive optima with an independent branch-and-bound solver, checks
maximal greedy sequences and curvature, and evaluates zero and nonuniform
marginal-shortfall bounds. It performs no estimator update or physical run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campaign_engine import (  # noqa: E402
    ContextGenerator,
    ElementBatch,
    assignment_violations,
    coverage_value,
    curvature,
    exact_optimum,
    load_assets,
)

SCHEMA = "exact-oracle-audit-v1"
CAMPAIGNS = ("satellite_campaign.json", "uav_campaign.json")
TOL = 2.0e-10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def first_design_seed(campaign_id: str) -> dict[str, str]:
    with (ROOT / "seeds" / "trials.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["campaign_id"] == campaign_id and row["partition"] == "design":
                return row
    raise RuntimeError(f"missing design seed for {campaign_id}")


def feasible_candidates(
    batch: ElementBatch,
    base: np.ndarray,
    selected: Sequence[int],
) -> list[int]:
    used_owner = {int(batch.owner[e]) for e in selected}
    used_task = {int(batch.task[e]) for e in selected}
    counts = np.zeros(len(batch.quota_caps), dtype=int)
    for e in selected:
        counts[int(batch.quota[e])] += 1
    return [
        int(e)
        for e in np.flatnonzero(base)
        if int(batch.owner[e]) not in used_owner
        and int(batch.task[e]) not in used_task
        and counts[int(batch.quota[e])] < int(batch.quota_caps[int(batch.quota[e])])
    ]


def greedy_path(
    batch: ElementBatch,
    base: np.ndarray,
    theta: np.ndarray,
    *,
    runner_up: bool,
) -> tuple[list[int], float, list[float]]:
    selected: list[int] = []
    fail = np.ones(batch.p_h.shape[1], dtype=float)
    shortfalls: list[float] = []
    while True:
        cand = feasible_candidates(batch, base, selected)
        if not cand:
            break
        ranked = sorted(
            (
                float(theta @ (batch.p_h[e] * fail)),
                int(batch.element_id[e]),
                int(e),
            )
            for e in cand
        )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        pos = 1 if runner_up and len(ranked) > 1 else 0
        best = ranked[0][0]
        chosen_marg, _, chosen = ranked[pos]
        shortfalls.append(max(0.0, best - chosen_marg))
        selected.append(chosen)
        fail *= 1.0 - batch.p_h[chosen]
    return selected, coverage_value(theta, batch.p_h, selected), shortfalls


@dataclass
class Enumeration:
    family_size: int = 0
    maximal_sets: int = 0
    maximum_cardinality: int = 0
    optimum_value: float = 0.0
    optimum_ids: tuple[int, ...] = ()
    violations: int = 0


def exhaustive_family(
    batch: ElementBatch,
    base: np.ndarray,
    theta: np.ndarray,
    max_sets: int,
) -> Enumeration:
    nowners = int(batch.owner.max()) + 1
    options = [list(map(int, np.flatnonzero(base & (batch.owner == owner)))) for owner in range(nowners)]
    out = Enumeration()

    def dfs(
        owner: int,
        selected: list[int],
        used_tasks: set[int],
        counts: np.ndarray,
        fail: np.ndarray,
        value: float,
    ) -> None:
        if owner == nowners:
            out.family_size += 1
            if out.family_size > max_sets:
                raise RuntimeError(f"exhaustive family exceeded configured limit {max_sets}")
            out.maximum_cardinality = max(out.maximum_cardinality, len(selected))
            out.violations += assignment_violations(batch, selected)
            ids = tuple(sorted(int(batch.element_id[e]) for e in selected))
            if value > out.optimum_value + 1.0e-13 or (
                abs(value - out.optimum_value) <= 1.0e-13
                and (not out.optimum_ids or ids < out.optimum_ids)
            ):
                out.optimum_value = float(value)
                out.optimum_ids = ids
            if not feasible_candidates(batch, base, selected):
                out.maximal_sets += 1
            return

        # Skip the current owner.
        dfs(owner + 1, selected, used_tasks, counts, fail, value)
        for e in options[owner]:
            task = int(batch.task[e])
            quota = int(batch.quota[e])
            if task in used_tasks or counts[quota] >= int(batch.quota_caps[quota]):
                continue
            marginal = float(theta @ (batch.p_h[e] * fail))
            selected.append(e)
            used_tasks.add(task)
            counts[quota] += 1
            dfs(owner + 1, selected, used_tasks, counts, fail * (1.0 - batch.p_h[e]), value + marginal)
            counts[quota] -= 1
            used_tasks.remove(task)
            selected.pop()

    dfs(0, [], set(), np.zeros(len(batch.quota_caps), dtype=int), np.ones(batch.p_h.shape[1]), 0.0)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered = {
                key: (format(value, ".17g") if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(rendered)


def build(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    campaign_summaries: list[dict[str, Any]] = []

    for cfg_name in CAMPAIGNS:
        cfg_path = ROOT / "configs" / cfg_name
        assets = load_assets(ROOT, cfg_path)
        cfg = assets.cfg
        cid = str(cfg["campaign_id"])
        spec = cfg["small_exact_oracle"]
        p = cfg["primary"]
        na = int(spec["agents"])
        nt = int(spec["tasks"])
        epochs = int(spec["epochs"])
        max_sets = int(spec["maximum_enumerated_feasible_sets"])
        q = int(p["q"])
        theta = np.asarray(p["theta_star"], dtype=float)
        resources = np.tile(np.asarray(p["resource_initial"], dtype=float), (na, 1))
        seed = first_design_seed(cid)
        generator = ContextGenerator(assets, int(seed["scenario_seed"]), na, nt, True)

        campaign_rows: list[dict[str, Any]] = []
        for epoch in range(1, epochs + 1):
            batch = generator.next(epoch)
            affordable = np.all(batch.robust_cost <= resources[batch.owner] + 1.0e-12, axis=1)
            base = batch.kinematic & affordable
            enum = exhaustive_family(batch, base, theta, max_sets)
            if enum.violations:
                raise RuntimeError(f"{cid} epoch {epoch}: enumerated family contains violations")

            bb_sel, bb_value, bb_leaves = exact_optimum(batch, base, theta, max_sets)
            exact_match = math.isclose(bb_value, enum.optimum_value, rel_tol=2.0e-12, abs_tol=2.0e-12)
            if not exact_match or assignment_violations(batch, bb_sel):
                raise RuntimeError(f"{cid} epoch {epoch}: exact optimum cross-check failed")

            greedy_sel, greedy_value, greedy_shortfalls = greedy_path(batch, base, theta, runner_up=False)
            approx_sel, approx_value, approx_shortfalls = greedy_path(batch, base, theta, runner_up=True)
            kappa = curvature(theta, batch.p_h[base])
            exact_rhs = enum.optimum_value / (q + kappa)
            approx_rhs = (enum.optimum_value - q * sum(approx_shortfalls)) / (q + kappa)
            exact_slack = greedy_value - exact_rhs
            approx_slack = approx_value - approx_rhs

            valid = (
                assignment_violations(batch, greedy_sel) == 0
                and assignment_violations(batch, approx_sel) == 0
                and len(feasible_candidates(batch, base, greedy_sel)) == 0
                and len(feasible_candidates(batch, base, approx_sel)) == 0
                and all(abs(x) <= TOL for x in greedy_shortfalls)
                and exact_slack >= -TOL
                and approx_slack >= -TOL
                and -TOL <= kappa <= 1.0 + TOL
            )
            if not valid:
                raise RuntimeError(f"{cid} epoch {epoch}: finite-channel audit failed")

            row = {
                "schema_version": SCHEMA,
                "campaign_id": cid,
                "epoch": epoch,
                "design_trial_index": int(seed["trial_index"]),
                "scenario_seed": int(seed["scenario_seed"]),
                "agents": na,
                "tasks": nt,
                "q": q,
                "screened_elements": int(np.sum(base)),
                "enumerated_feasible_sets": enum.family_size,
                "maximal_feasible_sets": enum.maximal_sets,
                "maximum_cardinality": enum.maximum_cardinality,
                "branch_and_bound_leaves": int(bb_leaves),
                "curvature": float(kappa),
                "exact_optimum_value": float(enum.optimum_value),
                "exact_score_greedy_value": float(greedy_value),
                "exact_score_greedy_ratio": float(greedy_value / enum.optimum_value if enum.optimum_value > 0 else 1.0),
                "zero_shortfall_certificate_slack": float(exact_slack),
                "approximate_greedy_value": float(approx_value),
                "approximate_shortfall_sum": float(sum(approx_shortfalls)),
                "nonuniform_channel_certificate_slack": float(approx_slack),
                "exact_optimum_crosscheck": 1,
                "feasibility_and_maximality_checks": 1,
                "finite_channel_checks": 1,
            }
            rows.append(row)
            campaign_rows.append(row)

        campaign_summaries.append(
            {
                "campaign_id": cid,
                "instances": len(campaign_rows),
                "design_trial_index": int(seed["trial_index"]),
                "scenario_seed": int(seed["scenario_seed"]),
                "enumerated_feasible_sets_total": int(sum(r["enumerated_feasible_sets"] for r in campaign_rows)),
                "enumerated_feasible_sets_min": int(min(r["enumerated_feasible_sets"] for r in campaign_rows)),
                "enumerated_feasible_sets_max": int(max(r["enumerated_feasible_sets"] for r in campaign_rows)),
                "screened_elements_min": int(min(r["screened_elements"] for r in campaign_rows)),
                "screened_elements_max": int(max(r["screened_elements"] for r in campaign_rows)),
                "minimum_zero_shortfall_certificate_slack": float(min(r["zero_shortfall_certificate_slack"] for r in campaign_rows)),
                "minimum_nonuniform_channel_certificate_slack": float(min(r["nonuniform_channel_certificate_slack"] for r in campaign_rows)),
                "minimum_exact_score_greedy_ratio": float(min(r["exact_score_greedy_ratio"] for r in campaign_rows)),
                "maximum_curvature": float(max(r["curvature"] for r in campaign_rows)),
                "all_checks_passed": True,
            }
        )

    csv_path = output_dir / "exact_oracle_instances.csv"
    write_csv(csv_path, rows)
    summary = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "scope": "small exact-oracle geometry audit; no learning updates or physical trajectory simulation",
        "instances_total": len(rows),
        "campaigns": campaign_summaries,
        "enumerated_feasible_sets_total": int(sum(r["enumerated_feasible_sets"] for r in rows)),
        "minimum_zero_shortfall_certificate_slack": float(min(r["zero_shortfall_certificate_slack"] for r in rows)),
        "minimum_nonuniform_channel_certificate_slack": float(min(r["nonuniform_channel_certificate_slack"] for r in rows)),
        "all_exact_optimum_crosschecks_passed": all(r["exact_optimum_crosscheck"] == 1 for r in rows),
        "all_feasibility_and_maximality_checks_passed": all(r["feasibility_and_maximality_checks"] == 1 for r in rows),
        "all_finite_channel_checks_passed": all(r["finite_channel_checks"] == 1 for r in rows),
        "inputs": {
            "satellite_config_sha256": sha256(ROOT / "configs" / "satellite_campaign.json"),
            "uav_config_sha256": sha256(ROOT / "configs" / "uav_campaign.json"),
            "trials_sha256": sha256(ROOT / "seeds" / "trials.csv"),
            "engine_sha256": sha256(ROOT / "src" / "campaign_engine.py"),
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            "exact_oracle_instances.csv": {
                "sha256": sha256(csv_path),
                "rows": len(rows),
            }
        },
    }
    summary_path = output_dir / "exact_oracle_audit.json"
    summary_path.write_text(canonical_json(summary), encoding="utf-8")
    print(canonical_json(summary), end="")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "exact_oracle",
    )
    args = parser.parse_args()
    build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
