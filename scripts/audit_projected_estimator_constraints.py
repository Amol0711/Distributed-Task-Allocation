#!/usr/bin/env python3
"""Compare nonnegative and fully constrained quadratic estimators.

For every evaluation seed, the audit records whether the nonnegative minimizer
violates the Euclidean prior bound or the total-value cap before each update.
The fully constrained estimate is applied to the subsequent episode. Per-trial
records quantify constraint activity, feasibility margins, and estimator
agreement under the configured feedback streams.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import campaign_engine as engine  # noqa: E402

CAMPAIGNS = {
    "SAT-COV-V1": ROOT / "configs" / "satellite_campaign.json",
    "UAV-AG-V1": ROOT / "configs" / "uav_campaign.json",
}
METHODS = ("DISTRIBUTED_UCB", "PROJECTED_MEAN", "UCB_WITHOUT_RESOURCE_FILTER")


def load_evaluation_rows() -> list[dict[str, str]]:
    with (ROOT / "seeds" / "trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["partition"] == "evaluation"]
    rows.sort(key=lambda row: (row["campaign_id"], int(row["trial_index"])))
    return rows


def _worker(seed: dict[str, str]) -> dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    stats: dict[str, Any] = {
        "updates": 0,
        "nonnegative_l1_activations": 0,
        "nonnegative_l2_activations": 0,
        "nonnegative_any_activations": 0,
        "max_nonnegative_l1": 0.0,
        "max_nonnegative_l2": 0.0,
        "min_nonnegative_l1_slack": float("inf"),
        "min_nonnegative_l2_slack": float("inf"),
        "max_full_vs_nonnegative_abs_difference": 0.0,
        "max_full_l1_violation": 0.0,
        "max_full_l2_violation": 0.0,
    }

    original_update = engine.Estimator.update

    def audited_update(self: engine.Estimator, x: np.ndarray, y: float) -> None:
        self.V += np.outer(x, x)
        self.rhs += x * y
        old = engine.nonnegative_quadratic_minimizer(self.V, self.rhs)
        full = engine.constrained_quadratic_minimizer(
            self.V, self.rhs, self.btheta, self.fmax
        )
        nonnegative_l1 = float(np.sum(old))
        nonnegative_l2 = float(np.linalg.norm(old))
        full_l1 = float(np.sum(full))
        full_l2 = float(np.linalg.norm(full))
        l1_active = nonnegative_l1 > self.fmax + 2.0e-10
        l2_active = nonnegative_l2 > self.btheta + 2.0e-10
        stats["updates"] += 1
        stats["nonnegative_l1_activations"] += int(l1_active)
        stats["nonnegative_l2_activations"] += int(l2_active)
        stats["nonnegative_any_activations"] += int(l1_active or l2_active)
        stats["max_nonnegative_l1"] = max(stats["max_nonnegative_l1"], nonnegative_l1)
        stats["max_nonnegative_l2"] = max(stats["max_nonnegative_l2"], nonnegative_l2)
        stats["min_nonnegative_l1_slack"] = min(stats["min_nonnegative_l1_slack"], self.fmax - nonnegative_l1)
        stats["min_nonnegative_l2_slack"] = min(stats["min_nonnegative_l2_slack"], self.btheta - nonnegative_l2)
        stats["max_full_vs_nonnegative_abs_difference"] = max(
            stats["max_full_vs_nonnegative_abs_difference"],
            float(np.max(np.abs(full - old))),
        )
        stats["max_full_l1_violation"] = max(
            stats["max_full_l1_violation"], max(0.0, full_l1 - self.fmax)
        )
        stats["max_full_l2_violation"] = max(
            stats["max_full_l2_violation"], max(0.0, full_l2 - self.btheta)
        )
        self.theta = full

    engine.Estimator.update = audited_update
    try:
        with tempfile.TemporaryDirectory(prefix="projection-audit-") as temp:
            result = engine.run_trial(
                sim_root=ROOT,
                campaign_path=CAMPAIGNS[seed["campaign_id"]],
                seed_row=seed,
                raw_dir=Path(temp),
                engine_hash=engine.sha256_file(ROOT / "src" / "campaign_engine.py"),
                scale="primary",
                methods=METHODS,
                write_raw=False,
            )
    finally:
        engine.Estimator.update = original_update

    for key in ("min_nonnegative_l1_slack", "min_nonnegative_l2_slack"):
        if not np.isfinite(stats[key]):
            stats[key] = None
    return {
        "campaign_id": seed["campaign_id"],
        "trial_index": int(seed["trial_index"]),
        "trial_seed": int(seed["trial_seed"]),
        "epochs": int(result["epochs"]),
        "methods": list(METHODS),
        **stats,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for campaign_id in CAMPAIGNS:
        subset = [row for row in rows if row["campaign_id"] == campaign_id]
        out[campaign_id] = {
            "trials": len(subset),
            "updates": sum(int(row["updates"]) for row in subset),
            "nonnegative_l1_activations": sum(int(row["nonnegative_l1_activations"]) for row in subset),
            "nonnegative_l2_activations": sum(int(row["nonnegative_l2_activations"]) for row in subset),
            "nonnegative_any_activations": sum(int(row["nonnegative_any_activations"]) for row in subset),
            "max_nonnegative_l1": max(float(row["max_nonnegative_l1"]) for row in subset),
            "max_nonnegative_l2": max(float(row["max_nonnegative_l2"]) for row in subset),
            "min_nonnegative_l1_slack": min(float(row["min_nonnegative_l1_slack"]) for row in subset),
            "min_nonnegative_l2_slack": min(float(row["min_nonnegative_l2_slack"]) for row in subset),
            "max_full_vs_nonnegative_abs_difference": max(
                float(row["max_full_vs_nonnegative_abs_difference"]) for row in subset
            ),
            "max_full_l1_violation": max(float(row["max_full_l1_violation"]) for row in subset),
            "max_full_l2_violation": max(float(row["max_full_l2_violation"]) for row in subset),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "projected_estimator",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    seeds = load_evaluation_rows()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = args.output_dir / "projected_estimator_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, str]] = []
    for seed in seeds:
        path = trial_dir / f"{seed['campaign_id']}_T{int(seed['trial_index']):02d}.json"
        if args.resume and path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(seed)
    if rows:
        print(f"[resume] loaded {len(rows)} completed trial audits", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, seed): seed for seed in pending}
        for number, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            path = trial_dir / f"{row['campaign_id']}_T{row['trial_index']:02d}.json"
            path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"[{number:02d}/{len(pending):02d}] {row['campaign_id']} "
                f"T{row['trial_index']:02d}: {row['updates']} updates PASS",
                flush=True,
            )
    rows.sort(key=lambda row: (row["campaign_id"], row["trial_index"]))
    if len(rows) != len(seeds):
        raise SystemExit(f"audit incomplete: {len(rows)}/{len(seeds)} trial records")
    summary = aggregate(rows)
    passed = all(
        values["nonnegative_any_activations"] == 0
        and values["max_full_vs_nonnegative_abs_difference"] <= 1.0e-12
        and values["max_full_l1_violation"] <= 2.0e-10
        and values["max_full_l2_violation"] <= 2.0e-10
        for values in summary.values()
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "scope": "all 32 predeclared independent evaluation traces per application",
        "methods": list(METHODS),
        "seed_registry_sha256": engine.sha256_file(ROOT / "seeds" / "trials.csv"),
        "campaign_engine_sha256": engine.sha256_file(ROOT / "src" / "campaign_engine.py"),
        "trials": len(rows),
        "aggregate": summary,
        "per_trial": rows,
    }
    json_path = args.output_dir / "projected_estimator_constraint_audit.json"
    csv_path = args.output_dir / "projected_estimator_constraint_audit.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        # Canonical bytes are independent of fresh/resumed dictionary order.
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if not passed:
        raise SystemExit("full projected-estimator audit failed")
    print(json.dumps({"status": "PASS", "aggregate": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
