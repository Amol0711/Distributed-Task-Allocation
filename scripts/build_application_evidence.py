#!/usr/bin/env python3
"""Aggregate application evaluations and export numerical plotting data.

The builder consumes validated evaluation records for exploration coefficients
0 and 0.25, checks value and exploration statistics against the reference
campaigns, and computes normalized certificate utilizations. Paired retention,
bootstrap intervals, and covariance diagnostics use the declared evaluation
seeds. Median curvature is descriptive and is not a certificate denominator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS = {
    "SAT-COV-V1": {"short": "SAT", "label": "Satellite coverage", "bootstrap_seed": 20260819},
    "UAV-AG-V1": {"short": "UAV", "label": "UAV allocation", "bootstrap_seed": 20260820},
}
C_VALUES = (0.0, 0.25)
BOOTSTRAP_REPLICATES = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered: dict[str, Any] = {}
            for field in fields:
                value = row.get(field, "")
                if isinstance(value, float):
                    rendered[field] = "" if not math.isfinite(value) else format(value, ".17g")
                else:
                    rendered[field] = value
            writer.writerow(rendered)


def mean_se(values: Iterable[float]) -> tuple[float, float]:
    sample = np.asarray(list(values), dtype=float)
    if sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("mean/SE input must be finite and nonempty")
    mean = float(sample.mean())
    se = float(sample.std(ddof=1) / math.sqrt(sample.size)) if sample.size > 1 else 0.0
    return mean, se


def close(actual: float, expected: float, *, atol: float = 5.0e-12, rtol: float = 5.0e-12) -> bool:
    return math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol)


def aggregate_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["schema_version"] != "exploration-campaign-v2":
            continue
        grouped[(row["campaign_id"], float(row["exploration_coefficient"]))].append(row)

    output: list[dict[str, Any]] = []
    scalar_metrics = (
        "exploration_fraction",
        "cumulative_reference_ratio",
        "direct_universal_ratio",
        "direct_raw_universal_ratio",
        "direct_clip_fraction",
        "direct_clip_fraction_all",
        "direct_clip_excess",
        "support_universal_ratio",
        "support_raw_universal_ratio",
        "support_clip_fraction",
        "support_clip_fraction_all",
        "support_clip_excess",
        "support_transfer_increment",
        "prior_universal_ratio",
        "prior_raw_universal_ratio",
        "prior_clip_fraction",
        "prior_clip_fraction_all",
        "prior_clip_excess",
        "prior_transfer_increment",
        "mean_contextual_comparator_factor",
        "direct_curvature_envelope_ratio",
        "support_curvature_envelope_ratio",
        "prior_curvature_envelope_ratio",
    )
    for app in APPLICATIONS:
        for c_exp in C_VALUES:
            subset = sorted(grouped[(app, c_exp)], key=lambda row: int(row["trial_index"]))
            if len(subset) != 32:
                raise ValueError(f"expected 32 held-out rows for {app}, c={c_exp}; got {len(subset)}")
            if any(int(row["validation_passed"]) != 1 for row in subset):
                raise ValueError(f"validation failure in {app}, c={c_exp}")
            record: dict[str, Any] = {
                "campaign_id": app,
                "application": APPLICATIONS[app]["label"],
                "c_exp": c_exp,
                "trials": len(subset),
                "epochs": int(subset[0]["epochs"]),
                "q": int(subset[0]["active_q"]),
                "f_max": float(subset[0]["f_max"]),
                "episode_cap": float(subset[0]["episode_cap"]),
                "universal_ceiling": float(subset[0]["universal_ceiling"]),
                "failed_trials": sum(1 - int(row["validation_passed"]) for row in subset),
            }
            gap_values = [
                float(row["cumulative_reference_value"]) - float(row["cumulative_true_value"])
                for row in subset
            ]
            record["posthoc_greedy_gap_mean"], record["posthoc_greedy_gap_se"] = mean_se(gap_values)
            for metric in scalar_metrics:
                record[f"{metric}_mean"], record[f"{metric}_se"] = mean_se(
                    float(row[metric]) for row in subset
                )
            output.append(record)
    return output


def verify_frozen_performance(
    aggregate: list[dict[str, Any]], reference_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    reference: dict[tuple[str, float], dict[str, str]] = {}
    for row in reference_rows:
        if row["method"] == "Exploitation-only UCB":
            reference[(row["campaign_id"], 0.0)] = row
        elif row["method"] == "Exploration UCB" and close(float(row["c_exp"]), 0.25):
            reference[(row["campaign_id"], 0.25)] = row
    checks: list[dict[str, Any]] = []
    for record in aggregate:
        key = (record["campaign_id"], record["c_exp"])
        old = reference[key]
        comparisons = {
            "cumulative_reference_ratio_mean": float(old["cumulative_exact_greedy_ratio_mean"]),
            "cumulative_reference_ratio_se": float(old["cumulative_exact_greedy_ratio_se"]),
            "posthoc_greedy_gap_mean": float(old["posthoc_exact_greedy_gap_mean"]),
            "posthoc_greedy_gap_se": float(old["posthoc_exact_greedy_gap_se"]),
            "exploration_percent_mean": float(old["exploration_percent_mean"]),
            "exploration_percent_se": float(old["exploration_percent_se"]),
        }
        actual = {
            "cumulative_reference_ratio_mean": record["cumulative_reference_ratio_mean"],
            "cumulative_reference_ratio_se": record["cumulative_reference_ratio_se"],
            "posthoc_greedy_gap_mean": record["posthoc_greedy_gap_mean"],
            "posthoc_greedy_gap_se": record["posthoc_greedy_gap_se"],
            "exploration_percent_mean": 100.0 * record["exploration_fraction_mean"],
            "exploration_percent_se": 100.0 * record["exploration_fraction_se"],
        }
        for metric, expected in comparisons.items():
            value = actual[metric]
            passed = close(value, expected, atol=2.0e-12, rtol=2.0e-12)
            checks.append({
                "campaign_id": key[0], "c_exp": key[1], "metric": metric,
                "reference_value": expected, "recomputed_value": value,
                "absolute_error": abs(value - expected), "status": "PASS" if passed else "FAIL",
            })
            if not passed:
                raise ValueError(
                    f"selection/performance changed for {key}, {metric}: {value} != {expected}"
                )
    return checks


def pooled_curvature(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for app in APPLICATIONS:
        curvatures: list[float] = []
        factors: list[float] = []
        paths = sorted(raw_root.glob(f"exploration_{app}_evaluation_*_C0_EXPLORATION_UCB.csv"))
        if len(paths) != 32:
            raise ValueError(f"expected 32 c=0 raw traces for {app}; got {len(paths)}")
        for path in paths:
            for row in read_csv(path):
                curvatures.append(float(row["empirical_curvature"]))
                factors.append(float(row["comparison_factor"]))
        rows.append({
            "campaign_id": app,
            "episodes": len(curvatures),
            "median_empirical_curvature": float(np.median(curvatures)),
            "median_contextual_comparator_factor": float(np.median(factors)),
            "mean_contextual_comparator_factor": float(np.mean(factors)),
            "normalization_role": "descriptive only; never a cumulative-certificate denominator",
        })
    return rows


def paired_retention(summary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key = {
        (row["campaign_id"], int(row["trial_index"]), float(row["exploration_coefficient"])): row
        for row in summary_rows
        if row["schema_version"] == "exploration-campaign-v2"
    }
    result: list[dict[str, Any]] = []
    for app in APPLICATIONS:
        ratios: list[float] = []
        for trial in range(1, 33):
            base = float(by_key[(app, trial, 0.0)]["cumulative_true_value"])
            shielded = float(by_key[(app, trial, 0.25)]["cumulative_true_value"])
            if base <= 0.0:
                raise ValueError("nonpositive unshielded cumulative true value")
            ratios.append(shielded / base)
        mean, se = mean_se(ratios)
        result.append({
            "campaign_id": app, "trials": len(ratios),
            "paired_shielded_to_unshielded_retention_mean": mean,
            "paired_shielded_to_unshielded_retention_se": se,
        })
    return result


def paired_bootstrap(
    *, app: str, baseline_root: Path, exploration_root: Path
) -> dict[str, Any]:
    realized_diffs: list[float] = []
    true_diffs: list[float] = []
    baseline_paths = sorted(
        baseline_root.glob(f"baseline_{app}_primary_evaluation_*_INSTANTANEOUS_MYOPIC.csv")
    )
    if len(baseline_paths) != 32:
        raise ValueError(f"expected 32 myopic baselines for {app}; got {len(baseline_paths)}")
    for path in baseline_paths:
        baseline = read_csv(path)
        trial = int(baseline[0]["trial_index"])
        seed = baseline[0]["trial_seed"]
        ucb_path = exploration_root / (
            f"exploration_{app}_evaluation_T{trial:02d}_S{seed}_C0_EXPLORATION_UCB.csv"
        )
        ucb = read_csv(ucb_path)
        realized_diffs.append(
            sum(float(row["realized_return"]) for row in ucb)
            - sum(float(row["realized_return"]) for row in baseline)
        )
        true_diffs.append(
            sum(float(row["true_value"]) for row in ucb)
            - sum(float(row["true_value"]) for row in baseline)
        )
    sample = np.asarray(realized_diffs, dtype=float)
    rng = np.random.default_rng(int(APPLICATIONS[app]["bootstrap_seed"]))
    indices = rng.integers(0, sample.size, size=(BOOTSTRAP_REPLICATES, sample.size))
    bootstrap_means = sample[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")
    return {
        "campaign_id": app,
        "comparison": "exploitation-only UCB minus fixed-H myopic",
        "paired_trials": sample.size,
        "mean_paired_realized_return_difference": float(sample.mean()),
        "bootstrap_method": "paired nonparametric percentile bootstrap of the trial-level mean",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": int(APPLICATIONS[app]["bootstrap_seed"]),
        "bootstrap_ci_level": 0.95,
        "bootstrap_ci95_low": float(low),
        "bootstrap_ci95_high": float(high),
        "mean_paired_true_value_difference": float(np.mean(true_diffs)),
    }


def covariance_nonactivation(reference_root: Path) -> list[dict[str, Any]]:
    cov = {row["campaign_id"]: row for row in read_csv(reference_root / "covariance_summary.csv")}
    learn = {row["campaign_id"]: row for row in read_csv(reference_root / "learning_configuration.csv")}
    diag = {row["campaign_id"]: row for row in read_csv(reference_root / "covariance_diagnostic.csv")}
    rows: list[dict[str, Any]] = []
    for app in APPLICATIONS:
        lam = float(cov[app]["mean_final_cumulative_covariance_min_eigenvalue"])
        pi_k = float(cov[app]["mean_expected_exploration_count"])
        d = int(learn[app]["dimension"])
        horizon = int(learn[app]["horizon_epochs"])
        delta = float(learn[app]["confidence_delta"])
        r_u = int(diag[app]["measured_identifiable_rank_rU"])
        p_bar = 0.25
        ell = math.log(3.0 * r_u * horizon / delta)
        penalty = d * math.sqrt(2.0 * pi_k * ell) + (2.0 * d * p_bar / 3.0) * ell
        necessary_mass = 2.0 * (r_u ** 2) * ell
        rows.append({
            "campaign_id": app,
            "horizon_epochs": horizon,
            "identifiable_rank": r_u,
            "expected_exploration_mass_Pi_K": pi_k,
            "freedman_log_factor": ell,
            "necessary_mass_for_positive_floor": necessary_mass,
            "exploration_mass_to_necessary_mass_ratio": pi_k / necessary_mass,
            "mean_terminal_cumulative_covariance_min_eigenvalue": lam,
            "terminal_matrix_freedman_penalty": penalty,
            "mean_covariance_to_penalty_ratio": lam / penalty,
            "positive_design_floor_trials": 0,
            "trials": 32,
            "structural_nonactivation_certified": int(pi_k <= necessary_mass),
            "interpretation": (
                "finite-horizon structural nonactivation on the evaluated traces; "
                "the eventual proportional-growth premise is not falsified"
            ),
        })
    return rows


def write_dat(path: Path, header: str, rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(header.rstrip() + "\n")
        for row in rows:
            handle.write(" ".join(format(value, ".12g") if isinstance(value, float) else str(value) for value in row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "results/generated/summary.csv")
    parser.add_argument("--validation", type=Path, default=ROOT / "results/generated/validation.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "results/generated/raw/exploration")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "results/generated/raw/baseline")
    parser.add_argument("--reference-root", type=Path, default=ROOT / "results/reference")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/generated/evidence")
    parser.add_argument(
        "--figure-data-dir", type=Path,
        default=ROOT / "results/generated/plot_data",
    )
    args = parser.parse_args()

    validation = json.loads(args.validation.read_text())
    if validation.get("status") != "PASS" or validation.get("certificate_v2_files") != 128:
        raise ValueError("the complete 128-file v2 validation must pass before evidence generation")

    summary_rows = read_csv(args.summary)
    aggregate = aggregate_summary(summary_rows)
    performance_checks = verify_frozen_performance(
        aggregate, read_csv(args.reference_root / "campaign_performance.csv")
    )
    curvature = pooled_curvature(args.raw_root)
    retention = paired_retention(summary_rows)
    bootstrap = [
        paired_bootstrap(app=app, baseline_root=args.baseline_root, exploration_root=args.raw_root)
        for app in APPLICATIONS
    ]
    covariance = covariance_nonactivation(args.reference_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_fields = list(aggregate[0].keys())
    write_csv(args.output_dir / "evaluation_certificate_summary.csv", aggregate, aggregate_fields)
    write_csv(args.output_dir / "selection_performance_invariance.csv", performance_checks, list(performance_checks[0].keys()))
    write_csv(args.output_dir / "curvature_descriptive_summary.csv", curvature, list(curvature[0].keys()))
    write_csv(args.output_dir / "paired_retention.csv", retention, list(retention[0].keys()))
    write_csv(args.output_dir / "paired_bootstrap_reconciliation.csv", bootstrap, list(bootstrap[0].keys()))
    write_csv(args.output_dir / "covariance_nonactivation.csv", covariance, list(covariance[0].keys()))

    lookup = {(row["campaign_id"], row["c_exp"]): row for row in aggregate}
    for app, filename in (("SAT-COV-V1", "sat_tradeoff.dat"), ("UAV-AG-V1", "uav_tradeoff.dat")):
        write_dat(
            args.figure_data_dir / filename,
            "c cert cert_se ratio ratio_se",
            [[
                c,
                lookup[(app, c)]["direct_universal_ratio_mean"],
                lookup[(app, c)]["direct_universal_ratio_se"],
                lookup[(app, c)]["cumulative_reference_ratio_mean"],
                lookup[(app, c)]["cumulative_reference_ratio_se"],
            ] for c in C_VALUES],
        )
    route_order = [
        (1, "SAT-COV-V1", 0.0), (2, "SAT-COV-V1", 0.25),
        (3, "UAV-AG-V1", 0.0), (4, "UAV-AG-V1", 0.25),
    ]
    write_dat(
        args.figure_data_dir / "certificate_route_totals.dat",
        "x direct direct_se support support_se prior prior_se",
        [[
            x,
            lookup[(app, c)]["direct_universal_ratio_mean"],
            lookup[(app, c)]["direct_universal_ratio_se"],
            lookup[(app, c)]["support_universal_ratio_mean"],
            lookup[(app, c)]["support_universal_ratio_se"],
            lookup[(app, c)]["prior_universal_ratio_mean"],
            lookup[(app, c)]["prior_universal_ratio_se"],
        ] for x, app, c in route_order],
    )

    all_files = [
        args.summary, args.validation, args.reference_root / "campaign_performance.csv",
        args.reference_root / "covariance_summary.csv",
        args.reference_root / "learning_configuration.csv",
        args.reference_root / "covariance_diagnostic.csv",
    ]
    output_files = sorted(args.output_dir.glob("*.csv")) + [
        args.figure_data_dir / "sat_tradeoff.dat",
        args.figure_data_dir / "uav_tradeoff.dat",
        args.figure_data_dir / "certificate_route_totals.dat",
    ]
    manifest = {
        "schema": "application-evidence-v1",
        "status": "PASS",
        "primary_normalization": "q B_K/(K F_max)",
        "known_universal_ceiling": "K F_max/q",
        "median_curvature_normalization_used": False,
        "exact_optimum_denominator_claimed": False,
        "posthoc_comparator": "centralized maximal greedy under exact true marginals",
        "fixed_trace_scope": "support-derived and prior-only routes recertify immutable traces; they are not target-policy runs",
        "validated_raw_files": validation["certificate_v2_files"],
        "validated_raw_rows": validation["rows_checked"],
        "held_out_trials_per_application": 32,
        "inputs": [{"path": path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.name, "sha256": sha256(path)} for path in all_files],
        "outputs": [{"path": path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.name, "sha256": sha256(path)} for path in output_files],
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "paired nonparametric percentile bootstrap of the trial-level mean",
            "seeds": {app: APPLICATIONS[app]["bootstrap_seed"] for app in APPLICATIONS},
        },
        "selection_performance_invariance_checks": len(performance_checks),
        "selection_performance_invariance_failures": sum(row["status"] != "PASS" for row in performance_checks),
    }
    manifest_path = args.output_dir / "EVIDENCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "aggregate_rows": len(aggregate),
        "invariance_checks": len(performance_checks),
        "manifest": str(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
