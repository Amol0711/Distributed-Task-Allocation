#!/usr/bin/env python3
"""Create compact, route-resolved summaries from generated simulation CSV files."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

SUMMARY_FIELDS = [
    "record_type", "schema_version", "campaign_id", "partition",
    "trial_index", "trial_seed", "method_or_variant",
    "exploration_coefficient", "epochs", "f_max", "episode_cap",
    "universal_ceiling", "cumulative_true_value",
    "cumulative_realized_return", "cumulative_reference_value",
    "cumulative_reference_ratio", "exploration_fraction", "active_q",
    "fixed_comparator_factor", "mean_contextual_comparator_factor",
    "curvature_aware_value_envelope",
    "direct_observable_certificate", "direct_universal_ratio",
    "direct_raw_universal_ratio", "direct_curvature_envelope_ratio",
    "direct_raw_exploitation_charge",
    "direct_clipped_exploitation_charge", "direct_clip_count",
    "direct_clip_fraction", "direct_clip_fraction_all", "direct_clip_excess",
    "support_observable_certificate", "support_universal_ratio",
    "support_raw_universal_ratio", "support_curvature_envelope_ratio",
    "support_raw_exploitation_charge",
    "support_clipped_exploitation_charge", "support_clip_count",
    "support_clip_fraction", "support_clip_fraction_all", "support_clip_excess",
    "support_transfer_increment",
    "prior_observable_certificate", "prior_universal_ratio",
    "prior_raw_universal_ratio", "prior_curvature_envelope_ratio",
    "prior_raw_exploitation_charge",
    "prior_clipped_exploitation_charge", "prior_clip_count",
    "prior_clip_fraction", "prior_clip_fraction_all", "prior_clip_excess",
    "prior_transfer_increment",
    "terminal_design_min_eigenvalue", "terminal_design_rank",
    "mean_tracking_rms", "maximum_tracking_peak", "resource_violations",
    "family_violations", "distributed_mismatches", "fallback_episodes",
    "validation_passed", "source_file",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty raw result file: {path}")
    return rows


def f(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def i(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def mean(values: Iterable[float]) -> float:
    values = [value for value in values if math.isfinite(value)]
    return sum(values) / len(values) if values else math.nan


def common_validation(rows: list[dict[str, str]]) -> tuple[int, int, int, int, bool]:
    violations = sum(i(row.get("resource_violations")) for row in rows)
    family = sum(i(row.get("complete_family_violations")) for row in rows)
    mismatches = sum(i(row.get("central_distributed_mismatch")) for row in rows)
    fallback = sum(i(row.get("fallback_episode")) for row in rows)
    margins_ok = all(
        f(row.get(name), math.inf) >= -1.0e-8
        for row in rows
        for name in (
            "finite_time_envelope_slack",
            "control_limit_margin",
            "model_validity_margin",
        )
    )
    valid = all(i(row.get("assignment_valid")) == 1 for row in rows)
    return violations, family, mismatches, fallback, bool(valid and margins_ok)


def summarize_baseline(
    path: Path, rows: list[dict[str, str]], raw_root: Path
) -> dict[str, object]:
    first = rows[0]
    true_total = sum(f(row.get("true_value")) for row in rows)
    return_total = sum(f(row.get("realized_return")) for row in rows)
    reference_total = sum(f(row.get("oracle_greedy_value")) for row in rows)
    ratio = true_total / reference_total if reference_total > 0.0 else 1.0
    violations, family, mismatches, fallback, common_ok = common_validation(rows)
    return {
        "record_type": "baseline",
        "schema_version": first["schema_version"],
        "campaign_id": first["campaign_id"],
        "partition": first["partition"],
        "trial_index": i(first["trial_index"]),
        "trial_seed": i(first["trial_seed"]),
        "method_or_variant": first["method_id"],
        "exploration_coefficient": "",
        "epochs": len(rows),
        "cumulative_true_value": true_total,
        "cumulative_realized_return": return_total,
        "cumulative_reference_value": reference_total,
        "cumulative_reference_ratio": ratio,
        "exploration_fraction": 0.0,
        "terminal_design_min_eigenvalue": "",
        "terminal_design_rank": "",
        "mean_tracking_rms": mean(f(row.get("tracking_rms"), math.nan) for row in rows),
        "maximum_tracking_peak": max(f(row.get("tracking_peak")) for row in rows),
        "resource_violations": violations,
        "family_violations": family,
        "distributed_mismatches": mismatches,
        "fallback_episodes": fallback,
        "validation_passed": int(common_ok and violations == family == mismatches == 0),
        "source_file": path.relative_to(raw_root).as_posix(),
    }


def route_metrics(
    *, rows: list[dict[str, str]], prefix: str, envelope: float,
    universal_ceiling: float
) -> dict[str, float | int]:
    last = rows[-1]
    if prefix:
        total = f(last[f"{prefix}cumulative_observable_bound"])
        ratio = f(last[f"{prefix}universal_normalized_observable_bound"])
        raw = f(last[f"{prefix}cumulative_raw_ucb_bound"])
        clipped = f(last[f"{prefix}cumulative_ucb_bound"])
        clips = i(last[f"{prefix}cumulative_clipped_episodes"])
        excess = f(last[f"{prefix}cumulative_clip_excess"])
        transfer = sum(f(row[f"{prefix}transfer_increment"]) for row in rows)
    else:
        total = f(last["cumulative_observable_bound"])
        ratio = f(last["universal_normalized_observable_bound"])
        raw = f(last["cumulative_raw_ucb_bound"])
        clipped = f(last["cumulative_ucb_bound"])
        clips = i(last["cumulative_clipped_episodes"])
        excess = f(last["cumulative_clip_excess"])
        transfer = 0.0
    exploration_total = f(last[f"{prefix}cumulative_exploration_bound"])
    raw_total = exploration_total + raw
    exploitation_episodes = sum(1 - i(row["exploration_indicator"]) for row in rows)
    return {
        "observable_certificate": total,
        "universal_ratio": ratio,
        "raw_universal_ratio": (raw_total / universal_ceiling
                                if universal_ceiling > 0.0 else math.nan),
        "curvature_envelope_ratio": total / envelope if envelope > 0.0 else math.nan,
        "raw_exploitation_charge": raw,
        "clipped_exploitation_charge": clipped,
        "clip_count": clips,
        "clip_fraction": (clips / exploitation_episodes
                          if exploitation_episodes > 0 else 0.0),
        "clip_fraction_all": clips / len(rows),
        "clip_excess": excess,
        "transfer_increment": transfer,
    }


def summarize_exploration_v2(
    path: Path, rows: list[dict[str, str]], raw_root: Path
) -> dict[str, object]:
    first, last = rows[0], rows[-1]
    violations, family, mismatches, fallback, common_ok = common_validation(rows)
    arithmetic = all(i(row.get("bound_arithmetic_holds"), 0) == 1 for row in rows)
    f_max = f(last["f_max"])
    q = i(last["active_q"])
    episode_cap = f(last["certificate_cap"])
    universal_ceiling = len(rows) * episode_cap
    envelope = f_max * sum(f(row["comparison_factor"]) for row in rows)
    direct = route_metrics(rows=rows, prefix="", envelope=envelope,
                           universal_ceiling=universal_ceiling)
    support = route_metrics(rows=rows, prefix="support_", envelope=envelope,
                            universal_ceiling=universal_ceiling)
    prior = route_metrics(rows=rows, prefix="prior_", envelope=envelope,
                          universal_ceiling=universal_ceiling)
    payload: dict[str, object] = {
        "record_type": "exploration",
        "schema_version": first["schema_version"],
        "campaign_id": first["campaign_id"],
        "partition": first["partition"],
        "trial_index": i(first["trial_index"]),
        "trial_seed": i(first["trial_seed"]),
        "method_or_variant": first["variant_id"],
        "exploration_coefficient": f(first["c_exp"]),
        "epochs": len(rows),
        "f_max": f_max,
        "episode_cap": episode_cap,
        "universal_ceiling": universal_ceiling,
        "cumulative_true_value": f(last["cumulative_true_value"]),
        "cumulative_realized_return": sum(f(row["realized_return"]) for row in rows),
        "cumulative_reference_value": f(last["cumulative_oracle_value"]),
        "cumulative_reference_ratio": f(last["cumulative_oracle_ratio"]),
        "exploration_fraction": f(last["realized_exploration_count"]) / len(rows),
        "active_q": q,
        "fixed_comparator_factor": f(last["fixed_comparator_factor"]),
        "mean_contextual_comparator_factor": mean(
            f(row["comparison_factor"]) for row in rows
        ),
        "curvature_aware_value_envelope": envelope,
        "terminal_design_min_eigenvalue": f(last["design_min_eigenvalue_unregularized"]),
        "terminal_design_rank": i(last["design_rank_unregularized"]),
        "mean_tracking_rms": mean(f(row["tracking_rms"]) for row in rows),
        "maximum_tracking_peak": max(f(row["tracking_peak"]) for row in rows),
        "resource_violations": violations,
        "family_violations": family,
        "distributed_mismatches": mismatches,
        "fallback_episodes": fallback,
        "validation_passed": int(
            common_ok
            and arithmetic
            and violations == family == mismatches == 0
            and 0.0 <= float(direct["universal_ratio"]) <= 1.0 + 1.0e-9
            and 0.0 <= float(support["universal_ratio"]) <= 1.0 + 1.0e-9
            and 0.0 <= float(prior["universal_ratio"]) <= 1.0 + 1.0e-9
        ),
        "source_file": path.relative_to(raw_root).as_posix(),
    }
    for label, metrics in (("direct", direct), ("support", support), ("prior", prior)):
        payload[f"{label}_observable_certificate"] = metrics["observable_certificate"]
        payload[f"{label}_universal_ratio"] = metrics["universal_ratio"]
        payload[f"{label}_raw_universal_ratio"] = metrics["raw_universal_ratio"]
        payload[f"{label}_curvature_envelope_ratio"] = metrics["curvature_envelope_ratio"]
        payload[f"{label}_raw_exploitation_charge"] = metrics["raw_exploitation_charge"]
        payload[f"{label}_clipped_exploitation_charge"] = metrics["clipped_exploitation_charge"]
        payload[f"{label}_clip_count"] = metrics["clip_count"]
        payload[f"{label}_clip_fraction"] = metrics["clip_fraction"]
        payload[f"{label}_clip_fraction_all"] = metrics["clip_fraction_all"]
        payload[f"{label}_clip_excess"] = metrics["clip_excess"]
        if label != "direct":
            payload[f"{label}_transfer_increment"] = metrics["transfer_increment"]
    return payload


def summarize_exploration_v1(
    path: Path, rows: list[dict[str, str]], raw_root: Path
) -> dict[str, object]:
    """Read the version-one raw simulation schema."""
    first, last = rows[0], rows[-1]
    violations, family, mismatches, fallback, common_ok = common_validation(rows)
    return {
        "record_type": "exploration-v1",
        "schema_version": first["schema_version"],
        "campaign_id": first["campaign_id"],
        "partition": first["partition"],
        "trial_index": i(first["trial_index"]),
        "trial_seed": i(first["trial_seed"]),
        "method_or_variant": first["variant_id"],
        "exploration_coefficient": f(first["c_exp"]),
        "epochs": len(rows),
        "cumulative_true_value": f(last["cumulative_true_value"]),
        "cumulative_realized_return": sum(f(row.get("realized_return")) for row in rows),
        "cumulative_reference_value": f(last["cumulative_oracle_value"]),
        "cumulative_reference_ratio": f(last["cumulative_oracle_ratio"]),
        "exploration_fraction": f(last["realized_exploration_count"]) / len(rows),
        "mean_tracking_rms": mean(f(row.get("tracking_rms"), math.nan) for row in rows),
        "maximum_tracking_peak": max(f(row.get("tracking_peak")) for row in rows),
        "resource_violations": violations,
        "family_violations": family,
        "distributed_mismatches": mismatches,
        "fallback_episodes": fallback,
        "validation_passed": int(common_ok and violations == family == mismatches == 0),
        "source_file": path.relative_to(raw_root).as_posix(),
    }


def render(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path,
        default=ROOT / "results" / "generated" / "raw",
        help="directory containing raw baseline and exploration CSV files",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "generated" / "summary.csv",
        help="summary CSV path",
    )
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    paths = sorted(raw_root.rglob("*.csv")) if raw_root.exists() else []
    if not paths:
        raise SystemExit(f"no raw CSV files found under {raw_root}")

    summaries: list[dict[str, object]] = []
    for path in paths:
        rows = read_rows(path)
        schema = rows[0].get("schema_version", "")
        if schema == "baseline-campaign-v1":
            summaries.append(summarize_baseline(path, rows, raw_root))
        elif schema == "exploration-campaign-v1":
            summaries.append(summarize_exploration_v1(path, rows, raw_root))
        elif schema == "exploration-campaign-v2":
            summaries.append(summarize_exploration_v2(path, rows, raw_root))
        else:
            raise ValueError(f"unsupported schema {schema!r} in {path}")

    summaries.sort(
        key=lambda row: (
            str(row["campaign_id"]), int(row["trial_index"]),
            str(row["record_type"]), str(row["method_or_variant"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: render(row.get(key, "")) for key in SUMMARY_FIELDS})
    print(f"PASS: wrote {len(summaries)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
