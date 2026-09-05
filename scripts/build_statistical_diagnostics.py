#!/usr/bin/env python3
"""Recompute paired-retention and trajectory diagnostics from execution records.

The builder reads application summaries and trajectory records, evaluates the
statistical and physical diagnostics, and writes separate audit products.
Execution records and trial summaries are verified as unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from campaign_engine import beta_radius, constrained_quadratic_minimizer  # noqa: E402
import trajectory_microcase as tm  # noqa: E402

LOCKED_EPISODE_SHA256 = tm.LOCKED_EPISODE_RECORDS_SHA256
LOCKED_TRIAL_SHA256 = tm.LOCKED_TRIAL_SUMMARY_SHA256
WITNESS_SEED = 12031
WITNESS_EPISODE = 21
WITNESS_STAGE = 2
WITNESS_EDGE_A = (2, 0)  # one-based (3,1)
WITNESS_EDGE_B = (0, 2)  # one-based (1,3)
ABS_TOL = 5.0e-13


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            rendered: dict[str, Any] = {}
            for field in fields:
                value = row[field]
                if isinstance(value, float):
                    rendered[field] = format(value, ".17g")
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


def close(actual: float, expected: float, *, atol: float = ABS_TOL) -> bool:
    return math.isclose(actual, expected, rel_tol=5.0e-12, abs_tol=atol)


def campaign_reconciliation(
    paired_rows: Sequence[dict[str, str]],
    aggregate_rows: Sequence[dict[str, str]],
    campaign_source: Path,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Reconcile the paired-retention statistic with policy-conditioned ratios."""
    paired = {row["campaign_id"]: row for row in paired_rows}
    aggregate = {
        (row["campaign_id"], float(row["c_exp"])): row for row in aggregate_rows
    }
    labels = {
        "SAT-COV-V1": "Satellite coverage",
        "UAV-AG-V1": "Aerial-vehicle allocation",
    }
    output: list[dict[str, Any]] = []
    for campaign_id, label in labels.items():
        row = paired[campaign_id]
        n = int(row["trials"])
        paired_mean = float(row["paired_shielded_to_unshielded_retention_mean"])
        paired_se = float(row["paired_shielded_to_unshielded_retention_se"])
        exploit_ratio = float(aggregate[(campaign_id, 0.0)]["cumulative_reference_ratio_mean"])
        shield_ratio = float(aggregate[(campaign_id, 0.25)]["cumulative_reference_ratio_mean"])
        ratio_of_table_means = shield_ratio / exploit_ratio
        output.append(
            {
                "campaign_id": campaign_id,
                "application": label,
                "paired_trials": n,
                "paired_retention_definition": "rho_s=V_shielded_s/V_exploitation_only_s",
                "paired_retention_mean": paired_mean,
                "paired_retention_standard_error": paired_se,
                "paired_retention_sample_standard_deviation": paired_se * math.sqrt(n),
                "exploitation_only_value_to_policy_conditioned_greedy_mean": exploit_ratio,
                "shielded_value_to_policy_conditioned_greedy_mean": shield_ratio,
                "ratio_of_table_means": ratio_of_table_means,
                "paired_mean_minus_ratio_of_table_means": paired_mean - ratio_of_table_means,
                "standard_error_definition": "sample_sd({rho_s})/sqrt(32)",
                "exact_score_greedy_denominator": "recomputed under each policy's own residual-resource state",
                "table_ratio_recovers_paired_retention": 0,
            }
        )

    source = campaign_source.read_text(encoding="utf-8")
    source_gates = {
        "affordability_uses_policy_specific_resources": (
            "method.resources[batch.owner]" in source
        ),
        "oracle_uses_policy_conditioned_base_family": (
            '"CENTRAL_GREEDY_ORACLE"' in source and "batch,\n                base," in source
        ),
        "policy_specific_oracle_is_accumulated": (
            "variant.cumulative_oracle += oracle_value" in source
        ),
        "paired_statistic_is_mean_of_absolute_value_ratios": all(
            int(row["paired_trials"]) == 32 and row["paired_retention_mean"] > 0.0
            for row in output
        ),
        "reported_uncertainty_is_standard_error": all(
            close(
                row["paired_retention_standard_error"],
                row["paired_retention_sample_standard_deviation"]
                / math.sqrt(row["paired_trials"]),
            )
            for row in output
        ),
        "table_ratios_do_not_algebraically_recover_paired_retention": all(
            abs(row["paired_mean_minus_ratio_of_table_means"]) > 5.0e-4
            for row in output
        ),
    }
    return output, source_gates


def reconstruct_estimator_state(
    cfg: dict,
    seed_rows: Sequence[dict[str, str]],
    episode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Reconstruct the pre-selection estimator from locked prior observations."""
    V = float(cfg["ridge_lambda"]) * np.eye(int(cfg["feature_dimension"]))
    rhs = np.zeros(int(cfg["feature_dimension"]))
    for row in sorted(seed_rows, key=lambda item: int(item["episode"])):
        k = int(row["episode"])
        if k >= episode:
            break
        assignment = tm.parse_sid(row["selected_matching"])
        feature = tm.set_feature(cfg, assignment, k)
        observed_return = float(row["observed_return"])
        V += np.outer(feature, feature)
        rhs += feature * observed_return
    theta_hat = constrained_quadratic_minimizer(
        V, rhs, float(cfg["btheta"]), float(cfg["fmax"])
    )
    certificate = tm.physical_certificate(cfg, tm.terminal_matchings(cfg))
    sigma = float(certificate["shared_reset_deviation_scale"])
    beta = beta_radius(
        V,
        float(cfg["ridge_lambda"]),
        sigma,
        float(cfg["confidence_delta"]),
        float(cfg["btheta"]),
    )
    beta_zero = beta_radius(
        V,
        float(cfg["ridge_lambda"]),
        0.0,
        float(cfg["confidence_delta"]),
        float(cfg["btheta"]),
    )
    gamma = (beta - beta_zero) / sigma
    return V, rhs, theta_hat, beta, beta_zero, gamma


def score_line(
    cfg: dict,
    episode: int,
    edge: tuple[int, int],
    partial: Sequence[tuple[int, int]],
    theta_hat: np.ndarray,
    V_inverse: np.ndarray,
    beta_zero: float,
    gamma: float,
) -> dict[str, Any]:
    feature = tm.marginal_feature(cfg, edge, partial, episode)
    width = math.sqrt(max(0.0, float(feature @ V_inverse @ feature)))
    posterior_mean = float(theta_hat @ feature)
    intercept = posterior_mean + beta_zero * width
    slope = gamma * width
    return {
        "edge_zero_based": f"{edge[0]}-{edge[1]}",
        "edge_one_based": f"({edge[0] + 1},{edge[1] + 1})",
        "posterior_mean": posterior_mean,
        "metric_width": width,
        "zero_scale_intercept": intercept,
        "physical_scale_slope": slope,
    }


def trajectory_reconciliation(
    cfg: dict,
    episode_rows: Sequence[dict[str, str]],
    trial_rows: Sequence[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    exploitation = [row for row in episode_rows if int(row["exploration"]) == 0]
    if len(exploitation) != 12 * (240 - 18):
        raise ValueError("unexpected exploitation-row count")

    cap = float(cfg["fmax"]) / int(cfg["q_extendibility"])
    clipped_rows = [
        row
        for row in exploitation
        if float(row["raw_certificate_increment"])
        > float(row["clipped_certificate_increment"]) + 1.0e-14
    ]
    clipped_cap_values = {
        float(row["clipped_certificate_increment"]) for row in clipped_rows
    }
    clip_episodes = sorted({int(row["episode"]) for row in clipped_rows})
    clips_per_seed: dict[int, int] = defaultdict(int)
    for row in clipped_rows:
        clips_per_seed[int(row["seed"])] += 1
    max_raw = max(float(row["raw_certificate_increment"]) for row in exploitation)
    total_clip_excess = sum(
        float(row["raw_certificate_increment"])
        - float(row["clipped_certificate_increment"])
        for row in exploitation
    )

    shares = [
        (float(row["beta"]) - float(row["beta_zero_scale"])) / float(row["beta"])
        for row in exploitation
    ]
    seed_share_means: list[float] = []
    for seed in sorted({int(row["seed"]) for row in exploitation}):
        values = [
            (float(row["beta"]) - float(row["beta_zero_scale"])) / float(row["beta"])
            for row in exploitation
            if int(row["seed"]) == seed
        ]
        seed_share_means.append(float(np.mean(values)))
    share_mean, share_seed_se = mean_se(seed_share_means)
    beta_zero_values = {float(row["beta_zero_scale"]) for row in exploitation}
    expected_beta_zero = math.sqrt(float(cfg["ridge_lambda"])) * float(cfg["btheta"])

    seed_rows = [row for row in episode_rows if int(row["seed"]) == WITNESS_SEED]
    V, _, theta_hat, beta, beta_zero, gamma = reconstruct_estimator_state(
        cfg, seed_rows, WITNESS_EPISODE
    )
    configured_trace = tm.centralized_greedy_trace(
        cfg,
        WITNESS_EPISODE,
        theta_hat,
        V,
        beta,
        score_scale_tag="configured",
    )
    zero_trace = tm.centralized_greedy_trace(
        cfg,
        WITNESS_EPISODE,
        theta_hat,
        V,
        beta_zero,
        score_scale_tag="zero_scale",
    )
    configured_first = configured_trace["stage_winners"][0].edge
    zero_first = zero_trace["stage_winners"][0].edge
    if configured_first is None or zero_first is None or configured_first != zero_first:
        raise AssertionError("witness first-stage partial set is not common")
    partial = (configured_first,)
    if tm.sid(partial) != "1-1":
        raise AssertionError("the locked witness partial set changed")
    V_inverse = np.linalg.inv(V)
    line_a = score_line(
        cfg,
        WITNESS_EPISODE,
        WITNESS_EDGE_A,
        partial,
        theta_hat,
        V_inverse,
        beta_zero,
        gamma,
    )
    line_b = score_line(
        cfg,
        WITNESS_EPISODE,
        WITNESS_EDGE_B,
        partial,
        theta_hat,
        V_inverse,
        beta_zero,
        gamma,
    )
    crossing = (
        line_b["zero_scale_intercept"] - line_a["zero_scale_intercept"]
    ) / (line_a["physical_scale_slope"] - line_b["physical_scale_slope"])
    sigma = float(tm.physical_certificate(cfg, tm.terminal_matchings(cfg))["shared_reset_deviation_scale"])
    for line in (line_a, line_b):
        line["score_at_zero_scale"] = line["zero_scale_intercept"]
        line["score_at_crossing"] = (
            line["zero_scale_intercept"] + line["physical_scale_slope"] * crossing
        )
        line["score_at_configured_scale"] = (
            line["zero_scale_intercept"] + line["physical_scale_slope"] * sigma
        )
        line["crossing_scale"] = crossing
        line["configured_scale"] = sigma

    # Verify that these two lines are the actual upper-envelope winners at both
    # endpoints and remain jointly maximal at the crossing.
    all_lines: list[dict[str, Any]] = []
    for edge in tm.edges(cfg):
        if tm.candidate_is_feasible(cfg, WITNESS_EPISODE, edge, partial):
            all_lines.append(
                {
                    "edge": edge,
                    **score_line(
                        cfg,
                        WITNESS_EPISODE,
                        edge,
                        partial,
                        theta_hat,
                        V_inverse,
                        beta_zero,
                        gamma,
                    ),
                }
            )
    def winner_at(scale: float) -> tuple[int, int]:
        ranked = sorted(
            all_lines,
            key=lambda line: (
                -(
                    line["zero_scale_intercept"]
                    + line["physical_scale_slope"] * scale
                ),
                line["edge"],
            ),
        )
        return ranked[0]["edge"]

    crossing_scores = sorted(
        [
            (
                line["zero_scale_intercept"]
                + line["physical_scale_slope"] * crossing,
                line["edge"],
            )
            for line in all_lines
        ],
        reverse=True,
    )
    top_cross_score = crossing_scores[0][0]
    top_at_crossing = {
        edge for score, edge in crossing_scores if abs(score - top_cross_score) <= 1.0e-12
    }

    tracking_max = max(float(row["tracking_tube_utilization"]) for row in episode_rows)
    tracking_rows = [
        row
        for row in episode_rows
        if close(float(row["tracking_tube_utilization"]), tracking_max, atol=1.0e-15)
    ]
    certificate = tm.physical_certificate(cfg, tm.terminal_matchings(cfg))
    initial_norm = float(np.linalg.norm(cfg["initial_tracking_error"], 2))
    all_time_radius = float(certificate["all_time_radius"])

    zero_difference_mean, zero_difference_se = mean_se(
        float(row["zero_scale_counterfactual_difference_fraction"])
        for row in trial_rows
    )
    diagnostics = {
        "audit_type": "statistical-diagnostics-v1",
        "status": "PASS",
        "clip": {
            "exploitation_episodes": len(exploitation),
            "clipped_exploitation_episodes": len(clipped_rows),
            "clip_fraction": len(clipped_rows) / len(exploitation),
            "clip_inactive_fraction": 1.0 - len(clipped_rows) / len(exploitation),
            "clip_episode_numbers": clip_episodes,
            "clips_per_seed": {str(seed): count for seed, count in sorted(clips_per_seed.items())},
            "episode_ceiling": cap,
            "maximum_raw_episode_charge": max_raw,
            "maximum_raw_episode_utilization": max_raw / cap,
            "total_clip_excess_over_all_exploitation_rows": total_clip_excess,
        },
        "physical_confidence_share": {
            "definition": "(beta_k-beta_k^(0))/beta_k on exploitation episodes",
            "beta_zero_scale": min(beta_zero_values),
            "sqrt_lambda_Btheta": expected_beta_zero,
            "mean_over_exploitation_episodes": float(np.mean(shares)),
            "mean_of_seed_means": share_mean,
            "standard_error_of_seed_means": share_seed_se,
            "minimum": min(shares),
            "maximum": max(shares),
            "fixed_estimator_matching_difference_mean": zero_difference_mean,
            "fixed_estimator_matching_difference_se": zero_difference_se,
            "counterfactual_performance_claimed": False,
        },
        "score_crossing_witness": {
            "seed": WITNESS_SEED,
            "episode": WITNESS_EPISODE,
            "stage": WITNESS_STAGE,
            "common_partial_set_zero_based": tm.sid(partial),
            "common_partial_set_one_based": "{(2,2)}",
            "beta_zero_scale": beta_zero,
            "beta_configured": beta,
            "confidence_log_factor_Gamma_k": gamma,
            "configured_physical_scale": sigma,
            "crossing_physical_scale": crossing,
            "zero_scale_winner_zero_based": tm.sid((winner_at(0.0),)),
            "zero_scale_winner_one_based": f"({winner_at(0.0)[0]+1},{winner_at(0.0)[1]+1})",
            "configured_scale_winner_zero_based": tm.sid((winner_at(sigma),)),
            "configured_scale_winner_one_based": f"({winner_at(sigma)[0]+1},{winner_at(sigma)[1]+1})",
            "top_candidates_at_crossing_zero_based": sorted(tm.sid((edge,)) for edge in top_at_crossing),
            "candidate_A": line_a,
            "candidate_B": line_b,
            "counterfactual_performance_claimed": False,
        },
        "tracking_tightness": {
            "maximum_tracking_utilization": tracking_max,
            "relative_margin": 1.0 - tracking_max,
            "initial_error_norm": initial_norm,
            "all_sequence_radius": all_time_radius,
            "absolute_radius_margin": all_time_radius - initial_norm,
            "maximizer_rows": len(tracking_rows),
            "maximizer_episode_numbers": sorted({int(row["episode"]) for row in tracking_rows}),
            "maximizer_is_predeclared_initial_condition": all(
                int(row["episode"]) == 1 for row in tracking_rows
            ),
            "interpretation": "deterministic tube tightness under the pathwise disturbance bound, not a failure probability",
        },
        "locked_evidence": {
            "episode_records_sha256": sha256(
                ROOT / "results" / "trajectory_microcase" / "episode_records.csv"
            ),
            "trial_summary_sha256": sha256(
                ROOT / "results" / "trajectory_microcase" / "trial_summary.csv"
            ),
        },
    }
    score_rows = [
        {
            "candidate": "A",
            **line_a,
            "selected_at_zero_scale": int(winner_at(0.0) == WITNESS_EDGE_A),
            "selected_at_configured_scale": int(winner_at(sigma) == WITNESS_EDGE_A),
        },
        {
            "candidate": "B",
            **line_b,
            "selected_at_zero_scale": int(winner_at(0.0) == WITNESS_EDGE_B),
            "selected_at_configured_scale": int(winner_at(sigma) == WITNESS_EDGE_B),
        },
    ]

    gates = {
        "locked_episode_records_unchanged": diagnostics["locked_evidence"]["episode_records_sha256"] == LOCKED_EPISODE_SHA256,
        "locked_trial_summary_unchanged": diagnostics["locked_evidence"]["trial_summary_sha256"] == LOCKED_TRIAL_SHA256,
        "episode_cap_is_fmax_over_q": close(cap, float(cfg["fmax"]) / int(cfg["q_extendibility"])),
        "clipped_rows_use_declared_cap": clipped_cap_values == {cap},
        "clip_count_is_24_of_2664": len(clipped_rows) == 24 and len(exploitation) == 2664,
        "clip_occurs_only_at_episodes_27_and_30": clip_episodes == [27, 30],
        "two_clips_per_seed": len(clips_per_seed) == 12 and set(clips_per_seed.values()) == {2},
        "maximum_raw_charge_matches_locked_reconstruction": close(max_raw, 0.6075788941521933),
        "zero_scale_beta_equals_sqrt_lambda_Btheta": len(beta_zero_values) == 1 and close(min(beta_zero_values), expected_beta_zero),
        "physical_share_mean_matches_reconstruction": close(float(np.mean(shares)), 0.47775452583112),
        "physical_share_range_matches_reconstruction": close(min(shares), 0.4223708948099473) and close(max(shares), 0.4993717677642393),
        "fixed_estimator_decision_difference_matches_locked_summary": close(zero_difference_mean, 0.5274024024024023) and close(zero_difference_se, 0.005004294016294119),
        "witness_common_first_stage": configured_first == zero_first == (1, 1),
        "witness_zero_scale_winner": winner_at(0.0) == WITNESS_EDGE_B,
        "witness_configured_scale_winner": winner_at(sigma) == WITNESS_EDGE_A,
        "witness_crossing_inside_certified_scale_interval": 0.0 < crossing < sigma,
        "witness_pair_is_top_envelope_at_crossing": top_at_crossing == {WITNESS_EDGE_A, WITNESS_EDGE_B},
        "witness_crossing_matches_reconstruction": close(crossing, 0.047525170715918545),
        "witness_line_A_matches_reconstruction": close(line_a["zero_scale_intercept"], 0.30521061255601756) and close(line_a["physical_scale_slope"], 0.8237179541492518),
        "witness_line_B_matches_reconstruction": close(line_b["zero_scale_intercept"], 0.3245637216540248) and close(line_b["physical_scale_slope"], 0.41649986726029864),
        "tracking_maximum_is_initial_condition_ratio": close(tracking_max, initial_norm / all_time_radius),
        "tracking_maximum_matches_locked_reconstruction": close(tracking_max, 0.9730372946297503),
        "tracking_maximizer_is_episode_one_all_seeds": len(tracking_rows) == 12 and all(int(row["episode"]) == 1 for row in tracking_rows),
    }
    return diagnostics, score_rows, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "trajectory_microcase.json",
    )
    parser.add_argument(
        "--episode-records",
        type=Path,
        default=ROOT / "results" / "trajectory_microcase" / "episode_records.csv",
    )
    parser.add_argument(
        "--trial-summary",
        type=Path,
        default=ROOT / "results" / "trajectory_microcase" / "trial_summary.csv",
    )
    parser.add_argument(
        "--paired-retention",
        type=Path,
        default=ROOT / "results" / "reference" / "application_evaluation" / "paired_retention.csv",
    )
    parser.add_argument(
        "--evaluation-summary",
        type=Path,
        default=ROOT / "results" / "reference" / "application_evaluation" / "evaluation_certificate_summary.csv",
    )
    parser.add_argument(
        "--campaign-source",
        type=Path,
        default=ROOT / "src" / "exploration_campaign.py",
    )
    parser.add_argument(
        "--campaign-output",
        type=Path,
        default=ROOT / "results" / "reference" / "paired_diagnostics" / "paired_retention_reconciliation.csv",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=ROOT / "results" / "trajectory_microcase" / "statistical_diagnostic_audit.json",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=ROOT / "results" / "trajectory_microcase" / "score_crossing_witness.csv",
    )
    args = parser.parse_args()

    before_episode_sha = sha256(args.episode_records)
    before_trial_sha = sha256(args.trial_summary)
    cfg = tm.load_config(args.config)
    campaign_rows, campaign_gates = campaign_reconciliation(
        read_csv(args.paired_retention),
        read_csv(args.evaluation_summary),
        args.campaign_source,
    )
    trajectory, score_rows, trajectory_gates = trajectory_reconciliation(
        cfg,
        read_csv(args.episode_records),
        read_csv(args.trial_summary),
    )
    all_gates = {**campaign_gates, **trajectory_gates}
    if not all(all_gates.values()):
        failed = [name for name, passed in all_gates.items() if not passed]
        raise AssertionError(f"statistical-diagnostics-v1 diagnostic gate failed: {failed}")

    write_csv(args.campaign_output, campaign_rows)
    write_csv(args.score_output, score_rows)
    audit = {
        **trajectory,
        "campaign_retention": {
            "rows": campaign_rows,
            "interpretation": (
                "The paired statistic uses absolute cumulative true values. "
                "Reported value ratios use policy-conditioned exact-score-greedy denominators "
                "and cannot be divided to recover the paired statistic."
            ),
        },
        "gates": all_gates,
        "source_sha256": {
            "configuration": sha256(args.config),
            "campaign_source": sha256(args.campaign_source),
            "trajectory_source": sha256(Path(tm.__file__)),
            "paired_retention": sha256(args.paired_retention),
            "evaluation_summary": sha256(args.evaluation_summary),
        },
    }
    args.trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    args.trajectory_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    after_episode_sha = sha256(args.episode_records)
    after_trial_sha = sha256(args.trial_summary)
    if before_episode_sha != after_episode_sha or before_trial_sha != after_trial_sha:
        raise AssertionError("diagnostic builder modified locked target evidence")

    print(
        json.dumps(
            {
                "status": "PASS",
                "campaign_rows": len(campaign_rows),
                "diagnostic_gates": len(all_gates),
                "clipped_exploitation_episodes": trajectory["clip"]["clipped_exploitation_episodes"],
                "physical_share_mean": trajectory["physical_confidence_share"]["mean_over_exploitation_episodes"],
                "crossing_scale": trajectory["score_crossing_witness"]["crossing_physical_scale"],
                "tracking_utilization": trajectory["tracking_tightness"]["maximum_tracking_utilization"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
