#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_EPISODE_RECORDS_SHA256 = (
    "fd953e541ae54c428096ffbbfd6a71d32bcaf724ee93459418cc0a0394be7b16"
)
EXPECTED_TRIAL_SUMMARY_SHA256 = (
    "6188e4a675d89c3f30c6df28a20f9313689ad29398d97951c420177cadfdb8df"
)

from trajectory_microcase import (
    allocation_graph_diameter,
    allocation_graph_neighbors,
    edges,
    execution_mode,
    execution_reference,
    execute,
    exploration_codebook_rows,
    exploration_schedule_indices,
    exploration_schedule_rows,
    fallback_mode,
    fallback_reference,
    feature_geometry_audit,
    hereditary_matchings,
    is_terminal_matching,
    load_config,
    public_exploration_codebook,
    resource_screened_edges,
    sha256,
    static_audit,
    terminal_matchings,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_matching(serialized: str) -> tuple[tuple[int, int], ...]:
    if not serialized:
        return ()
    return tuple(
        tuple(int(value) for value in item.split("-"))
        for item in serialized.split(";")
    )


if __name__ == "__main__":
    config_path = ROOT / "configs" / "trajectory_microcase.json"
    output_dir = ROOT / "results" / "trajectory_microcase"
    cfg = load_config(config_path)
    static = static_audit(cfg)
    geometry = feature_geometry_audit(cfg)
    summary = json.loads((output_dir / "summary.json").read_text())
    closure = json.loads((output_dir / "model_closure_audit.json").read_text())
    protocol_audit = json.loads(
        (output_dir / "distributed_reproducibility_audit.json").read_text()
    )
    manifest = json.loads((output_dir / "MANIFEST.json").read_text())

    require(
        allocation_graph_neighbors(cfg) == {0: (1,), 1: (0, 2), 2: (1,)},
        "allocation path graph",
    )
    require(allocation_graph_diameter(cfg) == 2, "allocation graph diameter")
    require(
        cfg["allocation_network"]["diameter_upper_bound"]
        == cfg["allocation_network"]["rounds_per_stage"]
        == 2,
        "diameter-round closure",
    )
    require(
        cfg["allocation_network"]["score_encoding"] == "identity"
        and cfg["allocation_network"]["encoding_error"] == 0.0,
        "identity encoder",
    )
    require(
        all(
            resource_screened_edges(cfg, episode) == edges(cfg)
            for episode in range(1, cfg["episodes"] + 1)
        ),
        "nonbinding resource screen",
    )

    codebook = public_exploration_codebook(cfg)
    schedule_indices = exploration_schedule_indices(cfg)
    require(codebook == terminal_matchings(cfg), "public codebook terminal family")
    require(len(codebook) == 60, "public codebook size")
    require(
        schedule_indices
        == (3, 10, 17, 24, 31, 38, 45, 52, 59, 6, 13, 20, 27, 34, 41, 48, 55, 2),
        "public exploration schedule",
    )
    codebook_path = ROOT / "seeds" / "trajectory_microcase_codebook.csv"
    schedule_path = ROOT / "seeds" / "trajectory_microcase_exploration_schedule.csv"
    with codebook_path.open(newline="", encoding="utf-8") as handle:
        codebook_asset = list(csv.DictReader(handle))
    with schedule_path.open(newline="", encoding="utf-8") as handle:
        schedule_asset = list(csv.DictReader(handle))
    require(
        codebook_asset
        == [
            {key: str(value) for key, value in row.items()}
            for row in exploration_codebook_rows(cfg)
        ],
        "public codebook asset",
    )
    require(
        schedule_asset
        == [
            {key: str(value) for key, value in row.items()}
            for row in exploration_schedule_rows(cfg)
        ],
        "public schedule asset",
    )

    require(len(hereditary_matchings(cfg)) == 136, "hereditary matching count")
    require(len(terminal_matchings(cfg)) == 60, "terminal matching count")
    require(
        geometry["hereditary_cardinality_counts"]
        == {"0": 1, "1": 15, "2": 60, "3": 60},
        "hereditary cardinality counts",
    )
    require(geometry["terminal_assignment_cardinality"] == 3, "terminal cardinality")
    require(static["matching_count"] == 60, "matching count alias")
    require(geometry["terminal_family_equals_maximal_family"], "terminal=maximal")
    require(
        geometry["edge_feature_coordinate_lower_bound"] == 0.12,
        "analytic edge-feature lower bound",
    )
    require(
        geometry["edge_feature_coordinate_upper_bound"] == 0.28,
        "analytic edge-feature upper bound",
    )
    require(
        geometry["analytical_feasible_raw_coordinate_upper_bound"]
        == 0.8400000000000001,
        "analytic hereditary-family cap bound",
    )
    require(
        geometry["analytical_ground_without_edge_coordinate_lower_bound"] == 1.68,
        "analytic full-ground-complement saturation bound",
    )
    require(
        geometry["maximum_feasible_raw_coordinate"]
        < cfg["feature_saturation_cap"],
        "saturation inactive on hereditary family",
    )
    require(
        geometry["maximum_feasible_saturation_residual"] == 0.0,
        "zero feasible saturation residual",
    )
    require(
        geometry["maximum_feasible_marginal_residual"] == 0.0,
        "implemented feasible marginals unchanged",
    )
    require(
        geometry["maximum_feasible_formula_marginal_residual"]
        <= 6e-17,
        "formula marginal floating residual",
    )
    require(
        math.isclose(
            geometry["maximum_feasible_raw_coordinate"],
            0.8400000000000001,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "maximum feasible raw coordinate",
    )
    require(
        math.isclose(
            geometry["feasible_saturation_cap_margin"],
            0.15999999999999992,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "feasible cap margin",
    )
    require(
        geometry["minimum_ground_without_edge_raw_coordinate"]
        > cfg["feature_saturation_cap"],
        "global terminal marginals saturated",
    )
    for name, expected in (
        ("minimum_ground_raw_coordinate", 1.9499072026525104),
        ("maximum_ground_raw_coordinate", 4.05162276916821),
        ("minimum_ground_without_edge_raw_coordinate", 1.7926979366123499),
        ("minimum_singleton_value", 0.19287919304680923),
    ):
        require(
            math.isclose(geometry[name], expected, rel_tol=0.0, abs_tol=5e-15),
            name,
        )
    require(geometry["curvature_equals_one_all_episodes"], "unit curvature")
    require(
        geometry["contextual_greedy_factor"]
        == geometry["fixed_greedy_factor"]
        == 1 / 3,
        "alpha=1/3",
    )
    require(cfg["q_extendibility"] == 2, "q")
    require(cfg["feature_saturation_cap"] == 1.0, "feature cap")
    require(cfg["terminal_assignment_cardinality"] == 3, "terminal cardinality config")
    require(cfg["physical_blocks"] == 1, "physical blocks")
    require(cfg["physical_state_dimension"] == 2, "physical state dimension")
    require(cfg["disturbance_dimension"] == 2, "disturbance dimension")
    require(cfg["fallback_value"] == 0.0, "zero-benefit fallback")
    require(cfg["fallback_mode"] == 0, "fallback mode")
    require(
        cfg["fallback_reference"].tolist() == [0.0, 0.0],
        "fallback reference",
    )
    require(execution_mode(cfg, tuple()) == fallback_mode(cfg), "fallback mode routing")
    require(
        execution_reference(cfg, tuple()).tolist()
        == fallback_reference(cfg).tolist(),
        "fallback reference routing",
    )
    require(cfg["episodes"] == 240, "K")
    require(cfg["physical_horizon"] == 8, "H")
    require(len(cfg["evaluation_seeds"]) == 12, "seeds")
    require(
        len(cfg["controller_templates"]) == len(cfg["disturbance_input_templates"]) == 4,
        "A/E template libraries",
    )
    require(
        math.isclose(
            static["shared_reset_deviation_scale"],
            0.0548271522404341,
            rel_tol=0.0,
            abs_tol=5e-15,
        ),
        "shared-reset deviation scale",
    )
    require(
        math.isclose(
            static["shared_reset_deviation_scale"],
            static["trajectory_sigma"],
            rel_tol=0.0,
            abs_tol=5e-15,
        ),
        "general/linear scale identity",
    )
    require(static["deviation_scale_identity_residual"] <= 5e-15, "identity residual")
    require(
        static["minimum_stage_reward_lower_bound"] > 0.0,
        "nonnegative stage-reward lower bound",
    )
    require(static["rho_H"] < 1.0, "episode-boundary contraction")

    require(
        sha256(output_dir / "episode_records.csv")
        == EXPECTED_EPISODE_RECORDS_SHA256,
        "locked episode-record bytes",
    )
    require(
        sha256(output_dir / "trial_summary.csv")
        == EXPECTED_TRIAL_SUMMARY_SHA256,
        "locked trial-summary bytes",
    )
    for name, expected_hash in manifest["files"].items():
        require(sha256(output_dir / name) == expected_hash, f"manifest {name}")

    rows = list(csv.DictReader((output_dir / "episode_records.csv").open()))
    trials = list(csv.DictReader((output_dir / "trial_summary.csv").open()))
    protocol_rows = list(
        csv.DictReader((output_dir / "distributed_protocol_records.csv").open())
    )
    feedback_rows = list(
        csv.DictReader((output_dir / "feedback_flood_records.csv").open())
    )
    require(len(rows) == 2880, "rows")
    require(len(trials) == 12, "trials")
    require(len(protocol_rows) == 5544, "distributed protocol rows")
    require(len(feedback_rows) == 2880, "feedback flood rows")
    for column in ("selected_matching", "zero_scale_matching", "optimal_matching"):
        require(
            all(is_terminal_matching(cfg, parse_matching(row[column])) for row in rows),
            f"all {column} assignments terminal",
        )
    require(closure["status"] == "PASS", "model closure status")
    require(closure["model_interface_version"] == "terminal-model-v1", "model closure version")
    require(closure["episode_rows"] == 2880, "model closure row count")
    require(closure["fallback_invocations"] == 0, "locked fallback invocations")
    require(
        closure["family"]
        == {
            "ground_edge_count": 15,
            "hereditary_assignment_count": 136,
            "hereditary_cardinality_counts": {
                "0": 1,
                "1": 15,
                "2": 60,
                "3": 60,
            },
            "terminal_assignment_count": 60,
            "terminal_cardinality": 3,
            "terminal_family_equals_maximal_family": True,
        },
        "model closure family",
    )
    structured = closure["structured_value"]
    require(
        structured["basis"] == "coordinatewise_min_of_additive_load_and_one",
        "closure capped basis",
    )
    require(structured["feature_saturation_cap"] == 1.0, "closure feature cap")
    require(
        structured["edge_feature_coordinate_lower_bound"] == 0.12,
        "closure analytic lower bound",
    )
    require(
        structured["edge_feature_coordinate_upper_bound"] == 0.28,
        "closure analytic upper bound",
    )
    require(
        structured["analytical_feasible_raw_coordinate_upper_bound"]
        == 0.8400000000000001,
        "closure analytic feasible bound",
    )
    require(
        structured["analytical_ground_without_edge_coordinate_lower_bound"] == 1.68,
        "closure analytic saturation bound",
    )
    require(
        structured["maximum_feasible_saturation_residual"] == 0.0,
        "closure saturation inactivity",
    )
    require(
        structured["maximum_feasible_marginal_residual"] == 0.0,
        "closure marginal preservation",
    )
    require(
        structured["minimum_contextual_curvature"]
        == structured["maximum_contextual_curvature"]
        == 1.0,
        "closure unit curvature",
    )
    require(
        structured["contextual_greedy_factor"]
        == structured["fixed_greedy_factor"]
        == 1 / 3,
        "closure alpha=1/3",
    )
    require(closure["physical"]["physical_blocks"] == 1, "closure physical blocks")
    require(closure["physical"]["joint_state_dimension"] == 2, "closure physical dimension")
    require(closure["physical"]["joint_mode_count_exercised"] == 4, "closure mode count")
    require(closure["physical"]["fallback_mode"] == 0, "closure fallback mode")
    require(
        closure["physical"]["fallback_reference"] == [0.0, 0.0],
        "closure fallback reference",
    )
    require(
        closure["physical"]["fallback_included_in_uniform_reference_certificate"],
        "fallback included in uniform reference certificate",
    )
    require(
        not closure["physical"]["per_agent_product_redesign_exercised"],
        "no per-agent redesign",
    )
    require(
        closure["gates"]["analytical_saturation_bounds_close"],
        "analytic saturation gate",
    )
    require(all(closure["gates"].values()), "all model closure gates")

    require(protocol_audit["audit_type"] == "distributed-protocol-v1", "protocol audit_type")
    require(protocol_audit["status"] == "PASS", "protocol closure status")
    require(
        protocol_audit["protocol_interface_version"] == "distributed-protocol-v1",
        "protocol closure version",
    )
    require(protocol_audit["network"]["diameter"] == 2, "audited graph diameter")
    require(
        protocol_audit["encoding"]["maximum_observed_residual"] == 0.0,
        "audited encoding residual",
    )
    require(
        protocol_audit["resource_screen"]["retained_edges_each_episode"] == 15
        and not protocol_audit["resource_screen"][
            "consumable_resource_channel_activated"
        ],
        "audited nonbinding resource screen",
    )
    require(
        protocol_audit["exploration"]["schedule_indices_one_based"]
        == [4, 11, 18, 25, 32, 39, 46, 53, 60, 7, 14, 21, 28, 35, 42, 49, 56, 3],
        "audited public schedule",
    )
    require(
        protocol_audit["counts"]
        == {
            "configured_scale_traces": 2664,
            "consensus_round_instances": 42624,
            "directed_proposal_transmissions": 170496,
            "directed_return_transmissions": 23040,
            "episode_rows": 2880,
            "evaluation_seeds": 12,
            "exploitation_episodes": 2664,
            "exploitation_stage_records": 21312,
            "exploitation_traces": 5328,
            "exploration_dispatch_records": 216,
            "feedback_floods": 2880,
            "feedback_round_instances": 5760,
            "zero_scale_traces": 2664,
        },
        "audited protocol counts",
    )
    require(all(protocol_audit["gates"].values()), "all protocol closure gates")
    require(
        all(row["centralized_distributed_equal"] == "1" for row in protocol_rows),
        "centralized/distributed matching equality",
    )
    require(
        all(row["resource_screen_nonbinding"] == "1" for row in protocol_rows),
        "protocol resource-screen equality",
    )
    exploitation_rows = [
        row
        for row in protocol_rows
        if row["protocol_branch"] in {"configured_scale", "zero_scale"}
    ]
    require(len(exploitation_rows) == 5328, "exploitation protocol rows")
    require(
        all(
            row["centralized_stage_sequence"] == row["distributed_stage_sequence"]
            and row["accepted_stage_count"] == "3"
            and row["null_stage"] == "4"
            and row["stage_count"] == "4"
            and row["all_agents_agree"] == "1"
            and row["all_tags_valid"] == "1"
            and row["discarded_invalid_tags"] == "0"
            and float(row["max_encoding_residual"]) == 0.0
            for row in exploitation_rows
        ),
        "all tagged max-consensus traces",
    )
    require(
        all(
            row["all_agents_received"] == "1"
            and row["all_agents_equal"] == "1"
            and row["flood_rounds"] == "2"
            and row["first_full_agreement_round"] == "2"
            and row["directed_return_transmissions"] == "8"
            for row in feedback_rows
        ),
        "all tagged return floods",
    )

    gate_columns = (
        "trajectory_deviation_utilization",
        "phase_deviation_tube_utilization",
        "tracking_tube_utilization",
        "reset_utilization",
        "actuator_utilization",
        "validity_utilization",
        "parameter_confidence_ratio",
        "max_edge_confidence_ratio",
        "cumulative_universal_utilization",
    )
    for column in gate_columns:
        require(max(float(row[column]) for row in rows) <= 1 + 1e-9, column)

    for row in rows:
        require(
            math.isclose(
                float(row["trajectory_deviation"]),
                float(row["trajectory_fluctuation"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "legacy deviation value alias",
        )
        require(
            math.isclose(
                float(row["shared_reset_deviation_scale"]),
                float(row["trajectory_sigma"]),
                rel_tol=0.0,
                abs_tol=5e-15,
            ),
            "legacy deviation scale alias",
        )
        require(
            math.isclose(
                float(row["trajectory_deviation_utilization"]),
                float(row["trajectory_fluctuation_utilization"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "legacy return utilization alias",
        )
        require(
            math.isclose(
                float(row["phase_deviation_tube_utilization"]),
                float(row["phase_fluctuation_tube_utilization"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "legacy phase utilization alias",
        )

    require(min(float(row["certificate_slack"]) for row in rows) >= -1e-9, "certificate")
    require(min(float(row["observed_return"]) for row in rows) >= -1e-12, "return")
    require(
        summary["status"]
        == "VERIFIED_DISTRIBUTED_EXECUTION",
        "status",
    )
    require(summary["calibration_interface_version"] == "shared-reset-calibration-v1", "calibration interface version")
    require(summary["model_interface_version"] == "terminal-model-v1", "model interface version")
    require(summary["protocol_interface_version"] == "distributed-protocol-v1", "protocol interface version")
    for scope_key in (
        "general_and_microcase_deviation_scales_identical",
        "globally_saturating_basis",
        "saturation_inactive_on_complete_hereditary_family",
        "unit_contextual_curvature_all_episodes",
        "physical_modes_and_returns_terminal_assignments_only",
        "absorbing_fallback_zero_benefit",
        "fallback_mode_and_reference_predeclared",
        "single_physical_block_base_case",
        "tagged_max_consensus_executed",
        "centralized_distributed_sequence_equality_all_exploitation_traces",
        "configured_and_zero_scale_protocols_checked",
        "identity_encoding_zero_error",
        "deterministic_public_assignment_codebook",
        "resource_screen_nonbinding_all_episodes",
        "tagged_feedback_flood_all_episodes",
    ):
        require(summary["scope"][scope_key], f"scope:{scope_key}")
    require(
        summary["results"]["zero_scale_counterfactual_difference_fraction_mean"] > 0,
        "zero-scale diagnostic",
    )

    with tempfile.TemporaryDirectory() as directory:
        replay_dir = Path(directory) / "out"
        execute(config_path, replay_dir)
        for name in (
            "episode_records.csv",
            "trial_summary.csv",
            "model_closure_audit.json",
            "distributed_protocol_records.csv",
            "feedback_flood_records.csv",
            "distributed_reproducibility_audit.json",
            "summary.json",
            "MANIFEST.json",
        ):
            require(
                sha256(replay_dir / name) == sha256(output_dir / name),
                f"replay {name}",
            )

    print("trajectory microcase distributed-protocol-v1 verification: PASS")
