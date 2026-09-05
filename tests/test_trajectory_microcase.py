from __future__ import annotations

import json
import math
import csv
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_microcase import (
    TaggedProposal,
    allocation_graph_diameter,
    allocation_graph_neighbors,
    centralized_greedy_trace,
    contextual_total_curvature,
    distributed_greedy_trace,
    edge_feature,
    edges,
    execution_mode,
    execution_reference,
    exploration_codebook_rows,
    exploration_schedule_indices,
    exploration_schedule_rows,
    fallback_mode,
    fallback_reference,
    feature_geometry_audit,
    hereditary_matchings,
    load_config,
    marginal_feature,
    matchings,
    mode,
    physical_certificate,
    public_exploration_codebook,
    raw_set_feature,
    reference,
    resource_screened_edges,
    set_feature,
    shared_reset_deviation_radii,
    shared_reset_deviation_scale,
    static_audit,
    tagged_feedback_flood,
    tagged_max_consensus,
    terminal_matchings,
    trajectory_linear_deviation_scale,
)


def test_shared_reset_radius_includes_disturbance_input_gain() -> None:
    a = (0.5 * np.eye(2),)
    e = (2.0 * np.eye(2),)
    a_max, e_max, radii = shared_reset_deviation_radii(
        a, e, disturbance_bound=0.1, horizon=3
    )
    assert a_max == 0.5
    assert e_max == 2.0
    assert np.allclose(radii, (0.0, 0.2, 0.3, 0.35), atol=1e-15, rtol=0.0)


def test_general_deviation_scale_formula() -> None:
    radii = (0.0, 0.2, 0.3)
    scale = shared_reset_deviation_scale(
        radii,
        discount_factor=0.9,
        running_lipschitz=1.5,
        terminal_lipschitz=2.0,
        trajectory_comparison=0.75,
    )
    expected = 0.75 * (1.5 * (0.0 + 0.9 * 0.2) + 0.9**2 * 2.0 * 0.3)
    assert math.isclose(scale, expected, rel_tol=0.0, abs_tol=1e-15)


def test_microcase_general_and_linear_scales_are_identical() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    cert = physical_certificate(cfg, matchings(cfg))
    radii = cert["deviation_phase_radii"]
    bnorm = float(np.linalg.norm(cfg["trajectory_return_vector"], 2))
    general = shared_reset_deviation_scale(
        radii,
        cfg["discount_factor"],
        running_lipschitz=bnorm / cfg["discount_factor"],
        terminal_lipschitz=bnorm / cfg["discount_factor"],
    )
    linear = trajectory_linear_deviation_scale(
        radii, cfg["discount_factor"], cfg["trajectory_return_vector"]
    )
    assert math.isclose(general, linear, rel_tol=0.0, abs_tol=5e-15)
    assert math.isclose(general, 0.0548271522404341, rel_tol=0.0, abs_tol=5e-15)
    assert cert["disturbance_input_norm_max"] == 1.0
    assert cert["deviation_scale_identity_residual"] <= 5e-15


def test_static_audit_closes() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    audit = static_audit(cfg)
    assert len(matchings(cfg)) == 60
    assert audit["rho_H"] < 1
    assert audit["maximum_full_matching_value"] <= cfg["fmax"] + 1e-12
    assert math.isclose(
        audit["shared_reset_deviation_scale"],
        audit["trajectory_sigma"],
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert math.isclose(
        audit["minimum_stage_baseline"],
        0.09152463026755847,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert math.isclose(
        audit["maximum_stage_deviation_bound"],
        0.012027974561191012,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert audit["minimum_stage_reward_lower_bound"] > 0.07949


def test_saturating_basis_is_inactive_on_complete_hereditary_family() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    cap = cfg["feature_saturation_cap"]
    for k in range(1, cfg["episodes"] + 1):
        for S in hereditary_matchings(cfg):
            raw = raw_set_feature(cfg, S, k)
            assert float(raw.max(initial=0.0)) < cap
            assert np.array_equal(set_feature(cfg, S, k), raw)
            for e in ((i, j) for i in range(cfg["agents"]) for j in range(cfg["tasks"])):
                if e in S or any(i == e[0] or j == e[1] for i, j in S):
                    continue
                assert np.array_equal(
                    marginal_feature(cfg, e, S, k), edge_feature(cfg, e, k)
                )


def test_global_capped_extension_has_unit_curvature() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    audit = feature_geometry_audit(cfg)
    assert audit["edge_feature_coordinate_lower_bound"] == 0.12
    assert audit["edge_feature_coordinate_upper_bound"] == 0.28
    assert audit["analytical_feasible_raw_coordinate_upper_bound"] == 0.8400000000000001
    assert audit["analytical_ground_without_edge_coordinate_lower_bound"] == 1.68
    assert audit["maximum_feasible_raw_coordinate"] == 0.8400000000000001
    assert audit["maximum_feasible_saturation_residual"] == 0.0
    assert audit["maximum_feasible_marginal_residual"] == 0.0
    assert audit["maximum_feasible_formula_marginal_residual"] <= 6e-17
    assert audit["minimum_ground_without_edge_raw_coordinate"] > 1.79
    assert audit["minimum_ground_raw_coordinate"] > 1.94
    assert audit["maximum_ground_raw_coordinate"] > 4.05
    ground = tuple(
        (i, j) for i in range(cfg["agents"]) for j in range(cfg["tasks"])
    )
    for k in range(1, cfg["episodes"] + 1):
        full_feature = set_feature(cfg, ground, k)
        assert np.array_equal(full_feature, np.ones(cfg["feature_dimension"]))
        for e in ground:
            without = tuple(item for item in ground if item != e)
            assert np.array_equal(
                set_feature(cfg, without, k), np.ones(cfg["feature_dimension"])
            )
    assert audit["curvature_equals_one_all_episodes"]
    assert all(
        contextual_total_curvature(cfg, k) == 1.0
        for k in range(1, cfg["episodes"] + 1)
    )
    assert audit["contextual_greedy_factor"] == audit["fixed_greedy_factor"] == 1 / 3


def test_terminal_family_is_exact_maximal_subfamily() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    hereditary = hereditary_matchings(cfg)
    terminal = terminal_matchings(cfg)
    assert len(hereditary) == 136
    assert len(terminal) == len(matchings(cfg)) == 60
    assert all(len(S) == 3 for S in terminal)
    assert feature_geometry_audit(cfg)["terminal_family_equals_maximal_family"]


def test_physical_interface_rejects_construction_only_sets() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    partial = ((0, 0), (1, 1))
    with np.testing.assert_raises(ValueError):
        mode(cfg, partial)
    with np.testing.assert_raises(ValueError):
        reference(partial, cfg)
    terminal = terminal_matchings(cfg)[0]
    assert isinstance(mode(cfg, terminal), int)
    assert reference(terminal, cfg).shape == (cfg["physical_state_dimension"],)
    assert execution_mode(cfg, terminal) == mode(cfg, terminal)
    assert np.array_equal(
        execution_reference(cfg, terminal), reference(terminal, cfg)
    )


def test_empty_execution_is_routed_to_predeclared_fallback() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    assert fallback_mode(cfg) == cfg["fallback_mode"] == 0
    assert np.array_equal(
        fallback_reference(cfg), np.zeros(cfg["physical_state_dimension"])
    )
    assert execution_mode(cfg, tuple()) == fallback_mode(cfg)
    assert np.array_equal(
        execution_reference(cfg, tuple()), fallback_reference(cfg)
    )
    # The defensive-copy contract prevents callers from mutating configuration.
    copied = fallback_reference(cfg)
    copied[0] = 1.0
    assert cfg["fallback_reference"][0] == 0.0
    cert = physical_certificate(cfg, terminal_matchings(cfg))
    assert cert["fallback_mode"] == 0
    assert cert["fallback_reference"] == [0.0, 0.0]
    assert cert["fallback_included_in_reference_library"]
    assert math.isclose(
        cert["reference_jump_bound"],
        0.02262741699796952,
        rel_tol=0.0,
        abs_tol=5e-15,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("feature_saturation_cap", 0.9, "cap exactly one"),
        ("terminal_assignment_cardinality", 2, "terminal cardinality"),
        ("physical_blocks", 3, "one physical block"),
        ("fallback_mode", 5, "fallback_mode"),
        ("fallback_reference", [0.0], "fallback_reference"),
        ("tasks", 2, "tasks >= agents"),
    ),
)
def test_locked_model_interface_rejects_invalid_configuration(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    source = ROOT / "configs" / "trajectory_microcase.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data[field] = value
    candidate = tmp_path / "invalid.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(candidate)



def test_model_closure_audit_is_self_contained_and_locked() -> None:
    audit = json.loads(
        (
            ROOT
            / "results"
            / "trajectory_microcase"
            / "model_closure_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["audit_type"] == "terminal-model-v1"
    assert audit["status"] == "PASS"
    assert audit["model_interface_version"] == "terminal-model-v1"
    assert audit["episode_rows"] == 2880
    assert audit["fallback_invocations"] == 0
    assert audit["family"] == {
        "ground_edge_count": 15,
        "hereditary_assignment_count": 136,
        "hereditary_cardinality_counts": {"0": 1, "1": 15, "2": 60, "3": 60},
        "terminal_assignment_count": 60,
        "terminal_cardinality": 3,
        "terminal_family_equals_maximal_family": True,
    }
    structured = audit["structured_value"]
    assert structured["basis"] == "coordinatewise_min_of_additive_load_and_one"
    assert structured["feature_saturation_cap"] == 1.0
    assert structured["edge_feature_coordinate_lower_bound"] == 0.12
    assert structured["edge_feature_coordinate_upper_bound"] == 0.28
    assert structured["analytical_feasible_raw_coordinate_upper_bound"] == 0.8400000000000001
    assert structured["analytical_ground_without_edge_coordinate_lower_bound"] == 1.68
    assert structured["maximum_feasible_raw_coordinate"] == 0.8400000000000001
    assert structured["maximum_feasible_saturation_residual"] == 0.0
    assert structured["maximum_feasible_marginal_residual"] == 0.0
    assert structured["minimum_contextual_curvature"] == 1.0
    assert structured["maximum_contextual_curvature"] == 1.0
    assert structured["contextual_greedy_factor"] == 1 / 3
    assert structured["fixed_greedy_factor"] == 1 / 3
    assert audit["physical"] == {
        "controller_template_count": 4,
        "fallback_included_in_uniform_reference_certificate": True,
        "fallback_mode": 0,
        "fallback_reference": [0.0, 0.0],
        "joint_mode_count_exercised": 4,
        "joint_state_dimension": 2,
        "per_agent_product_redesign_exercised": False,
        "physical_blocks": 1,
        "state_dimension_per_block": 2,
    }
    assert audit["file_sha256"] == {
        "episode_records.csv": (
            "fd953e541ae54c428096ffbbfd6a71d32bcaf724ee93459418cc0a0394be7b16"
        ),
        "trial_summary.csv": (
            "6188e4a675d89c3f30c6df28a20f9313689ad29398d97951c420177cadfdb8df"
        ),
    }
    assert audit["fingerprints"] == {
        "policy": (
            "4ef980f72c051e887ac85783f350c7255cf73e7b287080b8075c4bae9c2c2622"
        ),
        "selection_zero_scale_optimum": (
            "3aea33fe57fb2858bf797bc8a3d2b7749f4c1c3ff6d9b40e6d120cf40f14c55c"
        ),
        "primary_evidence": (
            "81672022ca28a206e5128d8660a29db717d563f658e529e0ce4e473f118fb3d0"
        ),
    }
    assert audit["gates"]
    assert audit["gates"]["analytical_saturation_bounds_close"]
    assert all(audit["gates"].values())

def test_locked_results_close_all_gates() -> None:
    summary = json.loads(
        (ROOT / "results" / "trajectory_microcase" / "summary.json").read_text()
    )
    results = summary["results"]
    static = summary["static_certificate"]
    assert summary["calibration_interface_version"] == "shared-reset-calibration-v1"
    assert summary["model_interface_version"] == "terminal-model-v1"
    assert summary["protocol_interface_version"] == "distributed-protocol-v1"
    assert summary["status"] == (
        "VERIFIED_DISTRIBUTED_"
        "EXECUTION"
    )
    assert results["parameter_confidence_trial_coverage"] == 1
    assert results["certificate_trial_coverage"] == 1
    assert results["final_universal_certificate_utilization_mean"] < 1
    assert results["zero_scale_counterfactual_difference_fraction_mean"] > 0
    assert math.isclose(
        static["shared_reset_deviation_scale"],
        0.0548271522404341,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert math.isclose(
        results["max_trajectory_deviation_utilization"],
        results["max_trajectory_fluctuation_utilization"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_path_graph_protocol_and_nonbinding_resource_screen() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    assert allocation_graph_neighbors(cfg) == {0: (1,), 1: (0, 2), 2: (1,)}
    assert allocation_graph_diameter(cfg) == 2
    assert cfg["allocation_network"]["diameter_upper_bound"] == 2
    assert cfg["allocation_network"]["rounds_per_stage"] == 2
    assert cfg["allocation_network"]["score_encoding"] == "identity"
    assert cfg["allocation_network"]["encoding_error"] == 0.0
    assert all(
        resource_screened_edges(cfg, episode) == edges(cfg)
        for episode in range(1, cfg["episodes"] + 1)
    )


def test_public_codebook_and_schedule_are_fixed_and_complete() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    codebook = public_exploration_codebook(cfg)
    indices = exploration_schedule_indices(cfg)
    assert codebook == terminal_matchings(cfg)
    assert len(codebook) == 60
    assert indices == (
        3,
        10,
        17,
        24,
        31,
        38,
        45,
        52,
        59,
        6,
        13,
        20,
        27,
        34,
        41,
        48,
        55,
        2,
    )
    assert len(indices) == len(set(indices)) == cfg["exploration_episodes"]


def test_public_protocol_assets_match_canonical_generators() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    seed_dir = ROOT / "seeds"
    with (seed_dir / "trajectory_microcase_codebook.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        actual_codebook = list(csv.DictReader(handle))
    with (seed_dir / "trajectory_microcase_exploration_schedule.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        actual_schedule = list(csv.DictReader(handle))
    expected_codebook = [
        {key: str(value) for key, value in row.items()}
        for row in exploration_codebook_rows(cfg)
    ]
    expected_schedule = [
        {key: str(value) for key, value in row.items()}
        for row in exploration_schedule_rows(cfg)
    ]
    assert actual_codebook == expected_codebook
    assert actual_schedule == expected_schedule


@pytest.mark.parametrize(("beta", "scale_tag"), ((0.75, "configured"), (0.0, "zero_scale")))
def test_centralized_and_distributed_stage_sequences_are_identical(
    beta: float, scale_tag: str
) -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    theta_hat = np.asarray([0.2, 0.15, 0.1, 0.08, 0.05, 0.02])
    V = cfg["ridge_lambda"] * np.eye(cfg["feature_dimension"])
    centralized = centralized_greedy_trace(
        cfg, 21, theta_hat, V, beta, score_scale_tag=scale_tag
    )
    distributed = distributed_greedy_trace(
        cfg, 21, theta_hat, V, beta, score_scale_tag=scale_tag
    )
    assert centralized["matching"] == distributed["matching"]
    assert centralized["widths"] == distributed["widths"]
    assert centralized["accepted_stage_count"] == 3
    assert centralized["null_stage"] == 4
    assert [proposal.edge for proposal in centralized["stage_winners"]] == [
        proposal.edge for proposal in distributed["stage_winners"]
    ]
    assert all(
        stage["all_agents_agree"]
        and stage["discarded_invalid_tags"] == 0
        and stage["rounds"] == 2
        for stage in distributed["stage_consensus"]
    )


def test_invalid_tagged_proposal_is_discarded_without_changing_valid_winner() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    valid = TaggedProposal(
        raw_score=1.0,
        encoded_score=1.0,
        edge=(0, 0),
        owner=0,
        identifier=0,
        width=0.5,
        episode=1,
        stage=1,
        partial_set_tag="",
        score_scale_tag="configured",
    )
    invalid = TaggedProposal(
        raw_score=10.0,
        encoded_score=10.0,
        edge=(1, 0),
        owner=1,
        identifier=5,
        width=0.5,
        episode=1,
        stage=99,
        partial_set_tag="",
        score_scale_tag="configured",
    )
    lower = TaggedProposal(
        raw_score=0.5,
        encoded_score=0.5,
        edge=(2, 0),
        owner=2,
        identifier=10,
        width=0.5,
        episode=1,
        stage=1,
        partial_set_tag="",
        score_scale_tag="configured",
    )
    result = tagged_max_consensus(
        cfg,
        {0: valid, 1: invalid, 2: lower},
        episode=1,
        stage=1,
        partial_set_tag="",
        score_scale_tag="configured",
    )
    assert result["discarded_invalid_tags"] > 0
    assert result["all_agents_agree"]
    assert result["winner"] == valid


def test_tagged_return_flood_reaches_all_agents_in_diameter_rounds() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    result = tagged_feedback_flood(cfg, value_=0.8125, episode=73)
    assert result["source_agent"] == 0
    assert result["rounds"] == 2
    assert result["first_full_agreement_round"] == 2
    assert result["directed_transmissions"] == 8
    assert result["all_agents_received"]
    assert result["all_agents_equal"]


def test_distributed_reproducibility_audit_closes_all_target_traces() -> None:
    audit = json.loads(
        (
            ROOT
            / "results"
            / "trajectory_microcase"
            / "distributed_reproducibility_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert audit["audit_type"] == "distributed-protocol-v1"
    assert audit["status"] == "PASS"
    assert audit["protocol_interface_version"] == "distributed-protocol-v1"
    assert audit["network"]["diameter"] == 2
    assert audit["network"]["rounds_per_stage"] == 2
    assert audit["encoding"]["maximum_observed_residual"] == 0.0
    assert audit["resource_screen"]["retained_edges_each_episode"] == 15
    assert not audit["resource_screen"]["consumable_resource_channel_activated"]
    assert audit["exploration"]["codebook_size"] == 60
    assert audit["exploration"]["schedule_indices_one_based"] == [
        4,
        11,
        18,
        25,
        32,
        39,
        46,
        53,
        60,
        7,
        14,
        21,
        28,
        35,
        42,
        49,
        56,
        3,
    ]
    assert audit["counts"] == {
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
    }
    assert all(audit["gates"].values())


def test_protocol_output_files_are_manifested_and_have_expected_rows() -> None:
    result_dir = ROOT / "results" / "trajectory_microcase"
    manifest = json.loads((result_dir / "MANIFEST.json").read_text())
    assert "distributed_protocol_records.csv" in manifest["files"]
    assert "feedback_flood_records.csv" in manifest["files"]
    assert "distributed_reproducibility_audit.json" in manifest["files"]
    with (result_dir / "distributed_protocol_records.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        protocol_rows = list(csv.DictReader(handle))
    with (result_dir / "feedback_flood_records.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        feedback_rows = list(csv.DictReader(handle))
    assert len(protocol_rows) == 5544
    assert len(feedback_rows) == 2880
    assert all(row["centralized_distributed_equal"] == "1" for row in protocol_rows)
    assert all(row["resource_screen_nonbinding"] == "1" for row in protocol_rows)
    assert all(row["all_agents_received"] == "1" for row in feedback_rows)
    assert all(row["all_agents_equal"] == "1" for row in feedback_rows)
