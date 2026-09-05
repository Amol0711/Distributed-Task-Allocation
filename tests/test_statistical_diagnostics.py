from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_statistical_diagnostics.py"


def _load_builder() -> Any:
    spec = importlib.util.spec_from_file_location("build_statistical_diagnostics", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def audit() -> dict[str, Any]:
    builder = _load_builder()
    cfg = builder.tm.load_config(ROOT / "configs" / "trajectory_microcase.json")
    campaign_rows, campaign_gates = builder.campaign_reconciliation(
        builder.read_csv(ROOT / "results" / "reference" / "application_evaluation" / "paired_retention.csv"),
        builder.read_csv(
            ROOT / "results" / "reference" / "application_evaluation" / "evaluation_certificate_summary.csv"
        ),
        ROOT / "src" / "exploration_campaign.py",
    )
    trajectory, score_rows, trajectory_gates = builder.trajectory_reconciliation(
        cfg,
        builder.read_csv(ROOT / "results" / "trajectory_microcase" / "episode_records.csv"),
        builder.read_csv(ROOT / "results" / "trajectory_microcase" / "trial_summary.csv"),
    )
    return {
        "builder": builder,
        "campaign_rows": campaign_rows,
        "campaign_gates": campaign_gates,
        "trajectory": trajectory,
        "score_rows": score_rows,
        "trajectory_gates": trajectory_gates,
    }


def test_paired_retention_is_trialwise_ratio_with_standard_error(audit: dict[str, Any]) -> None:
    rows = {row["campaign_id"]: row for row in audit["campaign_rows"]}
    expected = {
        "SAT-COV-V1": (0.9980609799238207, 0.0012283095127795783, 0.0069483678870590725),
        "UAV-AG-V1": (0.9964346511072375, 0.0014589930110019684, 0.008253310814266168),
    }
    for campaign_id, (mean, se, sd) in expected.items():
        row = rows[campaign_id]
        assert row["paired_trials"] == 32
        assert row["paired_retention_definition"] == "rho_s=V_shielded_s/V_exploitation_only_s"
        assert math.isclose(row["paired_retention_mean"], mean, rel_tol=0.0, abs_tol=5e-16)
        assert math.isclose(row["paired_retention_standard_error"], se, rel_tol=0.0, abs_tol=5e-16)
        assert math.isclose(row["paired_retention_sample_standard_deviation"], sd, rel_tol=0.0, abs_tol=5e-16)
        assert math.isclose(se, sd / math.sqrt(32), rel_tol=0.0, abs_tol=5e-16)
        assert row["table_ratio_recovers_paired_retention"] == 0
    assert all(audit["campaign_gates"].values())


def test_exact_score_greedy_denominators_are_policy_conditioned(audit: dict[str, Any]) -> None:
    gates = audit["campaign_gates"]
    assert gates["affordability_uses_policy_specific_resources"]
    assert gates["oracle_uses_policy_conditioned_base_family"]
    assert gates["policy_specific_oracle_is_accumulated"]
    for row in audit["campaign_rows"]:
        assert row["exact_score_greedy_denominator"] == (
            "recomputed under each policy's own residual-resource state"
        )
        assert abs(row["paired_mean_minus_ratio_of_table_means"]) > 5.0e-4


def test_clip_frequency_and_ceiling_are_reconstructed(audit: dict[str, Any]) -> None:
    clip = audit["trajectory"]["clip"]
    assert clip["exploitation_episodes"] == 2664
    assert clip["clipped_exploitation_episodes"] == 24
    assert math.isclose(clip["clip_fraction"], 24 / 2664, rel_tol=0.0, abs_tol=1e-18)
    assert math.isclose(clip["clip_inactive_fraction"], 2640 / 2664, rel_tol=0.0, abs_tol=1e-18)
    assert clip["clip_episode_numbers"] == [27, 30]
    assert set(clip["clips_per_seed"].values()) == {2}
    assert math.isclose(clip["episode_ceiling"], 0.6, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(
        clip["maximum_raw_episode_charge"], 0.6075788941521933, rel_tol=0.0, abs_tol=5e-16
    )
    assert math.isclose(
        clip["maximum_raw_episode_utilization"],
        1.0126314902536557,
        rel_tol=0.0,
        abs_tol=5e-16,
    )


def test_physical_confidence_share_uses_correct_ridge_floor(audit: dict[str, Any]) -> None:
    share = audit["trajectory"]["physical_confidence_share"]
    assert math.isclose(share["beta_zero_scale"], 0.275, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(share["sqrt_lambda_Btheta"], 0.275, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(
        share["mean_over_exploitation_episodes"], 0.47775452583112, rel_tol=0.0, abs_tol=5e-15
    )
    assert math.isclose(share["minimum"], 0.4223708948099473, rel_tol=0.0, abs_tol=5e-15)
    assert math.isclose(share["maximum"], 0.4993717677642393, rel_tol=0.0, abs_tol=5e-15)
    assert math.isclose(
        share["fixed_estimator_matching_difference_mean"],
        0.5274024024024023,
        rel_tol=0.0,
        abs_tol=5e-15,
    )
    assert share["counterfactual_performance_claimed"] is False


def test_concrete_score_crossing_is_actual_top_envelope_witness(audit: dict[str, Any]) -> None:
    witness = audit["trajectory"]["score_crossing_witness"]
    assert witness["seed"] == 12031
    assert witness["episode"] == 21
    assert witness["stage"] == 2
    assert witness["common_partial_set_one_based"] == "{(2,2)}"
    assert math.isclose(
        witness["crossing_physical_scale"], 0.047525170715918545, rel_tol=0.0, abs_tol=5e-16
    )
    assert 0.0 < witness["crossing_physical_scale"] < witness["configured_physical_scale"]
    assert witness["zero_scale_winner_one_based"] == "(1,3)"
    assert witness["configured_scale_winner_one_based"] == "(3,1)"
    assert witness["top_candidates_at_crossing_zero_based"] == ["0-2", "2-0"]
    a = witness["candidate_A"]
    b = witness["candidate_B"]
    assert math.isclose(a["zero_scale_intercept"], 0.30521061255601756, abs_tol=5e-16)
    assert math.isclose(a["physical_scale_slope"], 0.8237179541492518, abs_tol=5e-16)
    assert math.isclose(b["zero_scale_intercept"], 0.3245637216540248, abs_tol=5e-16)
    assert math.isclose(b["physical_scale_slope"], 0.41649986726029864, abs_tol=5e-16)
    assert math.isclose(a["score_at_crossing"], b["score_at_crossing"], abs_tol=1e-15)
    assert witness["counterfactual_performance_claimed"] is False


def test_maximum_tracking_utilization_is_deterministic_tightness(audit: dict[str, Any]) -> None:
    tracking = audit["trajectory"]["tracking_tightness"]
    assert math.isclose(
        tracking["maximum_tracking_utilization"], 0.9730372946297503, abs_tol=5e-16
    )
    assert math.isclose(
        tracking["maximum_tracking_utilization"],
        tracking["initial_error_norm"] / tracking["all_sequence_radius"],
        abs_tol=5e-16,
    )
    assert math.isclose(tracking["relative_margin"], 0.026962705370249695, abs_tol=5e-16)
    assert tracking["maximizer_rows"] == 12
    assert tracking["maximizer_episode_numbers"] == [1]
    assert tracking["maximizer_is_predeclared_initial_condition"]
    assert "not a failure probability" in tracking["interpretation"]


def test_builder_is_deterministic_and_does_not_modify_locked_evidence(tmp_path: Path) -> None:
    episode = ROOT / "results" / "trajectory_microcase" / "episode_records.csv"
    trial = ROOT / "results" / "trajectory_microcase" / "trial_summary.csv"
    episode_before = _sha256(episode)
    trial_before = _sha256(trial)
    output_hashes: list[tuple[str, str, str]] = []
    for index in (1, 2):
        campaign = tmp_path / f"campaign-{index}.csv"
        trajectory = tmp_path / f"trajectory-{index}.json"
        score = tmp_path / f"score-{index}.csv"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--campaign-output",
                str(campaign),
                "--trajectory-output",
                str(trajectory),
                "--score-output",
                str(score),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["status"] == "PASS"
        output_hashes.append((_sha256(campaign), _sha256(trajectory), _sha256(score)))
    assert output_hashes[0] == output_hashes[1]
    assert _sha256(episode) == episode_before
    assert _sha256(trial) == trial_before
