from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from campaign_engine import load_assets, run_trial as run_baseline, sha256_file  # noqa: E402
from certificate_arithmetic import (  # noqa: E402
    CertificateLedger,
    certificate_cap,
    certify_episode,
    clipped_transfer_increment,
    contextual_comparator_factor,
    fixed_comparator_factor,
    universal_normalization,
)
from exploration_campaign import run_trial as run_exploration  # noqa: E402
from validate_results import validate_file  # noqa: E402

CAMPAIGNS = {
    "SAT-COV-V1": ROOT / "configs" / "satellite_campaign.json",
    "UAV-AG-V1": ROOT / "configs" / "uav_campaign.json",
}


def source_bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class CertificateArithmeticTest(unittest.TestCase):
    def test_exploration_charge_is_fmax_over_q(self) -> None:
        episode = certify_episode(
            exploration=True,
            raw_exploitation_charge=100.0,
            f_max=9.0,
            q=3,
        )
        self.assertEqual(episode.cap, 3.0)
        self.assertEqual(episode.raw_charge, 3.0)
        self.assertEqual(episode.certified_charge, 3.0)
        self.assertFalse(episode.clipped)
        self.assertEqual(episode.clip_excess, 0.0)

    def test_exploitation_charge_below_ceiling_is_unchanged(self) -> None:
        episode = certify_episode(
            exploration=False,
            raw_exploitation_charge=0.4,
            f_max=6.0,
            q=3,
        )
        self.assertAlmostEqual(episode.raw_charge, 0.4)
        self.assertAlmostEqual(episode.certified_charge, 0.4)
        self.assertFalse(episode.clipped)

    def test_exploitation_charge_above_ceiling_is_clipped(self) -> None:
        episode = certify_episode(
            exploration=False,
            raw_exploitation_charge=4.0,
            f_max=6.0,
            q=3,
        )
        self.assertEqual(episode.raw_charge, 4.0)
        self.assertEqual(episode.certified_charge, 2.0)
        self.assertTrue(episode.clipped)
        self.assertEqual(episode.clip_excess, 2.0)

    def test_cumulative_arithmetic_and_universal_normalization(self) -> None:
        ledger = CertificateLedger()
        episodes = [
            certify_episode(
                exploration=True,
                raw_exploitation_charge=99.0,
                f_max=6.0,
                q=3,
            ),
            certify_episode(
                exploration=False,
                raw_exploitation_charge=0.4,
                f_max=6.0,
                q=3,
            ),
            certify_episode(
                exploration=False,
                raw_exploitation_charge=4.0,
                f_max=6.0,
                q=3,
            ),
        ]
        for episode in episodes:
            ledger.add(episode, exploration=episode.exploration)
        ledger.check(f_max=6.0, q=3)
        self.assertAlmostEqual(ledger.exploration, 2.0)
        self.assertAlmostEqual(ledger.raw_exploitation, 4.4)
        self.assertAlmostEqual(ledger.exploitation, 2.4)
        self.assertAlmostEqual(ledger.total, 4.4)
        self.assertEqual(ledger.clipped_episodes, 1)
        self.assertAlmostEqual(ledger.clip_excess, 2.0)
        self.assertAlmostEqual(
            universal_normalization(
                cumulative_charge=ledger.total,
                epoch=ledger.episodes,
                f_max=6.0,
                q=3,
            ),
            4.4 / 6.0,
        )

    def test_fixed_comparator_is_horizon_uniform(self) -> None:
        q = 3
        fixed = fixed_comparator_factor(q)
        self.assertEqual(fixed, 1.0 / 4.0)
        for kappa in (0.0, 0.1, 0.5, 1.0):
            alpha = contextual_comparator_factor(q=q, curvature=kappa)
            self.assertLessEqual(fixed, alpha)
            self.assertLessEqual(alpha, 1.0 / q)

    def test_fixed_trace_transfer_uses_episodewise_residual_capacity(self) -> None:
        self.assertAlmostEqual(
            clipped_transfer_increment(
                baseline_raw_charge=0.4,
                enlargement=0.7,
                cap=1.0,
            ),
            0.6,
        )
        self.assertEqual(
            clipped_transfer_increment(
                baseline_raw_charge=1.4,
                enlargement=0.7,
                cap=1.0,
            ),
            0.0,
        )
        self.assertEqual(
            clipped_transfer_increment(
                baseline_raw_charge=0.2,
                enlargement=0.3,
                cap=1.0,
            ),
            0.3,
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            certificate_cap(f_max=1.0, q=0)
        with self.assertRaises(ValueError):
            certify_episode(
                exploration=False,
                raw_exploitation_charge=-1.0,
                f_max=1.0,
                q=1,
            )
        with self.assertRaises(ValueError):
            contextual_comparator_factor(q=1, curvature=1.1)


class SimulationSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (ROOT / "seeds" / "trials.csv").open(newline="", encoding="utf-8") as handle:
            cls.seeds = list(csv.DictReader(handle))
        cls.exploration_config = json.loads(
            (ROOT / "configs" / "exploration.json").read_text(encoding="utf-8")
        )
        cls.exploration_engine_hash = source_bundle_sha256(
            [
                ROOT / "src" / "exploration_campaign.py",
                ROOT / "src" / "certificate_arithmetic.py",
            ]
        )
        cls.experiment_config_hash = sha256_file(ROOT / "configs" / "exploration.json")
        cls.seed_hash = sha256_file(ROOT / "seeds" / "trials.csv")

    def seed(self, campaign_id: str) -> dict[str, str]:
        return next(
            row
            for row in self.seeds
            if row["campaign_id"] == campaign_id
            and row["partition"] == "evaluation"
            and int(row["trial_index"]) == 1
        )

    @staticmethod
    def stable(result: dict) -> dict:
        result = json.loads(json.dumps(result))
        result.pop("elapsed_seconds", None)
        result.pop("files", None)
        return result

    def exploration_kwargs(
        self, campaign_id: str, raw_dir: Path, *, write_raw: bool
    ) -> dict:
        return dict(
            sim_root=ROOT,
            campaign_path=CAMPAIGNS[campaign_id],
            experiment_config=self.exploration_config,
            seed_row=self.seed(campaign_id),
            raw_dir=raw_dir,
            engine_hash=self.exploration_engine_hash,
            experiment_config_hash=self.experiment_config_hash,
            seed_registry_hash=self.seed_hash,
            c_values=(0.0, 0.25),
            epoch_limit=4,
            write_raw=write_raw,
        )

    def test_inputs_load(self) -> None:
        self.assertEqual({row["campaign_id"] for row in self.seeds}, set(CAMPAIGNS))
        for campaign_id, path in CAMPAIGNS.items():
            assets = load_assets(ROOT, path)
            self.assertEqual(assets.cfg["campaign_id"], campaign_id)
            self.assertGreater(len(assets.tracking.modes), 0)
            self.assertGreater(int(assets.cfg["primary"]["epochs"]), 0)

    def test_baseline_is_deterministic_and_feasible(self) -> None:
        engine_hash = sha256_file(ROOT / "src" / "campaign_engine.py")
        for campaign_id, path in CAMPAIGNS.items():
            seed = self.seed(campaign_id)
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                kwargs = dict(
                    sim_root=ROOT,
                    campaign_path=path,
                    seed_row=seed,
                    engine_hash=engine_hash,
                    scale="primary",
                    methods=("INSTANTANEOUS_MYOPIC",),
                    epoch_limit=4,
                    write_raw=False,
                )
                result_a = run_baseline(raw_dir=Path(first), **kwargs)
                result_b = run_baseline(raw_dir=Path(second), **kwargs)
            self.assertEqual(self.stable(result_a), self.stable(result_b))
            metrics = result_a["aggregates"]["INSTANTANEOUS_MYOPIC"]
            self.assertEqual(metrics["resource_violations"], 0)
            self.assertEqual(metrics["family_violations"], 0)
            self.assertEqual(metrics["mismatches"], 0)
            self.assertGreater(metrics["mean_oracle_ratio"], 0.0)

    def test_exploration_is_deterministic_feasible_and_universally_bounded(self) -> None:
        for campaign_id in CAMPAIGNS:
            with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
                result_a = run_exploration(
                    **self.exploration_kwargs(campaign_id, Path(first), write_raw=False)
                )
                result_b = run_exploration(
                    **self.exploration_kwargs(campaign_id, Path(second), write_raw=False)
                )
            self.assertEqual(self.stable(result_a), self.stable(result_b))
            for metrics in result_a["aggregates"].values():
                self.assertEqual(metrics["resource_violations"], 0)
                self.assertEqual(metrics["family_violations"], 0)
                self.assertEqual(metrics["mismatches"], 0)
                self.assertEqual(metrics["bound_failures"], 0)
                self.assertEqual(metrics["audit_feasibility_failures"], 0)
                self.assertEqual(metrics["audit_mismatches"], 0)
                self.assertGreater(metrics["cumulative_oracle_ratio"], 0.0)
                for route in ("", "support_", "prior_"):
                    self.assertLessEqual(
                        metrics[f"{route}universal_normalized_observable_bound"],
                        1.0 + 1.0e-10,
                    )
                self.assertLessEqual(
                    metrics["direct_clipped_episodes"],
                    metrics["epochs"] - metrics["realized_explorations"],
                )

    def test_v2_validator_detects_tampered_direct_arithmetic(self) -> None:
        campaign_id = "SAT-COV-V1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_exploration(
                **self.exploration_kwargs(campaign_id, root, write_raw=True)
            )
            raw_path = Path(result["files"][0]["relative_path"])
            self.assertEqual(validate_file(raw_path)["status"], "PASS")

            with raw_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            rows[0]["observable_bound_increment"] = format(
                float(rows[0]["observable_bound_increment"]) + 0.125,
                ".17g",
            )
            tampered = root / "tampered_direct.csv"
            with tampered.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                validate_file(tampered)

    def test_v2_validator_detects_tampered_primitive_charge(self) -> None:
        campaign_id = "SAT-COV-V1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_exploration(
                **self.exploration_kwargs(campaign_id, root, write_raw=True)
            )
            raw_path = Path(result["files"][0]["relative_path"])
            with raw_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            exploitation_index = next(
                index
                for index, row in enumerate(rows)
                if int(row["exploration_indicator"]) == 0
            )
            rows[exploitation_index]["selected_width_sum"] = format(
                float(rows[exploitation_index]["selected_width_sum"]) + 0.125,
                ".17g",
            )
            tampered = root / "tampered_primitive.csv"
            with tampered.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                validate_file(tampered)

    def test_v2_validator_detects_tampered_fixed_trace_arithmetic(self) -> None:
        campaign_id = "SAT-COV-V1"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_exploration(
                **self.exploration_kwargs(campaign_id, root, write_raw=True)
            )
            raw_path = Path(result["files"][0]["relative_path"])
            with raw_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = list(reader.fieldnames or [])
                rows = list(reader)
            rows[-1]["support_cumulative_observable_bound"] = format(
                float(rows[-1]["support_cumulative_observable_bound"]) + 0.125,
                ".17g",
            )
            tampered = root / "tampered_support.csv"
            with tampered.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                validate_file(tampered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
