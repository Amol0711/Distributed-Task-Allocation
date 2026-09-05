#!/usr/bin/env python3
"""Tests for the constructive reference-reset and separation interfaces."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reference_reset import (  # noqa: E402
    ReferenceResetError,
    load_reference_reset_certificates,
    mode_from_assignment_map,
)
from campaign_engine import load_assets, mode_for_demand  # noqa: E402


class ReferenceResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.certificates = load_reference_reset_certificates(ROOT)

    def test_both_application_libraries_close(self) -> None:
        self.assertEqual(set(self.certificates), {"SAT-COV-V1", "UAV-AG-V1"})
        for cert in self.certificates.values():
            self.assertAlmostEqual(
                cert.certified_jump_bound,
                cert.configured_jump_bound,
                places=12,
            )
            self.assertGreater(len(cert.maximizing_pairs), 0)

    def test_every_template_pair_is_certified(self) -> None:
        checks = 0
        for cert in self.certificates.values():
            for source in cert.mode_names:
                for target in cert.mode_names:
                    norm = cert.verify_jump(source, target)
                    self.assertLessEqual(norm, cert.certified_jump_bound + 5.0e-12)
                    checks += 1
        self.assertEqual(checks, 32)

    def test_reset_sign_matches_error_coordinate_change(self) -> None:
        cert = self.certificates["SAT-COV-V1"]
        physical_state = np.linspace(-0.01, 0.01, cert.state_dimension)
        source, target = "survey", "agile"
        old_error = physical_state - cert.offsets[source]
        new_error = physical_state - cert.offsets[target]
        self.assertTrue(
            np.allclose(new_error, old_error + cert.jump_vector(source, target), atol=1.0e-14)
        )

    def test_residual_reset_tamper_is_rejected(self) -> None:
        cert = self.certificates["UAV-AG-V1"]
        residual = np.zeros(cert.state_dimension)
        residual[0] = cert.residual_reset_bound + 1.0e-4
        with self.assertRaises(ReferenceResetError):
            cert.verify_jump("loiter", "mapping", residual)

    def test_library_level_tamper_is_rejected(self) -> None:
        source = json.loads((ROOT / "configs" / "reference_reset_library.json").read_text())
        source["applications"]["SAT-COV-V1"]["residual_reset_bound"] = 0.031
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ReferenceResetError):
                load_reference_reset_certificates(ROOT, path)



    def test_runtime_dispatch_uses_the_declared_partition(self) -> None:
        expected = {
            "SAT-COV-V1": ("idle_hold", "survey", "responsive", "agile"),
            "UAV-AG-V1": ("loiter", "mapping", "scouting", "urgent_inspection"),
        }
        probes = [
            (False, 0.0, 0),
            (False, 1.0, 0),
            (True, 0.0, 1),
            (True, 0.35, 1),
            (True, np.nextafter(0.35, 1.0), 2),
            (True, 0.65, 2),
            (True, np.nextafter(0.65, 1.0), 3),
            (True, 1.0, 3),
        ]
        library = json.loads((ROOT / "configs" / "reference_reset_library.json").read_text())
        for campaign_id, names in expected.items():
            campaign_path = ROOT / library["applications"][campaign_id]["campaign_config"]
            campaign = json.loads(campaign_path.read_text())
            assets = load_assets(ROOT, campaign_path)
            for assigned, demand, expected_index in probes:
                declared = mode_from_assignment_map(campaign_id, campaign, names, demand, assigned)
                runtime = mode_for_demand(assets, demand, assigned)
                self.assertEqual(declared, names[expected_index])
                self.assertEqual(runtime, declared)

    def test_assignment_template_map_tamper_is_rejected(self) -> None:
        source = json.loads((ROOT / "configs" / "reference_reset_library.json").read_text())
        source["applications"]["SAT-COV-V1"]["campaign_config"] = (
            "configs/satellite_tracking_model.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tampered_assignment_map.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ReferenceResetError):
                load_reference_reset_certificates(ROOT, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
