from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from return_model_scope import ReturnModelScope, ReturnModelScopeError, load_return_model_scope


class ReturnModelScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = ROOT / "configs" / "return_model_scope.json"
        self.raw = json.loads(self.path.read_text(encoding="utf-8"))

    def test_configured_scope_is_valid(self) -> None:
        scope = load_return_model_scope(self.path)
        self.assertEqual(scope.target_execution_route, "direct_operational")
        self.assertFalse(scope.trajectory_return_interface.enabled)
        self.assertEqual(scope.routes["support_fixed_trace"].status, "fixed_trace_recertification")

    def test_fixed_trace_route_cannot_be_promoted_to_target(self) -> None:
        raw = json.loads(json.dumps(self.raw))
        raw["target_execution_route"] = "support_fixed_trace"
        with self.assertRaises(ReturnModelScopeError):
            ReturnModelScope.from_dict(raw)

    def test_partial_trajectory_interface_is_rejected(self) -> None:
        raw = json.loads(json.dumps(self.raw))
        raw["trajectory_return_interface"]["nominal_return_specification"] = "nominal.json"
        with self.assertRaises(ReturnModelScopeError):
            ReturnModelScope.from_dict(raw)

    def test_enabled_trajectory_interface_requires_both_specs(self) -> None:
        raw = json.loads(json.dumps(self.raw))
        raw["trajectory_return_interface"] = {
            "enabled": True,
            "nominal_return_specification": "nominal.json",
            "tracking_to_return_lipschitz_specification": "lipschitz.json",
        }
        scope = ReturnModelScope.from_dict(raw)
        self.assertTrue(scope.trajectory_return_interface.enabled)

    def test_load_reports_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.json"
            bad.write_text("{", encoding="utf-8")
            with self.assertRaises(ReturnModelScopeError):
                load_return_model_scope(bad)


if __name__ == "__main__":
    unittest.main()
