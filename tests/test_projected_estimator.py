from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from campaign_engine import (  # noqa: E402
    Estimator,
    constrained_quadratic_minimizer,
    nonnegative_quadratic_minimizer,
)


class FullProjectedRidgeTest(unittest.TestCase):
    def test_inactive_constraints_preserve_nonnegative_solution_exactly(self) -> None:
        V = np.array([[3.0, 0.4], [0.4, 2.0]])
        rhs = np.array([1.2, 0.8])
        baseline = nonnegative_quadratic_minimizer(V, rhs)
        projected = constrained_quadratic_minimizer(V, rhs, btheta=5.0, fmax=8.0)
        np.testing.assert_array_equal(projected, baseline)

    def test_l1_constraint_is_enforced(self) -> None:
        V = np.eye(3)
        rhs = np.array([2.0, 1.5, 1.0])
        x = constrained_quadratic_minimizer(V, rhs, btheta=10.0, fmax=1.2)
        self.assertTrue(np.all(x >= -1.0e-12))
        self.assertAlmostEqual(float(x.sum()), 1.2, places=8)
        self.assertLessEqual(float(np.linalg.norm(x)), 10.0 + 1.0e-10)
        # Euclidean projection onto the simplex gives (0.85, 0.35, 0).
        np.testing.assert_allclose(x, np.array([0.85, 0.35, 0.0]), atol=3.0e-7)

    def test_l2_constraint_is_enforced(self) -> None:
        V = np.eye(2)
        rhs = np.array([3.0, 4.0])
        x = constrained_quadratic_minimizer(V, rhs, btheta=1.0, fmax=10.0)
        self.assertAlmostEqual(float(np.linalg.norm(x)), 1.0, places=8)
        np.testing.assert_allclose(x, np.array([0.6, 0.8]), atol=3.0e-7)

    def test_joint_constraints_and_estimator_update(self) -> None:
        est = Estimator(d=3, lam=1.0, sigma=0.1, delta=0.05, btheta=0.75, fmax=0.9)
        est.update(np.array([1.0, 1.0, 0.5]), 5.0)
        self.assertTrue(np.all(est.theta >= -1.0e-12))
        self.assertLessEqual(float(np.linalg.norm(est.theta)), 0.75 + 2.0e-8)
        self.assertLessEqual(float(est.theta.sum()), 0.9 + 2.0e-8)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            constrained_quadratic_minimizer(np.eye(2), np.ones(3), 1.0, 1.0)
        with self.assertRaises(ValueError):
            constrained_quadratic_minimizer(np.array([[1.0, 2.0], [0.0, 1.0]]), np.ones(2), 1.0, 1.0)
        with self.assertRaises(ValueError):
            constrained_quadratic_minimizer(np.eye(2), np.ones(2), -1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
