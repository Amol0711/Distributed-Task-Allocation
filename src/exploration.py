"""Deterministic public exploration-codebook utilities."""
from __future__ import annotations

import numpy as np


def public_nonnegative_codebook(
    dimension: int,
    *,
    seed: int = 123,
    directions_per_dimension: int = 4,
    alpha: float = 0.2,
) -> np.ndarray:
    """Return basis directions followed by nonnegative simplex directions."""
    if dimension < 1 or directions_per_dimension < 0 or alpha <= 0.0:
        raise ValueError("invalid codebook parameters")
    rng = np.random.default_rng(seed)
    extra = rng.dirichlet(
        np.full(dimension, alpha),
        size=directions_per_dimension * dimension,
    )
    return np.vstack((np.eye(dimension), extra))
