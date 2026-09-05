#!/usr/bin/env python3
"""Check the numerical Python environment used by the simulations."""
from __future__ import annotations

import json
import platform

import numpy as np
import scipy
import pytest


def main() -> None:
    major_minor = tuple(int(x) for x in np.__version__.split('.')[:2])
    if major_minor < (1, 24):
        raise SystemExit(f"NumPy >= 1.24 is required; found {np.__version__}")
    scipy_major_minor = tuple(int(x) for x in scipy.__version__.split('.')[:2])
    if scipy_major_minor < (1, 10):
        raise SystemExit(f"SciPy >= 1.10 is required; found {scipy.__version__}")
    if int(pytest.__version__.split(".")[0]) < 8:
        raise SystemExit("pytest >= 8 is required")
    print(json.dumps({
        "status": "PASS",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pytest": pytest.__version__,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
