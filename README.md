DISTRIBUTED TASK ALLOCATION SIMULATIONS
=======================================

PURPOSE
-------
This standalone computational repository contains deterministic simulation
code, synthetic benchmark data, seed registries, configuration files,
automated checks, and compact empirical results for learning-augmented
distributed task allocation. It contains simulation assets only.

BENCHMARK SCENARIOS
-------------------
1. SAT-COV-V1: synthetic micro-satellite sensing and coverage.
2. UAV-AG-V1: synthetic precision-agriculture UAV operation.

All datasets are synthetic. No real flight, orbital, remote-sensing,
agronomic, personal, or confidential data are included.

NUMERICAL ENVIRONMENT
---------------------
Tested environment:
- Python 3.13.5
- NumPy 2.3.5
- SciPy 1.17.0

Install compatible packages with:

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt

On Windows PowerShell, activate with:

  .venv\Scripts\Activate.ps1

QUICK VALIDATION
----------------
Run:

  ./run_quick.sh

This command checks the numerical environment and reference-file integrity,
runs deterministic smoke tests, executes one short trial per scenario,
summarizes the generated outputs, and validates assignment, resource,
communication, and tracking diagnostics.

COMPLETE SIMULATION RUN
-----------------------
Run:

  ./run_all.sh

To change the worker count:

  WORKERS=8 ./run_all.sh

Generated outputs are written under results/generated/. Compact empirical
summaries are stored under results/reference/.

REPOSITORY LAYOUT
-----------------
- configs/: scenario, tracking-model, and exploration settings
- datasets/: deterministic synthetic benchmark tables
- seeds/: evaluation seeds and public exploration codebooks
- src/: simulation engines and numerical helpers
- scripts/: command-line runners, summaries, and validation utilities
- tests/: deterministic standalone smoke tests
- results/reference/: compact empirical summaries and integrity metadata
