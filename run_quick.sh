#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
QUICK_ROOT="$ROOT/results/quick_generated"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
rm -rf "$QUICK_ROOT"
mkdir -p "$QUICK_ROOT/raw"
python "$ROOT/scripts/check_environment.py"
python "$ROOT/scripts/verify_reference_results.py"
python "$ROOT/scripts/verify_control_evidence.py"
python "$ROOT/scripts/verify_projected_estimator_audit.py"
python "$ROOT/scripts/verify_trajectory_microcase.py"
python "$ROOT/scripts/verify_exact_oracle_audit.py"
python "$ROOT/scripts/verify_package_integrity.py"
python -m pytest -q "$ROOT/tests"
python "$ROOT/scripts/run_baselines.py" \
  --partition evaluation --trials-per-campaign 1 --epochs 12 --workers 1 \
  --output-dir "$QUICK_ROOT/raw/baseline"
python "$ROOT/scripts/run_exploration.py" \
  --partition evaluation --trials-per-campaign 1 --epochs 12 \
  --c-exp 0.0 0.25 --workers 1 \
  --output-dir "$QUICK_ROOT/raw/exploration"
python "$ROOT/scripts/summarize_results.py" \
  --raw-root "$QUICK_ROOT/raw" --output "$QUICK_ROOT/summary.csv"
python "$ROOT/scripts/validate_results.py" \
  --raw-root "$QUICK_ROOT/raw" --output "$QUICK_ROOT/validation.json"
printf '%s\n' "Quick validation completed successfully under $QUICK_ROOT."
