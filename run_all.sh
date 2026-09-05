#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKERS="${WORKERS:-4}"
RESUME="${RESUME:-0}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  RESUME_ARGS=(--resume)
elif [[ "$RESUME" == "0" ]]; then
  rm -rf "$ROOT/results/generated"
else
  printf '%s\n' 'RESUME must be 0 or 1.' >&2
  exit 2
fi
python "$ROOT/scripts/check_environment.py"
python "$ROOT/scripts/verify_reference_results.py"
python "$ROOT/scripts/verify_package_integrity.py"
python "$ROOT/scripts/verify_trajectory_microcase.py"
python -m pytest -q "$ROOT/tests"
PYTHONPATH="$ROOT/src" python "$ROOT/tests/test_smoke.py"
python "$ROOT/scripts/run_baselines.py" \
  --partition evaluation --trials-per-campaign 0 --epochs 0 --workers "$WORKERS" \
  "${RESUME_ARGS[@]}"
python "$ROOT/scripts/run_exploration.py" \
  --partition evaluation --trials-per-campaign 0 --epochs 0 \
  --c-exp 0.0 0.25 --workers "$WORKERS" "${RESUME_ARGS[@]}"
python "$ROOT/scripts/summarize_results.py"
python "$ROOT/scripts/validate_results.py"
python "$ROOT/scripts/build_application_evidence.py"
printf '%s\n' 'Complete simulation run, evidence extraction, and validation finished successfully.'
