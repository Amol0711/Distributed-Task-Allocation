"""Regression tests for fresh, resumed, and mixed estimator-audit CSV output.

Workers are replaced by the frozen per-trial records. These tests verify only
serialization; the separate --reproduce audit re-executes all estimator updates.
"""
from __future__ import annotations
import importlib.util
import json
import sys
from concurrent.futures import Future
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT
EVIDENCE = SIM / 'results/reference/projected_estimator'

@pytest.mark.parametrize('mode', ['fresh', 'resumed', 'mixed'])
def test_canonical_csv(mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = importlib.util.spec_from_file_location('estimator_audit_serialization', SIM / 'scripts/audit_projected_estimator_constraints.py')
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = json.loads((EVIDENCE / 'projected_estimator_constraint_audit.json').read_text())['per_trial']
    lookup = {(r['campaign_id'], int(r['trial_index'])): dict(reversed(list(r.items()))) for r in rows}
    class ImmediateExecutor:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def submit(self, fn, seed):
            future = Future()
            future.set_result(lookup[(seed['campaign_id'], int(seed['trial_index']))])
            return future
    monkeypatch.setattr(module, 'ProcessPoolExecutor', ImmediateExecutor)
    out = tmp_path / mode
    trial = out / 'projected_estimator_trials'
    trial.mkdir(parents=True)
    selected = rows if mode == 'resumed' else rows[::2] if mode == 'mixed' else []
    for row in selected:
        (trial / f"{row['campaign_id']}_T{int(row['trial_index']):02d}.json").write_text(json.dumps(row, indent=2, sort_keys=True)+'\n')
    monkeypatch.setattr(sys, 'argv', ['audit', '--output-dir', str(out), '--workers', '1', '--resume'])
    module.main()
    for name in ['projected_estimator_constraint_audit.csv', 'projected_estimator_constraint_audit.json']:
        assert (out / name).read_bytes() == (EVIDENCE / name).read_bytes()
    assert b'\r\n' not in (out / 'projected_estimator_constraint_audit.csv').read_bytes()
