#!/usr/bin/env python3
"""Verify and optionally regenerate the projected-estimator audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "reference" / "projected_estimator"
SUMMARY_JSON = EVIDENCE / "projected_estimator_constraint_audit.json"
SUMMARY_CSV = EVIDENCE / "projected_estimator_constraint_audit.csv"
TRIAL_DIR = EVIDENCE / "projected_estimator_trials"
EXPECTED = {
    "SAT-COV-V1": {"trials": 32, "updates": 46080, "epochs": 480},
    "UAV-AG-V1": {"trials": 32, "updates": 48000, "epochs": 500},
}
METHODS = ["DISTRIBUTED_UCB", "PROJECTED_MEAN", "UCB_WITHOUT_RESOURCE_FILTER"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalized_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_committed() -> dict[str, Any]:
    require(SUMMARY_JSON.is_file(), f"missing {SUMMARY_JSON}")
    require(SUMMARY_CSV.is_file(), f"missing {SUMMARY_CSV}")
    payload: dict[str, Any] = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))
    require(payload.get("status") == "PASS", "audit status is not PASS")
    require(payload.get("trials") == 64, "audit must contain 64 evaluation trials")
    require(payload.get("methods") == METHODS, "method set mismatch")
    require(
        payload.get("scope") == "all 32 predeclared independent evaluation traces per application",
        "scope statement mismatch",
    )
    require(
        payload.get("seed_registry_sha256") == sha256_file(ROOT / "seeds" / "trials.csv"),
        "seed-registry hash mismatch",
    )
    require(
        payload.get("campaign_engine_sha256") == sha256_file(ROOT / "src" / "campaign_engine.py"),
        "campaign-engine hash mismatch",
    )

    per_trial = payload.get("per_trial")
    require(isinstance(per_trial, list) and len(per_trial) == 64, "per-trial record count mismatch")
    seen: set[tuple[str, int]] = set()
    for row in per_trial:
        campaign = row["campaign_id"]
        trial = int(row["trial_index"])
        require(campaign in EXPECTED, f"unexpected campaign {campaign}")
        require(1 <= trial <= 32, f"invalid trial index {campaign} T{trial}")
        require((campaign, trial) not in seen, f"duplicate trial {campaign} T{trial}")
        seen.add((campaign, trial))
        expected = EXPECTED[campaign]
        require(int(row["epochs"]) == expected["epochs"], f"epoch mismatch {campaign} T{trial}")
        require(int(row["updates"]) == expected["epochs"] * len(METHODS), f"update mismatch {campaign} T{trial}")
        require(row["methods"] == METHODS, f"methods mismatch {campaign} T{trial}")
        for field in ("nonnegative_l1_activations", "nonnegative_l2_activations", "nonnegative_any_activations"):
            require(int(row[field]) == 0, f"constraint activation in {campaign} T{trial}: {field}")
        for field in ("max_full_vs_nonnegative_abs_difference", "max_full_l1_violation", "max_full_l2_violation"):
            require(float(row[field]) <= 1.0e-12, f"nonzero {field} in {campaign} T{trial}")
        require(float(row["min_nonnegative_l1_slack"]) > 0.0, f"nonpositive l1 slack {campaign} T{trial}")
        require(float(row["min_nonnegative_l2_slack"]) > 0.0, f"nonpositive l2 slack {campaign} T{trial}")
        path = TRIAL_DIR / f"{campaign}_T{trial:02d}.json"
        require(path.is_file(), f"missing trial evidence {path}")
        require(json.loads(path.read_text(encoding="utf-8")) == row, f"trial JSON mismatch {path}")

    aggregate = payload.get("aggregate")
    require(isinstance(aggregate, dict) and set(aggregate) == set(EXPECTED), "aggregate campaigns mismatch")
    for campaign, expected in EXPECTED.items():
        row = aggregate[campaign]
        require(int(row["trials"]) == expected["trials"], f"aggregate trial count {campaign}")
        require(int(row["updates"]) == expected["updates"], f"aggregate update count {campaign}")
        for field in ("nonnegative_l1_activations", "nonnegative_l2_activations", "nonnegative_any_activations"):
            require(int(row[field]) == 0, f"aggregate activation {campaign}: {field}")
        for field in ("max_full_vs_nonnegative_abs_difference", "max_full_l1_violation", "max_full_l2_violation"):
            require(float(row[field]) <= 1.0e-12, f"aggregate nonzero {campaign}: {field}")
        require(float(row["min_nonnegative_l1_slack"]) > 0.0, f"aggregate l1 slack {campaign}")
        require(float(row["min_nonnegative_l2_slack"]) > 0.0, f"aggregate l2 slack {campaign}")

    csv_rows = normalized_csv_rows(SUMMARY_CSV)
    require(len(csv_rows) == 64, "summary CSV row count mismatch")
    csv_keys = {(row["campaign_id"], int(row["trial_index"])) for row in csv_rows}
    require(csv_keys == seen, "summary CSV trial set mismatch")
    return payload


def reproduce_byte_identically(workers: int) -> None:
    with tempfile.TemporaryDirectory(prefix="projection-replay-") as directory:
        target = Path(directory) / "projected_estimator"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "audit_projected_estimator_constraints.py"),
            "--workers",
            str(workers),
            "--output-dir",
            str(target),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        require(result.returncode == 0, "audit regeneration failed:\n" + result.stdout + result.stderr)
        committed = sorted(path.relative_to(EVIDENCE) for path in EVIDENCE.rglob("*") if path.is_file())
        regenerated = sorted(path.relative_to(target) for path in target.rglob("*") if path.is_file())
        require(committed == regenerated, "regenerated projected-audit file set differs")
        for rel in committed:
            require((EVIDENCE / rel).read_bytes() == (target / rel).read_bytes(), f"not byte reproducible: {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    payload = validate_committed()
    if args.reproduce:
        reproduce_byte_identically(args.workers)
    print(json.dumps({
        "status": "PASS",
        "trials": payload["trials"],
        "updates": sum(int(v["updates"]) for v in payload["aggregate"].values()),
        "byte_reproducible": bool(args.reproduce),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
