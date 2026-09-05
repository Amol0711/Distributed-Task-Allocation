#!/usr/bin/env python3
"""Verify the exact-oracle audit and optionally regenerate it.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results" / "reference" / "exact_oracle"
SCHEMA = "exact-oracle-audit-v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(directory: Path) -> dict[str, object]:
    summary_path = directory / "exact_oracle_audit.json"
    csv_path = directory / "exact_oracle_instances.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != SCHEMA or summary.get("status") != "PASS":
        raise RuntimeError("unsupported or failed exact-oracle audit")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(summary["instances_total"]):
        raise RuntimeError("exact-oracle row count mismatch")
    output = summary["outputs"]["exact_oracle_instances.csv"]
    if sha256(csv_path) != output["sha256"] or len(rows) != int(output["rows"]):
        raise RuntimeError("exact-oracle output hash/row mismatch")
    for row in rows:
        for field in (
            "exact_optimum_crosscheck",
            "feasibility_and_maximality_checks",
            "finite_channel_checks",
        ):
            if int(row[field]) != 1:
                raise RuntimeError(f"failed {field} at {row['campaign_id']} epoch {row['epoch']}")
        if float(row["zero_shortfall_certificate_slack"]) < -2.0e-10:
            raise RuntimeError("negative zero-shortfall certificate slack")
        if float(row["nonuniform_channel_certificate_slack"]) < -2.0e-10:
            raise RuntimeError("negative nonuniform-channel certificate slack")
    return {
        "status": "PASS",
        "instances_checked": len(rows),
        "enumerated_feasible_sets_total": int(summary["enumerated_feasible_sets_total"]),
        "minimum_zero_shortfall_certificate_slack": summary["minimum_zero_shortfall_certificate_slack"],
        "minimum_nonuniform_channel_certificate_slack": summary["minimum_nonuniform_channel_certificate_slack"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    result = verify(directory)
    if args.rerun:
        print("Rebuilding the exact-oracle audit in a temporary directory...", file=sys.stderr, flush=True)
        with tempfile.TemporaryDirectory(prefix="exact-oracle-") as tmp:
            tmpdir = Path(tmp)
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_exact_oracle_audit.py"), "--output-dir", str(tmpdir)],
                check=True,
                stdout=subprocess.DEVNULL,
                timeout=180,
            )
            for name in ("exact_oracle_audit.json", "exact_oracle_instances.csv"):
                if (directory / name).read_bytes() != (tmpdir / name).read_bytes():
                    raise RuntimeError(f"nondeterministic exact-oracle output: {name}")
        result["deterministic_rerun"] = True
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
