#!/usr/bin/env python3
"""Verify hashes, schemas, and basic consistency of compact reference results."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
MANIFEST = REFERENCE / "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def require_zero(rows: list[dict[str, str]], field: str, path: Path) -> None:
    if rows and field in rows[0]:
        bad = [index for index, row in enumerate(rows, 2) if float(row[field] or 0.0) != 0.0]
        if bad:
            raise ValueError(f"{path}: {field} is nonzero at rows {bad[:8]}")


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"missing reference manifest: {MANIFEST}")
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "reference-results-manifest-v1":
        raise SystemExit("unsupported reference manifest schema")

    entries = manifest.get("files", [])
    if not entries:
        raise SystemExit("reference manifest has no files")
    checked = 0
    for entry in entries:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe manifest path: {rel}")
        path = REFERENCE / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"size mismatch: {path}")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        if path.suffix == ".csv":
            fields, rows = csv_rows(path)
            if len(rows) != int(entry["rows"]):
                raise ValueError(f"row-count mismatch: {path}")
            if not fields:
                raise ValueError(f"missing CSV header: {path}")
            require_zero(rows, "validation_failed_trials", path)
            require_zero(rows, "validation_failures", path)
            require_zero(rows, "feasibility_failures", path)
            require_zero(rows, "full_consensus_mismatches", path)
            require_zero(rows, "catalogue_distributed_mismatches", path)
            require_zero(rows, "catalogue_feasibility_failures", path)
            require_zero(rows, "residual_distributed_mismatches", path)
            require_zero(rows, "residual_feasibility_failures", path)
        checked += 1
    print(json.dumps({"status": "PASS", "files_checked": checked}, sort_keys=True))


if __name__ == "__main__":
    main()
