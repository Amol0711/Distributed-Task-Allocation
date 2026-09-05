#!/usr/bin/env python3
"""Write deterministic file inventories for a prepared simulation release.

This command records bytes already present. It does not execute simulations or
assert that an unexecuted test has passed. Run the numerical verifiers before
refreshing an intentionally changed release.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference"
EXCLUDED_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "generated", "quick_generated",
}
EXCLUDED_ROOT_FILES = {"PACKAGE_MANIFEST.json", "SHA256SUMS.txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(path: Path, base: Path) -> dict:
    entry = {"path": path.relative_to(base).as_posix(),
             "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            entry["rows"] = sum(1 for _ in reader)
    return entry


def release_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            raise ValueError(f"symbolic links are not distributable: {relative}")
        if relative.as_posix() in EXCLUDED_ROOT_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return files


def main() -> None:
    reference_manifest = REFERENCE / "MANIFEST.json"
    reference_files = [p for p in sorted(REFERENCE.rglob("*"))
                       if p.is_file() and p != reference_manifest]
    reference_payload = {
        "schema_version": "reference-results-manifest-v1",
        "description": "Reference simulation tables and independently verifiable numerical audits.",
        "files": [inventory(p, REFERENCE) for p in reference_files],
    }
    reference_manifest.write_text(json.dumps(reference_payload, indent=2, sort_keys=True) + "\n")
    files = release_files()
    payload = {
        "package_name": "Distributed Task Allocation Simulations",
        "package_version": (ROOT / "VERSION").read_text().strip(),
        "schema_version": "simulation-package-manifest-v1",
        "scope": "Synthetic data, simulation code, numerical records, and executable verification.",
        "file_count": len(files),
        "files": [inventory(p, ROOT) for p in files],
    }
    manifest = ROOT / "PACKAGE_MANIFEST.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [f"{sha256_file(p)}  {p.relative_to(ROOT).as_posix()}"
             for p in sorted([manifest, *files])]
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "WRITTEN", "package_files": len(files),
                      "reference_files": len(reference_files),
                      "sha256_entries": len(lines)}, sort_keys=True))


if __name__ == "__main__":
    main()
