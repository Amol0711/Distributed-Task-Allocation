#!/usr/bin/env python3
"""Verify package-manifest completeness and SHA256SUMS consistency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"
SUMS = ROOT / "SHA256SUMS.txt"
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


def distributable_paths() -> set[str]:
    result: set[str] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.as_posix() in EXCLUDED_ROOT_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        result.add(rel.as_posix())
    return result


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = payload.get("files", [])
    if payload.get("file_count") != len(entries):
        raise SystemExit("PACKAGE_MANIFEST file_count mismatch")
    listed = {entry["path"] for entry in entries}
    actual = distributable_paths()
    if listed != actual:
        raise SystemExit(
            f"package manifest completeness failure: missing={sorted(actual-listed)}, "
            f"stale={sorted(listed-actual)}"
        )
    for entry in entries:
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise SystemExit(f"unsafe package path: {rel}")
        path = ROOT / rel
        if path.stat().st_size != int(entry["size_bytes"]):
            raise SystemExit(f"size mismatch: {rel}")
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit(f"hash mismatch: {rel}")

    parsed: dict[str, str] = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        parsed[rel] = digest
    expected_sums = listed | {"PACKAGE_MANIFEST.json"}
    if set(parsed) != expected_sums:
        raise SystemExit("SHA256SUMS path set mismatch")
    for rel, digest in parsed.items():
        if sha256_file(ROOT / rel) != digest:
            raise SystemExit(f"SHA256SUMS mismatch: {rel}")
    print(json.dumps({"status": "PASS", "package_files": len(entries),
                      "sha256_entries": len(parsed)}, sort_keys=True))


if __name__ == "__main__":
    main()
