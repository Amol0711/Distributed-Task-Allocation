#!/usr/bin/env python3
"""Verify and byte-reproduce controller and reference-reset certificates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "reference" / "control_certificates"
MANIFEST = EVIDENCE / "CONTROL_EVIDENCE_MANIFEST.json"
EXPECTED_APPLICATIONS = {"SAT-COV-V1", "UAV-AG-V1"}
EXPECTED_BOUNDS = {"SAT-COV-V1": 0.030, "UAV-AG-V1": 0.035}
ABS_TOL = 5.0e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_hashed_entry(base: Path, entry: dict[str, Any]) -> None:
    rel = Path(entry["path"])
    require(not rel.is_absolute() and ".." not in rel.parts, f"unsafe evidence path: {rel}")
    path = base / rel
    require(path.is_file(), f"missing hashed file: {path}")
    require(path.stat().st_size == int(entry["size_bytes"]), f"size mismatch: {path}")
    require(sha256_file(path) == entry["sha256"], f"hash mismatch: {path}")


def validate_committed_evidence() -> dict[str, Any]:
    require(MANIFEST.is_file(), f"missing control evidence manifest: {MANIFEST}")
    manifest: dict[str, Any] = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "control-evidence-v2", "unsupported schema")
    require(manifest.get("status") == "PASS", "control evidence status is not PASS")
    require(manifest.get("applications") == 2, "expected two application certificates")
    require(manifest.get("controller_templates_checked") == 8, "expected eight local templates")
    require(manifest.get("controller_within_mode_tests") == 8, "expected eight within-mode tests")
    require(manifest.get("controller_cross_mode_tests") == 32, "expected 32 controller-pair tests")
    require(manifest.get("reference_mode_pairs_checked") == 32, "expected 32 reference-pair tests")
    require(manifest.get("mode_pairs_checked") == 32, "compatibility mode-pair count mismatch")
    require(manifest.get("policy_trace_used") is False, "control evidence must not use a policy trace")
    require(manifest.get("certificate_timing") == "pre-execution", "certificate must be pre-execution")
    require(
        manifest.get("product_mode_lift") == "max-composed Lyapunov function under block-maximum norm",
        "inconsistent maximum-composed product-mode lift",
    )

    listed = {entry["path"]: entry for entry in manifest.get("files", [])}
    expected_files = {
        "reference_reset_summary.csv",
        "reference_reset_offsets.csv",
        "reference_reset_pairs.csv",
        "controller_template_summary.csv",
        "controller_mode_checks.csv",
        "controller_pair_checks.csv",
    }
    require(set(listed) == expected_files, f"unexpected control evidence file set: {sorted(listed)}")
    for entry in listed.values():
        check_hashed_entry(EVIDENCE, entry)

    source_entries = manifest.get("source_files", [])
    require(len(source_entries) >= 8, "source provenance is incomplete")
    for entry in source_entries:
        check_hashed_entry(ROOT, entry)

    reference_summary = read_csv(EVIDENCE / "reference_reset_summary.csv")
    offsets = read_csv(EVIDENCE / "reference_reset_offsets.csv")
    reference_pairs = read_csv(EVIDENCE / "reference_reset_pairs.csv")
    controller_summary = read_csv(EVIDENCE / "controller_template_summary.csv")
    controller_modes = read_csv(EVIDENCE / "controller_mode_checks.csv")
    controller_pairs = read_csv(EVIDENCE / "controller_pair_checks.csv")

    require(len(reference_summary) == 2, "reference summary must have two rows")
    require(len(offsets) == 8, "offset table must have eight rows")
    require(len(reference_pairs) == 32, "reference pair table must have 32 rows")
    require(len(controller_summary) == 2, "controller summary must have two rows")
    require(len(controller_modes) == 8, "controller mode table must have eight rows")
    require(len(controller_pairs) == 32, "controller pair table must have 32 rows")
    require(
        {row["campaign_id"] for row in reference_summary} == EXPECTED_APPLICATIONS,
        "application set mismatch",
    )

    for row in reference_summary:
        campaign = row["campaign_id"]
        expected = EXPECTED_BOUNDS[campaign]
        require(int(row["mode_count"]) == 4, f"{campaign}: expected four modes")
        require(int(row["pair_count"]) == 16, f"{campaign}: expected 16 reference pairs")
        require(row["assignment_map_verified"] == "1", f"{campaign}: assignment map not verified")
        require(row["all_pairs_certified"] == "1", f"{campaign}: reference-pair certification failed")
        require(float(row["residual_reset_bound"]) == 0.0, f"{campaign}: residual reset must be zero")
        for field in ("reference_diameter", "certified_jump_bound", "configured_jump_bound"):
            require(
                math.isclose(float(row[field]), expected, rel_tol=0.0, abs_tol=ABS_TOL),
                f"{campaign}: {field} does not match {expected}",
            )
        require(float(row["certification_margin"]) >= -ABS_TOL, f"{campaign}: negative margin")

    pair_counts: dict[str, int] = {campaign: 0 for campaign in EXPECTED_APPLICATIONS}
    for row in reference_pairs:
        campaign = row["campaign_id"]
        require(campaign in EXPECTED_APPLICATIONS, f"unknown campaign in reference pair table: {campaign}")
        pair_counts[campaign] += 1
        require(row["within_envelope"] == "1", f"{campaign}: reference pair outside envelope")
        require(
            float(row["jump_norm"]) <= float(row["certified_jump_bound"]) + ABS_TOL,
            f"{campaign}: reference pair norm exceeds envelope",
        )
    require(all(count == 16 for count in pair_counts.values()), f"reference-pair counts: {pair_counts}")

    for row in controller_summary:
        campaign = row["campaign_id"]
        require(campaign in EXPECTED_APPLICATIONS, f"unknown controller campaign: {campaign}")
        require(int(row["mode_count"]) == 4, f"{campaign}: controller mode count mismatch")
        require(int(row["within_mode_test_count"]) == 4, f"{campaign}: within-mode count mismatch")
        require(int(row["cross_mode_test_count"]) == 16, f"{campaign}: cross-mode count mismatch")
        for field in (
            "within_mode_checks_pass",
            "cross_mode_checks_pass",
            "dwell_check_pass",
            "actuator_gate_pass",
            "validity_gate_pass",
        ):
            require(row[field] == "1", f"{campaign}: {field} failed")
        require(float(row["lambda_floor"]) < float(row["lambda_c"]) < 1.0, f"{campaign}: lambda check")
        require(float(row["mu_floor"]) < float(row["mu"]), f"{campaign}: mu check")
        require(float(row["rho_h"]) < 1.0, f"{campaign}: dwell contraction failed")
        require(float(row["minimum_within_mode_block_margin"]) > 0.0, f"{campaign}: mode LMI margin")
        require(float(row["minimum_jump_block_margin"]) > 0.0, f"{campaign}: jump LMI margin")
        require(float(row["control_bound"]) < float(row["control_limit"]), f"{campaign}: actuator gate")
        require(
            float(row["tracking_radius_bound"]) < float(row["validity_radius"]),
            f"{campaign}: model-validity gate",
        )
        require(
            math.isclose(float(row["jump_bound"]), EXPECTED_BOUNDS[campaign], abs_tol=ABS_TOL),
            f"{campaign}: controller/reference jump envelope mismatch",
        )
        require(
            row["product_mode_lift"] == "max-composed Lyapunov function under block-maximum norm",
            f"{campaign}: product-mode lift mismatch",
        )

    for row in controller_modes:
        require(float(row["spectral_radius"]) < 1.0, "unstable controller template")
        require(float(row["contraction_floor"]) < float(row["lambda_c"]), "within-mode ratio failed")
        require(float(row["within_mode_block_min_eigenvalue"]) > 0.0, "within-mode block failed")
        require(float(row["p_min_eigenvalue"]) > 0.0, "nonpositive Lyapunov matrix")
    for row in controller_pairs:
        require(float(row["comparison_floor"]) < float(row["mu"]), "cross-mode ratio failed")
        require(float(row["jump_block_min_eigenvalue"]) > 0.0, "cross-mode block failed")

    return manifest


def reproduce_byte_identically() -> None:
    with tempfile.TemporaryDirectory(prefix="control-replay-") as tmp:
        regenerated = Path(tmp) / "control_certificates"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_control_evidence.py"),
            "--output-dir",
            str(regenerated),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        require(
            result.returncode == 0,
            "control evidence regeneration failed:\n" + result.stdout + result.stderr,
        )
        committed_names = sorted(path.name for path in EVIDENCE.iterdir() if path.is_file())
        regenerated_names = sorted(path.name for path in regenerated.iterdir() if path.is_file())
        require(committed_names == regenerated_names, "regenerated control evidence file set differs")
        for name in committed_names:
            require(
                (EVIDENCE / name).read_bytes() == (regenerated / name).read_bytes(),
                f"control evidence is not byte-reproducible: {name}",
            )


def main() -> None:
    manifest = validate_committed_evidence()
    reproduce_byte_identically()
    print(
        json.dumps(
            {
                "status": "PASS",
                "applications": manifest["applications"],
                "controller_templates_checked": manifest["controller_templates_checked"],
                "controller_cross_mode_tests": manifest["controller_cross_mode_tests"],
                "reference_mode_pairs_checked": manifest["reference_mode_pairs_checked"],
                "byte_reproducible": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
