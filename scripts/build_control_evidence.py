#!/usr/bin/env python3
"""Build deterministic controller and reference-reset certificates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reference_reset import load_reference_reset_certificates  # noqa: E402
from tracking_models import build_tracking_bounds, load_json  # noqa: E402

REFERENCE_CONFIG = ROOT / "configs" / "reference_reset_library.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.17g}"
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row[field]) for field in fields})


def safe_source_entry(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    root = ROOT.resolve()
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"source file escapes repository root: {path}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "reference" / "control_certificates",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    library_payload = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    library_entries = library_payload["applications"]
    certificates = load_reference_reset_certificates(ROOT)

    reference_summary_rows: list[dict[str, Any]] = []
    offset_rows: list[dict[str, Any]] = []
    reference_pair_rows: list[dict[str, Any]] = []
    controller_summary_rows: list[dict[str, Any]] = []
    controller_mode_rows: list[dict[str, Any]] = []
    controller_pair_rows: list[dict[str, Any]] = []

    source_paths: set[Path] = {
        REFERENCE_CONFIG,
        ROOT / "src" / "reference_reset.py",
        ROOT / "src" / "tracking_models.py",
        ROOT / "src" / "campaign_engine.py",
        Path(__file__).resolve(),
    }

    for campaign_id, cert in certificates.items():
        entry = library_entries[campaign_id]
        tracking_path = ROOT / cert.tracking_config_path
        campaign_path = ROOT / cert.campaign_config_path
        source_paths.update({tracking_path, campaign_path})

        reference_summary_rows.append(
            {
                "campaign_id": campaign_id,
                "application": cert.application,
                "state_dimension": cert.state_dimension,
                "mode_count": len(cert.mode_names),
                "pair_count": len(cert.mode_names) ** 2,
                "tracking_config_path": cert.tracking_config_path,
                "campaign_config_path": cert.campaign_config_path,
                "assignment_map_verified": cert.assignment_map_verified,
                "reference_diameter": cert.reference_diameter,
                "residual_reset_bound": cert.residual_reset_bound,
                "certified_jump_bound": cert.certified_jump_bound,
                "configured_jump_bound": cert.configured_jump_bound,
                "certification_margin": cert.certification_margin,
                "maximizing_pairs": "|".join(f"{a}->{b}" for a, b in cert.maximizing_pairs),
                "all_pairs_certified": 1,
            }
        )
        for mode in cert.mode_names:
            offset_rows.append(
                {
                    "campaign_id": campaign_id,
                    "mode": mode,
                    "offset": "|".join(f"{value:.17g}" for value in cert.offsets[mode]),
                    "offset_norm": float(np.linalg.norm(cert.offsets[mode])),
                }
            )
        for source in cert.mode_names:
            for target in cert.mode_names:
                vector = cert.jump_vector(source, target)
                reference_pair_rows.append(
                    {
                        "campaign_id": campaign_id,
                        "source_mode": source,
                        "target_mode": target,
                        "jump_vector": "|".join(f"{value:.17g}" for value in vector),
                        "jump_norm": cert.jump_norm(source, target),
                        "certified_jump_bound": cert.certified_jump_bound,
                        "within_envelope": 1,
                    }
                )

        tracking_config = load_json(tracking_path)
        bounds, mode_checks, pair_checks = build_tracking_bounds(tracking_config)
        controller_summary_rows.append(
            {
                "campaign_id": campaign_id,
                "application": bounds.application,
                "state_dimension": next(iter(bounds.modes.values())).a_cl.shape[0],
                "mode_count": len(bounds.modes),
                "within_mode_test_count": len(mode_checks),
                "cross_mode_test_count": len(pair_checks),
                "v_lower": bounds.v_lower,
                "v_upper": bounds.v_upper,
                "lambda_floor": bounds.lambda_floor,
                "lambda_c": bounds.lambda_c,
                "mu_floor": bounds.mu_floor,
                "mu": bounds.mu,
                "c_w": bounds.c_w,
                "c_jump": bounds.c_jump,
                "rho_h": bounds.rho_h,
                "horizon": bounds.horizon,
                "minimum_horizon": bounds.minimum_horizon,
                "w_bound": bounds.w_bound,
                "jump_bound": bounds.jump_bound,
                "initial_error_bound": bounds.initial_error_bound,
                "all_step_radius": bounds.all_step_radius,
                "tracking_radius_bound": bounds.tracking_radius_bound,
                "g_w": bounds.g_w,
                "g_jump": bounds.g_jump,
                "control_bound": bounds.control_bound,
                "control_limit": bounds.control_limit,
                "validity_radius": bounds.validity_radius,
                "minimum_within_mode_block_margin": bounds.minimum_within_mode_block_margin,
                "minimum_jump_block_margin": bounds.minimum_jump_block_margin,
                "within_mode_checks_pass": int(bounds.minimum_within_mode_block_margin > 0.0),
                "cross_mode_checks_pass": int(bounds.minimum_jump_block_margin > 0.0),
                "dwell_check_pass": int(bounds.rho_h < 1.0 and bounds.horizon >= bounds.minimum_horizon),
                "actuator_gate_pass": int(bounds.control_bound < bounds.control_limit),
                "validity_gate_pass": int(bounds.tracking_radius_bound < bounds.validity_radius),
                "product_mode_lift": "max-composed Lyapunov function under block-maximum norm",
            }
        )
        for row in mode_checks:
            controller_mode_rows.append({"campaign_id": campaign_id, **row})
        for row in pair_checks:
            controller_pair_rows.append({"campaign_id": campaign_id, **row})

        if set(entry["mode_levels"]) != set(bounds.modes):
            raise ValueError(f"{campaign_id}: reference modes and controller modes differ")

    outputs: list[Path] = []

    def emit(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
        path = output / name
        write_csv(path, rows, fields)
        outputs.append(path)

    emit(
        "reference_reset_summary.csv",
        reference_summary_rows,
        [
            "campaign_id", "application", "state_dimension", "mode_count", "pair_count",
            "tracking_config_path", "campaign_config_path", "assignment_map_verified",
            "reference_diameter", "residual_reset_bound", "certified_jump_bound",
            "configured_jump_bound", "certification_margin", "maximizing_pairs",
            "all_pairs_certified",
        ],
    )
    emit(
        "reference_reset_offsets.csv",
        offset_rows,
        ["campaign_id", "mode", "offset", "offset_norm"],
    )
    emit(
        "reference_reset_pairs.csv",
        reference_pair_rows,
        [
            "campaign_id", "source_mode", "target_mode", "jump_vector", "jump_norm",
            "certified_jump_bound", "within_envelope",
        ],
    )
    emit(
        "controller_template_summary.csv",
        controller_summary_rows,
        [
            "campaign_id", "application", "state_dimension", "mode_count",
            "within_mode_test_count", "cross_mode_test_count", "v_lower", "v_upper",
            "lambda_floor", "lambda_c", "mu_floor", "mu", "c_w", "c_jump",
            "rho_h", "horizon", "minimum_horizon", "w_bound", "jump_bound",
            "initial_error_bound", "all_step_radius", "tracking_radius_bound", "g_w",
            "g_jump", "control_bound", "control_limit", "validity_radius",
            "minimum_within_mode_block_margin", "minimum_jump_block_margin",
            "within_mode_checks_pass", "cross_mode_checks_pass", "dwell_check_pass",
            "actuator_gate_pass", "validity_gate_pass", "product_mode_lift",
        ],
    )
    emit(
        "controller_mode_checks.csv",
        controller_mode_rows,
        [
            "campaign_id", "application", "mode", "state_dimension", "input_dimension",
            "spectral_radius", "contraction_floor", "lambda_c", "c_w_mode",
            "within_mode_block_min_eigenvalue", "p_min_eigenvalue", "p_max_eigenvalue",
            "feedback_spectral_norm", "fallback_policy",
        ],
    )
    emit(
        "controller_pair_checks.csv",
        controller_pair_rows,
        [
            "campaign_id", "application", "source_mode", "target_mode",
            "comparison_floor", "mu", "c_jump_pair", "jump_block_min_eigenvalue",
        ],
    )

    manifest = {
        "schema": "control-evidence-v2",
        "status": "PASS",
        "applications": len(reference_summary_rows),
        "controller_templates_checked": len(controller_mode_rows),
        "controller_within_mode_tests": len(controller_mode_rows),
        "controller_cross_mode_tests": len(controller_pair_rows),
        "reference_mode_pairs_checked": len(reference_pair_rows),
        "mode_pairs_checked": len(reference_pair_rows),
        "policy_trace_used": False,
        "certificate_timing": "pre-execution",
        "product_mode_lift": "max-composed Lyapunov function under block-maximum norm",
        "source_files": [safe_source_entry(path) for path in sorted(source_paths)],
        "files": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in outputs
        ],
    }
    manifest_path = output / "CONTROL_EVIDENCE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "applications": manifest["applications"],
                "controller_templates_checked": manifest["controller_templates_checked"],
                "controller_cross_mode_tests": manifest["controller_cross_mode_tests"],
                "reference_mode_pairs_checked": manifest["reference_mode_pairs_checked"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
