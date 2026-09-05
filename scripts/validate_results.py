#!/usr/bin/env python3
"""Validate generated simulation files and independently replay certificate arithmetic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from certificate_arithmetic import (  # noqa: E402
    CertificateValidationError,
    certificate_cap,
    clipped_transfer_increment,
    contextual_comparator_factor,
    fixed_comparator_factor,
    replay_certificate_rows,
)

ABS_TOL = 5.0e-9
REL_TOL = 2.0e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_bundle_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


EXPECTED_ENGINE_HASH = source_bundle_sha256(
    [ROOT / "src" / "exploration_campaign.py", ROOT / "src" / "certificate_arithmetic.py"]
)
EXPECTED_CERTIFICATE_MODULE_HASH = sha256_file(
    ROOT / "src" / "certificate_arithmetic.py"
)

COMMON_REQUIRED = {
    "schema_version",
    "campaign_id",
    "partition",
    "trial_index",
    "trial_seed",
    "epoch",
    "assignment_valid",
    "resource_violations",
    "complete_family_violations",
    "central_distributed_mismatch",
    "allocation_rounds",
    "round_law_expected",
    "finite_time_envelope_slack",
    "control_limit_margin",
    "model_validity_margin",
}

EXPLORATION_V2_REQUIRED = {
    "variant_id",
    "c_exp",
    "engine_hash",
    "certificate_module_hash",
    "exploration_indicator",
    "active_q",
    "f_max",
    "selected_count",
    "empirical_curvature",
    "comparison_factor",
    "fixed_comparator_factor",
    "beta",
    "selected_width_sum",
    "score_quantization_epsilon",
    "cumulative_true_value",
    "cumulative_oracle_value",
    "cumulative_oracle_ratio",
    "realized_exploration_count",
    "bound_arithmetic_holds",
    "certificate_cap",
    "raw_exploitation_charge",
    "raw_observable_bound_increment",
    "observable_bound_increment",
    "certificate_clip_indicator",
    "certificate_clip_excess",
    "cumulative_exploration_bound",
    "cumulative_ucb_bound",
    "cumulative_raw_ucb_bound",
    "cumulative_observable_bound",
    "cumulative_clip_excess",
    "cumulative_clipped_episodes",
    "universal_normalized_observable_bound",
    "support_beta",
    "support_calibration_mismatch",
    "support_raw_exploitation_charge",
    "support_raw_observable_bound_increment",
    "support_observable_bound_increment",
    "support_certificate_clip_indicator",
    "support_certificate_clip_excess",
    "support_cumulative_exploration_bound",
    "support_cumulative_ucb_bound",
    "support_cumulative_raw_ucb_bound",
    "support_cumulative_observable_bound",
    "support_cumulative_clip_excess",
    "support_cumulative_clipped_episodes",
    "support_universal_normalized_observable_bound",
    "support_transfer_enlargement",
    "support_transfer_increment",
    "support_transfer_identity_holds",
    "prior_beta",
    "prior_calibration_mismatch",
    "prior_raw_exploitation_charge",
    "prior_raw_observable_bound_increment",
    "prior_observable_bound_increment",
    "prior_certificate_clip_indicator",
    "prior_certificate_clip_excess",
    "prior_cumulative_exploration_bound",
    "prior_cumulative_ucb_bound",
    "prior_cumulative_raw_ucb_bound",
    "prior_cumulative_observable_bound",
    "prior_cumulative_clip_excess",
    "prior_cumulative_clipped_episodes",
    "prior_universal_normalized_observable_bound",
    "prior_transfer_enlargement",
    "prior_transfer_increment",
    "prior_transfer_identity_holds",
}


def number(value: str, *, path: Path, field: str, row_number: int) -> float:
    if value is None or value == "":
        raise ValueError(f"{path}: row {row_number}: missing numeric field {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: row {row_number}: nonnumeric {field}={value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}: row {row_number}: nonfinite {field}={value!r}")
    return result


def integer(value: str, *, path: Path, field: str, row_number: int) -> int:
    result = number(value, path=path, field=field, row_number=row_number)
    if result != int(result):
        raise ValueError(f"{path}: row {row_number}: noninteger {field}={value!r}")
    return int(result)


def assert_close(
    actual: float,
    expected: float,
    *,
    path: Path,
    field: str,
    row_number: int,
    atol: float = ABS_TOL,
) -> None:
    if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=atol):
        raise ValueError(
            f"{path}: row {row_number}: {field}={actual:.17g}, "
            f"expected {expected:.17g}"
        )


def _check_certificate_primitives(rows: list[dict[str, str]], path: Path) -> None:
    """Recompute raw direct/support/prior charges from primitive logged fields."""
    engine_hashes = {row["engine_hash"] for row in rows}
    module_hashes = {row["certificate_module_hash"] for row in rows}
    if len(engine_hashes) != 1 or "" in engine_hashes:
        raise ValueError(f"{path}: engine hash is missing or changes within the trace")
    if len(module_hashes) != 1 or "" in module_hashes:
        raise ValueError(
            f"{path}: certificate-module hash is missing or changes within the trace"
        )
    if next(iter(engine_hashes)) != EXPECTED_ENGINE_HASH:
        raise ValueError(f"{path}: engine hash does not match the current source bundle")
    if next(iter(module_hashes)) != EXPECTED_CERTIFICATE_MODULE_HASH:
        raise ValueError(f"{path}: certificate-module hash does not match current source")

    for row_number, row in enumerate(rows, 2):
        q = integer(row["active_q"], path=path, field="active_q", row_number=row_number)
        f_max = number(row["f_max"], path=path, field="f_max", row_number=row_number)
        cap = certificate_cap(f_max=f_max, q=q)
        assert_close(
            number(row["certificate_cap"], path=path, field="certificate_cap", row_number=row_number),
            cap,
            path=path,
            field="certificate_cap",
            row_number=row_number,
        )

        kappa = number(
            row["empirical_curvature"],
            path=path,
            field="empirical_curvature",
            row_number=row_number,
        )
        expected_contextual = contextual_comparator_factor(q=q, curvature=kappa)
        expected_fixed = fixed_comparator_factor(q)
        contextual = number(
            row["comparison_factor"],
            path=path,
            field="comparison_factor",
            row_number=row_number,
        )
        fixed = number(
            row["fixed_comparator_factor"],
            path=path,
            field="fixed_comparator_factor",
            row_number=row_number,
        )
        assert_close(
            contextual,
            expected_contextual,
            path=path,
            field="comparison_factor",
            row_number=row_number,
        )
        assert_close(
            fixed,
            expected_fixed,
            path=path,
            field="fixed_comparator_factor",
            row_number=row_number,
        )
        if fixed > contextual + ABS_TOL or contextual > 1.0 / q + ABS_TOL:
            raise ValueError(f"{path}: row {row_number}: comparator ordering failed")

        selected_count = integer(
            row["selected_count"], path=path, field="selected_count", row_number=row_number
        )
        if selected_count < 0:
            raise ValueError(f"{path}: row {row_number}: negative selected_count")
        width = number(
            row["selected_width_sum"],
            path=path,
            field="selected_width_sum",
            row_number=row_number,
        )
        quant_eps = number(
            row["score_quantization_epsilon"],
            path=path,
            field="score_quantization_epsilon",
            row_number=row_number,
        )
        beta = number(row["beta"], path=path, field="beta", row_number=row_number)
        support_beta = number(
            row["support_beta"], path=path, field="support_beta", row_number=row_number
        )
        prior_beta = number(
            row["prior_beta"], path=path, field="prior_beta", row_number=row_number
        )
        support_mismatch = number(
            row["support_calibration_mismatch"],
            path=path,
            field="support_calibration_mismatch",
            row_number=row_number,
        )
        prior_mismatch = number(
            row["prior_calibration_mismatch"],
            path=path,
            field="prior_calibration_mismatch",
            row_number=row_number,
        )
        for field, value in (
            ("selected_width_sum", width),
            ("score_quantization_epsilon", quant_eps),
            ("beta", beta),
            ("support_beta", support_beta),
            ("prior_beta", prior_beta),
            ("support_calibration_mismatch", support_mismatch),
            ("prior_calibration_mismatch", prior_mismatch),
        ):
            if value < -ABS_TOL:
                raise ValueError(f"{path}: row {row_number}: negative {field}")
        if support_beta + ABS_TOL < beta or prior_beta + ABS_TOL < beta:
            raise ValueError(f"{path}: row {row_number}: target radius is not enlarged")

        quantization_charge = 2.0 * selected_count * quant_eps
        expected_raw = {
            "": 2.0 * beta * width + quantization_charge,
            "support_": 2.0 * support_beta * width
            + quantization_charge
            + support_mismatch,
            "prior_": 2.0 * prior_beta * width
            + quantization_charge
            + prior_mismatch,
        }
        for prefix, expected in expected_raw.items():
            field = f"{prefix}raw_exploitation_charge"
            assert_close(
                number(row[field], path=path, field=field, row_number=row_number),
                expected,
                path=path,
                field=field,
                row_number=row_number,
            )

        explore = bool(
            integer(
                row["exploration_indicator"],
                path=path,
                field="exploration_indicator",
                row_number=row_number,
            )
        )
        baseline_raw = expected_raw[""]
        direct_certified = number(
            row["observable_bound_increment"],
            path=path,
            field="observable_bound_increment",
            row_number=row_number,
        )
        for prefix in ("support_", "prior_"):
            target_raw = expected_raw[prefix]
            expected_enlargement = target_raw - baseline_raw
            if expected_enlargement < -ABS_TOL:
                raise ValueError(
                    f"{path}: row {row_number}: {prefix}raw target is smaller than baseline"
                )
            expected_enlargement = max(0.0, expected_enlargement)
            enlargement_field = f"{prefix}transfer_enlargement"
            assert_close(
                number(
                    row[enlargement_field],
                    path=path,
                    field=enlargement_field,
                    row_number=row_number,
                ),
                expected_enlargement,
                path=path,
                field=enlargement_field,
                row_number=row_number,
            )
            expected_transfer = (
                0.0
                if explore
                else clipped_transfer_increment(
                    baseline_raw_charge=baseline_raw,
                    enlargement=expected_enlargement,
                    cap=cap,
                )
            )
            transfer_field = f"{prefix}transfer_increment"
            assert_close(
                number(
                    row[transfer_field],
                    path=path,
                    field=transfer_field,
                    row_number=row_number,
                ),
                expected_transfer,
                path=path,
                field=transfer_field,
                row_number=row_number,
            )
            target_certified = number(
                row[f"{prefix}observable_bound_increment"],
                path=path,
                field=f"{prefix}observable_bound_increment",
                row_number=row_number,
            )
            assert_close(
                target_certified - direct_certified,
                expected_transfer,
                path=path,
                field=f"{prefix}clipped-transfer residual",
                row_number=row_number,
            )
            if integer(
                row[f"{prefix}transfer_identity_holds"],
                path=path,
                field=f"{prefix}transfer_identity_holds",
                row_number=row_number,
            ) != 1:
                raise ValueError(
                    f"{path}: row {row_number}: {prefix}transfer identity flag failed"
                )


def validate_file(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = COMMON_REQUIRED - fields
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: empty result file")

    schema = rows[0]["schema_version"]
    if schema not in {
        "baseline-campaign-v1",
        "exploration-campaign-v1",
        "exploration-campaign-v2",
    }:
        raise ValueError(f"{path}: unsupported schema {schema!r}")
    if schema == "baseline-campaign-v1" and "method_id" not in fields:
        raise ValueError(f"{path}: baseline file lacks method_id")
    if schema == "exploration-campaign-v1":
        required = {
            "variant_id",
            "c_exp",
            "cumulative_true_value",
            "cumulative_oracle_value",
            "cumulative_oracle_ratio",
            "realized_exploration_count",
            "bound_arithmetic_holds",
        }
        if required - fields:
            raise ValueError(
                f"{path}: exploration-v1 file lacks {sorted(required - fields)}"
            )
    if schema == "exploration-campaign-v2" and EXPLORATION_V2_REQUIRED - fields:
        raise ValueError(
            f"{path}: exploration-v2 file lacks "
            f"{sorted(EXPLORATION_V2_REQUIRED - fields)}"
        )

    constants = (
        "schema_version",
        "campaign_id",
        "partition",
        "trial_index",
        "trial_seed",
    )
    for field in constants:
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"{path}: field {field} is not constant")

    epochs = [
        integer(row["epoch"], path=path, field="epoch", row_number=n)
        for n, row in enumerate(rows, 2)
    ]
    if epochs != list(range(1, len(rows) + 1)):
        raise ValueError(f"{path}: epochs are not contiguous from one")

    failures: list[str] = []
    previous_true = -math.inf
    previous_reference = -math.inf
    previous_explorations = -1
    for row_number, row in enumerate(rows, 2):
        if (
            integer(
                row["assignment_valid"],
                path=path,
                field="assignment_valid",
                row_number=row_number,
            )
            != 1
        ):
            failures.append(f"row {row_number}: invalid assignment")
        for field in (
            "resource_violations",
            "complete_family_violations",
            "central_distributed_mismatch",
        ):
            if integer(row[field], path=path, field=field, row_number=row_number) != 0:
                failures.append(f"row {row_number}: {field} is nonzero")
        rounds = integer(
            row["allocation_rounds"],
            path=path,
            field="allocation_rounds",
            row_number=row_number,
        )
        expected = integer(
            row["round_law_expected"],
            path=path,
            field="round_law_expected",
            row_number=row_number,
        )
        if rounds != expected:
            failures.append(f"row {row_number}: allocation round count mismatch")
        for field in (
            "finite_time_envelope_slack",
            "control_limit_margin",
            "model_validity_margin",
        ):
            value = number(row[field], path=path, field=field, row_number=row_number)
            if value < -1.0e-8:
                failures.append(f"row {row_number}: negative {field}")

        if schema.startswith("exploration-campaign-"):
            if (
                integer(
                    row["bound_arithmetic_holds"],
                    path=path,
                    field="bound_arithmetic_holds",
                    row_number=row_number,
                )
                != 1
            ):
                failures.append(f"row {row_number}: cumulative arithmetic check failed")
            current_true = number(
                row["cumulative_true_value"],
                path=path,
                field="cumulative_true_value",
                row_number=row_number,
            )
            current_reference = number(
                row["cumulative_oracle_value"],
                path=path,
                field="cumulative_oracle_value",
                row_number=row_number,
            )
            current_explorations = integer(
                row["realized_exploration_count"],
                path=path,
                field="realized_exploration_count",
                row_number=row_number,
            )
            if current_true + 1.0e-10 < previous_true:
                failures.append(f"row {row_number}: cumulative true value decreased")
            if current_reference + 1.0e-10 < previous_reference:
                failures.append(f"row {row_number}: cumulative reference value decreased")
            if current_explorations < previous_explorations:
                failures.append(f"row {row_number}: exploration count decreased")
            previous_true = current_true
            previous_reference = current_reference
            previous_explorations = current_explorations

    if failures:
        raise ValueError(f"{path}: " + "; ".join(failures[:12]))

    certificate_replay: dict[str, Any] | None = None
    if schema == "exploration-campaign-v2":
        _check_certificate_primitives(rows, path)
        try:
            certificate_replay = {
                "direct": replay_certificate_rows(rows),
                "support": replay_certificate_rows(rows, prefix="support_"),
                "prior": replay_certificate_rows(rows, prefix="prior_"),
            }
        except CertificateValidationError as exc:
            raise ValueError(f"{path}: certificate replay failed: {exc}") from exc

    return {
        "path": path.as_posix(),
        "schema_version": schema,
        "rows": len(rows),
        "campaign_id": rows[0]["campaign_id"],
        "trial_index": int(rows[0]["trial_index"]),
        "status": "PASS",
        "certificate_replay": certificate_replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "results" / "generated" / "raw",
        help="directory containing generated raw CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "generated" / "validation.json",
        help="machine-readable validation report",
    )
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    paths = sorted(raw_root.rglob("*.csv")) if raw_root.exists() else []
    if not paths:
        raise SystemExit(f"no raw CSV files found under {raw_root}")

    records = [validate_file(path) for path in paths]
    for record in records:
        try:
            record["path"] = Path(record["path"]).relative_to(ROOT).as_posix()
        except ValueError:
            pass
    payload = {
        "status": "PASS",
        "files_checked": len(records),
        "rows_checked": sum(record["rows"] for record in records),
        "certificate_v2_files": sum(
            record["schema_version"] == "exploration-campaign-v2"
            for record in records
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "files_checked": len(records),
                "rows_checked": payload["rows_checked"],
                "certificate_v2_files": payload["certificate_v2_files"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
