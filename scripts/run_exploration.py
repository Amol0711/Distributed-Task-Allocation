#!/usr/bin/env python3
"""Run matched exploration simulations for the two scenarios."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from campaign_engine import sha256_file  # noqa: E402
from exploration_campaign import load_raw, raw_name, run_trial  # noqa: E402

CONFIG = ROOT / "configs" / "exploration.json"
CAMPAIGNS = {
    "SAT-COV-V1": ROOT / "configs" / "satellite_campaign.json",
    "UAV-AG-V1": ROOT / "configs" / "uav_campaign.json",
}


def source_bundle_sha256(paths: list[Path]) -> str:
    """Hash the complete exploration-certificate source bundle deterministically."""
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_seeds(partition: str, trials_per_campaign: int) -> list[dict[str, str]]:
    with (ROOT / "seeds" / "trials.csv").open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["partition"] == partition]
    selected: list[dict[str, str]] = []
    for campaign_id in CAMPAIGNS:
        subset = [r for r in rows if r["campaign_id"] == campaign_id]
        subset.sort(key=lambda r: int(r["trial_index"]))
        if trials_per_campaign > 0:
            subset = subset[:trials_per_campaign]
        selected.extend(subset)
    return selected


def existing_job_summary(*, seed: dict[str, str], output_dir: Path, c_values: tuple[float, ...], epoch_limit: int | None) -> dict[str, Any] | None:
    """Return a compact verified summary when every requested raw file exists.

    This resume path validates immutable identifiers, contiguous epochs, row
    counts, and file hashes.  The full arithmetic replay remains delegated to
    ``validate_results.py`` after the resumed campaign closes.
    """
    files: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, Any]] = {}
    expected_epochs: int | None = epoch_limit
    for c in c_values:
        path = output_dir / raw_name(
            seed["campaign_id"], seed["partition"], int(seed["trial_index"]),
            int(seed["trial_seed"]), c,
        )
        if not path.exists():
            return None
        rows = load_raw(path)
        if not rows:
            return None
        first, last = rows[0], rows[-1]
        if first.get("schema_version") != "exploration-campaign-v2":
            return None
        if first.get("campaign_id") != seed["campaign_id"]:
            return None
        if int(first.get("trial_index", -1)) != int(seed["trial_index"]):
            return None
        if int(first.get("trial_seed", -1)) != int(seed["trial_seed"]):
            return None
        if abs(float(first.get("c_exp", "nan")) - c) > 1.0e-14:
            return None
        if any(int(row["epoch"]) != index for index, row in enumerate(rows, 1)):
            return None
        if expected_epochs is not None and len(rows) != expected_epochs:
            return None
        if expected_epochs is None:
            expected_epochs = len(rows)
        files.append({
            "relative_path": path.as_posix(),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "c_exp": c,
        })
        aggregates[first["variant_id"]] = {
            "c_exp": c,
            "epochs": len(rows),
            "cumulative_oracle_ratio": float(last["cumulative_oracle_ratio"]),
            "universal_normalized_observable_bound": float(last["universal_normalized_observable_bound"]),
            "support_universal_normalized_observable_bound": float(last["support_universal_normalized_observable_bound"]),
            "prior_universal_normalized_observable_bound": float(last["prior_universal_normalized_observable_bound"]),
        }
    return {
        "status": "PASS",
        "resumed_from_existing": True,
        "campaign_id": seed["campaign_id"],
        "partition": seed["partition"],
        "trial_index": int(seed["trial_index"]),
        "trial_seed": int(seed["trial_seed"]),
        "epochs": expected_epochs,
        "files": files,
        "aggregates": aggregates,
    }


def worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return run_trial(
        sim_root=ROOT,
        campaign_path=CAMPAIGNS[job["seed"]["campaign_id"]],
        experiment_config=job["config"],
        seed_row=job["seed"],
        raw_dir=Path(job["output_dir"]),
        engine_hash=job["engine_hash"],
        experiment_config_hash=job["experiment_config_hash"],
        seed_registry_hash=job["seed_registry_hash"],
        c_values=job["c_values"],
        write_raw=True,
        epoch_limit=job["epoch_limit"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("design", "evaluation"), default="evaluation")
    parser.add_argument("--trials-per-campaign", type=int, default=0,
                        help="0 runs every available trial")
    parser.add_argument("--epochs", type=int, default=0,
                        help="0 uses the configured horizon")
    parser.add_argument("--c-exp", type=float, nargs="+", default=(0.0, 0.25),
                        help="sorted exploration coefficients")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true",
                        help="reuse complete deterministic v2 files and run only missing trials")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "generated" / "raw" / "exploration")
    args = parser.parse_args()
    c_values = tuple(float(x) for x in args.c_exp)
    if list(c_values) != sorted(set(c_values)) or any(x < 0.0 for x in c_values):
        raise SystemExit("--c-exp values must be unique, sorted, and nonnegative")
    if args.trials_per_campaign < 0 or args.epochs < 0 or args.workers < 1:
        raise SystemExit("trial, epoch, and worker arguments must be nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    seeds = load_seeds(args.partition, args.trials_per_campaign)
    engine_hash = source_bundle_sha256([
        ROOT / "src" / "exploration_campaign.py",
        ROOT / "src" / "certificate_arithmetic.py",
    ])
    experiment_config_hash = sha256_file(CONFIG)
    seed_registry_hash = sha256_file(ROOT / "seeds" / "trials.csv")
    summaries: list[dict[str, Any]] = []
    pending_seeds: list[dict[str, str]] = []
    for seed in seeds:
        existing = existing_job_summary(
            seed=seed,
            output_dir=args.output_dir,
            c_values=c_values,
            epoch_limit=args.epochs or None,
        ) if args.resume else None
        if existing is None:
            pending_seeds.append(seed)
        else:
            summaries.append(existing)
    jobs = [{
        "seed": seed,
        "output_dir": str(args.output_dir),
        "config": config,
        "engine_hash": engine_hash,
        "experiment_config_hash": experiment_config_hash,
        "seed_registry_hash": seed_registry_hash,
        "c_values": c_values,
        "epoch_limit": args.epochs or None,
    } for seed in pending_seeds]

    resumed_count = len(summaries)
    if resumed_count:
        print(f"[resume] verified {resumed_count} complete trial(s); "
              f"running {len(jobs)} missing trial(s)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, job): job for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            result = future.result()
            summaries.append(result)
            print(f"[{number:03d}/{len(jobs):03d}] {result['campaign_id']} "
                  f"T{result['trial_index']:02d} exploration PASS", flush=True)

    summaries.sort(key=lambda r: (r["campaign_id"], r["trial_index"]))
    payload = {
        "status": "PASS",
        "partition": args.partition,
        "trials": len(summaries),
        "resumed_trials": resumed_count,
        "executed_trials": len(jobs),
        "c_exp": c_values,
        "engine_sha256": engine_hash,
        "configuration_sha256": experiment_config_hash,
        "seed_registry_sha256": seed_registry_hash,
        "results": summaries,
    }
    output = args.output_dir.parent / "exploration_run_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "trials": len(summaries), "summary": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
