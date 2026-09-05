#!/usr/bin/env python3
"""Run deterministic baseline simulations for the two synthetic scenarios."""
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
from campaign_engine import RAW_SCHEMA, run_trial, sha256_file  # noqa: E402

CAMPAIGNS = {
    "SAT-COV-V1": ROOT / "configs" / "satellite_campaign.json",
    "UAV-AG-V1": ROOT / "configs" / "uav_campaign.json",
}
METHODS = ("INSTANTANEOUS_MYOPIC",)


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


def load_raw(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def expected_raw_path(output_dir: Path, seed: dict[str, str]) -> Path:
    return output_dir / (
        f"baseline_{seed['campaign_id']}_primary_{seed['partition']}_"
        f"T{int(seed['trial_index']):02d}_S{int(seed['trial_seed'])}_"
        "INSTANTANEOUS_MYOPIC.csv"
    )


def existing_job_summary(*, seed: dict[str, str], output_dir: Path,
                         epoch_limit: int | None) -> dict[str, Any] | None:
    """Return a compact verified summary for a complete deterministic trace."""
    path = expected_raw_path(output_dir, seed)
    if not path.exists():
        return None
    rows = load_raw(path)
    if not rows:
        return None
    first = rows[0]
    if first.get("schema_version") != RAW_SCHEMA:
        return None
    if first.get("campaign_id") != seed["campaign_id"]:
        return None
    if int(first.get("trial_index", -1)) != int(seed["trial_index"]):
        return None
    if int(first.get("trial_seed", -1)) != int(seed["trial_seed"]):
        return None
    if any(int(row["epoch"]) != index for index, row in enumerate(rows, 1)):
        return None
    if epoch_limit is not None and len(rows) != epoch_limit:
        return None
    last = rows[-1]
    return {
        "status": "PASS",
        "resumed_from_existing": True,
        "campaign_id": seed["campaign_id"],
        "scale": "primary",
        "partition": seed["partition"],
        "trial_index": int(seed["trial_index"]),
        "trial_seed": int(seed["trial_seed"]),
        "epochs": len(rows),
        "methods": list(METHODS),
        "files": [{
            "relative_path": path.as_posix(),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "method": "INSTANTANEOUS_MYOPIC",
        }],
        "aggregates": {
            "INSTANTANEOUS_MYOPIC": {
                "epochs": len(rows),
                "cumulative_true_value": sum(float(row["true_value"]) for row in rows),
                "cumulative_realized_return": sum(float(row["realized_return"]) for row in rows),
                "cumulative_oracle_value": sum(float(row["oracle_greedy_value"]) for row in rows),
                "cumulative_oracle_ratio": (
                    sum(float(row["true_value"]) for row in rows)
                    / sum(float(row["oracle_greedy_value"]) for row in rows)
                ),
                "terminal_oracle_ratio": float(last["oracle_value_ratio"]),
            }
        },
    }


def worker(job: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return run_trial(
        sim_root=ROOT,
        campaign_path=CAMPAIGNS[job["seed"]["campaign_id"]],
        seed_row=job["seed"],
        raw_dir=Path(job["output_dir"]),
        engine_hash=job["engine_hash"],
        scale="primary",
        methods=METHODS,
        epoch_limit=job["epoch_limit"],
        write_raw=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=("design", "evaluation"), default="evaluation")
    parser.add_argument("--trials-per-campaign", type=int, default=0,
                        help="0 runs every available trial")
    parser.add_argument("--epochs", type=int, default=0,
                        help="0 uses the configured horizon")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true",
                        help="reuse complete deterministic traces and run only missing trials")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "generated" / "raw" / "baseline")
    args = parser.parse_args()
    if args.trials_per_campaign < 0 or args.epochs < 0 or args.workers < 1:
        raise SystemExit("trial, epoch, and worker arguments must be nonnegative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds(args.partition, args.trials_per_campaign)
    engine_hash = sha256_file(ROOT / "src" / "campaign_engine.py")
    summaries: list[dict[str, Any]] = []
    pending_seeds: list[dict[str, str]] = []
    for seed in seeds:
        existing = existing_job_summary(
            seed=seed, output_dir=args.output_dir, epoch_limit=args.epochs or None
        ) if args.resume else None
        if existing is None:
            pending_seeds.append(seed)
        else:
            summaries.append(existing)
    jobs = [{
        "seed": seed,
        "output_dir": str(args.output_dir),
        "engine_hash": engine_hash,
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
                  f"T{result['trial_index']:02d} baseline PASS", flush=True)

    summaries.sort(key=lambda r: (r["campaign_id"], r["trial_index"]))
    payload = {
        "status": "PASS",
        "partition": args.partition,
        "trials": len(summaries),
        "resumed_trials": resumed_count,
        "executed_trials": len(jobs),
        "engine_sha256": engine_hash,
        "seed_registry_sha256": sha256_file(ROOT / "seeds" / "trials.csv"),
        "results": summaries,
    }
    output = args.output_dir.parent / "baseline_run_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "trials": len(summaries), "summary": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
