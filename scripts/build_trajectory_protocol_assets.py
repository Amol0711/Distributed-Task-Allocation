#!/usr/bin/env python3
"""Build the public trajectory-microcase codebook and exploration schedule."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trajectory_microcase import (  # noqa: E402
    exploration_codebook_rows,
    exploration_schedule_rows,
    load_config,
    write_csv,
)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "trajectory_microcase.json")
    seed_dir = ROOT / "seeds"
    codebook_path = seed_dir / "trajectory_microcase_codebook.csv"
    schedule_path = seed_dir / "trajectory_microcase_exploration_schedule.csv"
    write_csv(codebook_path, exploration_codebook_rows(cfg))
    write_csv(schedule_path, exploration_schedule_rows(cfg))
    print(f"wrote {codebook_path.relative_to(ROOT)}")
    print(f"wrote {schedule_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
