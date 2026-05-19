#!/usr/bin/env python3
"""Batch plot open_microwave trajectory analyses."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_NAME = "open_microwave"
ANALYZE_SCRIPT = Path(__file__).resolve().parent / f"analyze_trajectory_{TASK_NAME}.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plot" / TASK_NAME
DEFAULT_RAW_ROOTS = [
    Path("/data2/liujingzhi/robotwin/robotwin2_dataset"),
    Path("/mnt/data1/liujingzhi/dataset"),
]


def parse_episode_id(path: Path) -> Optional[int]:
    match = re.search(r"episode_(\d+)", str(path))
    if match:
        return int(match.group(1))
    return None


def infer_setting(processed_data_dir: Path) -> str:
    name = processed_data_dir.name
    if "clean_50" in name:
        return "aloha-agilex_clean_50"
    if "randomized_500" in name:
        return "aloha-agilex_randomized_500"
    return "aloha-agilex_clean_50"


def find_raw_episode_path(
    episode_id: int,
    processed_data_dir: Path,
    raw_root: Optional[Path] = None,
) -> Optional[Path]:
    setting = infer_setting(processed_data_dir)
    roots = [raw_root] if raw_root is not None else DEFAULT_RAW_ROOTS
    for root in roots:
        if root is None:
            continue
        candidate = root / TASK_NAME / setting / "data" / f"episode{episode_id}.hdf5"
        if candidate.exists():
            return candidate
    return None


def select_episode_dirs(data_dir: Path, episode_range: Optional[str]) -> list[Path]:
    episode_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("episode_")],
        key=lambda p: parse_episode_id(p) if parse_episode_id(p) is not None else -1,
    )
    if not episode_range:
        return episode_dirs

    selected_ids: set[int] = set()
    for part in episode_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            selected_ids.update(range(start, end + 1))
        else:
            selected_ids.add(int(part))
    return [p for p in episode_dirs if parse_episode_id(p) in selected_ids]


def process_all_episodes(
    data_dir: Path,
    output_dir: Path,
    episode_range: Optional[str] = None,
    raw_root: Optional[Path] = None,
) -> int:
    if not ANALYZE_SCRIPT.exists():
        print(f"Error: cannot find analysis script: {ANALYZE_SCRIPT}")
        return 1

    episode_dirs = select_episode_dirs(data_dir, episode_range)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(episode_dirs)} episodes in {data_dir}")
    print(f"Saving plots to {output_dir}")

    success_count = 0
    fail_count = 0

    for episode_dir in episode_dirs:
        hdf5_files = sorted(episode_dir.glob("episode_*.hdf5"))
        if not hdf5_files:
            print(f"{episode_dir.name}: no HDF5 file found, skipping")
            fail_count += 1
            continue

        hdf5_path = hdf5_files[0]
        episode_id = parse_episode_id(episode_dir)
        if episode_id is None:
            print(f"{episode_dir.name}: cannot parse episode id, skipping")
            fail_count += 1
            continue

        save_path = output_dir / f"{TASK_NAME}_episode_{episode_id}_analysis.png"
        cmd = [
            sys.executable,
            str(ANALYZE_SCRIPT),
            str(hdf5_path),
            "--save",
            str(save_path),
        ]

        raw_path = find_raw_episode_path(episode_id, data_dir, raw_root=raw_root)
        if raw_path is not None:
            cmd.extend(["--raw_episode", str(raw_path)])

        print(f"Processing {episode_dir.name}...", end=" ", flush=True)
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if result.returncode == 0:
            print("done")
            success_count += 1
        else:
            print("failed")
            print(result.stderr.strip())
            fail_count += 1

    print(f"Batch plotting complete. Success: {success_count}, Failed: {fail_count}")
    return 0 if fail_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch analyze open_microwave trajectories.")
    parser.add_argument("--data_dir", required=True, help="processed_data directory containing episode_* folders")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save analysis PNGs")
    parser.add_argument("--episode_range", default=None, help="Episode ids, e.g. '0-10' or '0,5,10'")
    parser.add_argument("--raw_root", default=None, help="Optional raw dataset root containing open_microwave/setting/data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data_dir does not exist: {data_dir}")
        return 1

    return process_all_episodes(
        data_dir=data_dir,
        output_dir=Path(args.output_dir),
        episode_range=args.episode_range,
        raw_root=Path(args.raw_root) if args.raw_root else None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
