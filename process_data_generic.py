#!/usr/bin/env python3
"""Generic subtask annotation entrypoint.

Example:
    python process_data_generic.py \
        --task_name adjust_bottle \
        --data_dir /data2/liujingzhi/robotwin_processed_pi0/adjust_bottle-aloha-agilex_clean_50-50
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_def.base_task import BaseTaskProcessor


DEFAULT_RAW_ROOTS = [
    Path("/data2/liujingzhi/robotwin/robotwin2_dataset"),
    Path("/mnt/data1/liujingzhi/dataset"),
]


def load_task_processor(task_name: str) -> BaseTaskProcessor:
    module = importlib.import_module(f"task_def.{task_name}")
    class_name = f"{task_name.title().replace('_', '')}Processor"
    if hasattr(module, class_name):
        return getattr(module, class_name)()

    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseTaskProcessor) and obj is not BaseTaskProcessor:
            candidates.append(obj)
    if not candidates:
        raise ValueError(f"No BaseTaskProcessor subclass found in task_def/{task_name}.py")
    return candidates[0]()


def parse_episode_id(episode_dir: Path) -> Optional[int]:
    match = re.search(r"episode_(\d+)", episode_dir.name)
    if match:
        return int(match.group(1))
    return None


def infer_setting_from_data_dir(data_dir: Path) -> str:
    name = data_dir.name
    if "aloha-agilex_clean_50" in name or "clean_50" in name:
        return "aloha-agilex_clean_50"
    if "aloha-agilex_randomized_500" in name or "randomized_500" in name:
        return "aloha-agilex_randomized_500"
    if "demo_clean" in name:
        return "demo_clean"
    if "demo_randomized" in name:
        return "demo_randomized"
    return "aloha-agilex_clean_50"


def find_raw_episode_path(
    task_name: str,
    episode_id: int,
    data_dir: Path,
    raw_root: Optional[Path] = None,
) -> Optional[Path]:
    setting = infer_setting_from_data_dir(data_dir)
    roots = [raw_root] if raw_root is not None else DEFAULT_RAW_ROOTS
    for root in roots:
        if root is None:
            continue
        candidate = root / task_name / setting / "data" / f"episode{episode_id}.hdf5"
        if candidate.exists():
            return candidate
    return None


def load_raw_endpose_xyz(raw_episode_path: Optional[Path], total_steps: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw_episode_path is None or not raw_episode_path.exists():
        return None, None
    try:
        with h5py.File(raw_episode_path, "r") as raw_f:
            if "endpose/left_endpose" not in raw_f or "endpose/right_endpose" not in raw_f:
                return None, None
            left_xyz = raw_f["endpose/left_endpose"][()][:total_steps, :3]
            right_xyz = raw_f["endpose/right_endpose"][()][:total_steps, :3]
            return _align_xyz(left_xyz, total_steps), _align_xyz(right_xyz, total_steps)
    except Exception as exc:
        print(f"  [Warn] Failed to read raw endpose from {raw_episode_path}: {exc}")
        return None, None


def load_raw_endpose_pose(raw_episode_path: Optional[Path], total_steps: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw_episode_path is None or not raw_episode_path.exists():
        return None, None
    try:
        with h5py.File(raw_episode_path, "r") as raw_f:
            if "endpose/left_endpose" not in raw_f or "endpose/right_endpose" not in raw_f:
                return None, None
            left_pose = np.asarray(raw_f["endpose/left_endpose"][()][:total_steps], dtype=np.float64)
            right_pose = np.asarray(raw_f["endpose/right_endpose"][()][:total_steps], dtype=np.float64)
            return _align_pose(left_pose, total_steps), _align_pose(right_pose, total_steps)
    except Exception as exc:
        print(f"  [Warn] Failed to read raw endpose pose from {raw_episode_path}: {exc}")
        return None, None


def _align_pose(pose: np.ndarray, total_steps: int) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if len(pose) >= total_steps:
        return pose[:total_steps]
    if len(pose) == 0:
        return pose
    pad = np.repeat(pose[-1:], total_steps - len(pose), axis=0)
    return np.concatenate([pose, pad], axis=0)


def _align_xyz(xyz: np.ndarray, total_steps: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) >= total_steps:
        return xyz[:total_steps, :3]
    if len(xyz) == 0:
        return xyz
    pad = np.repeat(xyz[-1:, :3], total_steps - len(xyz), axis=0)
    return np.concatenate([xyz[:, :3], pad], axis=0)


def process_episode(
    episode_dir: Path,
    data_dir: Path,
    task_name: str,
    task_processor: BaseTaskProcessor,
    raw_root: Optional[Path] = None,
) -> bool:
    hdf5_files = sorted(episode_dir.glob("episode_*.hdf5"))
    if not hdf5_files:
        print(f"Warning: no HDF5 file found in {episode_dir}")
        return False

    hdf5_path = hdf5_files[0]
    instructions_path = episode_dir / "instructions.json"
    if instructions_path.exists():
        with instructions_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"instructions": ["Complete the task."]}

    try:
        with h5py.File(hdf5_path, "r") as f:
            if "action" in f:
                total_steps = int(f["action"].shape[0])
            elif "observations/qpos" in f:
                total_steps = int(f["observations/qpos"].shape[0])
            else:
                total_steps = int(f["qpos"].shape[0])

            episode_id = parse_episode_id(episode_dir)
            raw_path = find_raw_episode_path(task_name, episode_id, data_dir, raw_root) if episode_id is not None else None
            left_xyz, right_xyz = load_raw_endpose_xyz(raw_path, total_steps)
            left_pose, right_pose = load_raw_endpose_pose(raw_path, total_steps)

            checkpoints = task_processor.get_phase_checkpoints(
                f,
                external_eef_xyz_left=left_xyz,
                external_eef_xyz_right=right_xyz,
                external_eef_pose_left=left_pose,
                external_eef_pose_right=right_pose,
            )
    except Exception as exc:
        print(f"Error processing {hdf5_path}: {exc}")
        return False

    checkpoints = task_processor.validate_checkpoints(checkpoints, total_steps)
    num_phases = len(checkpoints) + 1
    descriptions = task_processor.get_subtask_descriptions_for_phases(num_phases)

    instructions = data.get("instructions") or ["Complete the task."]
    if not isinstance(instructions, list):
        instructions = [str(instructions)]

    data["instructions"] = instructions
    data["subtasks"] = [descriptions.copy() for _ in instructions]
    phase_info = {
        "task_name": task_name,
        "checkpoints": [int(cp) for cp in checkpoints],
        "total_steps": int(total_steps),
        "num_phases": int(num_phases),
    }
    if hasattr(task_processor, "mode") and getattr(task_processor, "mode", None):
        phase_info["mode"] = str(task_processor.mode)
    if hasattr(task_processor, "cycle_modes") and getattr(task_processor, "cycle_modes", None):
        phase_info["cycle_modes"] = list(task_processor.cycle_modes)
    if hasattr(task_processor, "phase_kinds") and getattr(task_processor, "phase_kinds", None):
        phase_info["phase_kinds"] = list(task_processor.phase_kinds)
    if hasattr(task_processor, "object_arm") and getattr(task_processor, "object_arm", None):
        phase_info["object_arm"] = str(task_processor.object_arm)
    if hasattr(task_processor, "drawer_arm") and getattr(task_processor, "drawer_arm", None):
        phase_info["drawer_arm"] = str(task_processor.drawer_arm)
    if hasattr(task_processor, "scanner_arm") and getattr(task_processor, "scanner_arm", None):
        phase_info["scanner_arm"] = str(task_processor.scanner_arm)
    if hasattr(task_processor, "cycle_arms") and getattr(task_processor, "cycle_arms", None):
        phase_info["cycle_arms"] = list(task_processor.cycle_arms)
    data["phase_info"] = phase_info

    with instructions_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {episode_dir.name}: {num_phases} phases, checkpoints={checkpoints}")
    return True


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotate Robotwin subtasks with rule-based phase checkpoints.")
    parser.add_argument("--task_name", required=True, help="Task module name under task_def, e.g. adjust_bottle")
    parser.add_argument("--data_dir", required=True, help="processed_data directory containing episode_* folders")
    parser.add_argument("--episode_range", default=None, help="Episode ids, e.g. '0-10' or '0,5,10'")
    parser.add_argument("--raw_root", default=None, help="Optional raw dataset root containing task/setting/data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data_dir does not exist: {data_dir}")
        return 1

    processor = load_task_processor(args.task_name)
    raw_root = Path(args.raw_root) if args.raw_root else None
    episode_dirs = select_episode_dirs(data_dir, args.episode_range)
    if not episode_dirs:
        print(f"Warning: no episode_* directories found in {data_dir}")
        return 1

    print(f"Task: {args.task_name}")
    print(f"Data: {data_dir}")
    print(f"Episodes: {len(episode_dirs)}")

    success_count = 0
    for episode_dir in episode_dirs:
        if process_episode(episode_dir, data_dir, args.task_name, processor, raw_root=raw_root):
            success_count += 1

    print(f"Done. Successfully processed {success_count}/{len(episode_dirs)} episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
