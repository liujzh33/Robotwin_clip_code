#!/usr/bin/env python3
"""Enrich pick_dual_bottles subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach both bottles with both arms.",
        "Move both grippers toward the bottles.",
        "Guide each end-effector closer to a bottle.",
        "Bring the robot arms near the two bottles.",
        "Move into position around both bottles.",
        "Reach toward the two bottles.",
        "Position the grippers near the bottle bodies.",
        "Navigate both arms toward the red-capped bottle and the ridged bottle.",
        "Move closer to the bottles before grasping them.",
        "Approach the two bottles and prepare each gripper for grasping.",
    ],
    1: [
        "Grasp both bottles.",
        "Close both grippers.",
        "Pick up the bottles.",
        "Secure each bottle with a gripper.",
        "Clamp onto the two bottles.",
        "Take hold of both bottles.",
        "Grip the bottles firmly.",
        "Close the fingers around the bottle bodies.",
        "Hold the red-capped bottle and the ridged bottle.",
        "Complete the two-arm grasp so both bottles can be moved.",
    ],
    2: [
        "Move both bottles to the center.",
        "Carry the bottles inward.",
        "Transport the two bottles toward the middle.",
        "Move the grasped bottles to the central area.",
        "Bring both bottles to the center position.",
        "Guide the bottles into the middle of the workspace.",
        "Shift the bottles toward the center.",
        "Move each held bottle into the shared central area.",
        "Carry the two different bottles toward the center together.",
        "Complete the task by moving both grasped bottles to the central position.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Pick up both bottles and move them to the center."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 3


def enrich_episode(json_path: Path) -> bool:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("instructions") or [DEFAULT_INSTRUCTION]
    if not isinstance(instructions, list):
        instructions = [str(instructions)]

    num_phases = infer_num_phases(data)
    subtasks_list = []
    for _ in instructions:
        phase_texts = []
        for phase_idx in range(num_phases):
            variants = PHASE_VARIANTS.get(phase_idx, FALLBACK_VARIANTS)
            phase_texts.append(random.choice(variants))
        subtasks_list.append(phase_texts)

    data["instructions"] = instructions
    data["subtasks"] = subtasks_list

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich pick_dual_bottles subtask language templates.")
    parser.add_argument("--data_dir", required=True, help="processed_data directory containing episode_* folders")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: data_dir does not exist: {data_dir}")
        return 1

    episode_dirs = sorted(glob.glob(os.path.join(str(data_dir), "episode_*")))
    success_count = 0
    for episode_dir in episode_dirs:
        json_path = Path(episode_dir) / "instructions.json"
        if not json_path.exists():
            continue
        try:
            if enrich_episode(json_path):
                success_count += 1
        except Exception as exc:
            print(f"Error processing {json_path}: {exc}")

    print(f"Done. Enriched {success_count}/{len(episode_dirs)} episodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
