#!/usr/bin/env python3
"""Enrich place_object_stand subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the object.",
        "Move toward the item.",
        "Guide the gripper closer to the object.",
        "Bring the robot arm near the movable item.",
        "Move into position for grasping the object.",
        "Reach toward the item on the table.",
        "Position the gripper near the object body.",
        "Navigate to the object that will be placed on the stand.",
        "Move closer to the item before picking it up.",
        "Approach the movable object and prepare to place it on the display stand.",
    ],
    1: [
        "Grasp the object.",
        "Close the gripper.",
        "Pick up the item.",
        "Secure the object with the gripper.",
        "Clamp onto the movable item.",
        "Take hold of the object.",
        "Grip it firmly.",
        "Close the fingers around the item.",
        "Hold the object steady.",
        "Complete the grasp so the item can be moved to the stand.",
    ],
    2: [
        "Move the object above the display stand.",
        "Carry the item to the stand.",
        "Transport the object toward the stand.",
        "Move the grasped item over the display platform.",
        "Bring the object above the stand.",
        "Guide the item to the center of the display stand.",
        "Shift the object into position over the stand.",
        "Move the item to the placement area.",
        "Carry the object and align it above the display stand.",
        "Transfer the grasped item to the position directly above the stand.",
    ],
    3: [
        "Place the object on the display stand.",
        "Release the object.",
        "Open the gripper to set it down.",
        "Put the item onto the stand.",
        "Let go of the object on the display platform.",
        "Set it down.",
        "Drop the item gently onto the stand.",
        "Finish placing the object on the display stand.",
        "Open the fingers and leave the item on the stand.",
        "Complete the task by placing the movable object on the display stand.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the object on the display stand."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 4


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
    parser = argparse.ArgumentParser(description="Enrich place_object_stand subtask language templates.")
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
