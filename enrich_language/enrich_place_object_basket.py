#!/usr/bin/env python3
"""Enrich place_object_basket subtask language with phase-wise variants."""

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
        "Navigate to the object that will be placed in the basket.",
        "Move closer to the item before picking it up.",
        "Approach the movable object and prepare to place it into the basket.",
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
        "Complete the grasp so the item can be moved into the basket.",
    ],
    2: [
        "Move the object above the basket.",
        "Carry the item to the basket.",
        "Transport the object toward the basket opening.",
        "Move the grasped item over the basket.",
        "Bring the object above the basket.",
        "Guide the item to the basket interior.",
        "Shift the object into position over the basket.",
        "Move the item to the drop area.",
        "Carry the object and align it above the basket.",
        "Transfer the grasped item to the position directly above the basket.",
    ],
    3: [
        "Release the object into the basket.",
        "Drop the item.",
        "Open the gripper over the basket.",
        "Put the object into the basket.",
        "Let go of the item inside the basket.",
        "Set it down.",
        "Drop the object gently into the basket.",
        "Finish placing the item in the basket.",
        "Open the fingers and leave the object inside the basket.",
        "Complete the object placement by releasing it into the basket.",
    ],
    4: [
        "Approach the basket handle.",
        "Move toward the handle.",
        "Guide the other gripper closer to the basket handle.",
        "Bring the robot arm near the basket handle.",
        "Move into position for grasping the handle.",
        "Reach toward the basket handle.",
        "Position the gripper near the handle.",
        "Navigate to the basket handle after the item is placed.",
        "Move closer to the handle before lifting the basket.",
        "Approach the basket handle and prepare to grasp it.",
    ],
    5: [
        "Grasp the basket handle.",
        "Close the gripper.",
        "Hold the handle.",
        "Secure the basket handle with the gripper.",
        "Clamp onto the handle.",
        "Take hold of the basket.",
        "Grip the handle firmly.",
        "Close the fingers around the basket handle.",
        "Hold the basket handle steady.",
        "Complete the grasp so the basket can be lifted.",
    ],
    6: [
        "Lift the basket upward.",
        "Raise the basket.",
        "Pick up the basket by its handle.",
        "Lift the basket from the table.",
        "Move the handled basket upward.",
        "Hold the handle and raise the basket.",
        "Elevate the basket after the object is inside.",
        "Lift the basket while keeping the handle secured.",
        "Raise the basket with the object inside.",
        "Complete the task by placing the item in the basket and lifting the basket upward.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the object into the basket and lift the basket upward."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 7


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
    parser = argparse.ArgumentParser(description="Enrich place_object_basket subtask language templates.")
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
