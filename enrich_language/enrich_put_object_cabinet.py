#!/usr/bin/env python3
"""Enrich put_object_cabinet subtask language with phase-wise variants."""

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
        "Navigate to the object before picking it up.",
        "Move closer to the object before placing it in the cabinet.",
        "Approach the object and prepare to move it into the drawer.",
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
        "Complete the grasp so the object can be moved into the cabinet.",
    ],
    2: [
        "Approach the cabinet handle.",
        "Move toward the drawer handle.",
        "Guide the other gripper closer to the cabinet handle.",
        "Bring the robot arm near the handle.",
        "Move into position for grasping the drawer handle.",
        "Reach toward the cabinet handle.",
        "Position the gripper near the drawer pull.",
        "Navigate to the handle after the object is secured.",
        "Move closer to the cabinet handle before opening the drawer.",
        "Approach the drawer handle and prepare to pull it open.",
    ],
    3: [
        "Grasp the cabinet handle.",
        "Close the gripper.",
        "Hold the drawer handle.",
        "Secure the cabinet handle with the gripper.",
        "Clamp onto the drawer handle.",
        "Take hold of the handle.",
        "Grip the handle firmly.",
        "Close the fingers around the cabinet handle.",
        "Hold the drawer pull steady.",
        "Complete the handle grasp so the drawer can be opened.",
    ],
    4: [
        "Pull the cabinet drawer open.",
        "Open the drawer.",
        "Slide the drawer outward.",
        "Pull the handle to open the cabinet.",
        "Move the drawer outward.",
        "Draw the cabinet open.",
        "Open the cabinet by pulling the handle.",
        "Slide the drawer out to make space for the object.",
        "Pull the drawer open while the object remains held.",
        "Complete the drawer opening so the object can be placed inside.",
    ],
    5: [
        "Move the object above the opened drawer.",
        "Carry the object to the cabinet.",
        "Transport the item toward the open drawer.",
        "Move the grasped object over the cabinet opening.",
        "Bring the object above the drawer interior.",
        "Guide the item into position over the opened cabinet.",
        "Shift the object toward the drawer opening.",
        "Move the held item to the placement area inside the cabinet.",
        "Carry the object and align it above the open drawer.",
        "Transfer the object to the position directly above the opened cabinet drawer.",
    ],
    6: [
        "Place the object inside the cabinet drawer.",
        "Release the object.",
        "Open the gripper over the drawer.",
        "Put the item into the cabinet.",
        "Let go of the object inside the drawer.",
        "Set it down.",
        "Drop the item gently into the cabinet drawer.",
        "Finish placing the object inside the opened drawer.",
        "Open the fingers and leave the object in the cabinet.",
        "Complete the task by placing the object inside the open cabinet drawer.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Put the object into the cabinet drawer."


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
    parser = argparse.ArgumentParser(description="Enrich put_object_cabinet subtask language templates.")
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
