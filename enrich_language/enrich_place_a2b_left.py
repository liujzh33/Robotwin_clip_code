#!/usr/bin/env python3
"""Enrich place_a2b_left subtask language with generic phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the movable object.",
        "Move toward the object.",
        "Guide the gripper closer to the object to be moved.",
        "Bring the robot arm near the target object.",
        "Move into position for grasping the object.",
        "Reach toward the movable item.",
        "Position the gripper near the object.",
        "Navigate to the object on the table.",
        "Move closer to the object before picking it up.",
        "Approach the object that needs to be placed beside the reference item.",
    ],
    1: [
        "Grasp the object.",
        "Close the gripper.",
        "Pick up the object.",
        "Secure the object with the gripper.",
        "Clamp onto the movable item.",
        "Take hold of the object.",
        "Grip it firmly.",
        "Close the fingers around the object.",
        "Hold the object steady.",
        "Complete the grasp so the object can be moved to the left side.",
    ],
    2: [
        "Move the object to the left side of the reference object.",
        "Carry it left of the other object.",
        "Transport the held object toward the left side.",
        "Move the grasped object beside the reference item.",
        "Bring the object to the left of the object that stays on the table.",
        "Guide the object into position on the left side.",
        "Shift the movable item to the left of the reference object.",
        "Move the object while aligning it with the left side of the stationary item.",
        "Carry the object and place it near the left edge of the reference object.",
        "Transfer the held object to the target position left of the remaining table object.",
    ],
    3: [
        "Place the object to the left of the reference object.",
        "Release the object.",
        "Open the gripper to set it down.",
        "Put the object beside the reference item.",
        "Let go of the object on the left side.",
        "Set it down.",
        "Drop the object gently at the target position.",
        "Finish placing the object to the left.",
        "Open the fingers and leave the object beside the other item.",
        "Complete the task by placing the movable object to the left of the reference object.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the movable object to the left of the reference object."


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
    parser = argparse.ArgumentParser(description="Enrich place_a2b_left subtask language templates.")
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
