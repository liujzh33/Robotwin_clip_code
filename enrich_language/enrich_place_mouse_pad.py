#!/usr/bin/env python3
"""Enrich place_mouse_pad subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the mouse.",
        "Move toward the mouse.",
        "Guide the gripper closer to the computer mouse.",
        "Bring the robot arm near the dark gray mouse.",
        "Move into position for grasping the mouse.",
        "Reach toward the mouse on the table.",
        "Position the gripper near the mouse body.",
        "Navigate to the small computer mouse.",
        "Move closer to the mouse before picking it up.",
        "Approach the mouse with two buttons and prepare to grasp it.",
    ],
    1: [
        "Grasp the mouse.",
        "Close the gripper.",
        "Pick up the mouse.",
        "Secure the mouse with the gripper.",
        "Clamp onto the mouse body.",
        "Take hold of the computer mouse.",
        "Grip it firmly.",
        "Close the fingers around the mouse.",
        "Hold the mouse steady.",
        "Complete the grasp so the mouse can be moved to the mat.",
    ],
    2: [
        "Move the mouse above the gray mat.",
        "Carry the mouse to the mat.",
        "Transport the mouse toward the gray pad.",
        "Move the grasped mouse over the mat.",
        "Bring the mouse above the target mat.",
        "Guide the mouse to the center of the gray mat.",
        "Shift the mouse into position over the pad.",
        "Move the mouse to the placement area.",
        "Carry the dark gray mouse and align it above the mat.",
        "Transfer the computer mouse to the position directly above the gray mat.",
    ],
    3: [
        "Place the mouse on the gray mat.",
        "Release the mouse.",
        "Open the gripper to set down the mouse.",
        "Put the mouse onto the mat.",
        "Let go of the mouse on the gray pad.",
        "Set it down.",
        "Drop the mouse gently onto the mat.",
        "Finish placing the mouse on the gray mat.",
        "Open the fingers and leave the mouse on the pad.",
        "Complete the task by placing the computer mouse firmly on the gray mat.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the mouse on the gray mat."


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
    parser = argparse.ArgumentParser(description="Enrich place_mouse_pad subtask language templates.")
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
