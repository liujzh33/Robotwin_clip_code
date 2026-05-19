#!/usr/bin/env python3
"""Enrich move_stapler_pad subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the stapler.",
        "Move toward the stapler.",
        "Guide the gripper closer to the black stapler.",
        "Bring the robot arm near the stapler.",
        "Move into position for grasping the stapler.",
        "Reach toward the small stapler.",
        "Position the gripper near the stapler body.",
        "Navigate to the black stapler.",
        "Move closer to the stapler before picking it up.",
        "Approach the black stapler with the shiny metal tray and prepare to grasp it.",
    ],
    1: [
        "Grasp the stapler.",
        "Close the gripper.",
        "Pick up the stapler.",
        "Secure the stapler with the gripper.",
        "Clamp onto the black stapler.",
        "Take hold of the stapler.",
        "Grip it firmly.",
        "Close the fingers around the stapler.",
        "Hold the stapler steady.",
        "Complete the grasp so the stapler can be moved to the mat.",
    ],
    2: [
        "Move the stapler above the gray mat.",
        "Carry the stapler to the mat.",
        "Transport the stapler toward the gray pad.",
        "Move the grasped stapler over the mat.",
        "Bring the stapler to the target mat.",
        "Guide the stapler above the gray mat.",
        "Shift the stapler into position over the pad.",
        "Move the stapler while keeping its pose stable.",
        "Carry the black stapler and align it above the gray mat.",
        "Transfer the stapler with the silver inner tray to the position above the mat.",
    ],
    3: [
        "Place the stapler on the gray mat.",
        "Release the stapler.",
        "Open the gripper to set down the stapler.",
        "Put the stapler on the mat.",
        "Let go of the stapler on the gray pad.",
        "Set it down.",
        "Drop the stapler gently onto the mat.",
        "Finish placing the stapler on the gray mat.",
        "Open the fingers and leave the black stapler on the pad.",
        "Complete the task by placing the stapler with its metal tray on the gray mat.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Move the stapler onto the gray mat while keeping its pose stable."


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
    parser = argparse.ArgumentParser(description="Enrich move_stapler_pad subtask language templates.")
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
