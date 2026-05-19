#!/usr/bin/env python3
"""Enrich place_shoe subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the shoe.",
        "Move toward the shoe.",
        "Guide the gripper closer to the white shoe.",
        "Bring the robot arm near the shoe.",
        "Move into position for grasping the shoe.",
        "Reach toward the shoe on the table.",
        "Position the gripper near the shoe body.",
        "Navigate to the white shoe before lifting it.",
        "Move closer to the shoe before picking it up.",
        "Approach the rubber-soled shoe and prepare to grasp it.",
    ],
    1: [
        "Grasp the shoe.",
        "Close the gripper.",
        "Pick up the shoe.",
        "Secure the shoe with the gripper.",
        "Clamp onto the shoe.",
        "Take hold of the shoe.",
        "Grip it firmly.",
        "Close the fingers around the shoe.",
        "Hold the shoe steady.",
        "Complete the grasp so the shoe can be moved to the mat.",
    ],
    2: [
        "Move the shoe above the mat.",
        "Carry the shoe to the mat.",
        "Transport the shoe toward the mat.",
        "Move the grasped shoe over the mat.",
        "Bring the shoe above the target mat.",
        "Guide the shoe to the center of the mat.",
        "Shift the shoe into position over the mat.",
        "Move the shoe to the placement area.",
        "Carry the white shoe and align it above the mat.",
        "Transfer the shoe to the position directly above the mat.",
    ],
    3: [
        "Place the shoe on the mat.",
        "Release the shoe.",
        "Open the gripper to set down the shoe.",
        "Put the shoe onto the mat.",
        "Let go of the shoe on the mat.",
        "Set it down.",
        "Drop the shoe gently onto the mat.",
        "Finish placing the shoe on the mat.",
        "Open the fingers and leave the shoe on the mat.",
        "Complete the task by placing the white shoe securely on the mat.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the shoe on the mat."


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
    parser = argparse.ArgumentParser(description="Enrich place_shoe subtask language templates.")
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
