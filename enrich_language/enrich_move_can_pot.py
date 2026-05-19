#!/usr/bin/env python3
"""Enrich move_can_pot subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the sauce can.",
        "Move toward the can.",
        "Guide the gripper closer to the sauce can.",
        "Bring the robot arm near the yellow and brown can.",
        "Move into position for grasping the can.",
        "Reach toward the cylindrical sauce can.",
        "Position the gripper near the can.",
        "Navigate to the sauce can.",
        "Move closer to the can before picking it up.",
        "Approach the brown-topped sauce can and prepare to grasp it.",
    ],
    1: [
        "Grasp the sauce can.",
        "Close the gripper.",
        "Pick up the sauce can.",
        "Secure the can with the gripper.",
        "Clamp onto the cylindrical can.",
        "Take hold of the sauce can.",
        "Grip the can firmly.",
        "Close the fingers around the can.",
        "Hold the yellow and brown sauce can.",
        "Complete the grasp so the can can be moved.",
    ],
    2: [
        "Move the sauce can next to the pot.",
        "Carry the can to the pot.",
        "Transport the sauce can toward the kitchen pot.",
        "Move the grasped can beside the silver pot.",
        "Bring the can near the pot.",
        "Guide the can to the side of the pot.",
        "Shift the sauce can into position near the pot.",
        "Move the can to the placement area beside the pot.",
        "Carry the sauce can over and align it next to the medium-sized pot.",
        "Transfer the cylindrical can to a spot beside the pot with handles.",
    ],
    3: [
        "Place the sauce can beside the pot.",
        "Release the can.",
        "Open the gripper to set down the sauce can.",
        "Put the sauce can next to the pot.",
        "Let go of the can beside the kitchen pot.",
        "Set the can down.",
        "Drop the sauce can gently near the pot.",
        "Finish placing the can next to the silver pot.",
        "Open the fingers and leave the can beside the pot.",
        "Complete the task by placing the sauce can near the rounded pot.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Move the sauce can next to the pot and place it beside the pot."


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
    parser = argparse.ArgumentParser(description="Enrich move_can_pot subtask language templates.")
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
