#!/usr/bin/env python3
"""Enrich move_pillbottle_pad subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the pill bottle.",
        "Move toward the bottle.",
        "Guide the gripper closer to the white and orange bottle.",
        "Bring the robot arm near the pill bottle.",
        "Move into position for grasping the bottle.",
        "Reach toward the small storage bottle.",
        "Position the gripper near the bottle.",
        "Navigate to the palm-sized bottle.",
        "Move closer to the bottle before picking it up.",
        "Approach the white bottle with the orange label and prepare to grasp it.",
    ],
    1: [
        "Grasp the pill bottle.",
        "Close the gripper.",
        "Pick up the bottle.",
        "Secure the bottle with the gripper.",
        "Clamp onto the small bottle.",
        "Take hold of the pill bottle.",
        "Grip the bottle firmly.",
        "Close the fingers around the bottle.",
        "Hold the white and orange bottle.",
        "Complete the grasp so the bottle can be moved to the pad.",
    ],
    2: [
        "Move the pill bottle above the pad.",
        "Carry the bottle to the pad.",
        "Transport the bottle toward the placement pad.",
        "Move the grasped bottle over the pad.",
        "Bring the bottle to the target pad.",
        "Guide the bottle above the pad.",
        "Shift the pill bottle into position over the pad.",
        "Move the bottle to the placement area.",
        "Carry the white and orange bottle and align it above the pad.",
        "Transfer the palm-sized bottle to the position directly over the pad.",
    ],
    3: [
        "Place the pill bottle on the pad.",
        "Release the bottle.",
        "Open the gripper to set down the bottle.",
        "Put the bottle on the pad.",
        "Let go of the pill bottle on the pad.",
        "Set the bottle down.",
        "Drop the bottle gently onto the pad.",
        "Finish placing the bottle on the target pad.",
        "Open the fingers and leave the bottle on the pad.",
        "Complete the task by placing the white and orange bottle onto the pad.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Move the pill bottle onto the pad."


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
    parser = argparse.ArgumentParser(description="Enrich move_pillbottle_pad subtask language templates.")
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
