#!/usr/bin/env python3
"""Enrich place_burger_fries subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the burger and fries with both arms.",
        "Move both grippers toward the two food items.",
        "Guide each end-effector closer to a food item.",
        "Bring the robot arms near the burger and fries.",
        "Move into position around the burger and fries.",
        "Reach toward the two items.",
        "Position the grippers near the burger and the fries box.",
        "Navigate both arms toward the food items.",
        "Move closer to the burger and fries before grasping.",
        "Approach the burger and fries so both can be picked up together.",
    ],
    1: [
        "Grasp the burger and fries.",
        "Close both grippers.",
        "Pick up both food items.",
        "Secure the burger and fries with the grippers.",
        "Clamp onto the two items.",
        "Take hold of the burger and the fries box.",
        "Grip them firmly.",
        "Close the fingers around the burger and fries.",
        "Hold the burger and fries steady.",
        "Complete the two-arm grasp so both items can be moved to the tray.",
    ],
    2: [
        "Move the first item above the tray.",
        "Carry one item to the tray.",
        "Transport the first food item toward the tray.",
        "Move the first grasped item over the tray.",
        "Bring one food item to the tray opening.",
        "Guide the first item above the plastic tray.",
        "Shift one item into position over the tray.",
        "Move the first item into the drop area.",
        "Carry one of the food items and align it above the tray.",
        "Transfer the first grasped item to the position above the orange tray.",
    ],
    3: [
        "Release the first item onto the tray.",
        "Drop the first item.",
        "Open the gripper over the tray.",
        "Put the first item on the tray.",
        "Let go of the first food item inside the tray.",
        "Set the first item down.",
        "Drop the first item gently onto the tray.",
        "Finish placing the first item on the tray.",
        "Open the fingers and leave the first food item on the tray.",
        "Complete the first placement by releasing one item onto the tray.",
    ],
    4: [
        "Move the second item above the tray.",
        "Carry the other item to the tray.",
        "Transport the second food item toward the tray.",
        "Move the remaining grasped item over the tray.",
        "Bring the second item to the tray opening.",
        "Guide the other item above the plastic tray.",
        "Shift the remaining item into position over the tray.",
        "Move the second item into the drop area.",
        "Carry the remaining food item and align it above the tray.",
        "Transfer the second grasped item to the position above the orange tray.",
    ],
    5: [
        "Release the second item onto the tray.",
        "Drop the second item.",
        "Open the gripper over the tray.",
        "Put the second item on the tray.",
        "Let go of the second food item inside the tray.",
        "Set the second item down.",
        "Drop the second item gently onto the tray.",
        "Finish placing the second item on the tray.",
        "Open the fingers and leave the second food item on the tray.",
        "Complete the task by releasing the remaining item onto the tray.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the burger and fries onto the tray."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 6


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
    parser = argparse.ArgumentParser(description="Enrich place_burger_fries subtask language templates.")
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
