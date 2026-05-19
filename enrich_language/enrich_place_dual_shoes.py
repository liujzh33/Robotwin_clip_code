#!/usr/bin/env python3
"""Enrich place_dual_shoes subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach both shoes with both arms.",
        "Move both grippers toward the shoes.",
        "Guide each end-effector closer to a shoe.",
        "Bring the robot arms near the two shoes.",
        "Move into position around both shoes.",
        "Reach toward the shoes.",
        "Position the grippers near the shoe bodies.",
        "Navigate both arms toward the pair of shoes.",
        "Move closer to the shoes before grasping them.",
        "Approach the two white shoes and prepare both grippers for grasping.",
    ],
    1: [
        "Grasp both shoes.",
        "Close both grippers.",
        "Pick up the shoes.",
        "Secure each shoe with a gripper.",
        "Clamp onto the two shoes.",
        "Take hold of both shoes.",
        "Grip the shoes firmly.",
        "Close the fingers around the shoes.",
        "Hold the pair of shoes steady.",
        "Complete the two-arm grasp so both shoes can be moved to the box.",
    ],
    2: [
        "Move the first shoe above the shoe box.",
        "Carry one shoe to the box.",
        "Transport the first shoe toward the orange shoe box.",
        "Move the first grasped shoe over the box.",
        "Bring one shoe to the box opening.",
        "Guide the first shoe above the shoe box.",
        "Shift one shoe into position over the box.",
        "Move the first shoe into the drop area.",
        "Carry one shoe and align it above the shoe box with the toe facing left.",
        "Transfer the first grasped shoe to the position above the box while keeping its tip leftward.",
    ],
    3: [
        "Release the first shoe into the shoe box.",
        "Drop the first shoe.",
        "Open the gripper over the box.",
        "Put the first shoe into the shoe box.",
        "Let go of one shoe inside the box.",
        "Set the first shoe down.",
        "Drop the first shoe gently into the orange box.",
        "Finish placing the first shoe in the shoe box.",
        "Open the fingers and leave the first shoe inside the box.",
        "Complete the first placement by releasing one shoe into the box with its tip facing left.",
    ],
    4: [
        "Move the second shoe above the shoe box.",
        "Carry the other shoe to the box.",
        "Transport the second shoe toward the orange shoe box.",
        "Move the remaining grasped shoe over the box.",
        "Bring the second shoe to the box opening.",
        "Guide the other shoe above the shoe box.",
        "Shift the remaining shoe into position over the box.",
        "Move the second shoe into the drop area.",
        "Carry the remaining shoe and align it above the shoe box with the toe facing left.",
        "Transfer the second grasped shoe to the position above the box while keeping its tip leftward.",
    ],
    5: [
        "Release the second shoe into the shoe box.",
        "Drop the second shoe.",
        "Open the gripper over the box.",
        "Put the second shoe into the shoe box.",
        "Let go of the second shoe inside the box.",
        "Set the second shoe down.",
        "Drop the second shoe gently into the orange box.",
        "Finish placing the second shoe in the shoe box.",
        "Open the fingers and leave the remaining shoe inside the box.",
        "Complete the task by placing the second shoe into the box with its tip facing left.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place both shoes into the shoe box with tips facing left."


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
    parser = argparse.ArgumentParser(description="Enrich place_dual_shoes subtask language templates.")
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
