#!/usr/bin/env python3
"""Enrich place_cans_plasticbox subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach both cans with both arms.",
        "Move both grippers toward the cans.",
        "Guide each end-effector closer to a drink can.",
        "Bring the robot arms near the two cans.",
        "Move into position around both cans.",
        "Reach toward the cans.",
        "Position the grippers near the can bodies.",
        "Navigate both arms toward the two drink cans.",
        "Move closer to the cans before grasping them.",
        "Approach the two beverage cans and prepare both grippers for grasping.",
    ],
    1: [
        "Grasp both cans.",
        "Close both grippers.",
        "Pick up the cans.",
        "Secure each can with a gripper.",
        "Clamp onto the two cans.",
        "Take hold of both drink cans.",
        "Grip the cans firmly.",
        "Close the fingers around the can bodies.",
        "Hold the two cans steady.",
        "Complete the two-arm grasp so both cans can be moved to the plastic box.",
    ],
    2: [
        "Move the first can above the plastic box.",
        "Carry one can to the box.",
        "Transport the first can toward the blue plastic box.",
        "Move the first grasped can over the box.",
        "Bring one can to the box opening.",
        "Guide the first can above the plastic box.",
        "Shift one can into position over the box.",
        "Move the first can into the drop area.",
        "Carry one drink can and align it above the blue box.",
        "Transfer the first grasped can to the position above the plastic storage box.",
    ],
    3: [
        "Release the first can into the plastic box.",
        "Drop the first can.",
        "Open the gripper over the box.",
        "Put the first can into the plastic box.",
        "Let go of one can inside the box.",
        "Set the first can down.",
        "Drop the first can gently into the blue box.",
        "Finish placing the first can in the plastic box.",
        "Open the fingers and leave the first can inside the box.",
        "Complete the first placement by releasing one can into the plastic box.",
    ],
    4: [
        "Move the second can above the plastic box.",
        "Carry the other can to the box.",
        "Transport the second can toward the blue plastic box.",
        "Move the remaining grasped can over the box.",
        "Bring the second can to the box opening.",
        "Guide the other can above the plastic box.",
        "Shift the remaining can into position over the box.",
        "Move the second can into the drop area.",
        "Carry the remaining drink can and align it above the blue box.",
        "Transfer the second grasped can to the position above the plastic storage box.",
    ],
    5: [
        "Release the second can into the plastic box.",
        "Drop the second can.",
        "Open the gripper over the box.",
        "Put the second can into the plastic box.",
        "Let go of the second can inside the box.",
        "Set the second can down.",
        "Drop the second can gently into the blue box.",
        "Finish placing the second can in the plastic box.",
        "Open the fingers and leave the remaining can inside the box.",
        "Complete the task by releasing the second drink can into the plastic box.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place both cans into the plastic box."


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
    parser = argparse.ArgumentParser(description="Enrich place_cans_plasticbox subtask language templates.")
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
