#!/usr/bin/env python3
"""Enrich place_empty_cup subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the cup.",
        "Move toward the cup.",
        "Guide the gripper closer to the blue cup.",
        "Bring the robot arm near the cup.",
        "Move into position for grasping the cup.",
        "Reach toward the cup on the table.",
        "Position the gripper near the cup body.",
        "Navigate to the light blue cup.",
        "Move closer to the cup before picking it up.",
        "Approach the rounded-bottom blue cup and prepare to grasp it.",
    ],
    1: [
        "Grasp the cup.",
        "Close the gripper.",
        "Pick up the cup.",
        "Secure the cup with the gripper.",
        "Clamp onto the blue cup.",
        "Take hold of the cup.",
        "Grip it firmly.",
        "Close the fingers around the cup.",
        "Hold the cup steady.",
        "Complete the grasp so the cup can be moved to the coaster.",
    ],
    2: [
        "Move the cup above the coaster.",
        "Carry the cup to the coaster.",
        "Transport the cup toward the coaster.",
        "Move the grasped cup over the coaster.",
        "Bring the cup above the round coaster.",
        "Guide the cup to the coaster center.",
        "Shift the cup into position over the coaster.",
        "Move the cup to the placement area.",
        "Carry the blue cup and align it above the light gray coaster.",
        "Transfer the cup to the position directly above the small wooden coaster.",
    ],
    3: [
        "Place the cup on the coaster.",
        "Release the cup.",
        "Open the gripper to set down the cup.",
        "Put the cup onto the coaster.",
        "Let go of the cup on the coaster.",
        "Set it down.",
        "Drop the cup gently onto the coaster.",
        "Finish placing the cup on the coaster.",
        "Open the fingers and leave the cup on the coaster.",
        "Complete the task by placing the blue cup securely on the round coaster.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the cup on the coaster."


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
    parser = argparse.ArgumentParser(description="Enrich place_empty_cup subtask language templates.")
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
