#!/usr/bin/env python3
"""Enrich place_phone_stand subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the phone.",
        "Move toward the phone.",
        "Guide the gripper closer to the phone.",
        "Bring the robot arm near the phone.",
        "Move into position for grasping the phone.",
        "Reach toward the phone on the table.",
        "Position the gripper near the phone body.",
        "Navigate to the phone before lifting it.",
        "Move closer to the phone before picking it up.",
        "Approach the smooth phone and prepare to place it on the stand.",
    ],
    1: [
        "Grasp the phone.",
        "Close the gripper.",
        "Pick up the phone.",
        "Secure the phone with the gripper.",
        "Clamp onto the phone.",
        "Take hold of the phone.",
        "Grip it firmly.",
        "Close the fingers around the phone.",
        "Hold the phone steady.",
        "Complete the grasp so the phone can be moved to the stand.",
    ],
    2: [
        "Move the phone above the phone stand.",
        "Carry the phone to the stand.",
        "Transport the phone toward the phone holder.",
        "Move the grasped phone over the stand.",
        "Bring the phone above the stand slot.",
        "Guide the phone toward the phone stand.",
        "Shift the phone into position over the holder.",
        "Move and orient the phone for placement on the stand.",
        "Carry the phone and align it with the stand slot.",
        "Transfer the phone to the stand while adjusting its pose for placement.",
    ],
    3: [
        "Place the phone onto the phone stand.",
        "Release the phone.",
        "Open the gripper to set the phone down.",
        "Put the phone on the stand.",
        "Let go of the phone on the holder.",
        "Set it down.",
        "Drop the phone gently onto the stand.",
        "Finish placing the phone on the phone stand.",
        "Open the fingers and leave the phone on the stand.",
        "Complete the task by placing the phone securely onto the phone stand.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the phone onto the phone stand."


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
    parser = argparse.ArgumentParser(description="Enrich place_phone_stand subtask language templates.")
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
