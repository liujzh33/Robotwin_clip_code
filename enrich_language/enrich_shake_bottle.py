#!/usr/bin/env python3
"""Enrich shake_bottle subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the bottle.",
        "Move toward the bottle.",
        "Guide the gripper closer to the bottle.",
        "Bring the robot arm near the bottle.",
        "Move into position for grasping the bottle.",
        "Reach toward the bottle on the table.",
        "Position the gripper near the bottle body.",
        "Navigate to the handheld bottle.",
        "Move closer to the bottle before picking it up.",
        "Approach the bottle and prepare to grasp it for shaking.",
    ],
    1: [
        "Grasp the bottle.",
        "Close the gripper.",
        "Pick up the bottle.",
        "Secure the bottle with the gripper.",
        "Clamp onto the bottle.",
        "Take hold of the bottle.",
        "Grip the bottle firmly.",
        "Close the fingers around the bottle.",
        "Hold the bottle steady.",
        "Complete the grasp so the bottle can be lifted.",
    ],
    2: [
        "Lift the bottle upright.",
        "Raise the bottle.",
        "Move the bottle upward.",
        "Lift the grasped bottle from the table.",
        "Bring the bottle to a higher position.",
        "Raise the bottle into a vertical pose.",
        "Lift and stabilize the bottle before shaking.",
        "Move the bottle up and prepare it for shaking.",
        "Elevate the bottle while keeping it secured in the gripper.",
        "Lift the bottle to the shaking position and hold it steady.",
    ],
    3: [
        "Shake the bottle.",
        "Shake it.",
        "Move the bottle back and forth.",
        "Gently shake the bottle in the gripper.",
        "Oscillate the bottle after lifting it.",
        "Shake the held bottle repeatedly.",
        "Give the bottle a controlled shaking motion.",
        "Move the bottle with a periodic shaking pattern.",
        "Shake the raised bottle while keeping it firmly grasped.",
        "Complete the task by lifting the bottle and shaking it properly.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Shake the bottle."


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
    parser = argparse.ArgumentParser(description="Enrich shake_bottle subtask language templates.")
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
