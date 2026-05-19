#!/usr/bin/env python3
"""Enrich open_microwave subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the microwave handle.",
        "Move toward the handle.",
        "Guide the gripper closer to the microwave door handle.",
        "Bring the robot arm near the microwave handle.",
        "Move into position for grasping the handle.",
        "Reach toward the handle on the microwave door.",
        "Position the gripper near the gray microwave handle.",
        "Navigate to the front handle of the microwave.",
        "Move closer to the microwave handle before pulling it.",
        "Approach the countertop microwave handle and prepare to grasp it.",
    ],
    1: [
        "Grasp the microwave handle.",
        "Close the gripper.",
        "Hold the handle.",
        "Secure the microwave handle with the gripper.",
        "Clamp onto the door handle.",
        "Take hold of the microwave handle.",
        "Grip the handle firmly.",
        "Close the fingers around the handle.",
        "Hold the handle before pulling the door open.",
        "Complete the grasp so the microwave door can be opened.",
    ],
    2: [
        "Pull the microwave door open.",
        "Open the microwave.",
        "Pull the handle.",
        "Move the handle outward to open the door.",
        "Use the gripper to open the microwave door.",
        "Pull the gray microwave door away from the body.",
        "Swing the microwave door open by pulling the handle.",
        "Open the rectangular microwave with a pulling motion.",
        "Draw the microwave door outward from the front handle.",
        "Complete the task by pulling the handle and opening the countertop microwave.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Grasp the microwave handle and pull the door open."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 3


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
    parser = argparse.ArgumentParser(description="Enrich open_microwave subtask language templates.")
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
