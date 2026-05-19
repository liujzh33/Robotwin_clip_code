#!/usr/bin/env python3
"""Enrich place_fan subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the fan.",
        "Move toward the fan.",
        "Guide the gripper closer to the small fan.",
        "Bring the robot arm near the white fan.",
        "Move into position for grasping the fan.",
        "Reach toward the portable fan.",
        "Position the gripper near the fan body.",
        "Navigate to the lightweight fan.",
        "Move closer to the fan before picking it up.",
        "Approach the small adjustable fan and prepare to grasp it.",
    ],
    1: [
        "Grasp the fan.",
        "Close the gripper.",
        "Pick up the fan.",
        "Secure the fan with the gripper.",
        "Clamp onto the fan body.",
        "Take hold of the portable fan.",
        "Grip it firmly.",
        "Close the fingers around the fan.",
        "Hold the fan steady.",
        "Complete the grasp so the fan can be moved to the mat.",
    ],
    2: [
        "Move the fan onto the blue mat while facing the robot.",
        "Carry the fan to the mat.",
        "Transport the fan toward the blue mat.",
        "Move the grasped fan over the mat.",
        "Bring the fan to the target mat and orient it forward.",
        "Guide the fan above the blue mat.",
        "Shift the fan into position on the mat.",
        "Move and turn the fan so it faces the robot.",
        "Carry the white fan and align it on the blue mat.",
        "Transfer the fan to the blue mat while adjusting its front side toward the robot.",
    ],
    3: [
        "Place the fan on the blue mat.",
        "Release the fan.",
        "Open the gripper to set down the fan.",
        "Put the fan onto the mat.",
        "Let go of the fan on the blue mat.",
        "Set it down.",
        "Drop the fan gently onto the mat.",
        "Finish placing the fan facing the robot.",
        "Open the fingers and leave the fan on the blue mat.",
        "Complete the task by placing the fan on the blue mat with its front facing the robot.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the fan on the blue mat facing the robot."


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
    parser = argparse.ArgumentParser(description="Enrich place_fan subtask language templates.")
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
