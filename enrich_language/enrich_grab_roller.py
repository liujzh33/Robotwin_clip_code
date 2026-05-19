#!/usr/bin/env python3
"""Enrich grab_roller subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the wooden roller with both arms.",
        "Move both grippers toward the roller.",
        "Guide the two end-effectors to the ends of the roller.",
        "Bring both arms close to the brown wooden roller.",
        "Move into position around the roller.",
        "Reach toward both ends of the roller.",
        "Position the grippers near the roller ends.",
        "Navigate both arms toward the medium-sized roller.",
        "Move closer to the roller before grasping it.",
        "Approach the light brown roller from both sides and prepare to grasp it.",
    ],
    1: [
        "Grasp the roller with both grippers.",
        "Close both grippers around the roller.",
        "Secure the wooden roller from both ends.",
        "Hold the roller with both robot arms.",
        "Clamp onto the two ends of the roller.",
        "Take hold of the medium-sized roller.",
        "Grip the roller firmly with both grippers.",
        "Close the fingers to capture the roller.",
        "Hold the wooden roller steady before lifting.",
        "Complete the two-sided grasp so the roller can be lifted.",
    ],
    2: [
        "Lift the roller off the table.",
        "Raise the wooden roller upward.",
        "Pick up the roller with both arms.",
        "Lift the medium-sized roller from the table surface.",
        "Move the grasped roller upward.",
        "Hold the roller and raise it into the air.",
        "Elevate the light brown roller with both grippers.",
        "Lift the roller while keeping both ends secured.",
        "Raise the wooden roller away from the tabletop.",
        "Complete the task by lifting the roller off the table.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Grab and lift the roller with both arms."


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
    parser = argparse.ArgumentParser(description="Enrich grab_roller subtask language templates.")
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
