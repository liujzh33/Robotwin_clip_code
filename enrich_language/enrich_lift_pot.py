#!/usr/bin/env python3
"""Enrich lift_pot subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the pot with both arms.",
        "Move both grippers toward the kitchen pot.",
        "Guide the two end-effectors closer to the pot.",
        "Bring both arms near the gray pot.",
        "Move into position around the pot.",
        "Reach toward both sides of the kitchen pot.",
        "Position the grippers near the pot handles.",
        "Navigate both arms toward the dark gray pot.",
        "Move closer to the pot before grasping it.",
        "Approach the metal kitchen pot from both sides and prepare to grasp it.",
    ],
    1: [
        "Grasp the pot with both grippers.",
        "Close both grippers around the pot.",
        "Secure the kitchen pot from both sides.",
        "Hold the gray pot with both robot arms.",
        "Clamp onto the two sides of the pot.",
        "Take hold of the metal kitchen pot.",
        "Grip the pot firmly with both grippers.",
        "Close the fingers to capture the pot.",
        "Hold the pot steady before lifting.",
        "Complete the two-sided grasp so the pot can be lifted safely.",
    ],
    2: [
        "Lift the pot upward with both arms.",
        "Raise the kitchen pot off the table.",
        "Pick up the gray pot using both grippers.",
        "Lift the metal pot from the table surface.",
        "Move the grasped pot upward.",
        "Hold the pot and raise it into the air.",
        "Elevate the dark gray kitchen pot with both grippers.",
        "Lift the pot while keeping both sides secured.",
        "Raise the cylindrical pot away from the tabletop.",
        "Complete the task by lifting the kitchen pot steadily with both arms.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Lift the kitchen pot from the table with both arms."


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
    parser = argparse.ArgumentParser(description="Enrich lift_pot subtask language templates.")
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
