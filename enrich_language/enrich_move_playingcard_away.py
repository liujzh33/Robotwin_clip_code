#!/usr/bin/env python3
"""Enrich move_playingcard_away subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the playing card case.",
        "Move toward the cards.",
        "Guide the gripper closer to the blue playing card holder.",
        "Bring the robot arm near the rectangular card case.",
        "Move into position for grasping the playing card case.",
        "Reach toward the blue and white card packaging.",
        "Position the gripper near the playing card holder.",
        "Navigate to the smooth playing card case.",
        "Move closer to the card case before picking it up.",
        "Approach the rectangular blue playing card holder and prepare to grasp it.",
    ],
    1: [
        "Grasp the playing card case.",
        "Close the gripper.",
        "Pick up the cards.",
        "Secure the card case with the gripper.",
        "Clamp onto the rectangular case.",
        "Take hold of the playing card holder.",
        "Grip the blue card case firmly.",
        "Close the fingers around the card packaging.",
        "Hold the blue and white playing card case.",
        "Complete the grasp so the card case can be moved outward.",
    ],
    2: [
        "Move the playing card case outward.",
        "Carry the cards away.",
        "Transport the card case toward the outside edge.",
        "Move the grasped playing card holder outward.",
        "Bring the card case away from the table area.",
        "Guide the playing cards off the table.",
        "Shift the blue case outward from its starting position.",
        "Move the playing card packaging away from the workspace.",
        "Carry the rectangular case outward while keeping it secured.",
        "Transfer the blue and white playing card holder toward the outer side of the table.",
    ],
    3: [
        "Place the playing card case.",
        "Release the cards.",
        "Open the gripper to set down the card case.",
        "Put the playing card holder down.",
        "Let go of the card case after moving it outward.",
        "Set the cards down.",
        "Drop the card case gently at the outward position.",
        "Finish placing the playing card holder away from the table.",
        "Open the fingers and leave the blue card case outside the workspace.",
        "Complete the task by placing the playing card case after moving it outward.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Move the playing card case outward and place it away from the table."


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
    parser = argparse.ArgumentParser(description="Enrich move_playingcard_away subtask language templates.")
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
