#!/usr/bin/env python3
"""Enrich press_stapler subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the stapler pressing area.",
        "Move above the stapler.",
        "Guide the gripper over the stapler.",
        "Bring the robot arm to the top of the stapler.",
        "Move into position for pressing the stapler.",
        "Reach toward the stapler's pressing surface.",
        "Position the gripper above the blue stapler.",
        "Navigate to the area directly over the stapler.",
        "Move closer to the stapler before pressing it.",
        "Approach the top of the blue stapler and prepare for the press.",
    ],
    1: [
        "Close the gripper.",
        "Close the fingers.",
        "Prepare the gripper for pressing.",
        "Clamp the gripper slightly before pressing.",
        "Bring the fingers together above the stapler.",
        "Close the gripper over the stapler.",
        "Set the gripper into a pressing posture.",
        "Finish closing the gripper before applying force.",
        "Close the fingers to prepare for the downward push.",
        "Complete the gripper closure above the stapler pressing area.",
    ],
    2: [
        "Press the stapler downward.",
        "Push down.",
        "Apply pressure to the stapler.",
        "Press the top of the stapler.",
        "Move the gripper downward onto the stapler.",
        "Push the blue stapler down with the gripper.",
        "Apply downward force to activate the stapler.",
        "Lower the end-effector to press the stapler.",
        "Press the stapler firmly from above.",
        "Complete the task by pushing the stapler downward.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Press the stapler with the right arm."


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
    parser = argparse.ArgumentParser(description="Enrich press_stapler subtask language templates.")
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
