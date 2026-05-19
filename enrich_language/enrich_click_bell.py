#!/usr/bin/env python3
"""Enrich click_bell subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Move above the bell's top center.",
        "Position the end-effector over the bell.",
        "Move to the top center of the bell.",
        "Guide the gripper above the bell.",
        "Approach the bell from above.",
        "Align with the center of the bell.",
        "Move into position over the bell.",
        "Bring the gripper above the bell's top area.",
        "Reach the top of the bell before pressing.",
        "Navigate to the top center of the dome-shaped bell.",
    ],
    1: [
        "Close the gripper.",
        "Close the fingers before pressing.",
        "Bring the gripper fingers together.",
        "Prepare the gripper for pressing.",
        "Tighten the gripper into a closed shape.",
        "Close the gripper above the bell.",
        "Form a closed gripper before the downward press.",
        "Set the gripper into a pressing posture.",
        "Finish closing the gripper before touching the bell.",
        "Close the robot fingers to prepare for pressing the bell.",
    ],
    2: [
        "Press the bell downward.",
        "Push down on the bell's top center.",
        "Click the bell.",
        "Tap the top center of the bell.",
        "Move downward to press the bell.",
        "Apply a gentle downward press on the bell.",
        "Use the closed gripper to push the bell.",
        "Press the white bell at its top center.",
        "Lower the end-effector onto the bell.",
        "Activate the bell by pressing down on its top center.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Click the bell."


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
    parser = argparse.ArgumentParser(description="Enrich click_bell subtask language templates.")
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
