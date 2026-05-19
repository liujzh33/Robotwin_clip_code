#!/usr/bin/env python3
"""Enrich turn_switch subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Close the gripper.",
        "Close the fingers.",
        "Prepare the gripper for pressing.",
        "Bring the fingers together.",
        "Set the gripper into a pressing posture.",
        "Close the gripper before moving to the switch.",
        "Form a firm pressing shape with the gripper.",
        "Finish closing the gripper for the switch press.",
        "Close the fingers to prepare for pushing the switch.",
        "Complete the gripper closure before approaching the switch.",
    ],
    1: [
        "Move in front of the switch.",
        "Approach the switch.",
        "Move toward the switch face.",
        "Guide the closed gripper to the switch.",
        "Bring the gripper in front of the switch.",
        "Position the closed gripper before the switch.",
        "Move closer to the rectangular switch.",
        "Align the gripper with the switch surface.",
        "Navigate to the front of the switch before pressing.",
        "Move the closed gripper into position directly in front of the switch.",
    ],
    2: [
        "Press the switch forward.",
        "Push the switch.",
        "Click the switch.",
        "Activate the switch.",
        "Move forward to press the switch.",
        "Push the flat switch with the closed gripper.",
        "Apply forward pressure to the switch.",
        "Engage the switch by pressing it directly.",
        "Drive the gripper forward to click the switch.",
        "Complete the task by pushing the small switch into its active position.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Press the switch forward."


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
    parser = argparse.ArgumentParser(description="Enrich turn_switch subtask language templates.")
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
