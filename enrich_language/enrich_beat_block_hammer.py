#!/usr/bin/env python3
"""Enrich beat_block_hammer subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Move the gripper toward the hammer.",
        "Approach the hammer with the gripper.",
        "Guide the end-effector closer to the hammer.",
        "Move the robot arm toward the hammer.",
        "Bring the gripper near the hammer handle.",
        "Navigate the gripper to the hammer.",
        "Position the gripper close to the hammer.",
        "Move the arm into position near the hammer.",
        "Reach toward the hammer with the gripper.",
        "Advance the gripper toward the hammer.",
    ],
    1: [
        "Close the gripper to grasp the hammer.",
        "Clamp the gripper around the hammer handle.",
        "Secure the hammer with the gripper.",
        "Grip the hammer using the robot fingers.",
        "Close the fingers around the hammer.",
        "Take hold of the hammer with the gripper.",
        "Grasp the hammer firmly.",
        "Hold the hammer by closing the gripper.",
        "Engage the gripper to capture the hammer.",
        "Close the gripper until the hammer is secured.",
    ],
    2: [
        "Move the hammer above the block.",
        "Carry the hammer to the space above the block.",
        "Transport the hammer over the block.",
        "Position the hammer directly above the block.",
        "Move the grasped hammer to the top of the block.",
        "Guide the hammer toward the area above the block.",
        "Bring the hammer over the block for striking.",
        "Place the hammer in position above the block.",
        "Move the hammer into the striking position over the block.",
        "Align the hammer above the block.",
    ],
    3: [
        "Strike the block downward with the hammer.",
        "Swing the hammer down onto the block.",
        "Move the hammer downward to hit the block.",
        "Bring the hammer down to strike the block.",
        "Hammer the block with a downward motion.",
        "Drive the hammer downward onto the block.",
        "Lower the hammer quickly to hit the block.",
        "Use the hammer to strike the block from above.",
        "Hit the block by moving the hammer downward.",
        "Complete the task by striking the block with the hammer.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Use the hammer to strike the block."


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
    parser = argparse.ArgumentParser(description="Enrich beat_block_hammer subtask language templates.")
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
