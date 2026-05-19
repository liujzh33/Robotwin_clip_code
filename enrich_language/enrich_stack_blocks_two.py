#!/usr/bin/env python3
"""Enrich stack_blocks_two subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the red block.",
        "Move toward the red block.",
        "Guide the gripper closer to the red block.",
        "Bring the robot arm near the red block.",
        "Move into position for grasping the red block.",
        "Reach toward the red block on the table.",
        "Position the gripper near the red block.",
        "Navigate to the first block.",
        "Move closer to the red block before picking it up.",
        "Approach the red block and prepare to place it as the base.",
    ],
    1: [
        "Grasp the red block.",
        "Close the gripper.",
        "Pick up the red block.",
        "Secure the red block with the gripper.",
        "Clamp onto the red block.",
        "Take hold of the red block.",
        "Grip the first block firmly.",
        "Close the fingers around the red block.",
        "Hold the red block steady.",
        "Complete the grasp so the red block can be moved to the center.",
    ],
    2: [
        "Move the red block to the center.",
        "Carry the red block to the middle.",
        "Transport the red block toward the center point.",
        "Move the grasped red block to the stacking area.",
        "Bring the red block to the center of the table.",
        "Guide the red block into the base position.",
        "Shift the red block to the middle placement spot.",
        "Move the red block to where the stack will be built.",
        "Carry the red block and align it with the center target.",
        "Transfer the red block to the center to form the bottom layer.",
    ],
    3: [
        "Place the red block at the center.",
        "Release the red block.",
        "Open the gripper over the center.",
        "Put the red block down.",
        "Let go of the red block at the stack base.",
        "Set it down.",
        "Drop the red block gently at the center.",
        "Finish placing the red block as the bottom block.",
        "Open the fingers and leave the red block in the middle.",
        "Complete the base placement by setting the red block at the center.",
    ],
    4: [
        "Approach the green block.",
        "Move toward the green block.",
        "Guide the gripper closer to the green block.",
        "Bring the robot arm near the green block.",
        "Move into position for grasping the green block.",
        "Reach toward the green block on the table.",
        "Position the gripper near the green block.",
        "Navigate to the second block.",
        "Move closer to the green block before picking it up.",
        "Approach the green block and prepare to stack it on the red block.",
    ],
    5: [
        "Grasp the green block.",
        "Close the gripper.",
        "Pick up the green block.",
        "Secure the green block with the gripper.",
        "Clamp onto the green block.",
        "Take hold of the green block.",
        "Grip the top block firmly.",
        "Close the fingers around the green block.",
        "Hold the green block steady.",
        "Complete the grasp so the green block can be stacked on the red block.",
    ],
    6: [
        "Move the green block above the red block.",
        "Carry the green block to the stack.",
        "Transport the green block over the red block.",
        "Move the grasped green block above the base block.",
        "Bring the green block to the center stack position.",
        "Guide the green block over the red block.",
        "Shift the green block into position above the base.",
        "Move the green block to the top of the stack.",
        "Carry the green block and align it over the red block.",
        "Transfer the green block to the position directly above the red base block.",
    ],
    7: [
        "Place the green block on the red block.",
        "Release the green block.",
        "Open the gripper over the red block.",
        "Put the green block onto the red block.",
        "Let go of the green block on the stack.",
        "Set it down on the red block.",
        "Drop the green block gently onto the red block.",
        "Finish stacking the green block on the red block.",
        "Open the fingers and leave the green block as the top layer.",
        "Complete the stack by placing the green block on top of the red block.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Stack two blocks."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 8


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
    parser = argparse.ArgumentParser(description="Enrich stack_blocks_two subtask language templates.")
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
