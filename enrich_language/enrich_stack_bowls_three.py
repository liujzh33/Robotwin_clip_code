#!/usr/bin/env python3
"""Enrich stack_bowls_three subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the first bowl.",
        "Move toward the first bowl.",
        "Guide the gripper closer to the first bowl.",
        "Bring the robot arm near the first bowl.",
        "Move into position for grasping the first bowl.",
        "Reach toward the first bowl on the table.",
        "Position the gripper near the first bowl.",
        "Navigate to the bowl that will form the base.",
        "Move closer to the first bowl before picking it up.",
        "Approach the first bowl and prepare to place it as the base of the stack.",
    ],
    1: [
        "Grasp the first bowl.",
        "Close the gripper.",
        "Pick up the first bowl.",
        "Secure the first bowl with the gripper.",
        "Clamp onto the first bowl.",
        "Take hold of the base bowl.",
        "Grip the first bowl firmly.",
        "Close the fingers around the first bowl.",
        "Hold the first bowl steady.",
        "Complete the grasp so the first bowl can be moved to the center.",
    ],
    2: [
        "Move the first bowl to the center.",
        "Carry the first bowl to the middle.",
        "Transport the first bowl toward the center point.",
        "Move the grasped bowl to the stacking area.",
        "Bring the first bowl to the center of the table.",
        "Guide the first bowl into the base position.",
        "Shift the first bowl to the middle placement spot.",
        "Move the first bowl to where the stack will be built.",
        "Carry the first bowl and align it with the center target.",
        "Transfer the first bowl to the center to form the bottom layer.",
    ],
    3: [
        "Place the first bowl at the center.",
        "Release the first bowl.",
        "Open the gripper over the center.",
        "Put the first bowl down.",
        "Let go of the first bowl at the stack base.",
        "Set it down.",
        "Drop the first bowl gently at the center.",
        "Finish placing the first bowl as the bottom bowl.",
        "Open the fingers and leave the first bowl in the middle.",
        "Complete the base placement by setting the first bowl at the center.",
    ],
    4: [
        "Approach the second bowl.",
        "Move toward the second bowl.",
        "Guide the gripper closer to the second bowl.",
        "Bring the robot arm near the second bowl.",
        "Move into position for grasping the second bowl.",
        "Reach toward the second bowl on the table.",
        "Position the gripper near the second bowl.",
        "Navigate to the next bowl.",
        "Move closer to the second bowl before picking it up.",
        "Approach the second bowl and prepare to stack it on the first bowl.",
    ],
    5: [
        "Grasp the second bowl.",
        "Close the gripper.",
        "Pick up the second bowl.",
        "Secure the second bowl with the gripper.",
        "Clamp onto the second bowl.",
        "Take hold of the second bowl.",
        "Grip the next bowl firmly.",
        "Close the fingers around the second bowl.",
        "Hold the second bowl steady.",
        "Complete the grasp so the second bowl can be stacked on the first bowl.",
    ],
    6: [
        "Move the second bowl above the first bowl.",
        "Carry the second bowl to the stack.",
        "Transport the second bowl over the first bowl.",
        "Move the grasped second bowl above the base bowl.",
        "Bring the second bowl to the center stack position.",
        "Guide the second bowl over the first bowl.",
        "Shift the second bowl into position above the base.",
        "Move the second bowl to the second layer of the stack.",
        "Carry the second bowl and align it over the first bowl.",
        "Transfer the second bowl to the position directly above the first bowl.",
    ],
    7: [
        "Place the second bowl on the first bowl.",
        "Release the second bowl.",
        "Open the gripper over the first bowl.",
        "Put the second bowl onto the first bowl.",
        "Let go of the second bowl on the stack.",
        "Set it down on the first bowl.",
        "Drop the second bowl gently onto the first bowl.",
        "Finish stacking the second bowl on the first bowl.",
        "Open the fingers and leave the second bowl as the middle layer.",
        "Complete the second layer by placing the second bowl on the first bowl.",
    ],
    8: [
        "Approach the third bowl.",
        "Move toward the third bowl.",
        "Guide the gripper closer to the third bowl.",
        "Bring the robot arm near the third bowl.",
        "Move into position for grasping the third bowl.",
        "Reach toward the third bowl on the table.",
        "Position the gripper near the third bowl.",
        "Navigate to the final bowl.",
        "Move closer to the third bowl before picking it up.",
        "Approach the third bowl and prepare to place it on top of the stack.",
    ],
    9: [
        "Grasp the third bowl.",
        "Close the gripper.",
        "Pick up the third bowl.",
        "Secure the third bowl with the gripper.",
        "Clamp onto the third bowl.",
        "Take hold of the top bowl.",
        "Grip the final bowl firmly.",
        "Close the fingers around the third bowl.",
        "Hold the third bowl steady.",
        "Complete the grasp so the third bowl can be stacked on the second bowl.",
    ],
    10: [
        "Move the third bowl above the second bowl.",
        "Carry the third bowl to the stack.",
        "Transport the third bowl over the second bowl.",
        "Move the grasped third bowl above the middle bowl.",
        "Bring the third bowl to the top of the stack.",
        "Guide the third bowl over the second bowl.",
        "Shift the third bowl into position above the stack.",
        "Move the third bowl to the top layer.",
        "Carry the third bowl and align it over the second bowl.",
        "Transfer the third bowl to the position directly above the second bowl.",
    ],
    11: [
        "Place the third bowl on the second bowl.",
        "Release the third bowl.",
        "Open the gripper over the second bowl.",
        "Put the third bowl onto the second bowl.",
        "Let go of the third bowl on top of the stack.",
        "Set it down on the second bowl.",
        "Drop the third bowl gently onto the second bowl.",
        "Finish stacking the third bowl on the second bowl.",
        "Open the fingers and leave the third bowl as the top layer.",
        "Complete the stack by placing the third bowl on top of the second bowl.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Stack three bowls."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 12


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
    parser = argparse.ArgumentParser(description="Enrich stack_bowls_three subtask language templates.")
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
