#!/usr/bin/env python3
"""Enrich hanging_mug subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the mug with the left arm.",
        "Move the left gripper toward the black mug.",
        "Guide the left end-effector closer to the mug.",
        "Bring the left arm near the mug handle.",
        "Move into position for grasping the mug.",
        "Reach toward the black mug with the left arm.",
        "Position the left gripper near the mug.",
        "Navigate the left arm to the mug.",
        "Move closer to the mug before picking it up.",
        "Use the left arm to approach the mug and prepare for grasping.",
    ],
    1: [
        "Grasp the mug with the left arm.",
        "Close the left gripper around the mug.",
        "Pick up the black mug using the left gripper.",
        "Secure the mug with the left arm.",
        "Clamp the left gripper onto the mug.",
        "Take hold of the mug handle.",
        "Grip the mug firmly.",
        "Close the left fingers to capture the mug.",
        "Hold the mug with the left gripper.",
        "Complete the left-arm grasp of the black mug.",
    ],
    2: [
        "Move the mug to the center position.",
        "Carry the mug toward the middle.",
        "Transport the mug to the center.",
        "Move the held mug into the central area.",
        "Bring the mug to the middle of the workspace.",
        "Guide the mug to the center before the handoff.",
        "Shift the mug into the central position.",
        "Move the mug to where the right arm can reach it.",
        "Carry the mug with the left arm toward the center.",
        "Position the mug in the middle so it can be grasped again.",
    ],
    3: [
        "Place the mug at the center.",
        "Release the mug in the central position.",
        "Open the left gripper to set the mug down.",
        "Put the mug down in the middle.",
        "Let go of the mug at the center.",
        "Set the black mug down.",
        "Place the mug so the right arm can pick it up.",
        "Finish placing the mug in the center.",
        "Open the left fingers and leave the mug at the middle position.",
        "Set the mug in the central handoff area before the next grasp.",
    ],
    4: [
        "Approach the centered mug with the right arm.",
        "Move the right gripper toward the mug at the center.",
        "Guide the right end-effector closer to the centered mug.",
        "Bring the right arm near the placed mug.",
        "Move into position for the second grasp.",
        "Reach toward the mug with the right arm.",
        "Position the right gripper near the mug handle.",
        "Navigate the right arm to the mug in the middle.",
        "Move closer to the centered mug before lifting it.",
        "Use the right arm to approach the placed mug and prepare for hanging.",
    ],
    5: [
        "Grasp the mug with the right arm.",
        "Close the right gripper around the mug.",
        "Pick up the mug using the right gripper.",
        "Secure the mug with the right arm.",
        "Clamp the right gripper onto the mug handle.",
        "Take hold of the mug for hanging.",
        "Grip the black mug firmly with the right arm.",
        "Close the right fingers to capture the mug.",
        "Hold the mug with the right gripper.",
        "Complete the right-arm grasp before moving to the rack.",
    ],
    6: [
        "Hang the mug onto the rack.",
        "Move the mug to the rack and hang it.",
        "Carry the mug toward the rack rods.",
        "Place the mug handle onto the rack.",
        "Lift and turn the mug onto the hanging rack.",
        "Guide the mug handle onto the two rods.",
        "Rotate the mug and hang it on the dark gray rack.",
        "Move the mug into position and hook it onto the rack.",
        "Complete the task by hanging the mug from the rack.",
        "Carry, rotate, and hang the black mug onto the smooth rack.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Pick up the mug, place it at the center, then hang it on the rack with the right arm."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 7


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
    parser = argparse.ArgumentParser(description="Enrich hanging_mug subtask language templates.")
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
