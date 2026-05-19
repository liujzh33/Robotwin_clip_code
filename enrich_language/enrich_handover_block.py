#!/usr/bin/env python3
"""Enrich handover_block subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the red block with the left arm.",
        "Move the left gripper toward the red block.",
        "Guide the left end-effector closer to the red block.",
        "Bring the left arm near the red block.",
        "Move into position for grasping the red block.",
        "Reach toward the red block with the left arm.",
        "Position the left gripper above the red block.",
        "Navigate the left arm to the red block.",
        "Move closer to the red block before picking it up.",
        "Use the left arm to approach the red block and prepare for grasping.",
    ],
    1: [
        "Grasp the red block with the left arm.",
        "Close the left gripper around the red block.",
        "Pick up the red block using the left gripper.",
        "Secure the red block with the left arm.",
        "Clamp the left gripper onto the red block.",
        "Take hold of the red block with the left hand.",
        "Grip the red block firmly.",
        "Close the left fingers to capture the red block.",
        "Hold the red block with the left gripper.",
        "Complete the left-arm grasp of the red block.",
    ],
    2: [
        "Move the red block to the center handover position.",
        "Carry the red block toward the middle.",
        "Transport the red block to the center.",
        "Move the held red block into the handover area.",
        "Bring the red block to the central position.",
        "Guide the red block to the middle for transfer.",
        "Shift the red block into the center of the workspace.",
        "Move the red block to where the right arm can reach it.",
        "Carry the block with the left arm toward the handover point.",
        "Position the red block at the center so it can be passed to the right arm.",
    ],
    3: [
        "Approach the red block with the right arm.",
        "Move the right gripper toward the centered red block.",
        "Guide the right end-effector closer to the red block.",
        "Bring the right arm near the block at the center.",
        "Move into position for the handover grasp.",
        "Reach toward the red block with the right arm.",
        "Position the right gripper near the transferred block.",
        "Navigate the right arm to the centered red block.",
        "Move closer to the red block before taking it from the left arm.",
        "Use the right arm to approach the red block at the handover position.",
    ],
    4: [
        "Grasp the red block with the right arm.",
        "Close the right gripper around the red block.",
        "Take the red block from the left arm.",
        "Secure the red block with the right gripper.",
        "Clamp the right gripper onto the red block.",
        "Receive the red block with the right hand.",
        "Grip the red block firmly with the right arm.",
        "Close the right fingers to capture the block.",
        "Hold the red block with the right gripper.",
        "Complete the handover by grasping the red block with the right arm.",
    ],
    5: [
        "Move the red block above the blue pad.",
        "Carry the red block toward the blue pad.",
        "Transport the red block to the blue pad area.",
        "Move the grasped red block over the blue pad.",
        "Bring the red block to its placement target.",
        "Guide the red block above the blue pad.",
        "Shift the red block into position over the pad.",
        "Move the red block to the blue landing area.",
        "Carry the block with the right arm toward the blue pad.",
        "Position the red block over the blue pad before releasing it.",
    ],
    6: [
        "Place the red block on the blue pad.",
        "Release the red block onto the blue pad.",
        "Open the right gripper to set down the red block.",
        "Put the red block on the blue pad.",
        "Let go of the red block over the pad.",
        "Set the red block down on the blue target.",
        "Drop the red block gently onto the blue pad.",
        "Finish placing the red block on the pad.",
        "Open the right fingers and leave the red block on the blue pad.",
        "Complete the task by placing the red block on the blue pad.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Hand over the red block and place it on the blue pad."


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
    parser = argparse.ArgumentParser(description="Enrich handover_block subtask language templates.")
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
