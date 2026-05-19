#!/usr/bin/env python3
"""Enrich blocks_ranking_size subtask language without left/right arm wording."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the small block.",
        "Move the gripper toward the small block.",
        "Guide the end-effector closer to the small block.",
        "Bring the gripper near the small block.",
        "Move into position for the small block.",
        "Reach toward the small block.",
        "Position the gripper above the small block.",
        "Navigate to the small block.",
        "Move closer to the small block before grasping it.",
        "Approach the small block and prepare for grasping.",
    ],
    1: [
        "Grasp the small block.",
        "Close the gripper around the small block.",
        "Pick up the small block.",
        "Secure the small block with the gripper.",
        "Clamp onto the small block.",
        "Take hold of the small block.",
        "Grip the small block firmly.",
        "Close the fingers to capture the small block.",
        "Hold the small block with the gripper.",
        "Complete the grasp of the small block.",
    ],
    2: [
        "Move the small block to the far right.",
        "Carry the small block to the rightmost position.",
        "Transport the small block toward the far-right target.",
        "Move the grasped small block to the right side.",
        "Bring the small block to its far-right placement area.",
        "Guide the small block to the right end of the row.",
        "Shift the small block into the far-right position.",
        "Move the small block into place on the right.",
        "Carry the small block and align it as the rightmost block.",
        "Transfer the small block to the far-right side of the final line.",
    ],
    3: [
        "Place the small block on the far right.",
        "Release the small block at the rightmost position.",
        "Open the gripper to set down the small block.",
        "Put the small block in the far-right placement area.",
        "Let go of the small block on the right side.",
        "Set the small block down on the far right.",
        "Drop the small block gently at the right target.",
        "Finish placing the small block on the right.",
        "Open the fingers and leave the small block in the far-right position.",
        "Place the small block as the rightmost block in the row.",
    ],
    4: [
        "Approach the medium block.",
        "Move the gripper toward the medium block.",
        "Guide the end-effector closer to the medium block.",
        "Bring the gripper near the medium block.",
        "Move into position for the medium block.",
        "Reach toward the medium block.",
        "Position the gripper above the medium block.",
        "Navigate to the medium block.",
        "Move closer to the medium block before grasping it.",
        "Approach the medium block and prepare for grasping.",
    ],
    5: [
        "Grasp the medium block.",
        "Close the gripper around the medium block.",
        "Pick up the medium block.",
        "Secure the medium block with the gripper.",
        "Clamp onto the medium block.",
        "Take hold of the medium block.",
        "Grip the medium block firmly.",
        "Close the fingers to capture the medium block.",
        "Hold the medium block with the gripper.",
        "Complete the grasp of the medium block.",
    ],
    6: [
        "Move the medium block to the middle.",
        "Carry the medium block to the center position.",
        "Transport the medium block toward the middle target.",
        "Move the grasped medium block to the center.",
        "Bring the medium block to its middle placement area.",
        "Guide the medium block to the center of the row.",
        "Shift the medium block into the middle position.",
        "Move the medium block into place in the center.",
        "Carry the medium block and align it as the middle block.",
        "Transfer the medium block to the center of the final line.",
    ],
    7: [
        "Place the medium block in the middle.",
        "Release the medium block at the center position.",
        "Open the gripper to set down the medium block.",
        "Put the medium block in the middle placement area.",
        "Let go of the medium block in the center.",
        "Set the medium block down in the middle.",
        "Drop the medium block gently at the center target.",
        "Finish placing the medium block in the middle.",
        "Open the fingers and leave the medium block in the center position.",
        "Place the medium block as the center block in the row.",
    ],
    8: [
        "Approach the large block.",
        "Move the gripper toward the large block.",
        "Guide the end-effector closer to the large block.",
        "Bring the gripper near the large block.",
        "Move into position for the large block.",
        "Reach toward the large block.",
        "Position the gripper above the large block.",
        "Navigate to the large block.",
        "Move closer to the large block before grasping it.",
        "Approach the large block and prepare for grasping.",
    ],
    9: [
        "Grasp the large block.",
        "Close the gripper around the large block.",
        "Pick up the large block.",
        "Secure the large block with the gripper.",
        "Clamp onto the large block.",
        "Take hold of the large block.",
        "Grip the large block firmly.",
        "Close the fingers to capture the large block.",
        "Hold the large block with the gripper.",
        "Complete the grasp of the large block.",
    ],
    10: [
        "Move the large block to the far left.",
        "Carry the large block to the leftmost position.",
        "Transport the large block toward the far-left target.",
        "Move the grasped large block to the left side.",
        "Bring the large block to its far-left placement area.",
        "Guide the large block to the left end of the row.",
        "Shift the large block into the far-left position.",
        "Move the large block into place on the left.",
        "Carry the large block and align it as the leftmost block.",
        "Transfer the large block to the far-left side of the final line.",
    ],
    11: [
        "Place the large block on the far left.",
        "Release the large block at the leftmost position.",
        "Open the gripper to set down the large block.",
        "Put the large block in the far-left placement area.",
        "Let go of the large block on the left side.",
        "Set the large block down on the far left.",
        "Drop the large block gently at the left target.",
        "Finish placing the large block on the left.",
        "Open the fingers and leave the large block in the far-left position.",
        "Place the large block as the leftmost block in the row.",
    ],
    12: [
        "Return to the initial position.",
        "Move the robot back to the starting pose.",
        "Return to neutral.",
        "Retract the gripper after placing all blocks.",
        "Move away from the arranged blocks.",
        "Finish by returning to a safe position.",
        "Reset the robot after completing the row.",
        "Move the gripper back after placing the small, medium, and large blocks.",
        "Clear the workspace and return to the initial pose.",
        "Complete the task by moving back to the neutral position.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Arrange the small, medium, and large blocks by size."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases

    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])

    return 13


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
    parser = argparse.ArgumentParser(description="Enrich blocks_ranking_size subtask language templates.")
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
