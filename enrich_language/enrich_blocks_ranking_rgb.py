#!/usr/bin/env python3
"""Enrich blocks_ranking_rgb subtask language without left/right arm wording."""

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
        "Move the gripper toward the red block.",
        "Guide the end-effector closer to the red block.",
        "Bring the gripper near the red block.",
        "Move into position for the red block.",
        "Reach toward the red block.",
        "Position the gripper above the red block.",
        "Navigate to the red block.",
        "Move closer to the red block before grasping it.",
        "Approach the red block and prepare for grasping.",
    ],
    1: [
        "Grasp the red block.",
        "Close the gripper around the red block.",
        "Pick up the red block.",
        "Secure the red block with the gripper.",
        "Clamp onto the red block.",
        "Take hold of the red block.",
        "Grip the red block firmly.",
        "Close the fingers to capture the red block.",
        "Hold the red block with the gripper.",
        "Complete the grasp of the red block.",
    ],
    2: [
        "Move the red block to its target position.",
        "Carry the red block to the first placement area.",
        "Transport the red block toward the start of the row.",
        "Move the grasped red block into place.",
        "Bring the red block to its target.",
        "Guide the red block to the first position.",
        "Shift the red block into the row.",
        "Move the red block to the side-by-side arrangement.",
        "Carry the red block and align it as the first block.",
        "Transfer the red block to the first position in the final line.",
    ],
    3: [
        "Place the red block.",
        "Release the red block at its target position.",
        "Open the gripper to set down the red block.",
        "Put the red block in place.",
        "Let go of the red block.",
        "Set the red block down.",
        "Drop the red block gently at the target.",
        "Finish placing the red block.",
        "Open the fingers and leave the red block in position.",
        "Place the red block as the first block in the row.",
    ],
    4: [
        "Approach the green block.",
        "Move the gripper toward the green block.",
        "Guide the end-effector closer to the green block.",
        "Bring the gripper near the green block.",
        "Move into position for the green block.",
        "Reach toward the green block.",
        "Position the gripper above the green block.",
        "Navigate to the green block.",
        "Move closer to the green block before grasping it.",
        "Approach the green block and prepare for grasping.",
    ],
    5: [
        "Grasp the green block.",
        "Close the gripper around the green block.",
        "Pick up the green block.",
        "Secure the green block with the gripper.",
        "Clamp onto the green block.",
        "Take hold of the green block.",
        "Grip the green block firmly.",
        "Close the fingers to capture the green block.",
        "Hold the green block with the gripper.",
        "Complete the grasp of the green block.",
    ],
    6: [
        "Move the green block to its target position.",
        "Carry the green block to the middle placement area.",
        "Transport the green block toward the center of the row.",
        "Move the grasped green block into the middle.",
        "Bring the green block to its target.",
        "Guide the green block to the second position.",
        "Shift the green block into the center of the arrangement.",
        "Move the green block next to the red block.",
        "Carry the green block and align it as the second block.",
        "Transfer the green block to the middle position in the final line.",
    ],
    7: [
        "Place the green block.",
        "Release the green block at its target position.",
        "Open the gripper to set down the green block.",
        "Put the green block in place.",
        "Let go of the green block.",
        "Set the green block down.",
        "Drop the green block gently at the target.",
        "Finish placing the green block.",
        "Open the fingers and leave the green block in position.",
        "Place the green block as the second block in the row.",
    ],
    8: [
        "Approach the blue block.",
        "Move the gripper toward the blue block.",
        "Guide the end-effector closer to the blue block.",
        "Bring the gripper near the blue block.",
        "Move into position for the blue block.",
        "Reach toward the blue block.",
        "Position the gripper above the blue block.",
        "Navigate to the blue block.",
        "Move closer to the blue block before grasping it.",
        "Approach the blue block and prepare for grasping.",
    ],
    9: [
        "Grasp the blue block.",
        "Close the gripper around the blue block.",
        "Pick up the blue block.",
        "Secure the blue block with the gripper.",
        "Clamp onto the blue block.",
        "Take hold of the blue block.",
        "Grip the blue block firmly.",
        "Close the fingers to capture the blue block.",
        "Hold the blue block with the gripper.",
        "Complete the grasp of the blue block.",
    ],
    10: [
        "Move the blue block to its target position.",
        "Carry the blue block to the final placement area.",
        "Transport the blue block toward the end of the row.",
        "Move the grasped blue block into the last position.",
        "Bring the blue block to its target.",
        "Guide the blue block to the third position.",
        "Shift the blue block into the final spot.",
        "Move the blue block next to the arranged blocks.",
        "Carry the blue block and align it as the third block.",
        "Transfer the blue block to the final position in the row.",
    ],
    11: [
        "Place the blue block.",
        "Release the blue block at its target position.",
        "Open the gripper to set down the blue block.",
        "Put the blue block in place.",
        "Let go of the blue block.",
        "Set the blue block down.",
        "Drop the blue block gently at the target.",
        "Finish placing the blue block.",
        "Open the fingers and leave the blue block in position.",
        "Place the blue block as the third block in the row.",
    ],
    12: [
        "Return to the initial position.",
        "Move the robot back to the starting pose.",
        "Return to neutral.",
        "Retract the gripper after placing all blocks.",
        "Move away from the arranged blocks.",
        "Finish by returning to a safe position.",
        "Reset the robot after completing the row.",
        "Move the gripper back after placing the red, green, and blue blocks.",
        "Clear the workspace and return to the initial pose.",
        "Complete the task by moving back to the neutral position.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Arrange the red, green, and blue blocks in order."


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
    parser = argparse.ArgumentParser(description="Enrich blocks_ranking_rgb subtask language templates.")
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
