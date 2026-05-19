#!/usr/bin/env python3
"""Enrich dump_bin_bigbin subtask language for direct and transfer modes."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS_DIRECT = {
    0: [
        "Approach the small trashbin.",
        "Move the gripper toward the white trashbin.",
        "Guide the end-effector closer to the small bin.",
        "Bring the robot arm near the trashbin.",
        "Move into position beside the small trashbin.",
        "Reach toward the plastic trashbin.",
        "Position the gripper near the rim of the trashbin.",
        "Navigate to the small white bin.",
        "Move closer to the trashbin before grasping it.",
        "Approach the table trashbin and prepare to hold its rim.",
    ],
    1: [
        "Grasp the rim of the trashbin.",
        "Close the gripper on the small trashbin.",
        "Secure the trashbin with the gripper.",
        "Hold the edge of the white bin.",
        "Clamp onto the trashbin rim.",
        "Take hold of the small trashbin.",
        "Grip the bin firmly.",
        "Close the fingers around the trashbin edge.",
        "Hold the plastic bin before pouring.",
        "Complete the grasp on the trashbin so it can be lifted.",
    ],
    2: [
        "Move and tilt the trashbin to pour the balls into the large bin.",
        "Lift the small trashbin and dump its contents.",
        "Carry the trashbin toward the large bin and pour out the balls.",
        "Tilt the white bin to empty it.",
        "Pour the balls from the small trashbin into the big bin.",
        "Move the held trashbin and tip it over the large bin.",
        "Dump the contents of the table trashbin.",
        "Lift, move, and tilt the trashbin to release the balls.",
        "Empty the small bin into the larger container.",
        "Complete the task by pouring the balls out of the trashbin.",
    ],
}

PHASE_VARIANTS_TRANSFER_THEN_DUMP = {
    0: [
        "Approach the small trashbin.",
        "Move the gripper toward the white trashbin.",
        "Guide the end-effector closer to the small bin.",
        "Bring the robot arm near the trashbin.",
        "Move into position beside the small trashbin.",
        "Reach toward the plastic trashbin.",
        "Position the gripper near the trashbin.",
        "Navigate to the table trashbin.",
        "Move closer to the bin before grasping it.",
        "Approach the small white trashbin and prepare for the first grasp.",
    ],
    1: [
        "Grasp the small trashbin.",
        "Close the gripper around the trashbin.",
        "Secure the bin with the gripper.",
        "Take hold of the small white bin.",
        "Clamp onto the trashbin.",
        "Grip the plastic bin.",
        "Hold the trashbin firmly.",
        "Close the fingers to capture the bin.",
        "Pick up the small trashbin.",
        "Complete the first grasp of the trashbin.",
    ],
    2: [
        "Move the trashbin to a better pouring position.",
        "Carry the small trashbin to the handoff area.",
        "Transport the bin toward the center.",
        "Move the grasped trashbin into position.",
        "Bring the trashbin to a stable placement point.",
        "Shift the small bin for the next grasp.",
        "Reposition the trashbin before pouring.",
        "Move the bin closer to the large container.",
        "Carry the trashbin so it can be grasped again.",
        "Transfer the trashbin to a position suitable for dumping.",
    ],
    3: [
        "Place the trashbin down.",
        "Release the small trashbin.",
        "Open the gripper to set down the bin.",
        "Put the trashbin in the new position.",
        "Let go of the white bin.",
        "Set the plastic trashbin down.",
        "Drop the bin gently at the placement point.",
        "Finish placing the trashbin for the next grasp.",
        "Open the fingers and leave the trashbin in place.",
        "Place the small bin so it can be picked up for pouring.",
    ],
    4: [
        "Approach the placed trashbin.",
        "Move the gripper toward the repositioned bin.",
        "Guide the end-effector back to the trashbin.",
        "Bring the robot arm near the bin for pouring.",
        "Move into position beside the placed trashbin.",
        "Reach toward the trashbin again.",
        "Position the gripper near the rim of the bin.",
        "Navigate to the trashbin after it has been placed.",
        "Move closer to the bin before the pouring grasp.",
        "Approach the trashbin again and prepare to hold it for dumping.",
    ],
    5: [
        "Grasp the trashbin rim for pouring.",
        "Close the gripper on the edge of the trashbin.",
        "Secure the bin by its rim.",
        "Take hold of the trashbin for dumping.",
        "Clamp onto the rim of the small bin.",
        "Grip the bin firmly before pouring.",
        "Hold the trashbin at its edge.",
        "Close the fingers around the trashbin rim.",
        "Prepare the trashbin for the dumping motion.",
        "Complete the second grasp so the bin can be lifted and tilted.",
    ],
    6: [
        "Move and tilt the trashbin to pour the balls into the large bin.",
        "Lift the small trashbin and dump its contents.",
        "Carry the trashbin toward the large bin and pour out the balls.",
        "Tilt the white bin to empty it.",
        "Pour the balls from the small trashbin into the big bin.",
        "Move the held trashbin and tip it over the large bin.",
        "Dump the contents of the table trashbin.",
        "Lift, move, and tilt the trashbin to release the balls.",
        "Empty the small bin into the larger container.",
        "Complete the task by pouring the balls out of the trashbin.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Pour the balls from the small trashbin into the large bin."


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
    variants_by_phase = PHASE_VARIANTS_TRANSFER_THEN_DUMP if num_phases == 7 else PHASE_VARIANTS_DIRECT

    subtasks_list = []
    for _ in instructions:
        phase_texts = []
        for phase_idx in range(num_phases):
            variants = variants_by_phase.get(phase_idx, FALLBACK_VARIANTS)
            phase_texts.append(random.choice(variants))
        subtasks_list.append(phase_texts)

    data["instructions"] = instructions
    data["subtasks"] = subtasks_list

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich dump_bin_bigbin subtask language templates.")
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
