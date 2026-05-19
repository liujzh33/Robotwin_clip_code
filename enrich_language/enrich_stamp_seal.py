#!/usr/bin/env python3
"""Enrich stamp_seal subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the stamp object.",
        "Move toward the object.",
        "Guide the gripper closer to the stamp.",
        "Bring the robot arm near the object.",
        "Move into position for grasping the stamp object.",
        "Reach toward the object on the table.",
        "Position the gripper near the stamp handle.",
        "Navigate to the item that will be placed on the marked area.",
        "Move closer to the object before picking it up.",
        "Approach the stamp object and prepare to move it to the target area.",
    ],
    1: [
        "Grasp the stamp object.",
        "Close the gripper.",
        "Pick up the object.",
        "Secure the stamp with the gripper.",
        "Clamp onto the object.",
        "Take hold of the stamp object.",
        "Grip it firmly.",
        "Close the fingers around the object.",
        "Hold the item steady.",
        "Complete the grasp so the object can be moved to the marked area.",
    ],
    2: [
        "Move the stamp object above the marked area.",
        "Carry the object to the target area.",
        "Transport the object toward the special marked region.",
        "Move the grasped object over the Beige area.",
        "Bring the stamp object above the colored target.",
        "Guide the object to the marked placement area.",
        "Shift the object into position over the target region.",
        "Move the item above the area with the special label or color.",
        "Carry the object and align it above the Beige target.",
        "Transfer the stamp object to the position directly above the marked area.",
    ],
    3: [
        "Place the stamp object onto the marked area.",
        "Release the object.",
        "Open the gripper over the target area.",
        "Put the object onto the Beige region.",
        "Let go of the stamp object on the marked area.",
        "Set it down.",
        "Drop the object gently onto the target region.",
        "Finish placing the stamp object on the marked area.",
        "Open the fingers and leave the object on the Beige area.",
        "Complete the task by placing the stamp object onto the special marked region.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the stamp object onto the marked area."


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
    parser = argparse.ArgumentParser(description="Enrich stamp_seal subtask language templates.")
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
