#!/usr/bin/env python3
"""Enrich place_container_plate subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the ceramic bowl.",
        "Move toward the bowl.",
        "Guide the gripper closer to the gray bowl.",
        "Bring the robot arm near the ceramic bowl.",
        "Move into position for grasping the bowl.",
        "Reach toward the bowl on the table.",
        "Position the gripper near the bowl rim.",
        "Navigate to the light gray container.",
        "Move closer to the bowl before picking it up.",
        "Approach the smooth ceramic bowl and prepare to grasp it.",
    ],
    1: [
        "Grasp the ceramic bowl.",
        "Close the gripper.",
        "Pick up the bowl.",
        "Secure the bowl with the gripper.",
        "Clamp onto the bowl.",
        "Take hold of the container.",
        "Grip it firmly.",
        "Close the fingers around the ceramic bowl.",
        "Hold the gray bowl steady.",
        "Complete the grasp so the bowl can be moved to the plate.",
    ],
    2: [
        "Move the ceramic bowl above the plate.",
        "Carry the bowl to the plate.",
        "Transport the container toward the dinner plate.",
        "Move the grasped bowl over the plate.",
        "Bring the bowl above the flat plate.",
        "Guide the container to the center of the plate.",
        "Shift the bowl into position over the plate.",
        "Move the bowl to the placement area.",
        "Carry the light gray bowl and align it above the ceramic plate.",
        "Transfer the container to the position directly above the greenish-gray plate.",
    ],
    3: [
        "Release the ceramic bowl onto the plate.",
        "Drop the bowl.",
        "Open the gripper over the plate.",
        "Put the bowl onto the plate.",
        "Let go of the bowl on the plate.",
        "Set it down.",
        "Drop the container gently onto the dinner plate.",
        "Finish placing the bowl on the plate.",
        "Open the fingers and leave the ceramic bowl on the plate.",
        "Complete the task by placing the smooth gray bowl onto the flat ceramic plate.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the ceramic bowl onto the plate."


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
    parser = argparse.ArgumentParser(description="Enrich place_container_plate subtask language templates.")
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
