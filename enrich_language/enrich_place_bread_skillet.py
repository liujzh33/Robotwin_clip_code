#!/usr/bin/env python3
"""Enrich place_bread_skillet subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the skillet and the bread.",
        "Move both grippers toward the skillet and the bread.",
        "Guide one gripper to the skillet and the other to the bread.",
        "Bring the robot arms near the bread and the skillet.",
        "Move into position around the bread and the pan.",
        "Reach toward the bread and the skillet.",
        "Position the grippers near the two objects.",
        "Navigate each arm to its target object.",
        "Move closer to the bread and the skillet before grasping.",
        "Approach the bread and the skillet so both can be handled together.",
    ],
    1: [
        "Grasp the skillet and the bread.",
        "Close both grippers.",
        "Hold the skillet and the bread.",
        "Secure the pan and the loaf with the grippers.",
        "Clamp onto both objects.",
        "Take hold of the skillet and the bread.",
        "Grip them firmly.",
        "Close the fingers around the bread and the skillet.",
        "Hold the bread and keep the skillet secured.",
        "Complete the two-arm grasp so the skillet can be positioned under the bread.",
    ],
    2: [
        "Adjust the skillet under the bread.",
        "Move the skillet beneath the bread.",
        "Position the skillet opening below the loaf.",
        "Guide the pan under the held bread.",
        "Align the skillet with the bread.",
        "Move the pan into place below the bread.",
        "Shift the skillet so it sits under the bread.",
        "Bring the skillet mouth directly below the loaf.",
        "Adjust the skillet position while the bread remains held above it.",
        "Move the gray skillet into the correct position to receive the bread.",
    ],
    3: [
        "Release the bread into the skillet.",
        "Drop the bread.",
        "Open the bread-holding gripper.",
        "Put the bread into the skillet.",
        "Let go of the bread over the pan.",
        "Set the bread down.",
        "Drop the loaf gently into the skillet.",
        "Finish placing the bread in the skillet.",
        "Open the fingers and leave the bread inside the pan.",
        "Complete the task by releasing the bread into the gray skillet.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the bread into the skillet."


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
    parser = argparse.ArgumentParser(description="Enrich place_bread_skillet subtask language templates.")
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
