#!/usr/bin/env python3
"""Enrich rotate_qrcode subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the QR code sign.",
        "Move toward the payment sign.",
        "Guide the gripper closer to the QR code stand.",
        "Bring the robot arm near the tabletop sign.",
        "Move into position for grasping the payment sign.",
        "Reach toward the rectangular QR code sign.",
        "Position the gripper near the sign body.",
        "Navigate to the payment sign before lifting it.",
        "Move closer to the QR code sign before picking it up.",
        "Approach the tabletop payment sign and prepare to grasp it.",
    ],
    1: [
        "Grasp the QR code sign.",
        "Close the gripper.",
        "Pick up the payment sign.",
        "Secure the QR code stand with the gripper.",
        "Clamp onto the rectangular sign.",
        "Take hold of the tabletop payment sign.",
        "Grip it firmly.",
        "Close the fingers around the QR code sign.",
        "Hold the payment sign steady.",
        "Complete the grasp so the sign can be rotated toward the robot.",
    ],
    2: [
        "Rotate the QR code sign to face the robot.",
        "Turn the sign forward.",
        "Adjust the sign's angle toward the robot.",
        "Rotate the payment sign until the QR code faces forward.",
        "Move and turn the sign so the QR code faces the robot.",
        "Reorient the QR code stand toward the robot.",
        "Twist the grasped sign into the correct facing direction.",
        "Adjust the payment sign so its QR code is visible to the robot.",
        "Rotate the rectangular sign while keeping it secured in the gripper.",
        "Turn the tabletop payment sign until the QR code side points toward the robot.",
    ],
    3: [
        "Place the QR code sign down.",
        "Release the payment sign.",
        "Open the gripper to set down the sign.",
        "Put the sign back on the table.",
        "Let go of the QR code stand.",
        "Set it down.",
        "Drop the sign gently after rotating it.",
        "Finish placing the sign with the QR code facing forward.",
        "Open the fingers and leave the payment sign facing the robot.",
        "Complete the task by placing the QR code sign with its front side facing the robot.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Rotate the QR code sign to face the robot."


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
    parser = argparse.ArgumentParser(description="Enrich rotate_qrcode subtask language templates.")
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
