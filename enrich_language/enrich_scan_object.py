#!/usr/bin/env python3
"""Enrich scan_object subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the scanner and the object.",
        "Move both grippers toward the scanner and the item.",
        "Guide each arm closer to its target object.",
        "Bring one gripper near the scanner and the other near the object.",
        "Move into position around the barcode scanner and the item.",
        "Reach toward the scanner and the object.",
        "Position the grippers near the two objects.",
        "Navigate both arms toward the scanner and the scannable item.",
        "Move closer to both items before grasping them.",
        "Approach the barcode scanner and the object that needs to be scanned.",
    ],
    1: [
        "Grasp the scanner and the object.",
        "Close both grippers.",
        "Pick up both items.",
        "Secure the barcode scanner and the object.",
        "Clamp onto the scanner and the item.",
        "Take hold of both objects.",
        "Grip the scanner and the box firmly.",
        "Close the fingers around the scanner and the object.",
        "Hold the scanner and the scannable item steady.",
        "Complete the two-arm grasp so both items can be lifted for scanning.",
    ],
    2: [
        "Move and lift the scanner and the object.",
        "Lift both items.",
        "Raise the scanner and the object together.",
        "Move both grasped items upward.",
        "Bring the scanner and object to a higher scanning position.",
        "Lift the object and scanner away from the table.",
        "Move both held items into the scanning workspace.",
        "Raise both grippers while keeping the scanner and item secured.",
        "Carry the scanner and the object to a stable height.",
        "Move and elevate both items to prepare for barcode alignment.",
    ],
    3: [
        "Rotate the object to expose the barcode.",
        "Turn the object.",
        "Adjust the object so the barcode is visible.",
        "Rotate the scannable item toward the scanner.",
        "Move and turn the object to reveal its barcode.",
        "Reorient the box so the barcode faces outward.",
        "Twist the held object into a better scanning pose.",
        "Adjust the item's angle so the barcode can be scanned.",
        "Rotate the object while keeping it secured in the gripper.",
        "Turn the scannable object until its barcode side is easy for the scanner to read.",
    ],
    4: [
        "Orient the scanner toward the barcode and scan the object.",
        "Scan the barcode.",
        "Aim the scanner at the object.",
        "Point the barcode scanner toward the exposed code.",
        "Move the scanner into the correct scanning direction.",
        "Adjust the scanner so it faces the barcode.",
        "Align the scanner with the barcode on the object.",
        "Bring the scanner into position and complete the scan.",
        "Rotate the scanner toward the visible barcode and scan it.",
        "Complete the task by aiming the barcode scanner at the object and reading the code.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Scan the object with the barcode scanner."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 5


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
    parser = argparse.ArgumentParser(description="Enrich scan_object subtask language templates.")
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
