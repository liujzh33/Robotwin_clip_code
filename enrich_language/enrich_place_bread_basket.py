#!/usr/bin/env python3
"""Enrich place_bread_basket subtask language; mode inferred from num_phases or phase_info.mode."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS_SINGLE = {
    0: [
        "Approach the bread.",
        "Move toward the bread.",
        "Guide the gripper closer to the loaf.",
        "Bring the robot arm near the bread.",
        "Move into position for grasping the bread.",
        "Reach toward the bread on the table.",
        "Position the gripper near the bread.",
        "Navigate to the loaf.",
        "Move closer to the bread before picking it up.",
        "Approach the bread and prepare to place it into the basket.",
    ],
    1: [
        "Grasp the bread.",
        "Close the gripper.",
        "Pick up the bread.",
        "Secure the bread with the gripper.",
        "Clamp onto the loaf.",
        "Take hold of the bread.",
        "Grip it gently.",
        "Close the fingers around the bread.",
        "Hold the bread steady.",
        "Complete the grasp so the bread can be moved to the basket.",
    ],
    2: [
        "Move the bread above the basket.",
        "Carry the bread to the basket.",
        "Transport the loaf toward the breadbasket.",
        "Move the grasped bread over the basket.",
        "Bring the bread to the basket opening.",
        "Guide the bread above the basket.",
        "Shift the loaf into position over the basket.",
        "Move the bread to the drop area.",
        "Carry the bread and align it over the oval basket.",
        "Transfer the bread to the position above the basket.",
    ],
    3: [
        "Release the bread into the basket.",
        "Drop the bread.",
        "Open the gripper over the basket.",
        "Put the bread into the basket.",
        "Let go of the bread inside the basket.",
        "Set it down.",
        "Drop the loaf gently into the breadbasket.",
        "Finish placing the bread in the basket.",
        "Open the fingers and leave the bread inside the basket.",
        "Complete the task by placing the bread into the basket.",
    ],
}

PHASE_VARIANTS_SEQUENTIAL = {
    0: [
        "Approach the first bread.",
        "Move toward the first loaf.",
        "Guide the gripper closer to the first bread.",
        "Bring the robot arm near the first bread.",
        "Move into position for the first grasp.",
        "Reach toward the first bread.",
        "Position the gripper near the first loaf.",
        "Navigate to the first bread on the table.",
        "Move closer to the first bread before picking it up.",
        "Approach the first bread and prepare to move it into the basket.",
    ],
    1: [
        "Grasp the first bread.",
        "Close the gripper.",
        "Pick up the first bread.",
        "Secure the first loaf with the gripper.",
        "Clamp onto the first bread.",
        "Take hold of the first bread.",
        "Grip the first loaf gently.",
        "Close the fingers around the first bread.",
        "Hold the first bread steady.",
        "Complete the first grasp so the bread can be carried to the basket.",
    ],
    2: [
        "Move the first bread above the basket.",
        "Carry the first bread to the basket.",
        "Transport the first loaf toward the basket.",
        "Move the grasped first bread over the basket.",
        "Bring the first bread to the basket opening.",
        "Guide the first bread above the breadbasket.",
        "Shift the first loaf into position over the basket.",
        "Move the first bread to the drop area.",
        "Carry the first bread and align it over the oval basket.",
        "Transfer the first bread to the position above the basket.",
    ],
    3: [
        "Release the first bread into the basket.",
        "Drop the first bread.",
        "Open the gripper over the basket.",
        "Put the first bread into the basket.",
        "Let go of the first bread inside the basket.",
        "Set the first loaf down.",
        "Drop the first bread gently into the breadbasket.",
        "Finish placing the first bread in the basket.",
        "Open the fingers and leave the first bread inside the basket.",
        "Complete the first placement by releasing the bread into the basket.",
    ],
    4: [
        "Approach the second bread.",
        "Move toward the second loaf.",
        "Guide the gripper closer to the second bread.",
        "Bring the robot arm near the second bread.",
        "Move into position for the second grasp.",
        "Reach toward the second bread.",
        "Position the gripper near the second loaf.",
        "Navigate to the second bread on the table.",
        "Move closer to the second bread after placing the first one.",
        "Approach the second bread and prepare to place it into the basket.",
    ],
    5: [
        "Grasp the second bread.",
        "Close the gripper.",
        "Pick up the second bread.",
        "Secure the second loaf with the gripper.",
        "Clamp onto the second bread.",
        "Take hold of the second bread.",
        "Grip the second loaf gently.",
        "Close the fingers around the second bread.",
        "Hold the second bread steady.",
        "Complete the second grasp so the bread can be moved to the basket.",
    ],
    6: [
        "Move the second bread above the basket.",
        "Carry the second bread to the basket.",
        "Transport the second loaf toward the basket.",
        "Move the grasped second bread over the basket.",
        "Bring the second bread to the basket opening.",
        "Guide the second bread above the breadbasket.",
        "Shift the second loaf into position over the basket.",
        "Move the second bread to the drop area.",
        "Carry the second bread and align it over the oval basket.",
        "Transfer the second bread to the position above the basket.",
    ],
    7: [
        "Release the second bread into the basket.",
        "Drop the second bread.",
        "Open the gripper over the basket.",
        "Put the second bread into the basket.",
        "Let go of the second bread inside the basket.",
        "Set the second loaf down.",
        "Drop the second bread gently into the breadbasket.",
        "Finish placing the second bread in the basket.",
        "Open the fingers and leave the second bread inside the basket.",
        "Complete the task by releasing the second bread into the basket.",
    ],
}

PHASE_VARIANTS_DUAL = {
    0: [
        "Approach both breads with both arms.",
        "Move both grippers toward the breads.",
        "Guide each end-effector closer to a bread loaf.",
        "Bring the robot arms near the two breads.",
        "Move into position around both breads.",
        "Reach toward the breads on both sides.",
        "Position the grippers near the two loaves.",
        "Navigate both arms toward the breads.",
        "Move closer to both breads before grasping them.",
        "Approach the two breads and prepare both grippers for grasping.",
    ],
    1: [
        "Grasp both breads.",
        "Close both grippers.",
        "Pick up the two breads.",
        "Secure each bread with a gripper.",
        "Clamp onto both loaves.",
        "Take hold of the two breads.",
        "Grip both breads gently.",
        "Close the fingers around the breads.",
        "Hold the two loaves steady.",
        "Complete the two-arm grasp so both breads can be moved.",
    ],
    2: [
        "Move the first bread above the basket.",
        "Carry one bread to the basket.",
        "Transport the first loaf toward the breadbasket.",
        "Move the first grasped bread over the basket.",
        "Bring one bread to the basket opening.",
        "Guide the first bread above the basket.",
        "Shift the first loaf into position over the basket.",
        "Move the first bread into the drop area.",
        "Carry the first bread and align it over the oval basket.",
        "Transfer one of the breads to the position above the basket.",
    ],
    3: [
        "Release the first bread into the basket.",
        "Drop the first bread.",
        "Open the gripper over the basket.",
        "Put the first bread into the basket.",
        "Let go of the first bread inside the basket.",
        "Set the first loaf down.",
        "Drop the first bread gently into the breadbasket.",
        "Finish placing the first bread in the basket.",
        "Open the fingers and leave the first bread inside the basket.",
        "Complete the first drop by releasing the bread into the basket.",
    ],
    4: [
        "Move the second bread above the basket.",
        "Carry the other bread to the basket.",
        "Transport the second loaf toward the breadbasket.",
        "Move the remaining grasped bread over the basket.",
        "Bring the second bread to the basket opening.",
        "Guide the second bread above the basket.",
        "Shift the other loaf into position over the basket.",
        "Move the second bread into the drop area.",
        "Carry the remaining bread and align it over the oval basket.",
        "Transfer the second bread to the position above the basket.",
    ],
    5: [
        "Release the second bread into the basket.",
        "Drop the second bread.",
        "Open the gripper over the basket.",
        "Put the second bread into the basket.",
        "Let go of the second bread inside the basket.",
        "Set the second loaf down.",
        "Drop the second bread gently into the breadbasket.",
        "Finish placing the second bread in the basket.",
        "Open the fingers and leave the second bread inside the basket.",
        "Complete the task by releasing the remaining bread into the basket.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Place the bread into the basket."


def infer_mode(data: dict) -> str:
    phase_info = data.get("phase_info", {})
    mode = phase_info.get("mode")
    if isinstance(mode, str) and mode:
        return mode

    num_phases = phase_info.get("num_phases")
    if num_phases == 8:
        return "sequential_two_breads"
    if num_phases == 6:
        return "dual_breads"
    return "single_bread"


def variants_for_mode(mode: str) -> dict[int, list[str]]:
    if mode == "sequential_two_breads":
        return PHASE_VARIANTS_SEQUENTIAL
    if mode == "dual_breads":
        return PHASE_VARIANTS_DUAL
    return PHASE_VARIANTS_SINGLE


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

    mode = infer_mode(data)
    variants = variants_for_mode(mode)
    num_phases = infer_num_phases(data)

    subtasks_list = []
    for _ in instructions:
        phase_texts = []
        for phase_idx in range(num_phases):
            phase_variants = variants.get(phase_idx, FALLBACK_VARIANTS)
            phase_texts.append(random.choice(phase_variants))
        subtasks_list.append(phase_texts)

    data["instructions"] = instructions
    data["subtasks"] = subtasks_list

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich place_bread_basket subtask language templates.")
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
