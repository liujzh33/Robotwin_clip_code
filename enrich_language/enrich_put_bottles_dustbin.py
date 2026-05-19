#!/usr/bin/env python3
"""Enrich put_bottles_dustbin subtask language from phase_kinds in phase_info."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


APPROACH_BOTTLE_VARIANTS = [
    "Approach the bottle.",
    "Move toward the bottle.",
    "Guide the gripper closer to the bottle.",
    "Bring the robot arm near the bottle.",
    "Move into position for grasping the bottle.",
    "Reach toward the bottle on the table.",
    "Position the gripper near the bottle body.",
    "Navigate to the next bottle.",
    "Move closer to the bottle before picking it up.",
    "Approach the bottle and prepare to move it into the dustbin.",
]

GRASP_BOTTLE_VARIANTS = [
    "Grasp the bottle.",
    "Close the gripper.",
    "Pick up the bottle.",
    "Secure the bottle with the gripper.",
    "Clamp onto the bottle.",
    "Take hold of the bottle.",
    "Grip the bottle firmly.",
    "Close the fingers around the bottle.",
    "Hold the bottle steady.",
    "Complete the grasp so the bottle can be moved.",
]

MOVE_TO_CENTER_VARIANTS = [
    "Move the bottle to the center.",
    "Carry the bottle to the middle.",
    "Transport the bottle toward the center handover area.",
    "Move the grasped bottle into the central position.",
    "Bring the bottle to the center of the workspace.",
    "Guide the bottle to the handover point.",
    "Shift the bottle into the middle.",
    "Move the bottle where the bin-side arm can reach it.",
    "Carry the bottle to the central transfer position.",
    "Place the bottle in the center area for the other arm to take over.",
]

APPROACH_CENTER_BOTTLE_VARIANTS = [
    "Approach the centered bottle.",
    "Move toward the bottle in the center.",
    "Guide the receiving gripper closer to the bottle.",
    "Bring the bin-side arm near the centered bottle.",
    "Move into position for the handover grasp.",
    "Reach toward the bottle at the center.",
    "Position the gripper near the transferred bottle.",
    "Navigate to the bottle in the handover area.",
    "Move closer to the bottle before taking it from the other arm.",
    "Approach the centered bottle and prepare to move it to the dustbin.",
]

MOVE_TO_DUSTBIN_VARIANTS = [
    "Move the bottle above the dustbin.",
    "Carry the bottle to the dustbin.",
    "Transport the bottle toward the left-side dustbin.",
    "Move the grasped bottle over the dustbin.",
    "Bring the bottle above the trash bin opening.",
    "Guide the bottle to the dustbin.",
    "Shift the bottle into position over the bin.",
    "Move the bottle to the drop area.",
    "Carry the bottle and align it above the black dustbin.",
    "Transfer the bottle to the position directly above the left-side dustbin.",
]

RELEASE_BOTTLE_VARIANTS = [
    "Release the bottle into the dustbin.",
    "Drop the bottle.",
    "Open the gripper over the dustbin.",
    "Put the bottle into the trash bin.",
    "Let go of the bottle inside the dustbin.",
    "Set it down in the bin.",
    "Drop the bottle gently into the dustbin.",
    "Finish placing the bottle in the trash holder.",
    "Open the fingers and leave the bottle inside the dustbin.",
    "Complete the placement by releasing the bottle into the black dustbin.",
]

VARIANTS_BY_KIND = {
    "approach": APPROACH_BOTTLE_VARIANTS,
    "approach_pickup": APPROACH_BOTTLE_VARIANTS,
    "grasp": GRASP_BOTTLE_VARIANTS,
    "grasp_pickup": GRASP_BOTTLE_VARIANTS,
    "grasp_bin": GRASP_BOTTLE_VARIANTS,
    "move_center": MOVE_TO_CENTER_VARIANTS,
    "approach_center": APPROACH_CENTER_BOTTLE_VARIANTS,
    "move_dustbin": MOVE_TO_DUSTBIN_VARIANTS,
    "release": RELEASE_BOTTLE_VARIANTS,
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Put the bottles into the dustbin."


def infer_phase_kinds(data: dict, num_phases: int) -> list[str]:
    phase_info = data.get("phase_info", {})
    kinds = phase_info.get("phase_kinds")
    if isinstance(kinds, list) and len(kinds) == num_phases:
        return [str(k) for k in kinds]

    cycle_modes = phase_info.get("cycle_modes")
    if isinstance(cycle_modes, list) and len(cycle_modes) == 3:
        from task_def.put_bottles_dustbin import PHASE_KINDS_LEFT, PHASE_KINDS_RIGHT

        inferred: list[str] = []
        for mode in cycle_modes:
            if mode == "right":
                inferred.extend(PHASE_KINDS_RIGHT)
            else:
                inferred.extend(PHASE_KINDS_LEFT)
        if len(inferred) == num_phases:
            return inferred

    # Fallback from total phase count
    if num_phases == 12:
        return ["approach", "grasp", "move_dustbin", "release"] * 3
    if num_phases == 21:
        return (
            ["approach_pickup", "grasp_pickup", "move_center", "approach_center", "grasp_bin", "move_dustbin", "release"]
            * 3
        )
    return ["approach", "grasp", "move_dustbin", "release"] * max(1, num_phases // 4)


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 12


def enrich_episode(json_path: Path) -> bool:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    instructions = data.get("instructions") or [DEFAULT_INSTRUCTION]
    if not isinstance(instructions, list):
        instructions = [str(instructions)]

    num_phases = infer_num_phases(data)
    phase_kinds = infer_phase_kinds(data, num_phases)

    subtasks_list = []
    for _ in instructions:
        phase_texts = []
        for kind in phase_kinds:
            variants = VARIANTS_BY_KIND.get(kind, FALLBACK_VARIANTS)
            phase_texts.append(random.choice(variants))
        subtasks_list.append(phase_texts)

    data["instructions"] = instructions
    data["subtasks"] = subtasks_list

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich put_bottles_dustbin subtask language templates.")
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
