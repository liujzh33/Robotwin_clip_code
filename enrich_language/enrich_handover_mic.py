#!/usr/bin/env python3
"""Enrich handover_mic subtask language with phase-wise variants."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path


PHASE_VARIANTS = {
    0: [
        "Approach the microphone with the first arm.",
        "Move the first gripper toward the microphone.",
        "Guide the initial end-effector closer to the microphone.",
        "Bring the first arm near the microphone.",
        "Move into position for grasping the microphone.",
        "Reach toward the black and white microphone.",
        "Position the first gripper near the microphone handle.",
        "Navigate the first arm to the microphone.",
        "Move closer to the microphone before picking it up.",
        "Approach the microphone and prepare the first grasp.",
    ],
    1: [
        "Grasp the microphone with the first gripper.",
        "Close the first gripper around the microphone handle.",
        "Pick up the microphone with the first arm.",
        "Secure the microphone with the initial gripper.",
        "Clamp onto the microphone handle.",
        "Take hold of the black and white microphone.",
        "Grip the microphone firmly.",
        "Close the fingers to capture the microphone.",
        "Hold the microphone with the first gripper.",
        "Complete the initial grasp so the microphone can be transferred.",
    ],
    2: [
        "Move the microphone to the center handover position.",
        "Carry the microphone toward the middle.",
        "Transport the microphone to the center.",
        "Move the held microphone into the handover area.",
        "Bring the microphone to the central position.",
        "Guide the microphone to the middle for transfer.",
        "Shift the microphone into the center of the workspace.",
        "Move the microphone to where the receiving arm can reach it.",
        "Carry the microphone toward the handover point.",
        "Position the microphone at the center so it can be passed to the other gripper.",
    ],
    3: [
        "Approach the microphone with the receiving arm.",
        "Move the receiving gripper toward the centered microphone.",
        "Guide the other end-effector closer to the microphone.",
        "Bring the receiving arm near the microphone at the center.",
        "Move into position for the handover grasp.",
        "Reach toward the microphone with the receiving gripper.",
        "Position the second gripper near the microphone handle.",
        "Navigate the receiving arm to the centered microphone.",
        "Move closer to the microphone before taking it from the first gripper.",
        "Approach the microphone at the handover position and prepare to receive it.",
    ],
    4: [
        "Grasp the microphone with the receiving gripper.",
        "Close the receiving gripper around the microphone.",
        "Take the microphone from the first gripper.",
        "Secure the microphone with the second gripper.",
        "Clamp the receiving gripper onto the microphone handle.",
        "Receive the microphone with the other hand.",
        "Grip the microphone firmly with the receiving arm.",
        "Close the second gripper to capture the microphone.",
        "Hold the microphone with the receiving gripper.",
        "Complete the handover grasp by securing the microphone with the other gripper.",
    ],
    5: [
        "Release the microphone from the first gripper.",
        "Open the first gripper to let go of the microphone.",
        "Let the receiving gripper take over the microphone.",
        "Release the initial hold on the microphone.",
        "Open the original gripper after the handover.",
        "Free the microphone from the first gripper.",
        "Complete the transfer by opening the first gripper.",
        "Let go once the receiving gripper has secured the microphone.",
        "Disengage the first gripper from the microphone handle.",
        "Finish the handover by releasing the microphone from the original gripper.",
    ],
}

FALLBACK_VARIANTS = [
    "Continue with the next step.",
    "Proceed to the next phase.",
    "Complete this part of the task.",
]

DEFAULT_INSTRUCTION = "Hand over the microphone between the two arms."


def infer_num_phases(data: dict) -> int:
    phase_info = data.get("phase_info", {})
    num_phases = phase_info.get("num_phases")
    if isinstance(num_phases, int) and num_phases > 0:
        return num_phases
    subtasks = data.get("subtasks")
    if subtasks and isinstance(subtasks, list) and isinstance(subtasks[0], list):
        return len(subtasks[0])
    return 6


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
    parser = argparse.ArgumentParser(description="Enrich handover_mic subtask language templates.")
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
