#!/usr/bin/env python3
"""Plot one place_object_scale episode with predicted phase boundaries."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_def.place_object_scale import PlaceObjectScaleProcessor
from task_def.trajectory_analyzer import TrajectoryAnalyzer


TASK_NAME = "place_object_scale"
DEFAULT_RAW_ROOTS = [
    Path("/data2/liujingzhi/robotwin/robotwin2_dataset"),
    Path("/mnt/data1/liujingzhi/dataset"),
]


def infer_raw_path(hdf5_path: Path, raw_root: Optional[Path] = None) -> Optional[Path]:
    match = re.search(r"episode_(\d+)", hdf5_path.name)
    if not match:
        return None
    episode_id = match.group(1)
    data_dir_name = hdf5_path.parent.parent.name
    setting = "aloha-agilex_clean_50" if "clean_50" in data_dir_name else "aloha-agilex_randomized_500"

    roots = [raw_root] if raw_root is not None else DEFAULT_RAW_ROOTS
    for root in roots:
        if root is None:
            continue
        candidate = root / TASK_NAME / setting / "data" / f"episode{episode_id}.hdf5"
        if candidate.exists():
            return candidate
    return None


def load_raw_eef_xyz(raw_path: Optional[Path], total_steps: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if raw_path is None or not raw_path.exists():
        return None, None
    with h5py.File(raw_path, "r") as raw_f:
        if "endpose/left_endpose" not in raw_f or "endpose/right_endpose" not in raw_f:
            return None, None
        left_xyz = raw_f["endpose/left_endpose"][()][:total_steps, :3]
        right_xyz = raw_f["endpose/right_endpose"][()][:total_steps, :3]
    return align_xyz(left_xyz, total_steps), align_xyz(right_xyz, total_steps)


def align_xyz(xyz: np.ndarray, total_steps: int) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) >= total_steps:
        return xyz[:total_steps, :3]
    if len(xyz) == 0:
        return xyz
    pad = np.repeat(xyz[-1:, :3], total_steps - len(xyz), axis=0)
    return np.concatenate([xyz[:, :3], pad], axis=0)


def analyze_episode(
    hdf5_path: Path,
    save_path: Optional[Path] = None,
    raw_episode_path: Optional[Path] = None,
):
    analyzer = TrajectoryAnalyzer()
    task_processor = PlaceObjectScaleProcessor()

    with h5py.File(hdf5_path, "r") as f:
        left_gripper, right_gripper = analyzer.extract_gripper_states(f)
        total_steps = len(left_gripper)
        qpos = analyzer.extract_qpos(f)[:total_steps]

        raw_path = raw_episode_path or infer_raw_path(hdf5_path)
        left_xyz, right_xyz = load_raw_eef_xyz(raw_path, total_steps)

        checkpoints = task_processor.get_phase_checkpoints(
            f,
            external_eef_xyz_left=left_xyz,
            external_eef_xyz_right=right_xyz,
        )
        checkpoints = task_processor.validate_checkpoints(checkpoints, total_steps)

        active_arm = task_processor.active_arm or "left"
        active_gripper = left_gripper if active_arm == "left" else right_gripper
        left_xyz, right_xyz = analyzer.extract_left_right_eef_xyz(
            f,
            total_steps=total_steps,
            external_eef_xyz_left=left_xyz,
            external_eef_xyz_right=right_xyz,
        )
        active_xyz = left_xyz if active_arm == "left" else right_xyz
        active_z = active_xyz[:, 2] if active_xyz is not None else None
        active_joint_vel = analyzer.compute_velocity(
            qpos, arm_indices=(0, 6) if active_arm == "left" else (7, 13)
        )

        place_done = None
        if len(checkpoints) >= 3:
            place_done = task_processor.find_place_done(
                active_gripper,
                active_z,
                active_joint_vel,
                checkpoints[2],
                total_steps,
            )

    vel_left = analyzer.compute_velocity(qpos, arm_indices=(0, 6))
    vel_right = analyzer.compute_velocity(qpos, arm_indices=(7, 13))
    z_left = left_xyz[:, 2] if left_xyz is not None else None
    z_right = right_xyz[:, 2] if right_xyz is not None else None
    time_steps = np.arange(total_steps)

    fig, axes = plt.subplots(4, 1, figsize=(16, 15))
    fig.suptitle(
        f"{TASK_NAME} trajectory analysis ({hdf5_path.parent.name}, active={active_arm})",
        fontweight="bold",
    )

    axes[0].plot(time_steps, left_gripper, "b-", label="Left Gripper", alpha=0.8)
    axes[0].plot(time_steps, right_gripper, "g-", label="Right Gripper", alpha=0.8)
    axes[0].axhline(y=task_processor.gripper_open_value, color="orange", linestyle="--", alpha=0.5, label="Open value 1.0")
    axes[0].axhline(y=task_processor.close_threshold, color="k", linestyle="--", alpha=0.4, label="Close threshold 0.05")
    axes[0].set_ylabel("Gripper")
    axes[0].set_title("Gripper States")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    if z_left is not None:
        axes[1].plot(time_steps, z_left, "b-", label="Left EEF Z", linewidth=1.3)
    if z_right is not None:
        axes[1].plot(time_steps, z_right, "g-", label="Right EEF Z", linewidth=1.3)
    axes[1].set_ylabel("Z")
    axes[1].set_title("End-Effector Height")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_steps, vel_left, "b-", label="Left Joint Velocity", alpha=0.7)
    axes[2].plot(time_steps, vel_right, "g-", label="Right Joint Velocity", alpha=0.7)
    axes[2].axhline(y=task_processor.joint_velocity_threshold, color="r", linestyle="--", alpha=0.5, label="Velocity threshold")
    axes[2].set_ylabel("Velocity")
    axes[2].set_title("Joint Velocity")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlim(0, total_steps)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_title(f"Predicted Phases (Total: {len(checkpoints) + 1})")
    phase_edges = [0] + checkpoints + [total_steps]
    labels = ["Approach", "Grasp", "Move", "Place"]
    colors = ["#ffcccc", "#ffd9cc", "#ffe6cc", "#ccffcc"]
    for idx in range(len(phase_edges) - 1):
        start, end = phase_edges[idx], phase_edges[idx + 1]
        mid = (start + end) / 2
        axes[3].axvspan(start, end, color=colors[idx % len(colors)], alpha=0.55)
        axes[3].axvline(x=start, color="k", linestyle="-", linewidth=0.9)
        label = labels[idx] if idx < len(labels) else f"Phase {idx}"
        axes[3].text(mid, 0.5, f"P{idx}: {label}", ha="center", va="center", fontsize=10, fontweight="bold")

    checkpoint_names = ["c0", "c1", "c2"]
    for cp_idx, cp in enumerate(checkpoints):
        for ax in axes[:3]:
            ax.axvline(x=cp, color="red", linestyle="--", linewidth=0.9, alpha=0.85)
        name = checkpoint_names[cp_idx] if cp_idx < len(checkpoint_names) else f"c{cp_idx}"
        axes[3].text(cp, 0.92, name, ha="center", va="top", fontsize=8, color="red", fontweight="bold")

    if place_done is not None and 0 < place_done < total_steps:
        for ax in axes[:3]:
            ax.axvline(x=place_done, color="gray", linestyle=":", linewidth=1.0, alpha=0.9)
        axes[3].text(place_done, 0.08, "c3 (place done)", ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one place_object_scale trajectory.")
    parser.add_argument("hdf5_path", type=str, help="Path to episode_*/episode_*.hdf5")
    parser.add_argument("--save", type=str, default=None, help="Output PNG path")
    parser.add_argument("--raw_episode", type=str, default=None, help="Optional raw episode HDF5 path")
    args = parser.parse_args()

    analyze_episode(
        Path(args.hdf5_path),
        save_path=Path(args.save) if args.save else None,
        raw_episode_path=Path(args.raw_episode) if args.raw_episode else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
