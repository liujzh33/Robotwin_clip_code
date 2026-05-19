"""Rule-based phase segmentation for the click_alarmclock task.

Phases:
0. Move above the alarm-clock button.
1. Close the gripper.
2. Press the alarm-clock button downward.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class ClickAlarmclockProcessor(BaseTaskProcessor):
    """Three-phase splitter: move above target, close gripper, press downward."""

    def __init__(
        self,
        motion_weight: float = 0.1,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        min_drop_frames: int = 3,
        z_peak_eps: float = 0.005,
        down_delta_threshold: float = 0.005,
        velocity_threshold: float = 0.01,
        down_window: int = 5,
        lookback: int = 10,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=velocity_threshold)
        self.motion_weight = motion_weight
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.min_drop_frames = min_drop_frames
        self.z_peak_eps = z_peak_eps
        self.down_delta_threshold = down_delta_threshold
        self.velocity_threshold = velocity_threshold
        self.down_window = down_window
        self.lookback = lookback
        self.active_side = "left"

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: Optional[str] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
        **_,
    ) -> list[int]:
        qpos = self.analyzer.extract_qpos(hdf5_data)
        total_steps = len(qpos)
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)

        left_xyz, right_xyz = self.analyzer.extract_left_right_eef_xyz(
            hdf5_data,
            total_steps=total_steps,
            external_eef_xyz_left=external_eef_xyz_left,
            external_eef_xyz_right=external_eef_xyz_right,
        )
        left_z = left_xyz[:, 2] if left_xyz is not None else None
        right_z = right_xyz[:, 2] if right_xyz is not None else None

        left_joint_vel = self.analyzer.compute_velocity(qpos[:total_steps], arm_indices=(0, 6))
        right_joint_vel = self.analyzer.compute_velocity(qpos[:total_steps], arm_indices=(7, 13))

        if active_side is None:
            active_side = self._choose_active_arm(
                left_gripper,
                right_gripper,
                left_z,
                right_z,
                left_joint_vel,
                right_joint_vel,
            )
        self.active_side = active_side

        active_gripper = left_gripper if active_side == "left" else right_gripper
        active_z = left_z if active_side == "left" else right_z
        active_joint_vel = left_joint_vel if active_side == "left" else right_joint_vel

        c0 = self._find_gripper_close_start(active_gripper, start=0)
        c1 = self._find_press_start_after_gripper_close(
            gripper=active_gripper,
            z=active_z,
            joint_velocity=active_joint_vel,
            c0=c0,
            total_steps=total_steps,
        )
        if c1 <= c0:
            c1 = min(total_steps - 1, c0 + max(1, (total_steps - c0) // 3))

        return self.validate_checkpoints([c0, c1], total_steps)

    def _choose_active_arm(
        self,
        left_gripper: np.ndarray,
        right_gripper: np.ndarray,
        left_z: Optional[np.ndarray],
        right_z: Optional[np.ndarray],
        left_joint_vel: np.ndarray,
        right_joint_vel: np.ndarray,
    ) -> str:
        left_gripper_change = float(np.max(left_gripper) - np.min(left_gripper))
        right_gripper_change = float(np.max(right_gripper) - np.min(right_gripper))
        left_drop = self._z_drop_score(left_z)
        right_drop = self._z_drop_score(right_z)
        left_motion = float(np.sum(left_joint_vel))
        right_motion = float(np.sum(right_joint_vel))

        left_score = left_gripper_change + left_drop + self.motion_weight * left_motion
        right_score = right_gripper_change + right_drop + self.motion_weight * right_motion
        return "left" if left_score >= right_score else "right"

    @staticmethod
    def _z_drop_score(z: Optional[np.ndarray]) -> float:
        if z is None or len(z) == 0:
            return 0.0
        z = np.asarray(z, dtype=np.float64)
        return float(np.max(z) - np.min(z))

    def _find_gripper_close_start(self, gripper: np.ndarray, start: int = 0) -> int:
        """Find the first frame where the active gripper starts leaving open state."""
        total_steps = len(gripper)
        min_drop_frames = max(1, self.min_drop_frames)
        for t in range(start + 1, max(start + 1, total_steps - min_drop_frames)):
            was_open = np.isclose(gripper[t - 1], self.gripper_open_value, atol=self.gripper_open_atol)
            starts_closing = gripper[t] < gripper[t - 1] - self.gripper_open_atol
            continues_closing = gripper[t + min_drop_frames] < gripper[t] - self.gripper_open_atol
            if was_open and starts_closing and continues_closing:
                return int(t)

        candidates = np.where(~np.isclose(gripper[start:], self.gripper_open_value, atol=self.gripper_open_atol))[0]
        if len(candidates) > 0:
            return int(start + candidates[0])
        return max(1, int(total_steps * 0.5))

    def _find_press_start_after_gripper_close(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        c0: int,
        total_steps: int,
    ) -> int:
        """Find the high/stable point after gripper close and before downward pressing."""
        close_candidates = np.where(gripper[c0:] <= self.close_threshold)[0]
        close_done = c0 + int(close_candidates[0]) if len(close_candidates) > 0 else c0

        if z is None or len(z) == 0:
            return int(close_done)

        z = np.asarray(z[:total_steps], dtype=np.float64)
        down_window = max(1, self.down_window)
        if close_done >= total_steps - down_window - 1:
            return max(1, min(total_steps - 1, close_done))

        downward_candidates = []
        for t in range(max(0, close_done), total_steps - down_window):
            z_drop_ok = z[t] - z[t + down_window] > self.down_delta_threshold
            v_ok = np.any(joint_velocity[t : t + down_window] > self.velocity_threshold)
            if z_drop_ok and v_ok:
                downward_candidates.append(t)

        if not downward_candidates:
            return int(close_done)

        first_down = int(downward_candidates[0])
        local_start = max(close_done, first_down - self.lookback)
        local_end = first_down + 1
        local_z = z[local_start:local_end]
        local_max = float(np.max(local_z))
        peak_candidates = np.where(local_z >= local_max - self.z_peak_eps)[0]

        if len(peak_candidates) > 0:
            return int(local_start + peak_candidates[-1])
        return int(local_start + np.argmax(local_z))

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Move above the alarm-clock button.",
            "Close the gripper.",
            "Press the alarm-clock button downward.",
        ]
