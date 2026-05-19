"""Rule-based phase segmentation for the grab_roller task.

This is a two-arm cooperative task. Each checkpoint is computed separately for
left and right arms, then the later frame is used as the global checkpoint.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class GrabRollerProcessor(BaseTaskProcessor):
    """Three-phase splitter for approaching, grasping, and lifting a roller."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        min_drop_frames: int = 3,
        lookback: int = 15,
        z_eps: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 3,
        z_rise_threshold: float = 0.005,
        z_window: int = 5,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.min_drop_frames = min_drop_frames
        self.lookback = lookback
        self.z_eps = z_eps
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.z_rise_threshold = z_rise_threshold
        self.z_window = z_window
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: Optional[str] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
        **_,
    ) -> list[int]:
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)
        qpos = self.analyzer.extract_qpos(hdf5_data)[:total_steps]

        left_xyz, right_xyz = self.analyzer.extract_left_right_eef_xyz(
            hdf5_data,
            total_steps=total_steps,
            external_eef_xyz_left=external_eef_xyz_left,
            external_eef_xyz_right=external_eef_xyz_right,
        )
        left_z = left_xyz[:, 2] if left_xyz is not None else None
        right_z = right_xyz[:, 2] if right_xyz is not None else None
        left_joint_vel = self.analyzer.compute_velocity(qpos, arm_indices=(0, 6))
        right_joint_vel = self.analyzer.compute_velocity(qpos, arm_indices=(7, 13))
        left_eef_vel = self._compute_eef_velocity(left_xyz, total_steps)
        right_eef_vel = self._compute_eef_velocity(right_xyz, total_steps)

        left_close_start = self._find_close_start(left_gripper, start=0, total_steps=total_steps)
        right_close_start = self._find_close_start(right_gripper, start=0, total_steps=total_steps)

        left_c0 = self._find_approach_end_near_close_start(
            gripper=left_gripper,
            z=left_z,
            joint_velocity=left_joint_vel,
            start=0,
            total_steps=total_steps,
            close_start=left_close_start,
        )
        right_c0 = self._find_approach_end_near_close_start(
            gripper=right_gripper,
            z=right_z,
            joint_velocity=right_joint_vel,
            start=0,
            total_steps=total_steps,
            close_start=right_close_start,
        )
        c0 = max(left_c0, right_c0)

        left_lift_start = self._find_lift_start_after_close(
            gripper=left_gripper,
            z=left_z,
            joint_velocity=left_joint_vel,
            eef_velocity=left_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        right_lift_start = self._find_lift_start_after_close(
            gripper=right_gripper,
            z=right_z,
            joint_velocity=right_joint_vel,
            eef_velocity=right_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        c1 = max(left_lift_start, right_lift_start)

        return self.validate_checkpoints(self._enforce_order([c0, c1], total_steps), total_steps)

    def _find_close_start(self, gripper: np.ndarray, start: int, total_steps: int) -> Optional[int]:
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
        return None

    def _find_approach_end_near_close_start(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        total_steps: int,
        close_start: Optional[int],
    ) -> int:
        if close_start is None:
            return int(start)

        window = max(1, self.stable_window)
        search_start = max(start, close_start - self.lookback)
        search_end = min(total_steps, close_start)
        if search_end <= search_start + window or z is None:
            return int(max(start, close_start - 1))

        z_segment = np.asarray(z[search_start:search_end], dtype=np.float64)
        local_min_z = float(np.min(z_segment))
        candidates = []

        for t in range(search_start, search_end - window + 1):
            g_ok = np.all(np.isclose(gripper[t : t + window], self.gripper_open_value, atol=self.gripper_open_atol))
            z_ok = np.all(z[t : t + window] <= local_min_z + self.z_eps)
            v_ok = np.all(joint_velocity[t : t + window] <= self.stable_velocity_threshold)
            if g_ok and z_ok and v_ok:
                candidates.append(t)

        if candidates:
            return int(candidates[-1])
        return int(max(start, close_start - 1))

    def _find_lift_start_after_close(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        close_candidates = np.where(gripper[start:] <= self.close_threshold)[0]
        close_done = start + int(close_candidates[0]) if len(close_candidates) > 0 else start
        if z is None:
            return int(close_done)

        z = np.asarray(z[:total_steps], dtype=np.float64)
        z_window = max(1, self.z_window)
        for t in range(close_done, max(close_done, total_steps - z_window)):
            z_rise_ok = z[t + z_window] - z[t] > self.z_rise_threshold
            joint_move_ok = joint_velocity[t] > self.joint_velocity_threshold
            eef_move_ok = eef_velocity is not None and eef_velocity[t] > self.eef_velocity_threshold
            if z_rise_ok and (joint_move_ok or eef_move_ok):
                return int(t)
        return int(close_done)

    @staticmethod
    def _compute_eef_velocity(xyz: Optional[np.ndarray], total_steps: int) -> Optional[np.ndarray]:
        if xyz is None:
            return None
        velocity = np.linalg.norm(np.diff(xyz[:total_steps], axis=0), axis=1)
        return np.insert(velocity, 0, 0.0)

    @staticmethod
    def _enforce_order(points: list[int], total_steps: int) -> list[int]:
        ordered = []
        previous = 0
        for point in points:
            value = int(min(total_steps - 1, max(previous + 1, point)))
            ordered.append(value)
            previous = value
        return ordered

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the roller with both arms.",
            "Grasp the roller with both grippers.",
            "Lift the roller off the table.",
        ]
