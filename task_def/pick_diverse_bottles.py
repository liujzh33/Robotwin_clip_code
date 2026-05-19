"""Rule-based phase segmentation for the pick_diverse_bottles task.

Dual-arm parallel task: approach both bottles, grasp both, then move both to center.
Each checkpoint is computed per arm, then the later frame is used globally.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class PickDiverseBottlesProcessor(BaseTaskProcessor):
    """Three-phase splitter: dual approach, dual grasp, dual move to center."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        min_drop_frames: int = 3,
        lookback: int = 15,
        z_eps: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        xy_move_threshold: float = 0.015,
        xy_window: int = 5,
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
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.xy_move_threshold = xy_move_threshold
        self.xy_window = xy_window

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

        left_c1 = self._find_grasp_end_move_start(
            gripper=left_gripper,
            xyz=left_xyz,
            joint_velocity=left_joint_vel,
            eef_velocity=left_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        right_c1 = self._find_grasp_end_move_start(
            gripper=right_gripper,
            xyz=right_xyz,
            joint_velocity=right_joint_vel,
            eef_velocity=right_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        c1 = max(left_c1, right_c1)

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

    def _find_grasp_end_move_start(
        self,
        gripper: np.ndarray,
        xyz: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        close_candidates = np.where(gripper[start:] <= self.close_threshold)[0]
        close_done = start + int(close_candidates[0]) if len(close_candidates) > 0 else start

        if xyz is None:
            xyz = np.zeros((total_steps, 3), dtype=np.float64)

        xy = np.asarray(xyz[:total_steps, :2], dtype=np.float64)
        xy_window = max(1, self.xy_window)

        for t in range(close_done, max(close_done, total_steps - xy_window)):
            joint_ok = joint_velocity[t] > self.joint_velocity_threshold
            eef_ok = eef_velocity is not None and eef_velocity[t] > self.eef_velocity_threshold
            xy_ok = float(np.linalg.norm(xy[t + xy_window] - xy[t])) > self.xy_move_threshold
            if joint_ok or eef_ok or xy_ok:
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
            "Approach both bottles with both arms.",
            "Grasp both bottles with both grippers.",
            "Move both bottles to the center.",
        ]
