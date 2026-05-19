"""Rule-based phase segmentation for the lift_pot task.

Two-arm cooperative approach, grasp, and lift. Each checkpoint is computed per arm,
then the later frame is used as the global boundary.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class LiftPotProcessor(BaseTaskProcessor):
    """Three-phase splitter for dual-arm pot approach, grasp, and lift."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        semi_close_low: float = 0.3,
        semi_close_high: float = 0.7,
        drop_delta: float = 0.02,
        min_drop_frames: int = 3,
        search_window: int = 30,
        pre_offset: int = 2,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        z_lift_threshold: float = 0.005,
        z_window: int = 5,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.semi_close_low = semi_close_low
        self.semi_close_high = semi_close_high
        self.drop_delta = drop_delta
        self.min_drop_frames = min_drop_frames
        self.search_window = search_window
        self.pre_offset = pre_offset
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.z_lift_threshold = z_lift_threshold
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

        left_c0 = self._find_second_close_start_for_lift_pot(
            left_gripper, start=0, total_steps=total_steps
        )
        right_c0 = self._find_second_close_start_for_lift_pot(
            right_gripper, start=0, total_steps=total_steps
        )
        c0 = max(left_c0, right_c0)

        left_lift_start = self._find_grasp_end_lift_start(
            gripper=left_gripper,
            z=left_z,
            joint_velocity=left_joint_vel,
            eef_velocity=left_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        right_lift_start = self._find_grasp_end_lift_start(
            gripper=right_gripper,
            z=right_z,
            joint_velocity=right_joint_vel,
            eef_velocity=right_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        c1 = max(left_lift_start, right_lift_start)

        return self.validate_checkpoints(self._enforce_order([c0, c1], total_steps), total_steps)

    def _find_second_close_start_for_lift_pot(
        self,
        gripper: np.ndarray,
        start: int,
        total_steps: int,
    ) -> int:
        """
        Ignore the initial 1 -> ~0.5 pre-close; find when the gripper continues
        from the semi-closed platform down toward full close.
        """
        min_drop_frames = max(1, self.min_drop_frames)
        search_window = max(min_drop_frames + 1, self.search_window)

        for t in range(start + 1, max(start + 1, total_steps - search_window)):
            in_half_closed = self.semi_close_low <= gripper[t] <= self.semi_close_high
            starts_second_closing = gripper[t + min_drop_frames] < gripper[t] - self.drop_delta
            segment_end = min(total_steps, t + search_window)
            reaches_full_close = float(np.min(gripper[t:segment_end])) <= self.close_threshold
            if in_half_closed and starts_second_closing and reaches_full_close:
                return int(max(start, t - self.pre_offset))

        close_candidates = np.where(gripper[start:] <= self.close_threshold)[0]
        if len(close_candidates) > 0:
            close_done = start + int(close_candidates[0])
            return int(max(start, close_done - 5))

        return int(start)

    def _find_grasp_end_lift_start(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        close_done = self._first_consecutive_leq(gripper, self.close_threshold, start)
        if close_done is None:
            close_done = start

        z_window = max(1, self.z_window)
        for t in range(close_done, max(close_done, total_steps - z_window)):
            joint_ok = joint_velocity[t] > self.joint_velocity_threshold
            eef_ok = eef_velocity is not None and eef_velocity[t] > self.eef_velocity_threshold
            z_ok = False
            if z is not None:
                z_ok = float(z[t + z_window] - z[t]) > self.z_lift_threshold
            if joint_ok or eef_ok or z_ok:
                return int(t)
        return int(close_done)

    def _first_consecutive_leq(self, series: np.ndarray, threshold: float, start: int) -> Optional[int]:
        window = max(1, self.stable_window)
        for idx in range(max(0, start), max(0, len(series) - window + 1)):
            if np.all(series[idx : idx + window] <= threshold):
                return int(idx)
        candidates = np.where(series[max(0, start) :] <= threshold)[0]
        if len(candidates) > 0:
            return int(max(0, start) + candidates[0])
        return None

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
            "Approach the pot with both arms.",
            "Grasp the pot with both grippers.",
            "Lift the pot upward with both arms.",
        ]
