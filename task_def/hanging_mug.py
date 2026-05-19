"""Rule-based phase segmentation for the hanging_mug task.

Left arm approaches, grasps, moves the mug to center, and places it.
Right arm approaches the centered mug, grasps it, then hangs it on the rack.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class HangingMugProcessor(BaseTaskProcessor):
    """Seven-phase splitter for left place + right hang workflow."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        min_drop_frames: int = 3,
        open_delta: float = 0.01,
        lookback: int = 15,
        z_eps: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_change_threshold: float = 0.005,
        z_window: int = 5,
        search_gap: int = 3,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.min_drop_frames = min_drop_frames
        self.open_delta = open_delta
        self.lookback = lookback
        self.z_eps = z_eps
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_change_threshold = z_change_threshold
        self.z_window = z_window
        self.search_gap = search_gap

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
        if left_close_start is None:
            left_close_start = max(1, total_steps // 6)

        c0 = self._find_approach_end_near_close_start(
            gripper=left_gripper,
            z=left_z,
            joint_velocity=left_joint_vel,
            start=0,
            close_start=left_close_start,
            total_steps=total_steps,
        )
        c1 = self._find_move_start_after_grasp(
            gripper=left_gripper,
            joint_velocity=left_joint_vel,
            eef_velocity=left_eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        c2 = self._find_place_start_by_gripper_opening(
            gripper=left_gripper,
            start=c1,
            total_steps=total_steps,
        )

        c3 = self._find_right_approach_start(
            right_z=right_z,
            right_joint_velocity=right_joint_vel,
            right_eef_velocity=right_eef_vel,
            start=c2 + self.search_gap,
            total_steps=total_steps,
        )

        right_close_start = self._find_close_start(right_gripper, start=c3, total_steps=total_steps)
        if right_close_start is None:
            right_close_start = min(total_steps - 1, c3 + max(1, (total_steps - c3) // 4))

        c4 = self._find_approach_end_near_close_start(
            gripper=right_gripper,
            z=right_z,
            joint_velocity=right_joint_vel,
            start=c3,
            close_start=right_close_start,
            total_steps=total_steps,
        )
        c5 = self._find_move_start_after_grasp(
            gripper=right_gripper,
            joint_velocity=right_joint_vel,
            eef_velocity=right_eef_vel,
            start=c4,
            total_steps=total_steps,
        )

        checkpoints = self._enforce_order([c0, c1, c2, c3, c4, c5], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

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
        close_start: int,
        total_steps: int,
    ) -> int:
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

    def _find_move_start_after_grasp(
        self,
        gripper: np.ndarray,
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        close_done = self._first_consecutive_leq(gripper, self.close_threshold, start)
        if close_done is None:
            close_done = start

        if eef_velocity is not None:
            eef_hits = np.where(eef_velocity[close_done:] > self.eef_velocity_threshold)[0]
            if len(eef_hits) > 0:
                return int(close_done + eef_hits[0])

        joint_hits = np.where(joint_velocity[close_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(close_done + joint_hits[0])
        return int(close_done)

    def _find_right_approach_start(
        self,
        right_z: Optional[np.ndarray],
        right_joint_velocity: np.ndarray,
        right_eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        z_window = max(1, self.z_window)
        stable = max(1, self.stable_window)
        for t in range(max(0, start), max(start, total_steps - z_window)):
            joint_ok = np.any(right_joint_velocity[t : t + stable] > self.joint_velocity_threshold)
            eef_ok = right_eef_velocity is not None and np.any(right_eef_velocity[t : t + stable] > self.eef_velocity_threshold)
            z_ok = False
            if right_z is not None:
                z_ok = abs(float(right_z[t + z_window] - right_z[t])) > self.z_change_threshold
            if (joint_ok or eef_ok) and (z_ok or eef_ok):
                return int(t)
        return int(start)

    def _find_open_start(self, gripper: np.ndarray, start: int, total_steps: int) -> Optional[int]:
        for t in range(max(1, start), total_steps):
            if gripper[t - 1] <= self.close_threshold and gripper[t] > gripper[t - 1] + self.open_delta:
                return int(t)
        return None

    def _find_place_start_by_gripper_opening(self, gripper: np.ndarray, start: int, total_steps: int) -> int:
        return self._find_open_start(gripper, start, total_steps) or int(start)

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
            "Approach the mug with the left arm.",
            "Grasp the mug with the left arm.",
            "Move the mug to the center position.",
            "Place the mug at the center.",
            "Approach the centered mug with the right arm.",
            "Grasp the mug with the right arm.",
            "Hang the mug onto the rack.",
        ]
