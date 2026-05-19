"""Rule-based phase segmentation for the place_bread_skillet task.

Dual-arm cooperative: one arm holds/adjusts the skillet, the other grasps and
releases the bread. Bread arm is inferred as the gripper that opens after grasping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class PlaceBreadSkilletProcessor(BaseTaskProcessor):
    """Four-phase splitter: dual approach, dual grasp, adjust skillet, release bread."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.9,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        min_drop_frames: int = 3,
        lookback: int = 15,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_rise_threshold: float = 0.005,
        z_window: int = 5,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.open_done_threshold = open_done_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.min_drop_frames = min_drop_frames
        self.lookback = lookback
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_rise_threshold = z_rise_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.bread_arm: Optional[str] = None
        self.skillet_arm: Optional[str] = None

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
        arm_data = {
            "left": {
                "gripper": left_gripper,
                "z": left_xyz[:, 2] if left_xyz is not None else None,
                "joint_vel": self.analyzer.compute_velocity(qpos, arm_indices=(0, 6)),
                "eef_vel": self._compute_eef_velocity(left_xyz, total_steps),
            },
            "right": {
                "gripper": right_gripper,
                "z": right_xyz[:, 2] if right_xyz is not None else None,
                "joint_vel": self.analyzer.compute_velocity(qpos, arm_indices=(7, 13)),
                "eef_vel": self._compute_eef_velocity(right_xyz, total_steps),
            },
        }

        bread_arm, skillet_arm = self._assign_arm_roles(arm_data, total_steps)
        self.bread_arm = bread_arm
        self.skillet_arm = skillet_arm

        bread = arm_data[bread_arm]
        skillet = arm_data[skillet_arm]
        bread_close_start = self._find_close_start(bread["gripper"], 0) or max(1, total_steps // 8)
        skillet_close_start = self._find_close_start(skillet["gripper"], 0) or max(1, total_steps // 8)

        bread_c0 = self._find_approach_end_near_close_start(
            gripper=bread["gripper"],
            z=bread["z"],
            joint_velocity=bread["joint_vel"],
            start=0,
            total_steps=total_steps,
            close_start=bread_close_start,
        )
        skillet_c0 = self._find_approach_end_near_close_start(
            gripper=skillet["gripper"],
            z=skillet["z"],
            joint_velocity=skillet["joint_vel"],
            start=0,
            total_steps=total_steps,
            close_start=skillet_close_start,
        )
        c0 = max(bread_c0, skillet_c0)

        bread_c1 = self._find_move_start_after_grasp(
            bread["gripper"], bread["joint_vel"], bread["eef_vel"], c0, total_steps
        )
        skillet_c1 = self._find_move_start_after_grasp(
            skillet["gripper"], skillet["joint_vel"], skillet["eef_vel"], c0, total_steps
        )
        c1 = max(bread_c1, skillet_c1)

        c2 = self._find_place_start_by_gripper_opening(bread["gripper"], c1, total_steps)

        checkpoints = self._enforce_order([c0, c1, c2], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def find_place_done(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        total_steps: int,
    ) -> Optional[int]:
        """Optional visualization checkpoint after bread release."""
        open_done = self._first_consecutive_geq(gripper, self.open_done_threshold, start)
        if open_done is None:
            return None

        z_window = max(1, self.z_window)
        for idx in range(open_done, max(open_done, total_steps - z_window)):
            g_ok = gripper[idx] >= self.open_done_threshold
            joint_ok = joint_velocity[idx] > self.joint_velocity_threshold
            z_ok = z is not None and float(z[idx + z_window] - z[idx]) > self.z_rise_threshold
            if g_ok and (joint_ok or z_ok):
                return int(idx)
        return open_done

    def _assign_arm_roles(self, arm_data: dict, total_steps: int) -> Tuple[str, str]:
        left_gripper = arm_data["left"]["gripper"]
        right_gripper = arm_data["right"]["gripper"]

        left_close_start = self._find_close_start(left_gripper, 0) or 0
        right_close_start = self._find_close_start(right_gripper, 0) or 0
        left_close_done = self._first_consecutive_leq(left_gripper, self.close_threshold, left_close_start) or left_close_start
        right_close_done = self._first_consecutive_leq(right_gripper, self.close_threshold, right_close_start) or right_close_start

        left_open_start = self._find_open_start(left_gripper, left_close_done)
        right_open_start = self._find_open_start(right_gripper, right_close_done)

        if left_open_start is not None and right_open_start is None:
            return "left", "right"
        if right_open_start is not None and left_open_start is None:
            return "right", "left"

        if left_open_start is not None and right_open_start is not None:
            if left_open_start <= right_open_start:
                return "left", "right"
            return "right", "left"

        # Fallback: arm with larger gripper excursion after close is more likely bread.
        left_range = float(np.max(left_gripper) - np.min(left_gripper))
        right_range = float(np.max(right_gripper) - np.min(right_gripper))
        if left_range >= right_range:
            return "left", "right"
        return "right", "left"

    def _find_close_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        min_drop_frames = max(1, self.min_drop_frames)
        for t in range(start + 1, max(start + 1, len(gripper) - min_drop_frames)):
            was_open = np.isclose(gripper[t - 1], self.gripper_open_value, atol=self.gripper_open_atol)
            starts_closing = gripper[t] < gripper[t - 1] - self.open_delta
            continues_closing = gripper[t + min_drop_frames] < gripper[t] - self.open_delta
            if was_open and starts_closing and continues_closing:
                return int(t)

        candidates = np.where(~np.isclose(gripper[start:], self.gripper_open_value, atol=self.gripper_open_atol))[0]
        if len(candidates) > 0:
            return int(start + candidates[0])
        return None

    def _find_open_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        for idx in range(max(1, start), len(gripper)):
            if gripper[idx - 1] <= self.close_threshold and gripper[idx] > gripper[idx - 1] + self.open_delta:
                return int(idx)
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

    def _find_place_start_by_gripper_opening(self, gripper: np.ndarray, start: int, total_steps: int) -> int:
        open_start = self._find_open_start(gripper, start)
        return int(open_start) if open_start is not None else int(start)

    def _first_consecutive_leq(self, series: np.ndarray, threshold: float, start: int) -> Optional[int]:
        window = max(1, self.stable_window)
        for idx in range(max(0, start), max(0, len(series) - window + 1)):
            if np.all(series[idx : idx + window] <= threshold):
                return int(idx)
        candidates = np.where(series[max(0, start) :] <= threshold)[0]
        if len(candidates) > 0:
            return int(max(0, start) + candidates[0])
        return None

    def _first_consecutive_geq(self, series: np.ndarray, threshold: float, start: int) -> Optional[int]:
        window = max(1, self.stable_window)
        for idx in range(max(0, start), max(0, len(series) - window + 1)):
            if np.all(series[idx : idx + window] >= threshold):
                return int(idx)
        candidates = np.where(series[max(0, start) :] >= threshold)[0]
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
            "Approach the skillet and the bread.",
            "Grasp the skillet and the bread.",
            "Adjust the skillet under the bread.",
            "Release the bread into the skillet.",
        ]
