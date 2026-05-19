"""Rule-based phase segmentation for the dump_bin_bigbin task.

Two execution modes are supported:
- direct_dump: first active arm is left -> 3 phases
- transfer_then_dump: first active arm is right -> 7 phases
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class DumpBinBigbinProcessor(BaseTaskProcessor):
    """Split dump_bin_bigbin into direct-dump or transfer-then-dump phases."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.9,
        approach_pre_offset: int = 0,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        min_drop_frames: int = 3,
        open_delta: float = 0.01,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_rise_threshold: float = 0.005,
        z_window: int = 5,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        search_gap: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.open_done_threshold = open_done_threshold
        self.approach_pre_offset = approach_pre_offset
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.min_drop_frames = min_drop_frames
        self.open_delta = open_delta
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_rise_threshold = z_rise_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.search_gap = search_gap
        self.mode = "direct_dump"
        self.active_arm_sequence: list[str] = []

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
                "xyz": left_xyz,
                "z": left_xyz[:, 2] if left_xyz is not None else None,
                "joint_vel": self.analyzer.compute_velocity(qpos, arm_indices=(0, 6)),
                "eef_vel": self._compute_eef_velocity(left_xyz, total_steps),
            },
            "right": {
                "gripper": right_gripper,
                "xyz": right_xyz,
                "z": right_xyz[:, 2] if right_xyz is not None else None,
                "joint_vel": self.analyzer.compute_velocity(qpos, arm_indices=(7, 13)),
                "eef_vel": self._compute_eef_velocity(right_xyz, total_steps),
            },
        }

        first_arm, first_close_start = self._find_next_active_arm_by_close(
            left_gripper,
            right_gripper,
            start=0,
            total_steps=total_steps,
        )
        if first_arm is None or first_close_start is None:
            return self.validate_checkpoints([total_steps // 3, 2 * total_steps // 3], total_steps)

        self.active_arm_sequence = [first_arm]
        self.mode = "direct_dump" if first_arm == "left" else "transfer_then_dump"

        first_data = arm_data[first_arm]
        c0 = self._approach_end_from_close_start(start=0, close_start=first_close_start)
        c1 = self._find_move_start_after_grasp(
            gripper=first_data["gripper"],
            joint_velocity=first_data["joint_vel"],
            eef_velocity=first_data["eef_vel"],
            start=c0,
            total_steps=total_steps,
        )

        if self.mode == "direct_dump":
            return self.validate_checkpoints(self._enforce_order([c0, c1], total_steps), total_steps)

        c2 = self._find_place_start_by_gripper_opening(
            gripper=first_data["gripper"],
            start=c1,
            total_steps=total_steps,
        )
        c3 = self._find_place_end_after_release(
            gripper=first_data["gripper"],
            z=first_data["z"],
            joint_velocity=first_data["joint_vel"],
            start=c2,
            total_steps=total_steps,
        )

        second_arm, second_close_start = self._find_next_active_arm_by_close(
            left_gripper,
            right_gripper,
            start=min(total_steps - 1, c3 + self.search_gap),
            total_steps=total_steps,
        )
        if second_arm is None or second_close_start is None:
            fallback = self._enforce_order([c0, c1, c2, c3], total_steps)
            return self.validate_checkpoints(fallback, total_steps)

        self.active_arm_sequence.append(second_arm)
        second_data = arm_data[second_arm]
        c4 = self._approach_end_from_close_start(start=c3, close_start=second_close_start)
        c5 = self._find_move_start_after_grasp(
            gripper=second_data["gripper"],
            joint_velocity=second_data["joint_vel"],
            eef_velocity=second_data["eef_vel"],
            start=c4,
            total_steps=total_steps,
        )

        checkpoints = self._enforce_order([c0, c1, c2, c3, c4, c5], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_next_active_arm_by_close(
        self,
        left_gripper: np.ndarray,
        right_gripper: np.ndarray,
        start: int,
        total_steps: int,
    ) -> tuple[Optional[str], Optional[int]]:
        left_close = self._find_close_start(left_gripper, start, total_steps)
        right_close = self._find_close_start(right_gripper, start, total_steps)

        if left_close is None and right_close is None:
            return None, None
        if right_close is None or (left_close is not None and left_close < right_close):
            return "left", left_close
        return "right", right_close

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

    def _approach_end_from_close_start(self, start: int, close_start: int) -> int:
        """dump_bin_bigbin-specific approach end: gripper close_start, not global low-Z."""
        return int(max(start, close_start - self.approach_pre_offset))

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

    def _find_place_start_by_gripper_opening(
        self,
        gripper: np.ndarray,
        start: int,
        total_steps: int,
    ) -> int:
        for t in range(max(1, start), total_steps):
            if gripper[t - 1] <= self.close_threshold and gripper[t] > gripper[t - 1] + self.open_delta:
                return int(t)
        return int(start)

    def _find_place_end_after_release(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        total_steps: int,
    ) -> int:
        open_done = self._first_consecutive_geq(gripper, self.open_done_threshold, start)
        if open_done is None:
            open_done = start
        if z is None:
            return int(open_done)

        z = np.asarray(z[:total_steps], dtype=np.float64)
        z_window = max(1, self.z_window)
        search_stop = max(0, min(total_steps - z_window, len(z) - z_window))
        for t in range(max(0, open_done), search_stop):
            g_ok = gripper[t] >= self.open_done_threshold
            z_rise_ok = z[t + z_window] - z[t] > self.z_rise_threshold
            v_ok = joint_velocity[t] > self.joint_velocity_threshold
            if g_ok and z_rise_ok and v_ok:
                return int(t)
        return int(open_done)

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
        if self.mode == "transfer_then_dump":
            return [
                "Approach the small trashbin.",
                "Grasp the small trashbin.",
                "Move the small trashbin to a better pouring position.",
                "Place the small trashbin down.",
                "Approach the placed trashbin.",
                "Grasp the trashbin rim for pouring.",
                "Move and tilt the trashbin to pour the balls into the large bin.",
            ]

        return [
            "Approach the small trashbin.",
            "Grasp the rim of the small trashbin.",
            "Move and tilt the trashbin to pour the balls into the large bin.",
        ]
