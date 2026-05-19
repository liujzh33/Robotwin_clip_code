"""Rule-based phase segmentation for the place_empty_cup task.

Single-object pick-and-place with gripper pre-close: the active gripper may drop from
1.0 to ~0.6 before the formal grasp. c0 uses the second close-down (semi-closed platform
to full close), not the first gripper change.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class PlaceEmptyCupProcessor(BaseTaskProcessor):
    """Four-phase splitter: approach, grasp, move above coaster, release."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.9,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        semi_close_low: float = 0.4,
        semi_close_high: float = 0.8,
        drop_delta: float = 0.02,
        min_drop_frames: int = 3,
        search_window: int = 30,
        pre_offset: int = 2,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_rise_threshold: float = 0.005,
        z_window: int = 5,
        stable_window: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.open_done_threshold = open_done_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.semi_close_low = semi_close_low
        self.semi_close_high = semi_close_high
        self.drop_delta = drop_delta
        self.min_drop_frames = min_drop_frames
        self.search_window = search_window
        self.pre_offset = pre_offset
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_rise_threshold = z_rise_threshold
        self.z_window = z_window
        self.stable_window = stable_window
        self.active_arm: Optional[str] = None

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

        if active_side in ("left", "right"):
            active_arm = active_side
        else:
            left_score = float(np.max(left_gripper) - np.min(left_gripper))
            right_score = float(np.max(right_gripper) - np.min(right_gripper))
            active_arm = "right" if right_score >= left_score else "left"

        self.active_arm = active_arm
        data = arm_data[active_arm]
        gripper = data["gripper"]
        joint_vel = data["joint_vel"]
        eef_vel = data["eef_vel"]

        c0 = self._find_second_close_start_for_place_empty_cup(
            gripper=gripper,
            start=0,
            total_steps=total_steps,
        )
        c1 = self._find_move_start_after_grasp(
            gripper=gripper,
            joint_velocity=joint_vel,
            eef_velocity=eef_vel,
            start=c0,
            total_steps=total_steps,
        )
        c2 = self._find_place_start_by_gripper_opening(
            gripper=gripper,
            start=c1,
            total_steps=total_steps,
        )

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
        """Optional visualization checkpoint: release complete and arm retreats."""
        open_done = self._first_consecutive_geq(gripper, self.open_done_threshold, start)
        if open_done is None:
            return None

        z_window = max(1, self.z_window)
        velocity_window = max(1, self.stable_window)
        for idx in range(open_done, max(open_done, total_steps - z_window)):
            g_ok = gripper[idx] >= self.open_done_threshold
            z_rise_ok = z is not None and float(z[idx + z_window] - z[idx]) > self.z_rise_threshold
            v_end = min(len(joint_velocity), idx + velocity_window)
            v_ok = np.any(joint_velocity[idx:v_end] > self.joint_velocity_threshold)
            if g_ok and z_rise_ok and v_ok:
                return int(idx)
        return open_done

    def _find_second_close_start_for_place_empty_cup(
        self,
        gripper: np.ndarray,
        start: int,
        total_steps: int,
    ) -> int:
        """
        Ignore initial 1.0 -> ~0.6 pre-close; find when gripper continues from the
        semi-closed platform down toward full close.
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

    def _find_open_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        for idx in range(max(1, start), len(gripper)):
            if gripper[idx - 1] <= self.close_threshold and gripper[idx] > gripper[idx - 1] + self.open_delta:
                return int(idx)
        return None

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
            "Approach the cup.",
            "Grasp the cup.",
            "Move the cup above the coaster.",
            "Place the cup on the coaster.",
        ]
