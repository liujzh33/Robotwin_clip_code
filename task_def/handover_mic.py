"""Rule-based phase segmentation for the handover_mic task.

Two-arm handover: first arm approaches, grasps, moves mic to center; second arm
approaches, grasps; first arm releases after second arm grasps.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class HandoverMicProcessor(BaseTaskProcessor):
    """Six-phase splitter for microphone handover between two arms."""

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
        search_gap: int = 5,
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
        self.first_arm: Optional[str] = None
        self.second_arm: Optional[str] = None

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

        arm_data = {
            "left": {
                "gripper": left_gripper,
                "z": left_z,
                "joint_vel": left_joint_vel,
                "eef_vel": left_eef_vel,
            },
            "right": {
                "gripper": right_gripper,
                "z": right_z,
                "joint_vel": right_joint_vel,
                "eef_vel": right_eef_vel,
            },
        }

        first_arm, first_close_start = self._find_next_active_arm_by_close(
            left_gripper, right_gripper, start=0, total_steps=total_steps
        )
        if first_arm is None:
            return self.validate_checkpoints([total_steps // 5, 2 * total_steps // 5], total_steps)

        self.first_arm = first_arm
        first_data = arm_data[first_arm]

        c0 = self._find_approach_end_near_close_start(
            gripper=first_data["gripper"],
            z=first_data["z"],
            joint_velocity=first_data["joint_vel"],
            start=0,
            close_start=first_close_start,
            total_steps=total_steps,
        )
        c1 = self._find_move_start_after_grasp(
            gripper=first_data["gripper"],
            joint_velocity=first_data["joint_vel"],
            eef_velocity=first_data["eef_vel"],
            start=c0,
            total_steps=total_steps,
        )

        candidate_second = "right" if first_arm == "left" else "left"
        second_data = arm_data[candidate_second]

        c2 = self._find_other_arm_approach_start(
            z=second_data["z"],
            joint_velocity=second_data["joint_vel"],
            eef_velocity=second_data["eef_vel"],
            start=c1,
            total_steps=total_steps,
        )

        second_close_start = self._find_close_start(
            second_data["gripper"], start=c2, total_steps=total_steps
        )
        if second_close_start is None:
            second_close_start = min(total_steps - 1, c2 + max(1, (total_steps - c2) // 4))

        self.second_arm = candidate_second
        c3 = self._find_approach_end_near_close_start(
            gripper=second_data["gripper"],
            z=second_data["z"],
            joint_velocity=second_data["joint_vel"],
            start=c2,
            close_start=second_close_start,
            total_steps=total_steps,
        )

        second_close_done = self._first_consecutive_leq(
            second_data["gripper"], self.close_threshold, c3
        )
        if second_close_done is None:
            second_close_done = c3

        c4 = self._find_open_start(
            gripper=first_data["gripper"],
            start=second_close_done,
            total_steps=total_steps,
        )
        if c4 is None:
            open_hits = np.where(first_data["gripper"][second_close_done:] > self.close_threshold)[0]
            c4 = int(second_close_done + open_hits[0]) if len(open_hits) > 0 else int(min(total_steps - 1, c3 + self.search_gap))

        checkpoints = self._enforce_order([c0, c1, c2, c3, c4], total_steps)
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

    def _find_other_arm_approach_start(
        self,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        z_window = max(1, self.z_window)
        stable = max(1, self.stable_window)
        for t in range(max(0, start), max(start, total_steps - z_window)):
            joint_ok = np.any(joint_velocity[t : t + stable] > self.joint_velocity_threshold)
            eef_ok = eef_velocity is not None and np.any(eef_velocity[t : t + stable] > self.eef_velocity_threshold)
            z_ok = False
            if z is not None:
                z_ok = abs(float(z[t + z_window] - z[t])) > self.z_change_threshold
            if (joint_ok or eef_ok) and (z_ok or eef_ok):
                return int(t)
        return int(start)

    def _find_open_start(self, gripper: np.ndarray, start: int, total_steps: int) -> Optional[int]:
        for t in range(max(1, start), total_steps):
            if gripper[t - 1] <= self.close_threshold and gripper[t] > gripper[t - 1] + self.open_delta:
                return int(t)
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
            "Approach the microphone with the first arm.",
            "Grasp the microphone with the first gripper.",
            "Move the microphone to the center handover position.",
            "Approach the microphone with the receiving arm.",
            "Grasp the microphone with the receiving gripper.",
            "Release the microphone from the first gripper.",
        ]