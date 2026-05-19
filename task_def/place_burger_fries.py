"""Rule-based phase segmentation for the place_burger_fries task.

Dual-arm cooperative: grasp burger and fries, then place them onto the tray sequentially.
first_arm is the arm that opens its gripper first after grasping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class PlaceBurgerFriesProcessor(BaseTaskProcessor):
    """Six-phase splitter: dual approach/grasp, sequential place onto tray."""

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

        left_close_start = self._find_close_start(left_gripper, 0) or max(1, total_steps // 10)
        right_close_start = self._find_close_start(right_gripper, 0) or max(1, total_steps // 10)

        left_c0 = self._find_approach_end_near_close_start(
            left_gripper, arm_data["left"]["z"], arm_data["left"]["joint_vel"], 0, left_close_start, total_steps
        )
        right_c0 = self._find_approach_end_near_close_start(
            right_gripper, arm_data["right"]["z"], arm_data["right"]["joint_vel"], 0, right_close_start, total_steps
        )
        c0 = max(left_c0, right_c0)

        left_close_done = self._first_consecutive_leq(left_gripper, self.close_threshold, c0) or c0
        right_close_done = self._first_consecutive_leq(right_gripper, self.close_threshold, c0) or c0

        left_move_start = self._find_move_start_after_grasp(
            left_gripper, arm_data["left"]["joint_vel"], arm_data["left"]["eef_vel"], left_close_done, total_steps
        )
        right_move_start = self._find_move_start_after_grasp(
            right_gripper, arm_data["right"]["joint_vel"], arm_data["right"]["eef_vel"], right_close_done, total_steps
        )

        search_after_grasp = min(left_move_start, right_move_start)
        left_open_start = self._find_open_start(left_gripper, search_after_grasp)
        right_open_start = self._find_open_start(right_gripper, search_after_grasp)

        if left_open_start is None and right_open_start is None:
            if left_move_start <= right_move_start:
                first_arm, second_arm = "left", "right"
            else:
                first_arm, second_arm = "right", "left"
        elif left_open_start is None:
            first_arm, second_arm = "right", "left"
        elif right_open_start is None:
            first_arm, second_arm = "left", "right"
        elif left_open_start <= right_open_start:
            first_arm, second_arm = "left", "right"
        else:
            first_arm, second_arm = "right", "left"

        self.first_arm = first_arm
        self.second_arm = second_arm

        first_data = arm_data[first_arm]
        second_data = arm_data[second_arm]
        first_move_start = left_move_start if first_arm == "left" else right_move_start
        second_close_done = right_close_done if second_arm == "right" else left_close_done

        c1 = max(first_move_start, second_close_done)

        c2 = self._find_place_start_by_gripper_opening(first_data["gripper"], c1, total_steps)

        second_move_start = (
            right_move_start if second_arm == "right" else left_move_start
        )
        c3 = self._find_second_move_start_after_first_release(
            first_gripper=first_data["gripper"],
            second_gripper=second_data["gripper"],
            second_joint_vel=second_data["joint_vel"],
            second_eef_vel=second_data["eef_vel"],
            release_start=c2,
            fallback_move_start=second_move_start,
            total_steps=total_steps,
        )

        c4_open = self._find_open_start(second_data["gripper"], c3)
        c4 = int(c4_open) if c4_open is not None else int(c3)

        checkpoints = self._enforce_order([c0, c1, c2, c3, c4], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def find_place_done(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        total_steps: int,
    ) -> Optional[int]:
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

    def _find_second_move_start_after_first_release(
        self,
        first_gripper: np.ndarray,
        second_gripper: np.ndarray,
        second_joint_vel: np.ndarray,
        second_eef_vel: Optional[np.ndarray],
        release_start: int,
        fallback_move_start: int,
        total_steps: int,
    ) -> int:
        for idx in range(release_start, total_steps):
            first_open = first_gripper[idx] >= self.open_done_threshold
            second_closed = second_gripper[idx] <= self.close_threshold
            second_moving = second_joint_vel[idx] > self.joint_velocity_threshold
            second_eef_moving = (
                second_eef_vel is not None and second_eef_vel[idx] > self.eef_velocity_threshold
            )
            if first_open and second_closed and (second_moving or second_eef_moving):
                return int(idx)
        return int(max(release_start, fallback_move_start))

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
            "Approach the burger and fries with both arms.",
            "Grasp the burger and fries with both grippers.",
            "Move the first item above the tray.",
            "Release the first item onto the tray.",
            "Move the second item above the tray.",
            "Release the second item onto the tray.",
        ]
