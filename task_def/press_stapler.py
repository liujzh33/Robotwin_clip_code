"""Rule-based phase segmentation for the press_stapler task.

Three phases: approach pressing area, close gripper, press downward.
Active arm is inferred from gripper close plus subsequent EEF Z drop (no open event).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class PressStaplerProcessor(BaseTaskProcessor):
    """Three-phase splitter: approach, close gripper, press stapler down."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        lookback: int = 15,
        z_eps: float = 0.005,
        z_press_threshold: float = 0.005,
        z_window: int = 5,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        motion_weight: float = 0.1,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.lookback = lookback
        self.z_eps = z_eps
        self.z_press_threshold = z_press_threshold
        self.z_window = z_window
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.motion_weight = motion_weight
        self.active_arm: Optional[str] = None
        self.active_side: Optional[str] = None

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

        if active_side is not None:
            active_arm = active_side
            close_start = self._find_close_start(arm_data[active_arm]["gripper"], 0) or max(1, total_steps // 6)
        else:
            active_arm, close_start = self._find_active_arm_by_close_and_press(arm_data, total_steps)

        self.active_arm = active_arm
        self.active_side = active_arm
        data = arm_data[active_arm]

        c0 = self._find_approach_end_near_close_start(
            gripper=data["gripper"],
            z=data["z"],
            joint_velocity=data["joint_vel"],
            start=0,
            close_start=close_start,
            total_steps=total_steps,
        )
        c1 = self._find_press_start_after_gripper_close(
            gripper=data["gripper"],
            z=data["z"],
            joint_velocity=data["joint_vel"],
            eef_velocity=data["eef_vel"],
            start=c0,
            total_steps=total_steps,
        )
        if c1 <= c0:
            c1 = min(total_steps - 1, c0 + max(1, (total_steps - c0) // 3))

        return self.validate_checkpoints([c0, c1], total_steps)

    def _find_active_arm_by_close_and_press(
        self,
        arm_data: dict,
        total_steps: int,
    ) -> tuple[str, int]:
        best_arm = "right"
        best_close_start = max(1, total_steps // 6)
        best_score = -1.0

        for arm in ("left", "right"):
            gripper = arm_data[arm]["gripper"]
            z = arm_data[arm]["z"]
            joint_vel = arm_data[arm]["joint_vel"]

            close_start = self._find_close_start(gripper, 0)
            if close_start is None:
                continue

            close_done = self._first_leq(gripper, self.close_threshold, close_start)
            if close_done is None:
                close_done = int(close_start + np.argmin(gripper[close_start:]))

            gripper_drop = float(np.max(gripper) - np.min(gripper[close_start : close_done + 1]))
            press_frame, press_drop = self._find_z_press_after(gripper, z, close_done, total_steps)
            motion = float(np.sum(joint_vel[close_done:press_frame + 1])) if press_frame >= close_done else 0.0

            score = gripper_drop
            if press_drop > 0:
                score += press_drop + 0.5
            score += self.motion_weight * motion

            if score > best_score:
                best_score = score
                best_arm = arm
                best_close_start = int(close_start)

        return best_arm, best_close_start

    def _find_z_press_after(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        close_done: int,
        total_steps: int,
    ) -> tuple[int, float]:
        if z is None or len(z) == 0:
            return close_done, 0.0

        z = np.asarray(z[:total_steps], dtype=np.float64)
        window = max(1, self.z_window)
        for t in range(close_done, max(close_done, total_steps - window)):
            if z[t + window] - z[t] < -self.z_press_threshold:
                return int(t), float(z[t] - z[t + window])
        return close_done, 0.0

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

    def _find_press_start_after_gripper_close(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        close_done = self._first_leq(gripper, self.close_threshold, start)
        if close_done is None:
            segment = gripper[start:]
            close_done = start + int(np.argmin(segment)) if len(segment) > 0 else start

        window = max(1, self.z_window)
        for t in range(close_done, max(close_done, total_steps - window)):
            z_press_ok = False
            if z is not None and len(z) > t + window:
                z_press_ok = float(z[t + window] - z[t]) < -self.z_press_threshold

            joint_ok = float(joint_velocity[t]) > self.joint_velocity_threshold
            eef_ok = (
                eef_velocity is not None
                and len(eef_velocity) > t
                and float(eef_velocity[t]) > self.eef_velocity_threshold
            )
            if z_press_ok or joint_ok or eef_ok:
                return int(t)

        return int(close_done)

    def _find_close_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        open_mask = np.isclose(gripper, self.gripper_open_value, atol=self.gripper_open_atol)
        for idx in range(max(1, start), len(gripper)):
            was_open = open_mask[idx - 1]
            starts_closing = gripper[idx] < gripper[idx - 1] - self.open_delta
            leaves_open = not open_mask[idx]
            if was_open and (starts_closing or leaves_open):
                return int(idx)
        return None

    def _first_leq(self, series: np.ndarray, threshold: float, start: int) -> Optional[int]:
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

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the stapler pressing area.",
            "Close the gripper.",
            "Press the stapler downward.",
        ]
