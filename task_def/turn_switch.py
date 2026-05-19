"""Rule-based phase segmentation for the turn_switch task.

Three phases: close gripper, move in front of switch, press forward.
Active arm is inferred from gripper close plus subsequent move and press motion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class TurnSwitchProcessor(BaseTaskProcessor):
    """Three-phase splitter for close-then-move-then-forward-press."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        velocity_zero_threshold: float = 0.003,
        moving_window: int = 3,
        stop_window: int = 3,
        min_move_frames: int = 3,
        motion_weight: float = 0.1,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.velocity_zero_threshold = velocity_zero_threshold
        self.moving_window = moving_window
        self.stop_window = stop_window
        self.min_move_frames = min_move_frames
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

        if active_side is not None:
            active_arm = active_side
        else:
            active_arm = self._find_active_arm_by_gripper_close(arm_data, total_steps)

        self.active_arm = active_arm
        self.active_side = active_arm
        data = arm_data[active_arm]

        c0 = self._find_move_start_after_gripper_close(
            gripper=data["gripper"],
            joint_vel=data["joint_vel"],
            eef_vel=data["eef_vel"],
            start=0,
            total_steps=total_steps,
        )
        c1 = self._find_press_start_after_move_stop(
            gripper=data["gripper"],
            joint_vel=data["joint_vel"],
            start=c0,
            total_steps=total_steps,
        )

        if c1 <= c0:
            c1 = min(total_steps - 1, c0 + max(1, (total_steps - c0) // 3))

        checkpoints = self._enforce_order([c0, c1], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_active_arm_by_gripper_close(self, arm_data: dict, total_steps: int) -> str:
        best_arm = "left"
        best_score = -1.0

        for arm in ("left", "right"):
            gripper = arm_data[arm]["gripper"]
            joint_vel = arm_data[arm]["joint_vel"]
            xyz = arm_data[arm]["xyz"]

            close_start = self._find_close_start(gripper, 0)
            if close_start is None:
                continue

            segment = np.asarray(gripper[close_start:], dtype=np.float64)
            close_threshold = (
                min(self.close_threshold, float(np.min(segment) + 0.05))
                if len(segment) > 0
                else self.close_threshold
            )
            close_done = self._first_leq(gripper, close_threshold, close_start)
            if close_done is None:
                close_done = close_start + int(np.argmin(gripper[close_start:]))

            gripper_drop = float(np.max(gripper) - np.min(gripper[close_start : close_done + 1]))
            move_hits = np.where(joint_vel[close_done:total_steps] > self.joint_velocity_threshold)[0]
            motion = float(np.sum(joint_vel[close_done : close_done + 40])) if len(move_hits) > 0 else 0.0

            forward_motion = 0.0
            if xyz is not None and close_done < total_steps - 5:
                tail = np.asarray(xyz[close_done:total_steps, :2], dtype=np.float64)
                if len(tail) > 1:
                    forward_motion = float(np.max(np.linalg.norm(np.diff(tail, axis=0), axis=1)))

            score = gripper_drop + self.motion_weight * motion + 0.05 * forward_motion
            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def _find_move_start_after_gripper_close(
        self,
        gripper: np.ndarray,
        joint_vel: np.ndarray,
        eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        segment = np.asarray(gripper[start:total_steps], dtype=np.float64)
        close_threshold = (
            min(self.close_threshold, float(np.min(segment) + 0.05)) if len(segment) > 0 else self.close_threshold
        )

        close_candidates = np.where(gripper[start:total_steps] <= close_threshold)[0]
        if len(close_candidates) > 0:
            close_done = start + int(close_candidates[0])
        else:
            close_done = start + int(np.argmin(gripper[start:]))

        joint_vel_arr = self._joint_vel_norm(joint_vel, total_steps)
        joint_hits = np.where(joint_vel_arr[close_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(close_done + joint_hits[0])

        if eef_vel is not None:
            eef_hits = np.where(eef_vel[close_done:total_steps] > self.eef_velocity_threshold)[0]
            if len(eef_hits) > 0:
                return int(close_done + eef_hits[0])

        return int(close_done)

    def _find_press_start_after_move_stop(
        self,
        gripper: np.ndarray,
        joint_vel: np.ndarray,
        start: int,
        total_steps: int,
    ) -> int:
        """Move ends when gripper stays closed and joint velocity enters a stable near-zero segment."""
        segment = np.asarray(gripper[start:total_steps], dtype=np.float64)
        close_threshold = (
            min(self.close_threshold, float(np.min(segment) + 0.05)) if len(segment) > 0 else self.close_threshold
        )

        joint_vel = self._joint_vel_norm(joint_vel, total_steps)
        moving_window = max(1, self.moving_window)
        stop_window = max(1, self.stop_window)
        search_start = start + max(1, self.min_move_frames)

        for t in range(search_start, total_steps - stop_window):
            gripper_closed = gripper[t] <= close_threshold
            was_moving = float(np.max(joint_vel[max(start, t - moving_window) : t])) > self.joint_velocity_threshold
            now_stopped = float(np.max(joint_vel[t : t + stop_window])) <= self.velocity_zero_threshold

            if gripper_closed and was_moving and now_stopped:
                return int(t)

        return int(start)

    @staticmethod
    def _joint_vel_norm(joint_vel: np.ndarray, total_steps: int) -> np.ndarray:
        joint_vel = np.asarray(joint_vel[:total_steps], dtype=np.float64)
        if joint_vel.ndim > 1:
            return np.linalg.norm(joint_vel, axis=-1)
        return joint_vel

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
            "Close the gripper.",
            "Move in front of the switch.",
            "Press the switch forward.",
        ]
