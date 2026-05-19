"""Rule-based phase segmentation for the open_laptop task.

Approach the lid gap, grasp the lid, then lift/rotate to open. Active arm is inferred
from the first valid close event (full open is not required at episode end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


@dataclass
class CloseEvent:
    arm: str
    close_start: int
    close_done: int


class OpenLaptopProcessor(BaseTaskProcessor):
    """Three-phase splitter: approach gap, grasp lid, open lid upward."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        close_margin: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        min_drop_frames: int = 3,
        lookback: int = 15,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_lift_threshold: float = 0.005,
        z_window: int = 5,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.close_margin = close_margin
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.min_drop_frames = min_drop_frames
        self.lookback = lookback
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_lift_threshold = z_lift_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
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

        event = self._find_next_valid_close_event(arm_data, start=0, total_steps=total_steps)
        if event is None:
            fallback_arm = active_side or self.analyzer.active_side_from_grippers(left_gripper, right_gripper)
            close_start = self._find_close_start(arm_data[fallback_arm]["gripper"], 0) or max(1, total_steps // 6)
            event = CloseEvent(
                arm=fallback_arm,
                close_start=int(close_start),
                close_done=int(close_start),
            )

        self.active_arm = event.arm
        data = arm_data[event.arm]
        gripper = data["gripper"]
        active_z = data["z"]
        joint_vel = data["joint_vel"]
        eef_vel = data["eef_vel"]

        c0 = self._find_approach_end_near_close_start(
            gripper=gripper,
            z=active_z,
            joint_velocity=joint_vel,
            start=0,
            close_start=event.close_start,
            total_steps=total_steps,
        )
        c1 = self._find_open_laptop_start_after_grasp(
            gripper=gripper,
            z=active_z,
            joint_velocity=joint_vel,
            eef_velocity=eef_vel,
            start=c0,
            total_steps=total_steps,
        )

        checkpoints = self._enforce_order([c0, c1], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_next_valid_close_event(
        self,
        arm_data: dict,
        start: int,
        total_steps: int,
    ) -> Optional[CloseEvent]:
        events = []
        for arm in ("left", "right"):
            gripper = arm_data[arm]["gripper"]
            close_start = self._find_close_start(gripper, start)
            if close_start is None:
                continue
            segment = gripper[close_start:total_steps]
            if len(segment) == 0:
                continue
            close_candidates = np.where(segment <= self.close_threshold)[0]
            if len(close_candidates) == 0:
                close_done = int(close_start)
            else:
                close_done = int(close_start + close_candidates[0])
            events.append(CloseEvent(arm=arm, close_start=int(close_start), close_done=close_done))
        if not events:
            return None
        return min(events, key=lambda event: event.close_start)

    def _find_close_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        min_drop_frames = max(1, self.min_drop_frames)
        open_mask = np.isclose(gripper, self.gripper_open_value, atol=self.gripper_open_atol)
        for idx in range(max(1, start), len(gripper) - min_drop_frames):
            was_open = open_mask[idx - 1]
            starts_closing = gripper[idx] < gripper[idx - 1] - self.open_delta
            leaves_open = not open_mask[idx]
            continues_closing = gripper[idx + min_drop_frames] < gripper[idx] - self.open_delta
            if was_open and (starts_closing or leaves_open) and continues_closing:
                return int(idx)
            if was_open and (starts_closing or leaves_open):
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

    def _find_open_laptop_start_after_grasp(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        segment = np.asarray(gripper[start:total_steps], dtype=np.float64)
        adaptive_threshold = float(np.min(segment) + self.close_margin) if len(segment) > 0 else self.close_threshold
        close_threshold = min(self.close_threshold, adaptive_threshold)

        close_candidates = np.where(gripper[start:] <= close_threshold)[0]
        close_done = start + int(close_candidates[0]) if len(close_candidates) > 0 else start

        z_window = max(1, self.z_window)
        for t in range(close_done, max(close_done, total_steps - z_window)):
            z_lift_ok = z is not None and float(z[t + z_window] - z[t]) > self.z_lift_threshold
            joint_ok = joint_velocity[t] > self.joint_velocity_threshold
            eef_ok = eef_velocity is not None and eef_velocity[t] > self.eef_velocity_threshold
            if z_lift_ok or joint_ok or eef_ok:
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
            "Approach the laptop lid gap.",
            "Grasp the laptop lid.",
            "Open the laptop lid upward.",
        ]
