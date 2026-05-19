"""Rule-based phase segmentation for the blocks_ranking_rgb task.

Object order is fixed:
red block -> green block -> blue block.

The active arm is not fixed. For each object, the processor finds the next
valid close-open gripper event across both arms and uses that arm's signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


@dataclass
class GraspEvent:
    arm: str
    close_start: int
    close_done: int
    open_start: int
    open_done: int


class BlocksRankingRgbProcessor(BaseTaskProcessor):
    """Thirteen-phase splitter for red, green, blue block ranking."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.9,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_rise_threshold: float = 0.005,
        z_window: int = 5,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        min_event_gap: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.open_done_threshold = open_done_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_rise_threshold = z_rise_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.min_event_gap = min_event_gap
        self.current_arm_sequence: list[str] = []
        self.colors = ["red", "green", "blue"]

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

        left_xyz, right_xyz = self.analyzer.extract_left_right_eef_xyz(
            hdf5_data,
            total_steps=total_steps,
            external_eef_xyz_left=external_eef_xyz_left,
            external_eef_xyz_right=external_eef_xyz_right,
        )

        qpos = self.analyzer.extract_qpos(hdf5_data)[:total_steps]
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

        checkpoints: list[int] = []
        self.current_arm_sequence = []
        search_start = 0

        for _color in self.colors:
            event = self._find_next_valid_grasp_event(arm_data, search_start, total_steps)
            if event is None:
                break

            data = arm_data[event.arm]
            self.current_arm_sequence.append(event.arm)

            c_a = self._find_approach_end_before_close(
                gripper=data["gripper"],
                z=data["z"],
                joint_velocity=data["joint_vel"],
                start=search_start,
                close_start=event.close_start,
            )
            c_b = self._find_move_start_after_grasp(
                joint_velocity=data["joint_vel"],
                eef_velocity=data["eef_vel"],
                start=event.close_done,
                fallback=event.close_done,
            )
            c_c = event.open_start
            c_d = self._find_place_end_after_opening(
                gripper=data["gripper"],
                z=data["z"],
                joint_velocity=data["joint_vel"],
                start=c_c,
                total_steps=total_steps,
                fallback=event.open_done,
            )

            ordered = self._enforce_order([c_a, c_b, c_c, c_d], total_steps)
            checkpoints.extend(ordered)
            search_start = min(total_steps - 1, ordered[-1] + self.min_event_gap)

        if len(checkpoints) < 12:
            checkpoints.extend(self._fallback_missing_checkpoints(checkpoints, total_steps))

        return self.validate_checkpoints(checkpoints[:12], total_steps)

    def _find_next_valid_grasp_event(
        self,
        arm_data: dict,
        start: int,
        total_steps: int,
    ) -> Optional[GraspEvent]:
        events = []
        for arm in ("left", "right"):
            event = self._find_arm_grasp_event(
                arm=arm,
                gripper=arm_data[arm]["gripper"],
                start=start,
                total_steps=total_steps,
            )
            if event is not None:
                events.append(event)
        if not events:
            return None
        return min(events, key=lambda event: event.close_start)

    def _find_arm_grasp_event(
        self,
        arm: str,
        gripper: np.ndarray,
        start: int,
        total_steps: int,
    ) -> Optional[GraspEvent]:
        close_start = self._find_close_start(gripper, start)
        if close_start is None:
            return None

        close_done = self._first_consecutive_leq(gripper, self.close_threshold, close_start)
        if close_done is None:
            return None

        open_start = self._find_open_start(gripper, close_done)
        if open_start is None:
            return None

        open_done = self._first_consecutive_geq(gripper, self.open_done_threshold, open_start)
        if open_done is None:
            open_done = open_start

        if not (start <= close_start <= close_done <= open_start <= open_done < total_steps):
            return None

        return GraspEvent(
            arm=arm,
            close_start=int(close_start),
            close_done=int(close_done),
            open_start=int(open_start),
            open_done=int(open_done),
        )

    def _find_close_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        open_mask = np.isclose(gripper, self.gripper_open_value, atol=self.gripper_open_atol)
        for idx in range(max(1, start), len(gripper)):
            left_open = open_mask[idx - 1]
            starts_closing = gripper[idx] < gripper[idx - 1] - self.open_delta
            leaves_open = not open_mask[idx]
            if left_open and (starts_closing or leaves_open):
                return int(idx)
        return None

    def _find_open_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        for idx in range(max(1, start), len(gripper)):
            if gripper[idx] > self.close_threshold and gripper[idx] - gripper[idx - 1] > self.open_delta:
                return int(idx)
        return None

    def _find_approach_end_before_close(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        close_start: int,
    ) -> int:
        window = max(1, self.stable_window)
        search_start = max(0, start)
        search_end = max(close_start, search_start + window + 1)
        search_end = min(len(gripper), search_end)

        if z is None or search_end <= search_start:
            return max(1, min(len(gripper) - 1, close_start - 1))

        z_segment = np.asarray(z[search_start:search_end], dtype=np.float64)
        min_z = float(np.min(z_segment))
        open_mask = np.isclose(
            gripper,
            self.gripper_open_value,
            atol=self.gripper_open_atol,
        )

        for idx in range(search_start, max(search_start + 1, search_end - window + 1)):
            end = idx + window
            g_ok = np.all(open_mask[idx:end])
            z_ok = np.all(z[idx:end] <= min_z + self.z_eps)
            v_ok = np.all(joint_velocity[idx:end] <= self.stable_velocity_threshold)
            if g_ok and z_ok and v_ok:
                return int(idx)

        return int(search_start + np.argmin(z_segment))

    def _find_move_start_after_grasp(
        self,
        joint_velocity: np.ndarray,
        eef_velocity: Optional[np.ndarray],
        start: int,
        fallback: int,
    ) -> int:
        if eef_velocity is not None:
            eef_hits = np.where(eef_velocity[max(0, start) :] > self.eef_velocity_threshold)[0]
            if len(eef_hits) > 0:
                return int(max(0, start) + eef_hits[0])

        joint_hits = np.where(joint_velocity[max(0, start) :] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(max(0, start) + joint_hits[0])

        return int(fallback)

    def _find_place_end_after_opening(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        total_steps: int,
        fallback: int,
    ) -> int:
        """
        Place ends when the active arm starts leaving after release.

        The gripper must be open again, then EEF Z should rise and joint velocity
        should increase. This keeps the lift-away motion inside the Place phase.
        """
        open_done = self._first_consecutive_geq(gripper, self.open_done_threshold, start)
        if open_done is None:
            open_done = int(fallback)

        if z is None:
            return int(open_done)

        z = np.asarray(z[:total_steps], dtype=np.float64)
        z_window = max(1, self.z_window)
        velocity_window = max(1, self.stable_window)
        search_stop = max(0, min(total_steps - z_window, len(z) - z_window))

        for idx in range(max(0, open_done), search_stop):
            g_ok = gripper[idx] >= self.open_done_threshold
            z_rise_ok = z[idx + z_window] - z[idx] > self.z_rise_threshold
            v_end = min(len(joint_velocity), idx + velocity_window)
            v_ok = np.any(joint_velocity[idx:v_end] > self.joint_velocity_threshold)
            if g_ok and z_rise_ok and v_ok:
                return int(idx)

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

    def _fallback_missing_checkpoints(self, checkpoints: list[int], total_steps: int) -> list[int]:
        missing = 12 - len(checkpoints)
        if missing <= 0:
            return []
        start = checkpoints[-1] if checkpoints else 0
        remaining = max(1, total_steps - start - 1)
        return [min(total_steps - 1, start + (i + 1) * remaining // (missing + 1)) for i in range(missing)]

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the red block.",
            "Grasp the red block.",
            "Move the red block to its target position.",
            "Place the red block.",
            "Approach the green block.",
            "Grasp the green block.",
            "Move the green block to its target position.",
            "Place the green block.",
            "Approach the blue block.",
            "Grasp the blue block.",
            "Move the blue block to its target position.",
            "Place the blue block.",
            "Return to the initial position.",
        ]
