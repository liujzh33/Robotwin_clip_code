"""Rule-based phase segmentation for the stack_blocks_three task.

Three pick-and-place cycles (red base, green middle, blue top). Active arm is
inferred per cycle from the next valid close-open gripper event. Color order is
fixed by task semantics; arm assignment is data-driven.
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


BLOCK_COLORS = ("red", "green", "blue")

PHASE_DESCRIPTIONS = [
    "Approach the red block.",
    "Grasp the red block.",
    "Move the red block to the center.",
    "Place the red block at the center.",
    "Approach the green block.",
    "Grasp the green block.",
    "Move the green block above the red block.",
    "Place the green block on the red block.",
    "Approach the blue block.",
    "Grasp the blue block.",
    "Move the blue block above the green block.",
    "Place the blue block on the green block.",
]


class StackBlocksThreeProcessor(BaseTaskProcessor):
    """Twelve-phase splitter for three-block vertical stacking."""

    NUM_CYCLES = 3

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.9,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        min_close_drop: float = 0.15,
        lookback: int = 15,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_change_threshold: float = 0.005,
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
        self.min_close_drop = min_close_drop
        self.lookback = lookback
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_change_threshold = z_change_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.cycle_arms: list[str] = []
        self.block_colors: list[str] = list(BLOCK_COLORS)

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

        checkpoints: list[int] = []
        cycle_arms: list[str] = []
        cycle_start = 0

        for block_idx in range(self.NUM_CYCLES):
            event = self._find_next_valid_grasp_event(arm_data, cycle_start, total_steps)
            if event is None:
                break

            cycle_arms.append(event.arm)
            active = arm_data[event.arm]
            gripper = active["gripper"]
            active_z = active["z"]
            joint_vel = active["joint_vel"]
            eef_vel = active["eef_vel"]

            a_i = self._find_approach_end_near_close_start(
                gripper=gripper,
                z=active_z,
                joint_velocity=joint_vel,
                start=cycle_start,
                close_start=event.close_start,
                total_steps=total_steps,
            )
            b_i = self._find_move_start_after_grasp(
                gripper=gripper,
                joint_velocity=joint_vel,
                eef_velocity=eef_vel,
                start=a_i,
                total_steps=total_steps,
            )
            c_i = self._find_place_start_by_gripper_opening(gripper, b_i, total_steps)
            cycle_cps = self._enforce_order([a_i, b_i, c_i], total_steps)
            checkpoints.extend(cycle_cps)

            if block_idx < self.NUM_CYCLES - 1:
                next_pickup_arm = self._detect_next_pickup_arm(arm_data, cycle_cps[-1], total_steps)
                next_data = arm_data[next_pickup_arm]
                d_i = self._find_next_cycle_start_after_release(
                    released_gripper=gripper,
                    next_joint_vel=next_data["joint_vel"],
                    next_eef_vel=next_data["eef_vel"],
                    next_z=next_data["z"],
                    start=cycle_cps[-1],
                    total_steps=total_steps,
                )
                d_i = int(min(total_steps - 1, max(cycle_cps[-1] + 1, d_i)))
                checkpoints.append(d_i)
                cycle_start = d_i

        self.cycle_arms = cycle_arms
        final = self.validate_checkpoints(self._enforce_order(checkpoints, total_steps), total_steps)
        expected = self.NUM_CYCLES * 4 - 1
        if len(final) > expected:
            final = final[:expected]
        return final

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

        pre = gripper[max(start, close_start - 5) : close_start + 1]
        post = gripper[close_start : min(total_steps, close_start + 30)]
        if len(post) == 0:
            return None
        open_ref = float(np.max(pre)) if len(pre) > 0 else float(gripper[close_start - 1])
        drop = open_ref - float(np.min(post))
        if drop < self.min_close_drop:
            alt = self._find_close_start(gripper, close_start + 1)
            if alt is None:
                return None
            close_start = alt

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

    def _detect_next_pickup_arm(
        self,
        arm_data: dict,
        start: int,
        total_steps: int,
    ) -> str:
        left_close = self._find_close_start(arm_data["left"]["gripper"], start)
        right_close = self._find_close_start(arm_data["right"]["gripper"], start)

        if left_close is None and right_close is None:
            return "left"
        if right_close is None:
            return "left"
        if left_close is None:
            return "right"
        return "left" if left_close <= right_close else "right"

    def _find_next_cycle_start_after_release(
        self,
        released_gripper: np.ndarray,
        next_joint_vel: np.ndarray,
        next_eef_vel: Optional[np.ndarray],
        next_z: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        window = max(1, self.z_window)
        search_start = max(0, start)

        for t in range(search_start, max(search_start, total_steps - window)):
            release_done = float(released_gripper[t]) >= self.open_done_threshold
            if not release_done:
                continue

            next_arm_move = float(next_joint_vel[t]) > self.joint_velocity_threshold
            if next_eef_vel is not None and len(next_eef_vel) > t:
                next_arm_move = next_arm_move or float(next_eef_vel[t]) > self.eef_velocity_threshold
            if next_z is not None and len(next_z) > t + window:
                next_arm_move = next_arm_move or (
                    abs(float(next_z[t + window] - next_z[t])) > self.z_change_threshold
                )

            if next_arm_move:
                return int(t)

        open_done = self._first_consecutive_geq(released_gripper, self.open_done_threshold, search_start)
        if open_done is not None:
            return int(open_done)
        return int(search_start)

    def _find_close_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        open_mask = np.isclose(gripper, self.gripper_open_value, atol=self.gripper_open_atol)
        for idx in range(max(1, start), len(gripper)):
            was_open = open_mask[idx - 1]
            starts_closing = gripper[idx] < gripper[idx - 1] - self.open_delta
            leaves_open = not open_mask[idx]
            if was_open and (starts_closing or leaves_open):
                return int(idx)
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
        segment = np.asarray(gripper[start:total_steps], dtype=np.float64)
        close_threshold = (
            min(self.close_threshold, float(np.min(segment) + 0.05)) if len(segment) > 0 else self.close_threshold
        )

        close_done = self._first_consecutive_leq(gripper, close_threshold, start)
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
        return list(PHASE_DESCRIPTIONS)
