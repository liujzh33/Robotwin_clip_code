"""Rule-based dynamic phase segmentation for put_bottles_dustbin.

Three bottle cycles; each cycle is either 4 phases (left pickup) or 7 phases
(right pickup with center handover to left arm). Between bottles, an extra
checkpoint splits release end from the next approach start.

Total phases (with inter-cycle boundaries): 12 / 17 / 20 / 23.
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


PHASE_KINDS_LEFT = ["approach", "grasp", "move_dustbin", "release"]
PHASE_KINDS_RIGHT = [
    "approach_pickup",
    "grasp_pickup",
    "move_center",
    "approach_center",
    "grasp_bin",
    "move_dustbin",
    "release",
]

PHASE_DESCRIPTIONS_LEFT = [
    "Approach the bottle.",
    "Grasp the bottle.",
    "Move the bottle above the left-side dustbin.",
    "Release the bottle into the dustbin.",
]

PHASE_DESCRIPTIONS_RIGHT = [
    "Approach the bottle with the pickup arm.",
    "Grasp the bottle with the pickup arm.",
    "Move the bottle to the center handover position.",
    "Approach the centered bottle with the bin-side arm.",
    "Grasp the bottle with the bin-side arm.",
    "Move the bottle above the left-side dustbin.",
    "Release the bottle into the dustbin.",
]


class PutBottlesDustbinProcessor(BaseTaskProcessor):
    """Three dynamic bottle cycles with left-direct or right-handover paths."""

    NUM_BOTTLES = 3

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
        cycle_gap: int = 5,
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
        self.cycle_gap = cycle_gap
        self.cycle_modes: list[str] = []
        self.phase_kinds: list[str] = []
        self._current_descriptions: list[str] = []

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
        descriptions: list[str] = []
        cycle_modes: list[str] = []
        phase_kinds: list[str] = []
        cycle_start = 0

        for bottle_idx in range(self.NUM_BOTTLES):
            pickup_arm, pickup_event = self._find_pickup_arm_at_cycle_start(
                arm_data, cycle_start, total_steps
            )
            cycle_modes.append(pickup_arm)

            if pickup_arm == "left":
                cps, kinds = self._left_pick_cycle(
                    arm_data["left"], pickup_event.close_start, cycle_start, total_steps
                )
                descriptions.extend(PHASE_DESCRIPTIONS_LEFT)
            else:
                cps, kinds = self._right_pick_cycle(
                    arm_data, pickup_event.close_start, cycle_start, total_steps
                )
                descriptions.extend(PHASE_DESCRIPTIONS_RIGHT)

            checkpoints.extend(cps)
            phase_kinds.extend(kinds)
            release_start = cps[-1]
            release_gripper = arm_data["left"]["gripper"]

            if bottle_idx < self.NUM_BOTTLES - 1:
                next_pickup_arm = self._detect_next_pickup_arm(
                    arm_data=arm_data,
                    start=release_start,
                    total_steps=total_steps,
                )
                next_arm = arm_data[next_pickup_arm]
                next_cycle_start = self._find_next_cycle_start_after_release(
                    released_gripper=release_gripper,
                    next_joint_vel=next_arm["joint_vel"],
                    next_eef_vel=next_arm["eef_vel"],
                    next_z=next_arm["z"],
                    start=release_start,
                    total_steps=total_steps,
                )
                checkpoints.append(next_cycle_start)
                cycle_start = next_cycle_start

        final = self.validate_checkpoints(self._enforce_order(checkpoints, total_steps), total_steps)
        self.cycle_modes = cycle_modes
        self.phase_kinds = phase_kinds
        self._current_descriptions = descriptions
        if len(self.phase_kinds) > len(final) + 1:
            self.phase_kinds = self.phase_kinds[: len(final) + 1]
        if len(self._current_descriptions) > len(final) + 1:
            self._current_descriptions = self._current_descriptions[: len(final) + 1]
        return final

    def _left_pick_cycle(
        self,
        data: dict,
        close_start: int,
        cycle_start: int,
        total_steps: int,
    ) -> tuple[list[int], list[str]]:
        c0 = self._find_approach_end_near_close_start(
            gripper=data["gripper"],
            z=data["z"],
            joint_velocity=data["joint_vel"],
            start=cycle_start,
            close_start=close_start,
            total_steps=total_steps,
        )
        c1 = self._find_move_start_after_grasp(
            data["gripper"], data["joint_vel"], data["eef_vel"], c0, total_steps
        )
        c2 = self._find_place_start_by_gripper_opening(data["gripper"], c1, total_steps)
        return [c0, c1, c2], list(PHASE_KINDS_LEFT)

    def _right_pick_cycle(
        self,
        arm_data: dict,
        right_close_start: int,
        cycle_start: int,
        total_steps: int,
    ) -> tuple[list[int], list[str]]:
        right = arm_data["right"]
        left = arm_data["left"]

        b0 = self._find_approach_end_near_close_start(
            gripper=right["gripper"],
            z=right["z"],
            joint_velocity=right["joint_vel"],
            start=cycle_start,
            close_start=right_close_start,
            total_steps=total_steps,
        )
        b1 = self._find_move_start_after_grasp(
            right["gripper"], right["joint_vel"], right["eef_vel"], b0, total_steps
        )
        b2 = self._find_left_start_after_right_handover(
            right_gripper=right["gripper"],
            left_z=left["z"],
            left_joint_vel=left["joint_vel"],
            left_eef_vel=left["eef_vel"],
            start=b1,
            total_steps=total_steps,
        )

        left_close_start = self._find_close_start(left["gripper"], b2)
        if left_close_start is None:
            left_close_start = min(total_steps - 1, b2 + max(1, (total_steps - b2) // 6))

        b3 = self._find_approach_end_near_close_start(
            gripper=left["gripper"],
            z=left["z"],
            joint_velocity=left["joint_vel"],
            start=b2,
            close_start=left_close_start,
            total_steps=total_steps,
        )
        b4 = self._find_move_start_after_grasp(
            left["gripper"], left["joint_vel"], left["eef_vel"], b3, total_steps
        )
        b5 = self._find_place_start_by_gripper_opening(left["gripper"], b4, total_steps)
        return [b0, b1, b2, b3, b4, b5], list(PHASE_KINDS_RIGHT)

    def _find_pickup_arm_at_cycle_start(
        self,
        arm_data: dict,
        cycle_start: int,
        total_steps: int,
    ) -> tuple[str, CloseEvent]:
        left_event = self._find_next_close_event(arm_data["left"], cycle_start, total_steps)
        right_event = self._find_next_close_event(arm_data["right"], cycle_start, total_steps)

        if left_event is not None and right_event is not None:
            if left_event.close_start <= right_event.close_start:
                return "left", left_event
            return "right", right_event
        if left_event is not None:
            return "left", left_event
        if right_event is not None:
            return "right", right_event

        fallback_arm = "left"
        gripper = arm_data[fallback_arm]["gripper"]
        close_start = self._find_close_start(gripper, cycle_start) or max(cycle_start + 1, total_steps // 8)
        return fallback_arm, CloseEvent(arm=fallback_arm, close_start=int(close_start))

    def _find_next_close_event(
        self,
        data: dict,
        start: int,
        total_steps: int,
    ) -> Optional[CloseEvent]:
        gripper = data["gripper"]
        joint_vel = data["joint_vel"]
        close_start = self._find_close_start(gripper, start)
        if close_start is None:
            return None

        pre_segment = gripper[max(start, close_start - 5) : close_start + 1]
        post_segment = gripper[close_start : min(total_steps, close_start + 30)]
        if len(post_segment) == 0:
            return None
        open_ref = float(np.max(pre_segment)) if len(pre_segment) > 0 else float(gripper[close_start - 1])
        drop = open_ref - float(np.min(post_segment))
        if drop < self.min_close_drop:
            alt = self._find_close_start(gripper, close_start + 1)
            if alt is None:
                return None
            close_start = alt

        close_done = self._first_consecutive_leq(gripper, self.close_threshold, close_start)
        if close_done is None:
            return None

        motion_hits = np.where(joint_vel[close_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(motion_hits) == 0:
            return None

        return CloseEvent(arm="", close_start=int(close_start))

    def _find_left_start_after_right_handover(
        self,
        right_gripper: np.ndarray,
        left_z: Optional[np.ndarray],
        left_joint_vel: np.ndarray,
        left_eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        window = max(1, self.z_window)
        for t in range(start, max(start, total_steps - window)):
            if right_gripper[t] > self.close_threshold:
                continue
            joint_ok = float(left_joint_vel[t]) > self.joint_velocity_threshold
            eef_ok = (
                left_eef_vel is not None
                and len(left_eef_vel) > t
                and float(left_eef_vel[t]) > self.eef_velocity_threshold
            )
            z_ok = False
            if left_z is not None and len(left_z) > t + window:
                z_ok = abs(float(left_z[t + window] - left_z[t])) > self.z_change_threshold
            if joint_ok or eef_ok or z_ok:
                return int(t)
        return int(start)

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
        """Release done / next bottle approach start: gripper open + next arm moves."""
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
            joint_hits = np.where(next_joint_vel[open_done:total_steps] > self.joint_velocity_threshold)[0]
            if len(joint_hits) > 0:
                return int(open_done + joint_hits[0])
            return int(open_done)

        return int(min(total_steps - 1, search_start + self.cycle_gap))

    def _first_consecutive_geq(self, series: np.ndarray, threshold: float, start: int) -> Optional[int]:
        window = max(1, self.stable_window)
        for idx in range(max(0, start), max(0, len(series) - window + 1)):
            if np.all(series[idx : idx + window] >= threshold):
                return int(idx)
        candidates = np.where(series[max(0, start) :] >= threshold)[0]
        if len(candidates) > 0:
            return int(max(0, start) + candidates[0])
        return None

    def _find_close_start(self, gripper: np.ndarray, start: int) -> Optional[int]:
        open_mask = np.isclose(gripper, self.gripper_open_value, atol=self.gripper_open_atol)
        for idx in range(max(1, start), len(gripper)):
            was_open = open_mask[idx - 1]
            starts_closing = gripper[idx] < gripper[idx - 1] - self.open_delta
            leaves_open = not open_mask[idx]
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
        for idx in range(max(1, start), total_steps):
            if gripper[idx - 1] <= self.close_threshold and gripper[idx] > gripper[idx - 1] + self.open_delta:
                return int(idx)
        return int(start)

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

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        descriptions = self._current_descriptions
        if len(descriptions) >= num_phases:
            return descriptions[:num_phases]
        padded = list(descriptions)
        while len(padded) < num_phases:
            padded.append("Complete the next step.")
        return padded

    def get_subtask_descriptions(self) -> list[str]:
        return list(PHASE_DESCRIPTIONS_LEFT)
