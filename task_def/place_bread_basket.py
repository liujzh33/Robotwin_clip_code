"""Rule-based multi-mode phase segmentation for place_bread_basket.

Modes:
- single_bread: 1 bread, 4 phases / 3 checkpoints
- sequential_two_breads: 2 breads same side, one arm twice, 8 phases / 7 checkpoints
- dual_breads: 2 breads on both sides, 6 phases / 5 checkpoints
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


PHASE_DESCRIPTIONS = {
    "single_bread": [
        "Approach the bread.",
        "Grasp the bread.",
        "Move the bread above the basket.",
        "Release the bread into the basket.",
    ],
    "sequential_two_breads": [
        "Approach the first bread.",
        "Grasp the first bread.",
        "Move the first bread above the basket.",
        "Release the first bread into the basket.",
        "Approach the second bread.",
        "Grasp the second bread.",
        "Move the second bread above the basket.",
        "Release the second bread into the basket.",
    ],
    "dual_breads": [
        "Approach both breads with both arms.",
        "Grasp both breads with both grippers.",
        "Move the first bread above the basket.",
        "Release the first bread into the basket.",
        "Move the second bread above the basket.",
        "Release the second bread into the basket.",
    ],
}


class PlaceBreadBasketProcessor(BaseTaskProcessor):
    """Multi-mode bread-in-basket splitter with automatic mode detection."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.9,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        lookback: int = 15,
        sync_window: int = 30,
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
        self.lookback = lookback
        self.sync_window = sync_window
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_rise_threshold = z_rise_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.mode: str = "single_bread"
        self._current_descriptions: Optional[list[str]] = None
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

        left_events = self._find_all_arm_grasp_events("left", left_gripper, total_steps)
        right_events = self._find_all_arm_grasp_events("right", right_gripper, total_steps)
        all_events = sorted(left_events + right_events, key=lambda event: event.close_start)

        mode, mode_events = self._detect_mode(left_events, right_events, all_events)
        self.mode = mode

        if mode == "single_bread":
            event = mode_events if isinstance(mode_events, GraspEvent) else self._fallback_event(
                arm_data, active_side, left_gripper, right_gripper, total_steps
            )
            self.active_arm = event.arm
            checkpoints = self._single_bread_checkpoints(arm_data[event.arm], event, total_steps)
        elif mode == "sequential_two_breads":
            event1, event2 = mode_events
            self.active_arm = event1.arm
            checkpoints = self._sequential_checkpoints(arm_data[event1.arm], event1, event2, total_steps)
        else:
            left_ev, right_ev = mode_events
            self.active_arm = "dual"
            checkpoints = self._dual_bread_checkpoints(arm_data, left_ev, right_ev, total_steps)

        self._current_descriptions = list(PHASE_DESCRIPTIONS[mode])
        return self.validate_checkpoints(self._enforce_order(checkpoints, total_steps), total_steps)

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        descriptions = self._current_descriptions or PHASE_DESCRIPTIONS.get(self.mode, PHASE_DESCRIPTIONS["single_bread"])
        if len(descriptions) >= num_phases:
            return descriptions[:num_phases]
        padded = list(descriptions)
        while len(padded) < num_phases:
            padded.append("Complete the next step.")
        return padded

    def get_subtask_descriptions(self) -> list[str]:
        return list(PHASE_DESCRIPTIONS["single_bread"])

    def find_place_done(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        total_steps: int,
    ) -> Optional[int]:
        """Optional visualization checkpoint: release complete and arm retreats."""
        return self._find_release_done_and_arm_moves(gripper, z, joint_velocity, start, total_steps)

    def _detect_mode(
        self,
        left_events: list[GraspEvent],
        right_events: list[GraspEvent],
        all_events: list[GraspEvent],
    ) -> tuple[str, object]:
        if not all_events:
            return "single_bread", None

        if len(all_events) == 1:
            return "single_bread", all_events[0]

        for arm, events in (("left", left_events), ("right", right_events)):
            if len(events) >= 2:
                return "sequential_two_breads", (events[0], events[1])

        left_ev = left_events[0] if left_events else None
        right_ev = right_events[0] if right_events else None
        if left_ev is not None and right_ev is not None:
            if abs(left_ev.close_start - right_ev.close_start) <= self.sync_window:
                return "dual_breads", (left_ev, right_ev)
            return "dual_breads", (left_ev, right_ev)

        return "single_bread", all_events[0]

    def _fallback_event(
        self,
        arm_data: dict,
        active_side: Optional[str],
        left_gripper: np.ndarray,
        right_gripper: np.ndarray,
        total_steps: int,
    ) -> GraspEvent:
        fallback_arm = active_side or self.analyzer.active_side_from_grippers(left_gripper, right_gripper)
        close_start = self._find_close_start(arm_data[fallback_arm]["gripper"], 0) or max(1, total_steps // 6)
        return GraspEvent(
            arm=fallback_arm,
            close_start=int(close_start),
            close_done=int(close_start),
            open_start=min(total_steps - 1, close_start + max(1, total_steps // 4)),
            open_done=min(total_steps - 1, close_start + max(1, total_steps // 3)),
        )

    def _single_bread_checkpoints(self, data: dict, event: GraspEvent, total_steps: int) -> list[int]:
        c0, c1, c2 = self._pick_place_cycle(data, event, search_start=0, total_steps=total_steps)
        return [c0, c1, c2]

    def _sequential_checkpoints(
        self,
        data: dict,
        event1: GraspEvent,
        event2: GraspEvent,
        total_steps: int,
    ) -> list[int]:
        c0, c1, c2, c3 = self._pick_place_cycle(
            data, event1, search_start=0, total_steps=total_steps, include_release_done=True
        )
        c4, c5, c6 = self._pick_place_cycle(
            data, event2, search_start=c3, total_steps=total_steps, include_release_done=False
        )
        return [c0, c1, c2, c3, c4, c5, c6]

    def _dual_bread_checkpoints(
        self,
        arm_data: dict,
        left_ev: GraspEvent,
        right_ev: GraspEvent,
        total_steps: int,
    ) -> list[int]:
        left = arm_data["left"]
        right = arm_data["right"]

        left_c0 = self._find_approach_end_near_close_start(
            gripper=left["gripper"],
            z=left["z"],
            joint_velocity=left["joint_vel"],
            start=0,
            close_start=left_ev.close_start,
            total_steps=total_steps,
        )
        right_c0 = self._find_approach_end_near_close_start(
            gripper=right["gripper"],
            z=right["z"],
            joint_velocity=right["joint_vel"],
            start=0,
            close_start=right_ev.close_start,
            total_steps=total_steps,
        )
        c0 = max(left_c0, right_c0)

        left_c1 = self._find_move_start_after_grasp(
            left["gripper"], left["joint_vel"], left["eef_vel"], c0, total_steps
        )
        right_c1 = self._find_move_start_after_grasp(
            right["gripper"], right["joint_vel"], right["eef_vel"], c0, total_steps
        )

        if left_c1 <= right_c1:
            first_arm, second_arm = "left", "right"
            first_c1, second_c1 = left_c1, right_c1
        else:
            first_arm, second_arm = "right", "left"
            first_c1, second_c1 = right_c1, left_c1

        other_arm = "right" if first_arm == "left" else "left"
        other_close_done = self._first_consecutive_leq(
            arm_data[other_arm]["gripper"], self.close_threshold, c0
        )
        if other_close_done is None:
            other_close_done = c0
        c1 = max(first_c1, other_close_done)

        first_data = arm_data[first_arm]
        second_data = arm_data[second_arm]
        c2 = self._find_place_start_by_gripper_opening(first_data["gripper"], c1, total_steps)
        c3 = self._find_second_move_start_after_first_release(
            first_gripper=first_data["gripper"],
            second_gripper=second_data["gripper"],
            second_joint_vel=second_data["joint_vel"],
            release_start=c2,
            search_start=c0,
            fallback_move_start=second_c1,
            total_steps=total_steps,
        )
        c4 = self._find_place_start_by_gripper_opening(second_data["gripper"], c3, total_steps)
        return [c0, c1, c2, c3, c4]

    def _pick_place_cycle(
        self,
        data: dict,
        event: GraspEvent,
        search_start: int,
        total_steps: int,
        include_release_done: bool = False,
    ) -> tuple[int, ...]:
        gripper = data["gripper"]
        z = data["z"]
        joint_vel = data["joint_vel"]
        eef_vel = data["eef_vel"]

        c0 = self._find_approach_end_near_close_start(
            gripper=gripper,
            z=z,
            joint_velocity=joint_vel,
            start=search_start,
            close_start=event.close_start,
            total_steps=total_steps,
        )
        c1 = self._find_move_start_after_grasp(gripper, joint_vel, eef_vel, c0, total_steps)
        c2 = self._find_place_start_by_gripper_opening(gripper, c1, total_steps)
        if not include_release_done:
            return c0, c1, c2
        c3 = self._find_release_done_and_arm_moves(gripper, z, joint_vel, c2, total_steps)
        return c0, c1, c2, c3

    def _find_release_done_and_arm_moves(
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

        z_window = max(1, self.z_window)
        for idx in range(open_done, max(open_done, total_steps - z_window)):
            g_ok = gripper[idx] >= self.open_done_threshold
            joint_ok = joint_velocity[idx] > self.joint_velocity_threshold
            z_ok = z is not None and float(z[idx + z_window] - z[idx]) > self.z_rise_threshold
            if g_ok and (joint_ok or z_ok):
                return int(idx)

        joint_hits = np.where(joint_velocity[open_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(open_done + joint_hits[0])
        return int(open_done)

    def _find_second_move_start_after_first_release(
        self,
        first_gripper: np.ndarray,
        second_gripper: np.ndarray,
        second_joint_vel: np.ndarray,
        release_start: int,
        search_start: int,
        fallback_move_start: int,
        total_steps: int,
    ) -> int:
        for idx in range(release_start, total_steps):
            first_open = first_gripper[idx] >= self.open_done_threshold
            second_closed = second_gripper[idx] <= self.close_threshold
            second_moving = second_joint_vel[idx] > self.joint_velocity_threshold
            if first_open and second_closed and second_moving:
                return int(idx)
        return int(max(release_start, fallback_move_start))

    def _find_all_arm_grasp_events(self, arm: str, gripper: np.ndarray, total_steps: int) -> list[GraspEvent]:
        events: list[GraspEvent] = []
        start = 0
        while start < total_steps - 5:
            event = self._find_arm_grasp_event(arm, gripper, start, total_steps)
            if event is None:
                break
            events.append(event)
            start = int(event.open_done) + 1
        return events

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
