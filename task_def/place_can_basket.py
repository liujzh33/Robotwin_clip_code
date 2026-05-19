"""Rule-based phase segmentation for the place_can_basket task.

can_arm: pick and place the beverage can (first full close-open event).
basket_arm: approach handle, grasp, and lift the basket after can release.
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


class PlaceCanBasketProcessor(BaseTaskProcessor):
    """Seven-phase splitter: place can, then grasp handle and lift basket."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        close_margin: float = 0.05,
        open_done_threshold: float = 0.9,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        lookback: int = 15,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_lift_threshold: float = 0.005,
        z_change_threshold: float = 0.005,
        z_window: int = 5,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.close_margin = close_margin
        self.open_done_threshold = open_done_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.lookback = lookback
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_lift_threshold = z_lift_threshold
        self.z_change_threshold = z_change_threshold
        self.z_window = z_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.can_arm: Optional[str] = None
        self.basket_arm: Optional[str] = None

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

        can_event = self._find_next_valid_grasp_event(arm_data, start=0, total_steps=total_steps)
        if can_event is None:
            fallback_arm = active_side or self.analyzer.active_side_from_grippers(left_gripper, right_gripper)
            close_start = self._find_close_start(arm_data[fallback_arm]["gripper"], 0) or max(1, total_steps // 8)
            can_event = GraspEvent(
                arm=fallback_arm,
                close_start=int(close_start),
                close_done=int(close_start),
                open_start=min(total_steps - 1, close_start + total_steps // 3),
                open_done=min(total_steps - 1, close_start + total_steps // 2),
            )

        self.can_arm = can_event.arm
        self.basket_arm = "right" if can_event.arm == "left" else "left"

        can = arm_data[self.can_arm]
        basket = arm_data[self.basket_arm]

        c0 = self._find_approach_end_near_close_start(
            gripper=can["gripper"],
            z=can["z"],
            joint_velocity=can["joint_vel"],
            start=0,
            close_start=can_event.close_start,
            total_steps=total_steps,
        )
        c1 = self._find_move_start_after_grasp(
            can["gripper"], can["joint_vel"], can["eef_vel"], c0, total_steps
        )
        c2 = self._find_place_start_by_gripper_opening(can["gripper"], c1, total_steps)
        c3 = self._find_other_arm_start_after_release(
            released_gripper=can["gripper"],
            other_z=basket["z"],
            other_joint_vel=basket["joint_vel"],
            other_eef_vel=basket["eef_vel"],
            start=c2,
            total_steps=total_steps,
        )

        basket_close_start = self._find_close_start(basket["gripper"], c3)
        if basket_close_start is None:
            basket_close_start = c3 + max(1, total_steps // 10)

        c4 = self._find_approach_end_near_close_start(
            gripper=basket["gripper"],
            z=basket["z"],
            joint_velocity=basket["joint_vel"],
            start=c3,
            close_start=basket_close_start,
            total_steps=total_steps,
        )
        c5 = self._find_lift_start_after_grasp(
            basket["gripper"], basket["z"], basket["joint_vel"], basket["eef_vel"], c4, total_steps
        )

        checkpoints = self._enforce_order([c0, c1, c2, c3, c4, c5], total_steps)
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
            if gripper[idx] >= self.open_done_threshold:
                joint_ok = joint_velocity[idx] > self.joint_velocity_threshold
                z_ok = z is not None and float(z[idx + z_window] - z[idx]) > self.z_change_threshold
                if joint_ok or z_ok:
                    return int(idx)
        return open_done

    def _find_other_arm_start_after_release(
        self,
        released_gripper: np.ndarray,
        other_z: Optional[np.ndarray],
        other_joint_vel: np.ndarray,
        other_eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        z_window = max(1, self.z_window)
        for idx in range(start, max(start, total_steps - z_window)):
            can_open = released_gripper[idx] >= self.open_done_threshold
            joint_ok = other_joint_vel[idx] > self.joint_velocity_threshold
            eef_ok = other_eef_vel is not None and other_eef_vel[idx] > self.eef_velocity_threshold
            z_ok = other_z is not None and float(other_z[idx + z_window] - other_z[idx]) > self.z_change_threshold
            if can_open and (joint_ok or eef_ok or z_ok):
                return int(idx)

        open_done = self._first_consecutive_geq(released_gripper, self.open_done_threshold, start)
        if open_done is not None:
            joint_hits = np.where(other_joint_vel[open_done:total_steps] > self.joint_velocity_threshold)[0]
            if len(joint_hits) > 0:
                return int(open_done + joint_hits[0])
            return int(open_done)
        return int(start)

    def _find_lift_start_after_grasp(
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

        close_done = self._first_consecutive_leq(gripper, close_threshold, start)
        if close_done is None:
            close_done = start

        z_window = max(1, self.z_window)
        for idx in range(close_done, max(close_done, total_steps - z_window)):
            z_lift_ok = z is not None and float(z[idx + z_window] - z[idx]) > self.z_lift_threshold
            joint_ok = joint_velocity[idx] > self.joint_velocity_threshold
            eef_ok = eef_velocity is not None and eef_velocity[idx] > self.eef_velocity_threshold
            if z_lift_ok or joint_ok or eef_ok:
                return int(idx)
        return int(close_done)

    def _find_next_valid_grasp_event(
        self,
        arm_data: dict,
        start: int,
        total_steps: int,
    ) -> Optional[GraspEvent]:
        events = []
        for arm in ("left", "right"):
            event = self._find_arm_grasp_event(arm, arm_data[arm]["gripper"], start, total_steps)
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

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the can.",
            "Grasp the can.",
            "Move the can above the basket.",
            "Release the can into the basket.",
            "Approach the basket handle.",
            "Grasp the basket handle.",
            "Lift the basket upward.",
        ]
