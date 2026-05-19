"""Rule-based phase segmentation for the put_object_cabinet task.

object_arm: first arm to grasp the table object and release it into the drawer.
drawer_arm: other arm that grasps the handle and pulls the drawer open.
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


class PutObjectCabinetProcessor(BaseTaskProcessor):
    """Seven-phase dual-arm splitter: grasp object, open drawer, place object."""

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
        z_change_threshold: float = 0.005,
        z_window: int = 5,
        xy_move_threshold: float = 0.015,
        xy_window: int = 5,
        min_pull_frames: int = 8,
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
        self.z_change_threshold = z_change_threshold
        self.z_window = z_window
        self.xy_move_threshold = xy_move_threshold
        self.xy_window = xy_window
        self.min_pull_frames = min_pull_frames
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.object_arm: Optional[str] = None
        self.drawer_arm: Optional[str] = None

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

        object_event = self._find_next_valid_grasp_event(arm_data, start=0, total_steps=total_steps)
        if object_event is None:
            fallback_arm = active_side or self.analyzer.active_side_from_grippers(left_gripper, right_gripper)
            close_start = self._find_close_start(arm_data[fallback_arm]["gripper"], 0) or max(1, total_steps // 8)
            object_event = GraspEvent(
                arm=fallback_arm,
                close_start=int(close_start),
                close_done=int(close_start),
                open_start=min(total_steps - 1, close_start + total_steps // 3),
                open_done=min(total_steps - 1, close_start + total_steps // 2),
            )

        self.object_arm = object_event.arm
        self.drawer_arm = "right" if object_event.arm == "left" else "left"

        obj = arm_data[self.object_arm]
        drawer = arm_data[self.drawer_arm]

        c0 = self._find_approach_end_near_close_start(
            gripper=obj["gripper"],
            z=obj["z"],
            joint_velocity=obj["joint_vel"],
            start=0,
            close_start=object_event.close_start,
            total_steps=total_steps,
        )
        c1 = self._find_drawer_start_after_object_grasp(
            object_gripper=obj["gripper"],
            drawer_z=drawer["z"],
            drawer_joint_vel=drawer["joint_vel"],
            drawer_eef_vel=drawer["eef_vel"],
            start=c0,
            total_steps=total_steps,
        )

        drawer_close_start = self._find_close_start(drawer["gripper"], c1)
        if drawer_close_start is None:
            drawer_close_start = c1 + max(1, total_steps // 12)

        c2 = self._find_approach_end_near_close_start(
            gripper=drawer["gripper"],
            z=drawer["z"],
            joint_velocity=drawer["joint_vel"],
            start=c1,
            close_start=drawer_close_start,
            total_steps=total_steps,
        )
        c3 = self._find_pull_start_after_handle_grasp(
            gripper=drawer["gripper"],
            xyz=drawer["xyz"],
            joint_velocity=drawer["joint_vel"],
            eef_velocity=drawer["eef_vel"],
            start=c2,
            total_steps=total_steps,
        )
        c4 = self._find_object_move_start_after_drawer_open(
            drawer_gripper=drawer["gripper"],
            object_z=obj["z"],
            object_joint_vel=obj["joint_vel"],
            object_eef_vel=obj["eef_vel"],
            start=c3,
            total_steps=total_steps,
        )
        c5 = self._find_place_start_by_gripper_opening(obj["gripper"], c4, total_steps)

        checkpoints = self._enforce_order([c0, c1, c2, c3, c4, c5], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_drawer_start_after_object_grasp(
        self,
        object_gripper: np.ndarray,
        drawer_z: Optional[np.ndarray],
        drawer_joint_vel: np.ndarray,
        drawer_eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        object_close_done = self._first_consecutive_leq(object_gripper, self.close_threshold, start)
        if object_close_done is None:
            object_close_done = start

        z_window = max(1, self.z_window)
        for idx in range(object_close_done, max(object_close_done, total_steps - z_window)):
            if object_gripper[idx] > self.close_threshold:
                continue
            joint_ok = float(drawer_joint_vel[idx]) > self.joint_velocity_threshold
            eef_ok = (
                drawer_eef_vel is not None
                and len(drawer_eef_vel) > idx
                and float(drawer_eef_vel[idx]) > self.eef_velocity_threshold
            )
            z_ok = (
                drawer_z is not None
                and len(drawer_z) > idx + z_window
                and abs(float(drawer_z[idx + z_window] - drawer_z[idx])) > self.z_change_threshold
            )
            if joint_ok or eef_ok or z_ok:
                return int(idx)

        joint_hits = np.where(drawer_joint_vel[object_close_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(object_close_done + joint_hits[0])
        return int(object_close_done)

    def _find_pull_start_after_handle_grasp(
        self,
        gripper: np.ndarray,
        xyz: Optional[np.ndarray],
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

        xy_window = max(1, self.xy_window)
        for idx in range(close_done, max(close_done, total_steps - xy_window)):
            joint_ok = float(joint_velocity[idx]) > self.joint_velocity_threshold
            eef_ok = eef_velocity is not None and float(eef_velocity[idx]) > self.eef_velocity_threshold
            xy_ok = False
            if xyz is not None and len(xyz) > idx + xy_window:
                delta = xyz[idx + xy_window, :2] - xyz[idx, :2]
                xy_ok = float(np.linalg.norm(delta)) > self.xy_move_threshold
            if joint_ok or eef_ok or xy_ok:
                return int(idx)
        return int(close_done)

    def _find_object_move_start_after_drawer_open(
        self,
        drawer_gripper: np.ndarray,
        object_z: Optional[np.ndarray],
        object_joint_vel: np.ndarray,
        object_eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        search_start = min(total_steps - 1, start + self.min_pull_frames)
        z_window = max(1, self.z_window)

        for idx in range(search_start, max(search_start, total_steps - z_window)):
            if drawer_gripper[idx] > self.close_threshold:
                continue
            joint_ok = float(object_joint_vel[idx]) > self.joint_velocity_threshold
            eef_ok = (
                object_eef_vel is not None
                and len(object_eef_vel) > idx
                and float(object_eef_vel[idx]) > self.eef_velocity_threshold
            )
            z_ok = (
                object_z is not None
                and len(object_z) > idx + z_window
                and abs(float(object_z[idx + z_window] - object_z[idx])) > self.z_change_threshold
            )
            if joint_ok or eef_ok or z_ok:
                return int(idx)

        joint_hits = np.where(object_joint_vel[search_start:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(search_start + joint_hits[0])
        return int(search_start)

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
            "Approach the object.",
            "Grasp the object.",
            "Approach the cabinet handle.",
            "Grasp the cabinet handle.",
            "Pull the cabinet drawer open.",
            "Move the object above the opened cabinet drawer.",
            "Place the object inside the cabinet drawer.",
        ]
