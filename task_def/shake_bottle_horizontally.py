"""Rule-based phase segmentation for the shake_bottle_horizontally task.

Four phases: approach, grasp, move and orient horizontally, horizontal shake.
Active arm is inferred from gripper close, Z lift, and horizontal x/y oscillation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class ShakeBottleHorizontallyProcessor(BaseTaskProcessor):
    """Four-phase splitter with lift-orient-then-horizontal-shake detection."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        min_close_drop: float = 0.15,
        lookback: int = 15,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        z_lift_threshold: float = 0.005,
        z_window: int = 5,
        z_high_margin: float = 0.015,
        z_stable_threshold: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 8,
        shake_window: int = 35,
        min_shake_peaks: int = 3,
        min_turning_points: int = 3,
        motion_weight: float = 0.1,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.min_close_drop = min_close_drop
        self.lookback = lookback
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.z_lift_threshold = z_lift_threshold
        self.z_window = z_window
        self.z_high_margin = z_high_margin
        self.z_stable_threshold = z_stable_threshold
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.shake_window = shake_window
        self.min_shake_peaks = min_shake_peaks
        self.min_turning_points = min_turning_points
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
            close_start = self._find_close_start(arm_data[active_arm]["gripper"], 0) or max(1, total_steps // 8)
        else:
            active_arm, close_start = self._find_active_arm_by_close_lift_horizontal_shake(arm_data, total_steps)

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
        c1 = self._find_lift_start_after_grasp(
            gripper=data["gripper"],
            z=data["z"],
            joint_velocity=data["joint_vel"],
            eef_velocity=data["eef_vel"],
            start=c0,
            total_steps=total_steps,
        )
        c2 = self._find_horizontal_shake_start_after_lift(
            gripper=data["gripper"],
            z=data["z"],
            xyz=data["xyz"],
            joint_vel=data["joint_vel"],
            eef_vel=data["eef_vel"],
            start=c1,
            total_steps=total_steps,
        )

        checkpoints = self._enforce_order([c0, c1, c2], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_active_arm_by_close_lift_horizontal_shake(
        self,
        arm_data: dict,
        total_steps: int,
    ) -> tuple[str, int]:
        best_arm = "left"
        best_close_start = max(1, total_steps // 8)
        best_score = -1.0

        for arm in ("left", "right"):
            gripper = arm_data[arm]["gripper"]
            z = arm_data[arm]["z"]
            joint_vel = arm_data[arm]["joint_vel"]
            xyz = arm_data[arm]["xyz"]

            close_start = self._find_close_start(gripper, 0)
            if close_start is None:
                continue

            pre = gripper[max(0, close_start - 5) : close_start + 1]
            post = gripper[close_start : min(total_steps, close_start + 30)]
            if len(post) == 0:
                continue
            open_ref = float(np.max(pre)) if len(pre) > 0 else float(gripper[close_start - 1])
            drop = open_ref - float(np.min(post))
            if drop < self.min_close_drop:
                alt = self._find_close_start(gripper, close_start + 1)
                if alt is None:
                    continue
                close_start = alt

            close_done = self._first_leq(gripper, self.close_threshold, close_start)
            if close_done is None:
                continue

            z_lift = 0.0
            if z is not None and close_done < total_steps - self.z_window:
                z_lift = float(np.max(z[close_done : min(total_steps, close_done + 80)]) - z[close_done])

            shake_score = 0.0
            if xyz is not None and self._has_horizontal_eef_oscillation(xyz, close_done, total_steps):
                shake_score = 2.0
            elif self._has_periodic_velocity_motion(joint_vel, close_done, total_steps):
                shake_score = 1.0

            score = drop + z_lift + self.motion_weight * shake_score
            if score > best_score:
                best_score = score
                best_arm = arm
                best_close_start = int(close_start)

        return best_arm, best_close_start

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
        close_threshold = (
            min(self.close_threshold, float(np.min(segment) + 0.05)) if len(segment) > 0 else self.close_threshold
        )

        close_done = self._first_leq(gripper, close_threshold, start)
        if close_done is None:
            close_done = start

        window = max(1, self.z_window)
        if z is not None:
            z_arr = np.asarray(z[:total_steps], dtype=np.float64)
            for t in range(close_done, max(close_done, total_steps - window)):
                if float(z_arr[t + window] - z_arr[t]) > self.z_lift_threshold:
                    return int(t)

        if eef_velocity is not None:
            eef_hits = np.where(eef_velocity[close_done:] > self.eef_velocity_threshold)[0]
            if len(eef_hits) > 0:
                return int(close_done + eef_hits[0])

        joint_hits = np.where(joint_velocity[close_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(close_done + joint_hits[0])
        return int(close_done)

    def _find_horizontal_shake_start_after_lift(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        xyz: Optional[np.ndarray],
        joint_vel: np.ndarray,
        eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        if z is None:
            return self._fallback_horizontal_shake_start(xyz, joint_vel, eef_vel, start, total_steps)

        z = np.asarray(z[:total_steps], dtype=np.float64)
        segment = np.asarray(gripper[start:total_steps], dtype=np.float64)
        close_threshold = (
            min(self.close_threshold, float(np.min(segment) + 0.05)) if len(segment) > 0 else self.close_threshold
        )

        high_z_level = float(np.max(z[start:])) - self.z_high_margin
        stable_window = max(1, self.stable_window)
        shake_window = max(5, self.shake_window)

        search_end = max(start, total_steps - max(stable_window, shake_window))
        for t in range(start, search_end):
            if gripper[t] > close_threshold:
                continue
            if z[t] < high_z_level:
                continue
            if float(np.std(z[t : t + stable_window])) >= self.z_stable_threshold:
                continue

            horizontal_periodic = (
                xyz is not None and self._has_horizontal_eef_oscillation(xyz, t, total_steps)
            )
            joint_periodic = self._has_periodic_velocity_motion(joint_vel, t, total_steps)
            eef_periodic = (
                eef_vel is not None
                and self._has_periodic_velocity_motion(
                    eef_vel, t, total_steps, threshold=self.eef_velocity_threshold
                )
            )

            if horizontal_periodic or joint_periodic or eef_periodic:
                return int(t)

        high_candidates = np.where(z[start:] >= high_z_level)[0]
        if len(high_candidates) > 0:
            return int(start + high_candidates[0])
        return int(start)

    def _fallback_horizontal_shake_start(
        self,
        xyz: Optional[np.ndarray],
        joint_vel: np.ndarray,
        eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        for t in range(start, total_steps - 5):
            if xyz is not None and self._has_horizontal_eef_oscillation(xyz, t, total_steps):
                return int(t)
            if self._has_periodic_velocity_motion(joint_vel, t, total_steps):
                return int(t)
            if eef_vel is not None and self._has_periodic_velocity_motion(
                eef_vel, t, total_steps, threshold=self.eef_velocity_threshold
            ):
                return int(t)
        joint_hits = np.where(joint_vel[start:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(start + joint_hits[0])
        return int(start)

    def _has_horizontal_eef_oscillation(
        self,
        xyz: np.ndarray,
        start: int,
        total_steps: int,
        axis: Optional[int] = None,
    ) -> bool:
        end = min(total_steps, start + self.shake_window)
        axes = [0, 1] if axis is None else [axis]

        for ax in axes:
            segment = np.asarray(xyz[start:end, ax], dtype=np.float64)
            if len(segment) < 6:
                continue

            turning_points = 0
            for idx in range(1, len(segment) - 1):
                if (segment[idx] - segment[idx - 1]) * (segment[idx + 1] - segment[idx]) < 0:
                    turning_points += 1
            if turning_points >= self.min_turning_points:
                return True
        return False

    def _has_periodic_velocity_motion(
        self,
        velocity: np.ndarray,
        start: int,
        total_steps: int,
        threshold: Optional[float] = None,
    ) -> bool:
        if threshold is None:
            threshold = self.joint_velocity_threshold
        end = min(total_steps, start + self.shake_window)
        segment = np.asarray(velocity[start:end], dtype=np.float64)
        if len(segment) < 5:
            return False

        peaks = 0
        for idx in range(1, len(segment) - 1):
            if segment[idx] > threshold and segment[idx] > segment[idx - 1] and segment[idx] > segment[idx + 1]:
                peaks += 1
        return peaks >= self.min_shake_peaks

    def _find_approach_end_near_close_start(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
        start: int,
        close_start: int,
        total_steps: int,
    ) -> int:
        window = max(1, min(5, self.stable_window))
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
            "Approach the bottle.",
            "Grasp the bottle.",
            "Move and orient the bottle horizontally.",
            "Shake the bottle horizontally.",
        ]
