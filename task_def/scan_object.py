"""Rule-based phase segmentation for the scan_object task.

Dual-arm协同: approach, grasp, move and lift together, object rotates barcode
into view, then scanner arm adjusts to scan.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class ScanObjectProcessor(BaseTaskProcessor):
    """Five-phase dual-arm splitter for barcode scanning."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        open_delta: float = 0.01,
        lookback: int = 15,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        angular_velocity_threshold: float = 0.01,
        z_eps: float = 0.005,
        z_lift_threshold: float = 0.005,
        z_window: int = 5,
        z_high_margin: float = 0.015,
        z_stable_threshold: float = 0.005,
        stable_window: int = 5,
        stable_velocity_threshold: float = 0.01,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_threshold = close_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.open_delta = open_delta
        self.lookback = lookback
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.angular_velocity_threshold = angular_velocity_threshold
        self.z_eps = z_eps
        self.z_lift_threshold = z_lift_threshold
        self.z_window = z_window
        self.z_high_margin = z_high_margin
        self.z_stable_threshold = z_stable_threshold
        self.stable_window = stable_window
        self.stable_velocity_threshold = stable_velocity_threshold
        self.object_arm: Optional[str] = None
        self.scanner_arm: Optional[str] = None

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: Optional[str] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
        external_eef_pose_left: Optional[np.ndarray] = None,
        external_eef_pose_right: Optional[np.ndarray] = None,
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
                "angular_vel": self._compute_eef_angular_velocity(
                    hdf5_data, "left", total_steps, external_eef_pose_left
                ),
            },
            "right": {
                "gripper": right_gripper,
                "z": right_xyz[:, 2] if right_xyz is not None else None,
                "joint_vel": self.analyzer.compute_velocity(qpos, arm_indices=(7, 13)),
                "eef_vel": self._compute_eef_velocity(right_xyz, total_steps),
                "angular_vel": self._compute_eef_angular_velocity(
                    hdf5_data, "right", total_steps, external_eef_pose_right
                ),
            },
        }

        left_close_start = self._find_close_start(left_gripper, 0) or max(1, total_steps // 10)
        right_close_start = self._find_close_start(right_gripper, 0) or max(1, total_steps // 10)

        left_approach_end = self._find_approach_end_near_close_start(
            left_gripper, arm_data["left"]["z"], arm_data["left"]["joint_vel"], 0, left_close_start, total_steps
        )
        right_approach_end = self._find_approach_end_near_close_start(
            right_gripper, arm_data["right"]["z"], arm_data["right"]["joint_vel"], 0, right_close_start, total_steps
        )
        c0 = max(left_approach_end, right_approach_end)

        left_close_done = self._find_close_done(left_gripper, c0, total_steps)
        right_close_done = self._find_close_done(right_gripper, c0, total_steps)

        left_lift_start = self._find_lift_or_move_start_after_grasp(
            arm_data["left"]["gripper"],
            arm_data["left"]["z"],
            arm_data["left"]["joint_vel"],
            arm_data["left"]["eef_vel"],
            left_close_done,
            total_steps,
        )
        right_lift_start = self._find_lift_or_move_start_after_grasp(
            arm_data["right"]["gripper"],
            arm_data["right"]["z"],
            arm_data["right"]["joint_vel"],
            arm_data["right"]["eef_vel"],
            right_close_done,
            total_steps,
        )
        c1 = max(left_lift_start, right_lift_start)

        left_lift_done = self._find_lift_done_high_stable(
            arm_data["left"]["gripper"], arm_data["left"]["z"], c1, total_steps
        )
        right_lift_done = self._find_lift_done_high_stable(
            arm_data["right"]["gripper"], arm_data["right"]["z"], c1, total_steps
        )
        lift_ready = max(left_lift_done, right_lift_done)

        left_rotate_start = self._find_post_lift_motion_start(
            arm_data["left"]["joint_vel"],
            arm_data["left"]["eef_vel"],
            arm_data["left"]["angular_vel"],
            lift_ready,
            total_steps,
        )
        right_rotate_start = self._find_post_lift_motion_start(
            arm_data["right"]["joint_vel"],
            arm_data["right"]["eef_vel"],
            arm_data["right"]["angular_vel"],
            lift_ready,
            total_steps,
        )

        if left_rotate_start <= right_rotate_start:
            self.object_arm = "left"
            self.scanner_arm = "right"
            c2 = left_rotate_start
            object_data = arm_data["left"]
            scanner_data = arm_data["right"]
        else:
            self.object_arm = "right"
            self.scanner_arm = "left"
            c2 = right_rotate_start
            object_data = arm_data["right"]
            scanner_data = arm_data["left"]

        c3 = self._find_scanner_adjust_start_after_object_rotation(
            object_joint_vel=object_data["joint_vel"],
            object_eef_vel=object_data["eef_vel"],
            scanner_joint_vel=scanner_data["joint_vel"],
            scanner_eef_vel=scanner_data["eef_vel"],
            object_angular_vel=object_data["angular_vel"],
            scanner_angular_vel=scanner_data["angular_vel"],
            start=c2,
            total_steps=total_steps,
        )

        checkpoints = self._enforce_order([c0, c1, c2, c3], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _find_lift_or_move_start_after_grasp(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_vel: np.ndarray,
        eef_vel: Optional[np.ndarray],
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

        window = max(1, self.z_window)
        if z is not None:
            z_arr = np.asarray(z[:total_steps], dtype=np.float64)
            for t in range(close_done, max(close_done, total_steps - window)):
                if float(z_arr[t + window] - z_arr[t]) > self.z_lift_threshold:
                    return int(t)

        if eef_vel is not None:
            eef_hits = np.where(eef_vel[close_done:] > self.eef_velocity_threshold)[0]
            if len(eef_hits) > 0:
                return int(close_done + eef_hits[0])

        joint_hits = np.where(joint_vel[close_done:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(close_done + joint_hits[0])
        return int(close_done)

    def _find_lift_done_high_stable(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        if z is None:
            return int(start)

        z = np.asarray(z[:total_steps], dtype=np.float64)
        high_z_level = float(np.max(z[start:])) - self.z_high_margin
        stable_window = max(1, self.stable_window)

        segment = np.asarray(gripper[start:total_steps], dtype=np.float64)
        close_threshold = (
            min(self.close_threshold, float(np.min(segment) + 0.05)) if len(segment) > 0 else self.close_threshold
        )

        for t in range(start, max(start, total_steps - stable_window)):
            if gripper[t] > close_threshold:
                continue
            if z[t] < high_z_level:
                continue
            if float(np.std(z[t : t + stable_window])) >= self.z_stable_threshold:
                continue
            return int(t)

        high_candidates = np.where(z[start:] >= high_z_level)[0]
        if len(high_candidates) > 0:
            return int(start + high_candidates[0])
        return int(start)

    def _find_scanner_adjust_start_after_object_rotation(
        self,
        object_joint_vel: np.ndarray,
        object_eef_vel: Optional[np.ndarray],
        scanner_joint_vel: np.ndarray,
        scanner_eef_vel: Optional[np.ndarray],
        object_angular_vel: Optional[np.ndarray],
        scanner_angular_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        window = max(1, self.stable_window)
        object_eef = object_eef_vel if object_eef_vel is not None else np.zeros(total_steps, dtype=np.float64)
        scanner_eef = scanner_eef_vel if scanner_eef_vel is not None else np.zeros(total_steps, dtype=np.float64)

        for t in range(start, max(start, total_steps - window)):
            object_stop = (
                float(np.max(object_joint_vel[t : t + window])) < self.joint_velocity_threshold
                and float(np.max(object_eef[t : t + window])) < self.eef_velocity_threshold
            )
            if object_angular_vel is not None:
                object_stop = object_stop and (
                    float(np.max(object_angular_vel[t : t + window])) < self.angular_velocity_threshold
                )

            scanner_move = float(scanner_joint_vel[t]) > self.joint_velocity_threshold
            scanner_move = scanner_move or float(scanner_eef[t]) > self.eef_velocity_threshold
            if scanner_angular_vel is not None:
                scanner_move = scanner_move or float(scanner_angular_vel[t]) > self.angular_velocity_threshold

            if object_stop and scanner_move:
                return int(t)

        scanner_hits = np.where(scanner_joint_vel[start:total_steps] > self.joint_velocity_threshold)[0]
        if len(scanner_hits) > 0:
            return int(start + scanner_hits[0])
        return int(start)

    def _find_post_lift_motion_start(
        self,
        joint_vel: np.ndarray,
        eef_vel: Optional[np.ndarray],
        angular_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        for t in range(start, total_steps):
            if float(joint_vel[t]) > self.joint_velocity_threshold:
                return int(t)
            if eef_vel is not None and float(eef_vel[t]) > self.eef_velocity_threshold:
                return int(t)
            if angular_vel is not None and float(angular_vel[t]) > self.angular_velocity_threshold:
                return int(t)

        joint_hits = np.where(joint_vel[start:total_steps] > self.joint_velocity_threshold)[0]
        if len(joint_hits) > 0:
            return int(start + joint_hits[0])
        return int(start)

    def _find_close_done(self, gripper: np.ndarray, start: int, total_steps: int) -> int:
        done = self._first_consecutive_leq(gripper, self.close_threshold, start)
        if done is not None:
            return int(done)
        segment = gripper[start:total_steps]
        if len(segment) == 0:
            return int(start)
        return int(start + np.argmin(segment))

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

    def _first_consecutive_leq(self, series: np.ndarray, threshold: float, start: int) -> Optional[int]:
        window = max(1, self.stable_window)
        for idx in range(max(0, start), max(0, len(series) - window + 1)):
            if np.all(series[idx : idx + window] <= threshold):
                return int(idx)
        candidates = np.where(series[max(0, start) :] <= threshold)[0]
        if len(candidates) > 0:
            return int(max(0, start) + candidates[0])
        return None

    def _compute_eef_angular_velocity(
        self,
        hdf5_data,
        side: str,
        total_steps: int,
        external_pose: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        pose = external_pose
        key = f"endpose/{side}_endpose"
        if pose is None and key in hdf5_data:
            pose = hdf5_data[key][()][:total_steps]
        if pose is None:
            return None

        pose = np.asarray(pose, dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] < 7:
            return None
        if len(pose) < 2:
            return np.zeros(total_steps, dtype=np.float64)

        quat = pose[:, 3:7]
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        quat = quat / norms

        angles = np.zeros(len(quat), dtype=np.float64)
        for idx in range(1, len(quat)):
            dot = float(np.clip(np.abs(np.dot(quat[idx - 1], quat[idx])), -1.0, 1.0))
            angles[idx] = 2.0 * np.arccos(dot)

        if len(angles) >= total_steps:
            return angles[:total_steps]
        pad = np.zeros(total_steps - len(angles), dtype=np.float64)
        return np.concatenate([angles, pad])

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
            "Approach the scanner and the object.",
            "Grasp the scanner and the object.",
            "Move and lift the scanner and the object.",
            "Rotate the object to expose the barcode.",
            "Orient the scanner toward the barcode and scan the object.",
        ]
