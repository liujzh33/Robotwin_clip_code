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
        min_prev_motion_frames: int = 5,
        next_move_window: int = 4,
        motion_lookback: int = 20,
        plateau_min_gap: int = 3,
        post_lift_settle: int = 8,
        min_motion_segment_len: int = 12,
        max_motion_gap: int = 3,
        strong_motion_threshold: float = 0.03,
        min_rotate_duration: int = 20,
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
        self.min_prev_motion_frames = min_prev_motion_frames
        self.next_move_window = next_move_window
        self.motion_lookback = motion_lookback
        self.plateau_min_gap = plateau_min_gap
        self.post_lift_settle = post_lift_settle
        self.min_motion_segment_len = min_motion_segment_len
        self.max_motion_gap = max_motion_gap
        self.strong_motion_threshold = strong_motion_threshold
        self.min_rotate_duration = min_rotate_duration
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
        c2, c3 = self._compute_rotate_scan_checkpoints(arm_data, lift_ready, total_steps)

        checkpoints = self._enforce_order([c0, c1, c2, c3], total_steps)
        return self.validate_checkpoints(checkpoints, total_steps)

    def _compute_rotate_scan_checkpoints(
        self,
        arm_data: dict,
        lift_ready: int,
        total_steps: int,
    ) -> tuple[int, int]:
        """c2/c3 from post-lift motion segments (not single-frame velocity spikes)."""
        post_lift_start = min(total_steps - 1, lift_ready + self.post_lift_settle)

        all_segments: list[dict] = []
        for arm in ("left", "right"):
            for seg in self._find_motion_segments(
                joint_vel=arm_data[arm]["joint_vel"],
                eef_vel=arm_data[arm]["eef_vel"],
                angular_vel=arm_data[arm]["angular_vel"],
                start=post_lift_start,
                total_steps=total_steps,
            ):
                seg = dict(seg)
                seg["arm"] = arm
                all_segments.append(seg)
        all_segments.sort(key=lambda x: x["start"])

        if len(all_segments) >= 1:
            c2, c3 = self._checkpoints_from_motion_segments(all_segments)
            if c3 - c2 < self.min_rotate_duration:
                later_segments = [
                    seg
                    for seg in all_segments
                    if seg["start"] > c2 and seg["length"] >= self.min_motion_segment_len
                ]
                if len(later_segments) >= 2:
                    c2_retry, c3_retry = self._checkpoints_from_motion_segments(later_segments)
                    if c3_retry - c2_retry >= self.min_rotate_duration:
                        c2, c3 = c2_retry, c3_retry
            return c2, c3

        return self._fallback_rotate_scan_checkpoints(arm_data, lift_ready, total_steps)

    def _checkpoints_from_motion_segments(self, segments: list[dict]) -> tuple[int, int]:
        object_seg = segments[0]
        object_arm = object_seg["arm"]
        scanner_arm = "right" if object_arm == "left" else "left"

        self.object_arm = object_arm
        self.scanner_arm = scanner_arm

        c2 = object_seg["start"]

        scanner_seg = None
        for seg in segments[1:]:
            if seg["arm"] == scanner_arm and seg["start"] >= object_seg["end"] + self.plateau_min_gap:
                scanner_seg = seg
                break

        if scanner_seg is not None:
            c3 = int((object_seg["end"] + scanner_seg["start"]) // 2)
        else:
            c3 = int(object_seg["end"])

        return c2, c3

    def _fallback_rotate_scan_checkpoints(
        self,
        arm_data: dict,
        lift_ready: int,
        total_steps: int,
    ) -> tuple[int, int]:
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

        c3 = self._find_boundary_between_two_motion_segments(
            prev_z=object_data["z"],
            prev_joint_vel=object_data["joint_vel"],
            prev_eef_vel=object_data["eef_vel"],
            next_joint_vel=scanner_data["joint_vel"],
            next_eef_vel=scanner_data["eef_vel"],
            start=c2,
            total_steps=total_steps,
            prev_angular_vel=object_data["angular_vel"],
            next_angular_vel=scanner_data["angular_vel"],
        )
        return c2, c3

    def _find_motion_segments(
        self,
        joint_vel: np.ndarray,
        eef_vel: Optional[np.ndarray],
        angular_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> list[dict]:
        """Find sustained post-lift motion segments; filter short noise / lift tail."""
        eef = eef_vel if eef_vel is not None else np.zeros(total_steps, dtype=np.float64)
        ang = angular_vel if angular_vel is not None else np.zeros(total_steps, dtype=np.float64)

        motion_mask = (
            (joint_vel[:total_steps] > self.joint_velocity_threshold)
            | (eef[:total_steps] > self.eef_velocity_threshold)
            | (ang[:total_steps] > self.angular_velocity_threshold)
        )

        segments: list[dict] = []
        in_segment = False
        seg_start: Optional[int] = None
        last_active: Optional[int] = None

        for t in range(start, total_steps):
            if motion_mask[t]:
                if not in_segment:
                    in_segment = True
                    seg_start = t
                last_active = t
            elif in_segment and last_active is not None:
                if t - last_active <= self.max_motion_gap:
                    continue

                seg_end = last_active + 1
                seg = self._motion_segment_dict(
                    seg_start, seg_end, joint_vel, eef, ang
                )
                if seg is not None:
                    segments.append(seg)

                in_segment = False
                seg_start = None
                last_active = None

        if in_segment and seg_start is not None and last_active is not None:
            seg_end = last_active + 1
            seg = self._motion_segment_dict(
                seg_start, seg_end, joint_vel, eef, ang
            )
            if seg is not None:
                segments.append(seg)

        return segments

    def _motion_segment_dict(
        self,
        seg_start: int,
        seg_end: int,
        joint_vel: np.ndarray,
        eef: np.ndarray,
        ang: np.ndarray,
    ) -> Optional[dict]:
        length = seg_end - seg_start
        joint_peak = float(np.max(joint_vel[seg_start:seg_end]))
        eef_peak = float(np.max(eef[seg_start:seg_end]))
        ang_peak = float(np.max(ang[seg_start:seg_end]))

        strong_enough = (
            joint_peak >= self.strong_motion_threshold
            or eef_peak >= self.eef_velocity_threshold * 2
            or ang_peak >= self.angular_velocity_threshold * 2
        )
        long_enough = length >= self.min_motion_segment_len

        if not (long_enough and strong_enough):
            return None

        return {
            "start": int(seg_start),
            "end": int(seg_end),
            "length": int(length),
            "joint_peak": joint_peak,
            "eef_peak": eef_peak,
            "angular_peak": ang_peak,
        }

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

    def _find_boundary_between_two_motion_segments(
        self,
        prev_z: Optional[np.ndarray],
        prev_joint_vel: np.ndarray,
        prev_eef_vel: Optional[np.ndarray],
        next_joint_vel: np.ndarray,
        next_eef_vel: Optional[np.ndarray],
        start: int,
        total_steps: int,
        prev_angular_vel: Optional[np.ndarray] = None,
        next_angular_vel: Optional[np.ndarray] = None,
    ) -> int:
        """Midpoint of static plateau between prev_arm stop and next_arm start (Rotate -> Scan)."""
        stable_window = max(1, self.stable_window)
        next_move_window = max(1, self.next_move_window)
        search_start = start + max(1, self.min_prev_motion_frames)

        prev_eef = (
            prev_eef_vel if prev_eef_vel is not None else np.zeros(total_steps, dtype=np.float64)
        )
        next_eef = (
            next_eef_vel if next_eef_vel is not None else np.zeros(total_steps, dtype=np.float64)
        )

        prev_stop: Optional[int] = None
        for t in range(search_start, max(search_start, total_steps - stable_window)):
            prev_joint_stop = (
                float(np.max(prev_joint_vel[t : t + stable_window])) <= self.joint_velocity_threshold
            )
            prev_eef_stop = float(np.max(prev_eef[t : t + stable_window])) <= self.eef_velocity_threshold
            prev_stop_ok = prev_joint_stop and prev_eef_stop

            if prev_z is not None:
                z_slice = np.asarray(prev_z[:total_steps], dtype=np.float64)[t : t + stable_window]
                prev_z_stable = float(np.max(z_slice) - np.min(z_slice)) <= self.z_stable_threshold
                prev_stop_ok = prev_stop_ok and prev_z_stable

            if prev_angular_vel is not None:
                prev_ang_stop = (
                    float(np.max(prev_angular_vel[t : t + stable_window]))
                    <= self.angular_velocity_threshold
                )
                prev_stop_ok = prev_stop_ok and prev_ang_stop

            lookback_start = max(start, t - self.motion_lookback)
            prev_had_motion = (
                float(np.max(prev_joint_vel[lookback_start:t])) > self.joint_velocity_threshold
                or float(np.max(prev_eef[lookback_start:t])) > self.eef_velocity_threshold
            )
            if prev_angular_vel is not None:
                prev_had_motion = prev_had_motion or (
                    float(np.max(prev_angular_vel[lookback_start:t])) > self.angular_velocity_threshold
                )

            if prev_stop_ok and prev_had_motion:
                prev_stop = int(t)
                break

        if prev_stop is None:
            return int(search_start)

        next_start: Optional[int] = None
        for t in range(prev_stop + self.plateau_min_gap, max(prev_stop + 1, total_steps - next_move_window)):
            next_joint_move = (
                float(np.max(next_joint_vel[t : t + next_move_window])) > self.joint_velocity_threshold
            )
            next_eef_move = float(np.max(next_eef[t : t + next_move_window])) > self.eef_velocity_threshold
            next_move_ok = next_joint_move or next_eef_move

            if next_angular_vel is not None:
                next_ang_move = (
                    float(np.max(next_angular_vel[t : t + next_move_window]))
                    > self.angular_velocity_threshold
                )
                next_move_ok = next_move_ok or next_ang_move

            if next_move_ok:
                next_start = int(t)
                break

        if next_start is None:
            return int(prev_stop)

        if next_start - prev_stop >= self.plateau_min_gap:
            return int((prev_stop + next_start) // 2)
        return int(prev_stop)

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
