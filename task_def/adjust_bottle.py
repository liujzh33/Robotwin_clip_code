"""Rule-based phase segmentation for the adjust_bottle task.

Phases:
0. Approach the bottle.
1. Close the gripper and grasp the bottle.
2. Move the bottle back to its initial/upright state.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class AdjustBottleProcessor(BaseTaskProcessor):
    """Three-phase adjust_bottle splitter based on gripper closure and motion onset."""

    def __init__(
        self,
        open_gripper_threshold: float = 0.98,
        close_gripper_threshold: float = 0.05,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        consecutive_frames: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.open_gripper_threshold = open_gripper_threshold
        self.close_gripper_threshold = close_gripper_threshold
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.stable_velocity_threshold = stable_velocity_threshold
        self.consecutive_frames = consecutive_frames
        self.active_side = "left"

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: Optional[str] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
        **_,
    ) -> list[int]:
        """
        Return [c0, c1].

        c0: approach end while the gripper is still open, selected from the
            low-Z, low-velocity stable platform before gripper closing starts.
        c1: after the gripper is closed, first clear motion onset; fallback to close_done.
        """
        left_gripper, right_gripper = self.analyzer.extract_gripper_states(hdf5_data)
        total_steps = len(left_gripper)

        if active_side is None:
            active_side = self.analyzer.active_side_from_grippers(left_gripper, right_gripper)
        self.active_side = active_side

        gripper = left_gripper if active_side == "left" else right_gripper

        active_xyz = self._get_active_eef_xyz(
            hdf5_data,
            active_side=active_side,
            total_steps=total_steps,
            external_eef_xyz_left=external_eef_xyz_left,
            external_eef_xyz_right=external_eef_xyz_right,
        )
        active_z = active_xyz[:, 2] if active_xyz is not None else None
        joint_velocity = self._get_active_joint_velocity(
            hdf5_data,
            active_side=active_side,
            total_steps=total_steps,
        )

        c0 = self._find_stable_approach_end(
            gripper=gripper,
            z=active_z,
            joint_velocity=joint_velocity,
        )

        close_done = self._first_consecutive_leq(
            gripper,
            threshold=self.close_gripper_threshold,
            start=c0,
        )
        if close_done is None:
            close_done = min(total_steps - 1, max(c0 + 1, int(total_steps * 0.5)))

        c1 = self._find_motion_start(
            start=close_done,
            active_xyz=active_xyz,
            joint_velocity=joint_velocity,
        )
        if c1 is None:
            c1 = close_done

        if c1 <= c0:
            c1 = min(total_steps - 1, c0 + max(1, (total_steps - c0) // 4))

        return self.validate_checkpoints([c0, c1], total_steps)

    def _find_stable_approach_end(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
    ) -> int:
        """Find the open-gripper, low-Z, low-velocity platform before closing starts."""
        total_steps = len(gripper)
        window = max(1, self.consecutive_frames)

        closing_candidates = np.where(gripper < self.open_gripper_threshold)[0]
        close_start = int(closing_candidates[0]) if len(closing_candidates) > 0 else total_steps
        search_end = min(total_steps, max(close_start, window + 1))

        if z is None or len(z) < search_end:
            return max(1, min(total_steps - 1, close_start - 1 if close_start > 1 else int(total_steps * 0.3)))

        z_before_close = np.asarray(z[:search_end], dtype=np.float64)
        min_z = float(np.min(z_before_close))

        for idx in range(0, max(1, search_end - window + 1)):
            end = idx + window
            g_ok = np.all(gripper[idx:end] >= self.open_gripper_threshold)
            z_ok = np.all(z[idx:end] <= min_z + self.z_eps)
            v_ok = np.all(joint_velocity[idx:end] <= self.stable_velocity_threshold)
            if g_ok and z_ok and v_ok:
                return int(idx)

        return int(np.argmin(z_before_close))

    def _first_consecutive_leq(
        self,
        series: np.ndarray,
        threshold: float,
        start: int,
    ) -> Optional[int]:
        """Find first index where ``consecutive_frames`` values stay below threshold."""
        n = len(series)
        window = max(1, self.consecutive_frames)
        for idx in range(max(0, start), max(0, n - window + 1)):
            if np.all(series[idx : idx + window] <= threshold):
                return int(idx)
        candidates = np.where(series[max(0, start) :] <= threshold)[0]
        if len(candidates) > 0:
            return int(max(0, start) + candidates[0])
        return None

    def _get_active_eef_xyz(
        self,
        hdf5_data,
        active_side: str,
        total_steps: int,
        external_eef_xyz_left: Optional[np.ndarray],
        external_eef_xyz_right: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Get active arm end-effector XYZ from raw endpose when available."""
        left_xyz, right_xyz = self.analyzer.extract_left_right_eef_xyz(
            hdf5_data,
            total_steps=total_steps,
            external_eef_xyz_left=external_eef_xyz_left,
            external_eef_xyz_right=external_eef_xyz_right,
        )
        return left_xyz if active_side == "left" else right_xyz

    def _get_active_joint_velocity(
        self,
        hdf5_data,
        active_side: str,
        total_steps: int,
    ) -> np.ndarray:
        qpos = self.analyzer.extract_qpos(hdf5_data)
        arm_indices = (0, 6) if active_side == "left" else (7, 13)
        return self.analyzer.compute_velocity(qpos[:total_steps], arm_indices=arm_indices)

    def _find_motion_start(
        self,
        start: int,
        active_xyz: Optional[np.ndarray],
        joint_velocity: np.ndarray,
    ) -> Optional[int]:
        """Use end-effector velocity first, then joint velocity, to detect post-grasp motion."""
        if active_xyz is not None:
            eef_velocity = np.linalg.norm(np.diff(active_xyz, axis=0), axis=1)
            eef_velocity = np.insert(eef_velocity, 0, 0.0)
            hit = self._first_velocity_above(eef_velocity, self.eef_velocity_threshold, start)
            if hit is not None:
                return hit

        return self._first_velocity_above(joint_velocity, self.joint_velocity_threshold, start)

    @staticmethod
    def _first_velocity_above(
        velocity: np.ndarray,
        threshold: float,
        start: int,
    ) -> Optional[int]:
        candidates = np.where(velocity[max(0, start) :] > threshold)[0]
        if len(candidates) == 0:
            return None
        return int(max(0, start) + candidates[0])

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the bottle.",
            "Close the gripper and grasp the bottle.",
            "Restore the bottle to its initial state.",
        ]
