"""Rule-based phase segmentation for the beat_block_hammer task.

Phases:
0. Approach the hammer.
1. Grasp the hammer.
2. Move the hammer above the block.
3. Strike the block downward with the hammer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base_task import BaseTaskProcessor
from .trajectory_analyzer import TrajectoryAnalyzer


class BeatBlockHammerProcessor(BaseTaskProcessor):
    """Four-phase splitter using open-gripper low-Z approach and post-grasp Z peak."""

    def __init__(
        self,
        close_gripper_threshold: float = 0.05,
        gripper_open_value: float = 1.0,
        gripper_open_atol: float = 1e-4,
        joint_velocity_threshold: float = 0.01,
        eef_velocity_threshold: float = 0.005,
        z_eps: float = 0.005,
        stable_velocity_threshold: float = 0.01,
        stable_window: int = 5,
        z_peak_eps: float = 0.005,
        down_delta_threshold: float = 0.005,
        down_window: int = 5,
    ):
        self.analyzer = TrajectoryAnalyzer(velocity_threshold=joint_velocity_threshold)
        self.close_gripper_threshold = close_gripper_threshold
        self.gripper_open_value = gripper_open_value
        self.gripper_open_atol = gripper_open_atol
        self.joint_velocity_threshold = joint_velocity_threshold
        self.eef_velocity_threshold = eef_velocity_threshold
        self.z_eps = z_eps
        self.stable_velocity_threshold = stable_velocity_threshold
        self.stable_window = stable_window
        self.z_peak_eps = z_peak_eps
        self.down_delta_threshold = down_delta_threshold
        self.down_window = down_window
        self.active_side = "left"

    def get_phase_checkpoints(
        self,
        hdf5_data,
        active_side: Optional[str] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
        **_,
    ) -> list[int]:
        """Return [c0, c1, c2] for approach, grasp, move, and strike phases."""
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
        joint_velocity = self._get_active_joint_velocity(hdf5_data, active_side, total_steps)

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
            close_done = min(total_steps - 1, max(c0 + 1, int(total_steps * 0.4)))

        c1 = self._find_motion_start(
            start=close_done,
            active_xyz=active_xyz,
            joint_velocity=joint_velocity,
        )
        if c1 is None:
            c1 = close_done
        if c1 <= c0:
            c1 = min(total_steps - 1, c0 + max(1, (total_steps - c0) // 5))

        c2 = self._find_strike_start_by_z_peak(
            z=active_z,
            start=c1,
            total_steps=total_steps,
        )
        if c2 <= c1:
            c2 = min(total_steps - 1, c1 + max(1, (total_steps - c1) // 2))

        return self.validate_checkpoints([c0, c1, c2], total_steps)

    def _find_stable_approach_end(
        self,
        gripper: np.ndarray,
        z: Optional[np.ndarray],
        joint_velocity: np.ndarray,
    ) -> int:
        """
        Find approach end while the gripper is still fully open.

        Conditions:
        - gripper is 1.0, using isclose for float robustness
        - active EEF Z is on the lowest pre-closing platform
        - active arm joint velocity is near zero
        """
        total_steps = len(gripper)
        window = max(1, self.stable_window)
        open_mask = np.isclose(
            gripper,
            self.gripper_open_value,
            atol=self.gripper_open_atol,
        )
        closing_candidates = np.where(~open_mask)[0]
        close_start = int(closing_candidates[0]) if len(closing_candidates) > 0 else total_steps
        search_end = min(total_steps, max(close_start, window + 1))

        if z is None or len(z) < search_end:
            if close_start > 1:
                return int(close_start - 1)
            return max(1, int(total_steps * 0.25))

        z_before_close = np.asarray(z[:search_end], dtype=np.float64)
        min_z = float(np.min(z_before_close))

        for idx in range(0, max(1, search_end - window + 1)):
            end = idx + window
            g_ok = np.all(open_mask[idx:end])
            z_ok = np.all(z[idx:end] <= min_z + self.z_eps)
            v_ok = np.all(joint_velocity[idx:end] <= self.stable_velocity_threshold)
            if g_ok and z_ok and v_ok:
                return int(idx)

        return int(np.argmin(z_before_close))

    def _find_strike_start_by_z_peak(
        self,
        z: Optional[np.ndarray],
        start: int,
        total_steps: int,
    ) -> int:
        """Find the post-grasp Z peak immediately before the downward strike."""
        if z is None or start >= total_steps - self.down_window - 1:
            return max(1, min(total_steps - 1, start))

        z = np.asarray(z[:total_steps], dtype=np.float64)
        search_start = max(0, start)
        z_segment = z[search_start:total_steps]
        if len(z_segment) == 0:
            return max(1, min(total_steps - 1, start))

        max_z = float(np.max(z_segment))
        peak_candidates = np.where(z_segment >= max_z - self.z_peak_eps)[0]
        if len(peak_candidates) == 0:
            return int(search_start + np.argmax(z_segment))

        for idx in peak_candidates:
            t = search_start + int(idx)
            if t + self.down_window < total_steps:
                if z[t] - z[t + self.down_window] > self.down_delta_threshold:
                    return int(t)

        return int(search_start + peak_candidates[-1])

    def _first_consecutive_leq(
        self,
        series: np.ndarray,
        threshold: float,
        start: int,
    ) -> Optional[int]:
        """Find first index where ``stable_window`` values stay below threshold."""
        n = len(series)
        window = max(1, self.stable_window)
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
            "Approach the hammer.",
            "Grasp the hammer.",
            "Move the hammer above the block.",
            "Strike the block downward with the hammer.",
        ]
