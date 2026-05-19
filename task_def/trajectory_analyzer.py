"""Shared trajectory utilities used by task rules and plotting scripts."""

from typing import Optional, Tuple

import numpy as np


class TrajectoryAnalyzer:
    """Extract gripper, end-effector, and velocity signals from HDF5 episodes."""

    def __init__(self, gripper_threshold: float = 0.5, velocity_threshold: float = 0.01):
        self.gripper_threshold = gripper_threshold
        self.velocity_threshold = velocity_threshold

    def extract_qpos(self, hdf5_data) -> np.ndarray:
        """Read qpos from either processed_data or a simpler HDF5 layout."""
        if "observations/qpos" in hdf5_data:
            return hdf5_data["observations/qpos"][()]
        if "qpos" in hdf5_data:
            return hdf5_data["qpos"][()]
        raise ValueError("Cannot find qpos data in HDF5 file.")

    def extract_gripper_states(self, hdf5_data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract left and right gripper trajectories.

        Expected processed qpos layout:
        [left_arm..., left_gripper, right_arm..., right_gripper]
        """
        qpos = self.extract_qpos(hdf5_data)

        if "observations/left_arm_dim" in hdf5_data and "observations/right_arm_dim" in hdf5_data:
            left_arm_dim = int(hdf5_data["observations/left_arm_dim"][0])
            right_arm_dim = int(hdf5_data["observations/right_arm_dim"][0])
            left_gripper_idx = left_arm_dim
            right_gripper_idx = left_arm_dim + 1 + right_arm_dim
        else:
            left_gripper_idx = 6
            right_gripper_idx = 13

        if qpos.shape[1] <= max(left_gripper_idx, right_gripper_idx):
            raise ValueError(f"qpos has shape {qpos.shape}, cannot extract both grippers.")

        return qpos[:, left_gripper_idx], qpos[:, right_gripper_idx]

    def compute_velocity(
        self,
        qpos: np.ndarray,
        arm_indices: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Compute per-frame L2 joint velocity for the selected arm joints."""
        if arm_indices is None:
            arm_indices = (0, min(6, qpos.shape[1]))

        start_idx, end_idx = arm_indices
        arm_qpos = qpos[:, start_idx:end_idx]
        velocity = np.linalg.norm(np.diff(arm_qpos, axis=0), axis=1)
        return np.insert(velocity, 0, 0.0)

    def extract_left_right_eef_xyz(
        self,
        hdf5_data,
        total_steps: Optional[int] = None,
        external_eef_xyz_left: Optional[np.ndarray] = None,
        external_eef_xyz_right: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return left/right end-effector XYZ if available, otherwise qpos approximations."""
        if total_steps is None:
            total_steps = len(self.extract_qpos(hdf5_data))

        left_xyz = external_eef_xyz_left
        right_xyz = external_eef_xyz_right

        if left_xyz is None and "endpose/left_endpose" in hdf5_data:
            left_xyz = hdf5_data["endpose/left_endpose"][()][:total_steps, :3]
        if right_xyz is None and "endpose/right_endpose" in hdf5_data:
            right_xyz = hdf5_data["endpose/right_endpose"][()][:total_steps, :3]

        if (left_xyz is None or right_xyz is None) and ("observations/eef_pos" in hdf5_data):
            eef = hdf5_data["observations/eef_pos"][()]
            if eef.ndim == 2 and eef.shape[1] >= 3:
                left_xyz = left_xyz if left_xyz is not None else eef[:total_steps, :3]

        if left_xyz is None or right_xyz is None:
            qpos = self.extract_qpos(hdf5_data)
            if qpos.shape[1] >= 13:
                left_xyz = left_xyz if left_xyz is not None else qpos[:total_steps, 0:3]
                right_xyz = right_xyz if right_xyz is not None else qpos[:total_steps, 7:10]

        return self._align_xyz(left_xyz, total_steps), self._align_xyz(right_xyz, total_steps)

    def active_side_from_grippers(self, left_gripper: np.ndarray, right_gripper: np.ndarray) -> str:
        """Choose the arm whose gripper changes more during the episode."""
        left_delta = float(np.max(left_gripper) - np.min(left_gripper))
        right_delta = float(np.max(right_gripper) - np.min(right_gripper))
        return "left" if left_delta >= right_delta else "right"

    @staticmethod
    def _align_xyz(xyz: Optional[np.ndarray], total_steps: int) -> Optional[np.ndarray]:
        if xyz is None:
            return None
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            return None
        if len(xyz) >= total_steps:
            return xyz[:total_steps, :3]
        if len(xyz) == 0:
            return None
        pad = np.repeat(xyz[-1:, :3], total_steps - len(xyz), axis=0)
        return np.concatenate([xyz[:, :3], pad], axis=0)
