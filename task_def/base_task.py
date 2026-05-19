"""Base interface for rule-based Robotwin subtask segmentation."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTaskProcessor(ABC):
    """
    Every task processor returns ordered phase checkpoints and language labels.

    If checkpoints are [c0, c1], the phases are:
    - phase 0: [0, c0)
    - phase 1: [c0, c1)
    - phase 2: [c1, total_steps)
    """

    @abstractmethod
    def get_phase_checkpoints(self, hdf5_data, **kwargs) -> list[int]:
        """Return ordered checkpoint frame indices for a single episode."""
        raise NotImplementedError

    @abstractmethod
    def get_subtask_descriptions(self) -> list[str]:
        """Return one language label for each phase."""
        raise NotImplementedError

    def get_subtask_descriptions_for_phases(self, num_phases: int) -> list[str]:
        """Return exactly ``num_phases`` labels, padding or truncating if needed."""
        descriptions = list(self.get_subtask_descriptions())
        while len(descriptions) < num_phases:
            descriptions.append("Complete the next step.")
        return descriptions[:num_phases]

    def validate_checkpoints(self, checkpoints: list[int], total_steps: int) -> list[int]:
        """Filter, deduplicate, and sort checkpoints so they are valid frame indices."""
        valid = [int(cp) for cp in checkpoints if 0 < int(cp) < total_steps]
        return sorted(set(valid))
