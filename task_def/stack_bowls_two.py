"""Rule-based phase segmentation for the stack_bowls_two task.

Two pick-and-place cycles for nested bowl stacking. Logic matches
stack_blocks_two; bowl order is fixed by cycle index, active arm is data-driven.
"""

from __future__ import annotations

from .stack_blocks_two import StackBlocksTwoProcessor
from .stack_bowls_three import PHASE_DESCRIPTIONS as _THREE_BOWL_DESCRIPTIONS


BOWL_INDICES = ("first", "second")

PHASE_DESCRIPTIONS = list(_THREE_BOWL_DESCRIPTIONS[:8])


class StackBowlsTwoProcessor(StackBlocksTwoProcessor):
    """Eight-phase splitter for two-bowl nested stacking."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bowl_indices = list(BOWL_INDICES)
        self.block_colors = list(BOWL_INDICES)

    def get_subtask_descriptions(self) -> list[str]:
        return list(PHASE_DESCRIPTIONS)
