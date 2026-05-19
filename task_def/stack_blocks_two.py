"""Rule-based phase segmentation for the stack_blocks_two task.

Two pick-and-place cycles (red base, green top). Active arm is inferred per
cycle from the next valid close-open gripper event.
"""

from __future__ import annotations

from .stack_blocks_three import PHASE_DESCRIPTIONS as _THREE_PHASE_DESCRIPTIONS
from .stack_blocks_three import StackBlocksThreeProcessor


BLOCK_COLORS = ("red", "green")

PHASE_DESCRIPTIONS = list(_THREE_PHASE_DESCRIPTIONS[:8])


class StackBlocksTwoProcessor(StackBlocksThreeProcessor):
    """Eight-phase splitter for two-block vertical stacking."""

    NUM_CYCLES = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block_colors = list(BLOCK_COLORS)

    def get_subtask_descriptions(self) -> list[str]:
        return list(PHASE_DESCRIPTIONS)
