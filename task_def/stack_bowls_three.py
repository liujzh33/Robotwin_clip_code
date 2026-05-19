"""Rule-based phase segmentation for the stack_bowls_three task.

Three pick-and-place cycles for nested bowl stacking. Logic matches
stack_blocks_three; bowl order is fixed by cycle index, active arm is data-driven.
"""

from __future__ import annotations

from .stack_blocks_three import StackBlocksThreeProcessor


BOWL_INDICES = ("first", "second", "third")

PHASE_DESCRIPTIONS = [
    "Approach the first bowl.",
    "Grasp the first bowl.",
    "Move the first bowl to the center.",
    "Place the first bowl at the center.",
    "Approach the second bowl.",
    "Grasp the second bowl.",
    "Move the second bowl above the first bowl.",
    "Place the second bowl on the first bowl.",
    "Approach the third bowl.",
    "Grasp the third bowl.",
    "Move the third bowl above the second bowl.",
    "Place the third bowl on the second bowl.",
]


class StackBowlsThreeProcessor(StackBlocksThreeProcessor):
    """Twelve-phase splitter for three-bowl nested stacking."""

    NUM_CYCLES = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bowl_indices = list(BOWL_INDICES)
        self.block_colors = list(BOWL_INDICES)

    def get_subtask_descriptions(self) -> list[str]:
        return list(PHASE_DESCRIPTIONS)
