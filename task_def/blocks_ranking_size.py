"""Rule-based phase segmentation for the blocks_ranking_size task.

This task uses the same event-based splitter as blocks_ranking_rgb:
for each object, the active arm is inferred from the next valid close-open
gripper event. The object order is fixed by task semantics:
small block -> medium block -> large block.
"""

from __future__ import annotations

from .blocks_ranking_rgb import BlocksRankingRgbProcessor


class BlocksRankingSizeProcessor(BlocksRankingRgbProcessor):
    """Thirteen-phase splitter for small, medium, large block ranking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colors = ["small", "medium", "large"]

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the small block.",
            "Grasp the small block.",
            "Move the small block to the far right.",
            "Place the small block on the far right.",
            "Approach the medium block.",
            "Grasp the medium block.",
            "Move the medium block to the middle.",
            "Place the medium block in the middle.",
            "Approach the large block.",
            "Grasp the large block.",
            "Move the large block to the far left.",
            "Place the large block on the far left.",
            "Return to the initial position.",
        ]
