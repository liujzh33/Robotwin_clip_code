"""Rule-based phase segmentation for the place_fan task.

Standard single-object pick-and-place onto a blue mat with orientation semantics in
phase 2. Active arm is inferred from the first valid close-open gripper event.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class PlaceFanProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move-and-orient, place on blue mat."""

    def __init__(
        self,
        close_threshold: float = 0.05,
        open_done_threshold: float = 0.85,
        **kwargs,
    ):
        super().__init__(
            close_threshold=close_threshold,
            open_done_threshold=open_done_threshold,
            **kwargs,
        )

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the fan.",
            "Grasp the fan.",
            "Move and orient the fan on the blue mat facing the robot.",
            "Place the fan on the blue mat.",
        ]
