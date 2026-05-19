"""Rule-based phase segmentation for the place_phone_stand task.

Standard single-object pick-and-place onto a phone stand with orientation semantics
in phase 2. Active arm is inferred from the first valid close-open gripper event.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class PlacePhoneStandProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move-and-orient, place on phone stand."""

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
            "Approach the phone.",
            "Grasp the phone.",
            "Move and orient the phone above the phone stand.",
            "Place the phone onto the phone stand.",
        ]
