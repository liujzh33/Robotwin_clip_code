"""Rule-based phase segmentation for the place_shoe task.

Standard single-object pick-and-place onto a mat. Active arm is inferred
from the first valid close-open gripper event across both arms.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class PlaceShoeProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move above mat, place."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the shoe.",
            "Grasp the shoe.",
            "Move the shoe above the mat.",
            "Place the shoe on the mat.",
        ]
