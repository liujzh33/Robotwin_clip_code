"""Rule-based phase segmentation for the place_mouse_pad task.

Standard single-object pick-and-place onto a gray mat. Active arm is inferred
from the first valid close-open gripper event across both arms.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class PlaceMousePadProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move above gray mat, place."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the mouse.",
            "Grasp the mouse.",
            "Move the mouse above the gray mat.",
            "Place the mouse on the gray mat.",
        ]
