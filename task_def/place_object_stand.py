"""Rule-based phase segmentation for the place_object_stand task.

Standard single-object pick-and-place onto a display stand.
Active arm is inferred from the first valid close-open gripper event across both arms.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class PlaceObjectStandProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move above display stand, place."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the object.",
            "Grasp the object.",
            "Move the object above the display stand.",
            "Place the object on the display stand.",
        ]
