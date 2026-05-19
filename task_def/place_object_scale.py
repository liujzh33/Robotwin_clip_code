"""Rule-based phase segmentation for the place_object_scale task.

Standard single-object pick-and-place onto an electronic scale platform.
Active arm is inferred from the first valid close-open gripper event across both arms.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class PlaceObjectScaleProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move above scale platform, place."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the object.",
            "Grasp the object.",
            "Move the object above the scale platform.",
            "Place the object on the scale platform.",
        ]
