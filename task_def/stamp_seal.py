"""Rule-based phase segmentation for the stamp_seal task.

Standard single-object pick-and-place onto a marked target area (e.g. Beige region).
Active arm is inferred from the first valid close-open gripper event across both arms.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class StampSealProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, move above marked area, place."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the stamp object.",
            "Grasp the stamp object.",
            "Move the stamp object above the marked area.",
            "Place the stamp object onto the marked area.",
        ]
