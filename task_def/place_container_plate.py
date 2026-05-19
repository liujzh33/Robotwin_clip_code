"""Rule-based phase segmentation for the place_container_plate task.

Single-object pick-and-place: grasp the ceramic bowl and place it onto the plate.
Active arm is inferred from the first valid close-open gripper event across both arms.
"""

from __future__ import annotations

from .place_a2b_left import PlaceA2BLeftProcessor


class PlaceContainerPlateProcessor(PlaceA2BLeftProcessor):
    """Four-phase splitter: approach, grasp, move above plate, release."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the ceramic bowl.",
            "Grasp the ceramic bowl.",
            "Move the ceramic bowl above the plate.",
            "Release the ceramic bowl onto the plate.",
        ]
