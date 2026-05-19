"""Rule-based phase segmentation for the place_object_basket task.

object_arm: pick and place the movable object (first full close-open event).
basket_arm: approach handle, grasp, and lift the basket after object release.
"""

from __future__ import annotations

from typing import Optional

from .place_can_basket import PlaceCanBasketProcessor


class PlaceObjectBasketProcessor(PlaceCanBasketProcessor):
    """Seven-phase splitter: place object, then grasp handle and lift basket."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.object_arm: Optional[str] = None

    def get_phase_checkpoints(self, hdf5_data, active_side=None, **kwargs) -> list[int]:
        checkpoints = super().get_phase_checkpoints(hdf5_data, active_side=active_side, **kwargs)
        self.object_arm = self.can_arm
        return checkpoints

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the object.",
            "Grasp the object.",
            "Move the object above the basket.",
            "Release the object into the basket.",
            "Approach the basket handle.",
            "Grasp the basket handle.",
            "Lift the basket upward.",
        ]
