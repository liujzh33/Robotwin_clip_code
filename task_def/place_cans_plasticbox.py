"""Rule-based phase segmentation for the place_cans_plasticbox task.

Dual-arm cooperative: grasp two cans, then place them into the plastic box sequentially.
first_arm is the arm that opens its gripper first after grasping.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .place_burger_fries import PlaceBurgerFriesProcessor


class PlaceCansPlasticboxProcessor(PlaceBurgerFriesProcessor):
    """Six-phase splitter: dual approach/grasp, sequential place into plastic box."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach both cans with both arms.",
            "Grasp both cans with both grippers.",
            "Move the first can above the plastic box.",
            "Release the first can into the plastic box.",
            "Move the second can above the plastic box.",
            "Release the second can into the plastic box.",
        ]
