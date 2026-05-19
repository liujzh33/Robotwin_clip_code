"""Rule-based phase segmentation for the place_dual_shoes task.

Dual-arm cooperative: grasp two shoes, then place them into the shoe box sequentially.
first_arm is the arm that opens its gripper first after grasping.
"""

from __future__ import annotations

from .place_burger_fries import PlaceBurgerFriesProcessor


class PlaceDualShoesProcessor(PlaceBurgerFriesProcessor):
    """Six-phase splitter: dual approach/grasp, sequential place into shoe box."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach both shoes with both arms.",
            "Grasp both shoes with both grippers.",
            "Move the first shoe above the shoe box.",
            "Release the first shoe into the shoe box.",
            "Move the second shoe above the shoe box.",
            "Release the second shoe into the shoe box.",
        ]
