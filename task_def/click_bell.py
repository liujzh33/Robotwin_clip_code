"""Rule-based phase segmentation for the click_bell task."""

from __future__ import annotations

from .click_alarmclock import ClickAlarmclockProcessor


class ClickBellProcessor(ClickAlarmclockProcessor):
    """Three-phase splitter for moving above, closing, and pressing the bell."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Move above the bell's top center.",
            "Close the gripper.",
            "Press the bell downward.",
        ]
