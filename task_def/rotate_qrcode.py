"""Rule-based phase segmentation for the rotate_qrcode task.

Standard single-object manipulation: approach, grasp, rotate sign toward robot,
then place. Active arm is inferred from the first valid close-open gripper event.
"""

from __future__ import annotations

from .move_stapler_pad import MoveStaplerPadProcessor


class RotateQrcodeProcessor(MoveStaplerPadProcessor):
    """Four-phase splitter: approach, grasp, rotate QR sign, place down."""

    def get_subtask_descriptions(self) -> list[str]:
        return [
            "Approach the QR code sign.",
            "Grasp the QR code sign.",
            "Rotate the QR code sign to face the robot.",
            "Place the QR code sign down.",
        ]
