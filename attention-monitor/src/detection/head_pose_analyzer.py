from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class HeadPose:
    yaw: float    # normalized; positive = face turned right in image
    pitch: float  # normalized; positive = face tilted down


class HeadPoseAnalyzer:
    """
    Estimates head orientation from the displacement of the nose tip
    relative to the geometric center of the face.

    All coordinates must be in the same unit (pixels or normalized [0,1]).
    The result is normalized by face width (yaw) and face height (pitch),
    making it resolution-independent.
    """

    YAW_THRESHOLD: float = 0.12
    PITCH_THRESHOLD: float = 0.07

    @staticmethod
    def compute(
        nose: Tuple[float, float],
        left_cheek: Tuple[float, float],
        right_cheek: Tuple[float, float],
        forehead: Tuple[float, float],
        chin: Tuple[float, float],
    ) -> HeadPose:
        face_width = abs(right_cheek[0] - left_cheek[0])
        face_height = abs(chin[1] - forehead[1])

        if face_width < 1e-6 or face_height < 1e-6:
            return HeadPose(yaw=0.0, pitch=0.0)

        center_x = (left_cheek[0] + right_cheek[0]) / 2.0
        center_y = (forehead[1] + chin[1]) / 2.0

        yaw = (nose[0] - center_x) / face_width
        pitch = (nose[1] - center_y) / face_height

        return HeadPose(yaw=yaw, pitch=pitch)

    @staticmethod
    def is_looking_away(
        pose: HeadPose,
        yaw_threshold: float = 0.12,
        pitch_threshold: float = 0.07,
    ) -> bool:
        return abs(pose.yaw) > yaw_threshold or abs(pose.pitch) > pitch_threshold
