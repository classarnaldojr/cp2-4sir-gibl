from __future__ import annotations

import pytest

from src.detection.head_pose_analyzer import HeadPose, HeadPoseAnalyzer


def _frontal() -> HeadPose:
    """Landmarks for a face looking straight at the camera."""
    return HeadPoseAnalyzer.compute(
        nose        = (5.0, 5.0),
        left_cheek  = (0.0, 5.0),
        right_cheek = (10.0, 5.0),
        forehead    = (5.0, 0.0),
        chin        = (5.0, 10.0),
    )


class TestHeadPoseCompute:
    def test_frontal_yaw_near_zero(self):
        assert abs(_frontal().yaw) < 0.01

    def test_frontal_pitch_near_zero(self):
        assert abs(_frontal().pitch) < 0.01

    def test_turning_right_positive_yaw(self):
        pose = HeadPoseAnalyzer.compute(
            nose=(7.0, 5.0),
            left_cheek=(0.0, 5.0), right_cheek=(10.0, 5.0),
            forehead=(5.0, 0.0), chin=(5.0, 10.0),
        )
        assert pose.yaw > 0

    def test_turning_left_negative_yaw(self):
        pose = HeadPoseAnalyzer.compute(
            nose=(3.0, 5.0),
            left_cheek=(0.0, 5.0), right_cheek=(10.0, 5.0),
            forehead=(5.0, 0.0), chin=(5.0, 10.0),
        )
        assert pose.yaw < 0

    def test_looking_down_positive_pitch(self):
        pose = HeadPoseAnalyzer.compute(
            nose=(5.0, 7.0),
            left_cheek=(0.0, 5.0), right_cheek=(10.0, 5.0),
            forehead=(5.0, 0.0), chin=(5.0, 10.0),
        )
        assert pose.pitch > 0

    def test_looking_up_negative_pitch(self):
        pose = HeadPoseAnalyzer.compute(
            nose=(5.0, 3.0),
            left_cheek=(0.0, 5.0), right_cheek=(10.0, 5.0),
            forehead=(5.0, 0.0), chin=(5.0, 10.0),
        )
        assert pose.pitch < 0

    def test_zero_face_width_returns_zero_pose(self):
        pose = HeadPoseAnalyzer.compute(
            nose=(5.0, 5.0),
            left_cheek=(5.0, 5.0), right_cheek=(5.0, 5.0),
            forehead=(5.0, 0.0), chin=(5.0, 10.0),
        )
        assert pose.yaw == pytest.approx(0.0)
        assert pose.pitch == pytest.approx(0.0)

    def test_zero_face_height_returns_zero_pose(self):
        pose = HeadPoseAnalyzer.compute(
            nose=(5.0, 5.0),
            left_cheek=(0.0, 5.0), right_cheek=(10.0, 5.0),
            forehead=(5.0, 5.0), chin=(5.0, 5.0),
        )
        assert pose.yaw == pytest.approx(0.0)
        assert pose.pitch == pytest.approx(0.0)

    def test_known_yaw_value(self):
        # nose 2.5 right of center on a face of width 10 → yaw = 2.5/10 = 0.25
        pose = HeadPoseAnalyzer.compute(
            nose=(7.5, 5.0),
            left_cheek=(0.0, 5.0), right_cheek=(10.0, 5.0),
            forehead=(5.0, 0.0), chin=(5.0, 10.0),
        )
        assert pose.yaw == pytest.approx(0.25, abs=1e-9)


class TestHeadPoseIsLookingAway:
    def test_frontal_not_looking_away(self):
        assert HeadPoseAnalyzer.is_looking_away(_frontal()) is False

    def test_large_yaw_is_looking_away(self):
        assert HeadPoseAnalyzer.is_looking_away(HeadPose(yaw=0.25, pitch=0.0)) is True

    def test_large_pitch_is_looking_away(self):
        assert HeadPoseAnalyzer.is_looking_away(HeadPose(yaw=0.0, pitch=0.15)) is True

    def test_custom_threshold(self):
        pose = HeadPose(yaw=0.10, pitch=0.0)
        assert HeadPoseAnalyzer.is_looking_away(pose, yaw_threshold=0.05) is True
        assert HeadPoseAnalyzer.is_looking_away(pose, yaw_threshold=0.15) is False
