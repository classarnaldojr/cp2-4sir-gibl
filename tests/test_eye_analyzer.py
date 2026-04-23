from __future__ import annotations

from typing import List, Tuple

import pytest

from src.detection.eye_analyzer import EyeAnalyzer


def make_eye(vertical_gap: float, horizontal_span: float = 6.0) -> List[Tuple[float, float]]:
    """
    Build a synthetic 6-point eye where both vertical distances equal
    vertical_gap and the horizontal span equals horizontal_span.

    Layout:
      p0(outer), p1(upper-inner), p2(upper-outer),
      p3(inner),  p4(lower-outer), p5(lower-inner)
    """
    half = horizontal_span / 2.0
    half_v = vertical_gap / 2.0
    return [
        (0.0,            0.0),
        (half - 1.0,    -half_v),
        (half + 1.0,    -half_v),
        (horizontal_span, 0.0),
        (half + 1.0,     half_v),
        (half - 1.0,     half_v),
    ]


class TestEyeAnalyzerCompute:
    def test_known_value(self):
        # Symmetric eye: two vertical distances of 2.0, horizontal of 4.0
        # EAR = (2 + 2) / (2 * 4) = 0.5
        pts = [
            (0.0, 0.0),
            (1.5, -1.0),
            (2.5, -1.0),
            (4.0,  0.0),
            (2.5,  1.0),
            (1.5,  1.0),
        ]
        assert EyeAnalyzer.compute_ear(pts) == pytest.approx(0.5, abs=1e-9)

    def test_open_eye_above_threshold(self):
        ear = EyeAnalyzer.compute_ear(make_eye(vertical_gap=3.0))
        assert ear > EyeAnalyzer.CLOSED_THRESHOLD

    def test_nearly_closed_eye_below_threshold(self):
        ear = EyeAnalyzer.compute_ear(make_eye(vertical_gap=0.2))
        assert ear < EyeAnalyzer.CLOSED_THRESHOLD

    def test_fully_closed_eye_is_zero(self):
        ear = EyeAnalyzer.compute_ear(make_eye(vertical_gap=0.0))
        assert ear == pytest.approx(0.0, abs=1e-9)

    def test_zero_horizontal_span_returns_zero(self):
        pts = [(0.0, i) for i in range(6)]
        assert EyeAnalyzer.compute_ear(pts) == 0.0

    def test_ear_proportional_to_vertical_gap(self):
        ear_wide   = EyeAnalyzer.compute_ear(make_eye(vertical_gap=4.0))
        ear_narrow = EyeAnalyzer.compute_ear(make_eye(vertical_gap=1.0))
        assert ear_wide > ear_narrow

    def test_ear_invariant_to_horizontal_scaling(self):
        # EAR is a ratio, so scaling everything uniformly should not change it
        base = make_eye(vertical_gap=2.0, horizontal_span=6.0)
        scaled = [(x * 2, y * 2) for x, y in base]
        assert EyeAnalyzer.compute_ear(base) == pytest.approx(
            EyeAnalyzer.compute_ear(scaled), abs=1e-6
        )


class TestEyeAnalyzerIsClosed:
    def test_below_default_threshold_is_closed(self):
        assert EyeAnalyzer.is_closed(0.20) is True

    def test_above_default_threshold_is_open(self):
        assert EyeAnalyzer.is_closed(0.30) is False

    def test_exactly_at_threshold_is_open(self):
        assert EyeAnalyzer.is_closed(0.25) is False

    def test_custom_threshold_respected(self):
        assert EyeAnalyzer.is_closed(0.28, threshold=0.30) is True
        assert EyeAnalyzer.is_closed(0.28, threshold=0.25) is False
