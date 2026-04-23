from __future__ import annotations

import pytest

from src.detection.head_pose_analyzer import HeadPose
from src.detection.types import DetectionResult
from src.scoring.attention_scorer import AttentionScorer, AttentionState, ScorerConfig


# ─── Frame factories ──────────────────────────────────────────────────────────

def attentive_frame() -> DetectionResult:
    return DetectionResult(
        face_detected=True,
        mean_ear=0.35,
        head_pose=HeadPose(yaw=0.0, pitch=0.0),
    )


def distracted_frame() -> DetectionResult:
    return DetectionResult(
        face_detected=True,
        mean_ear=0.35,
        head_pose=HeadPose(yaw=0.25, pitch=0.0),
    )


def drowsy_frame() -> DetectionResult:
    return DetectionResult(
        face_detected=True,
        mean_ear=0.15,
        head_pose=HeadPose(yaw=0.0, pitch=0.0),
    )


def absent_frame() -> DetectionResult:
    return DetectionResult(face_detected=False)


# ─── Initial state ────────────────────────────────────────────────────────────

class TestInitialState:
    def test_starts_attentive(self):
        scorer = AttentionScorer()
        assert scorer.stats.current_state == AttentionState.ATTENTIVE

    def test_starts_at_full_score(self):
        scorer = AttentionScorer()
        assert scorer.stats.attention_score == pytest.approx(100.0)

    def test_starts_with_zero_frames(self):
        scorer = AttentionScorer()
        assert scorer.stats.total_frames == 0


# ─── Score dynamics ───────────────────────────────────────────────────────────

class TestScoreDynamics:
    def test_attentive_recovers_score(self):
        scorer = AttentionScorer(ScorerConfig(score_recovery=1.0))
        scorer.stats.attention_score = 90.0
        scorer.update(attentive_frame())
        assert scorer.stats.attention_score == pytest.approx(91.0)

    def test_score_caps_at_100(self):
        scorer = AttentionScorer(ScorerConfig(score_recovery=5.0))
        scorer.stats.attention_score = 99.0
        scorer.update(attentive_frame())
        assert scorer.stats.attention_score == pytest.approx(100.0)

    def test_distracted_reduces_score(self):
        scorer = AttentionScorer(ScorerConfig(score_decay_distracted=1.0))
        scorer.update(distracted_frame())
        assert scorer.stats.attention_score < 100.0

    def test_score_floors_at_zero(self):
        scorer = AttentionScorer(
            ScorerConfig(absent_frame_count=1, score_decay_absent=500.0)
        )
        scorer.update(absent_frame())
        assert scorer.stats.attention_score == pytest.approx(0.0)

    def test_decay_is_faster_than_recovery(self):
        cfg = ScorerConfig()
        assert cfg.score_decay_distracted > cfg.score_recovery


# ─── State transitions ────────────────────────────────────────────────────────

class TestStateTransitions:
    def test_attentive_frame_sets_attentive(self):
        scorer = AttentionScorer()
        scorer.update(attentive_frame())
        assert scorer.stats.current_state == AttentionState.ATTENTIVE

    def test_distracted_frame_sets_distracted(self):
        scorer = AttentionScorer()
        scorer.update(distracted_frame())
        assert scorer.stats.current_state == AttentionState.DISTRACTED

    def test_absent_triggers_after_threshold(self):
        cfg = ScorerConfig(absent_frame_count=3)
        scorer = AttentionScorer(cfg)
        for _ in range(2):
            scorer.update(absent_frame())
        assert scorer.stats.current_state != AttentionState.ABSENT
        scorer.update(absent_frame())
        assert scorer.stats.current_state == AttentionState.ABSENT

    def test_drowsy_triggers_after_threshold_frames(self):
        cfg = ScorerConfig(drowsy_frame_count=5, ear_threshold=0.25)
        scorer = AttentionScorer(cfg)
        for _ in range(4):
            scorer.update(drowsy_frame())
        assert scorer.stats.current_state != AttentionState.DROWSY
        scorer.update(drowsy_frame())
        assert scorer.stats.current_state == AttentionState.DROWSY

    def test_recovery_from_distraction(self):
        scorer = AttentionScorer()
        scorer.update(distracted_frame())
        assert scorer.stats.current_state == AttentionState.DISTRACTED
        scorer.update(attentive_frame())
        assert scorer.stats.current_state == AttentionState.ATTENTIVE


# ─── Frame accounting ─────────────────────────────────────────────────────────

class TestFrameAccounting:
    def test_frame_counts_are_mutually_exclusive(self):
        scorer = AttentionScorer()
        scorer.update(attentive_frame())
        scorer.update(distracted_frame())
        scorer.update(attentive_frame())
        total_classified = (
            scorer.stats.attentive_frames
            + scorer.stats.distracted_frames
            + scorer.stats.drowsy_frames
            + scorer.stats.absent_frames
        )
        assert total_classified == scorer.stats.total_frames

    def test_attention_percentage_calculation(self):
        scorer = AttentionScorer()
        scorer.update(attentive_frame())
        scorer.update(attentive_frame())
        scorer.update(distracted_frame())
        pct = scorer.attention_percentage()
        assert pct == pytest.approx(66.67, abs=0.1)

    def test_total_frames_increments(self):
        scorer = AttentionScorer()
        for _ in range(5):
            scorer.update(attentive_frame())
        assert scorer.stats.total_frames == 5
