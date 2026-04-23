from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..detection.eye_analyzer import EyeAnalyzer
from ..detection.head_pose_analyzer import HeadPoseAnalyzer
from ..detection.types import DetectionResult


class AttentionState(str, Enum):
    ATTENTIVE  = "ATENTO"
    DISTRACTED = "DISTRAIDO"
    DROWSY     = "SONOLENTO"
    ABSENT     = "AUSENTE"


@dataclass
class ScorerConfig:
    ear_threshold: float = 0.25
    drowsy_frame_count: int = 20       # consecutive eye-closed frames before DROWSY
    yaw_threshold: float = 0.12        # normalized horizontal nose offset
    pitch_threshold: float = 0.07      # normalized vertical nose offset
    absent_frame_count: int = 30       # frames without a face before ABSENT
    score_decay_distracted: float = 0.30
    score_decay_drowsy: float = 0.50
    score_decay_absent: float = 0.70
    score_recovery: float = 0.15


@dataclass
class SessionStats:
    total_frames: int = 0
    attentive_frames: int = 0
    distracted_frames: int = 0
    drowsy_frames: int = 0
    absent_frames: int = 0
    attention_score: float = 100.0
    current_state: AttentionState = AttentionState.ATTENTIVE
    _closed_counter: int = field(default=0, repr=False)
    _no_face_counter: int = field(default=0, repr=False)


class AttentionScorer:
    """
    Frame-by-frame state machine that classifies attention and maintains
    a rolling score in [0, 100].

    Score decay is asymmetric: losing attention drops the score faster
    than recovering it, which reflects the real cost of distraction.
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self.config = config or ScorerConfig()
        self.stats = SessionStats()
        self._start = time.monotonic()

    def update(self, detection: DetectionResult) -> SessionStats:
        self.stats.total_frames += 1
        cfg = self.config
        st = self.stats

        if not detection.face_detected:
            st._no_face_counter += 1
            st._closed_counter = 0
            if st._no_face_counter >= cfg.absent_frame_count:
                st.current_state = AttentionState.ABSENT
                st.absent_frames += 1
                st.attention_score = max(0.0, st.attention_score - cfg.score_decay_absent)
            return st

        st._no_face_counter = 0

        eyes_closed = EyeAnalyzer.is_closed(detection.mean_ear, cfg.ear_threshold)
        if eyes_closed:
            st._closed_counter += 1
        else:
            st._closed_counter = max(0, st._closed_counter - 3)

        looking_away = (
            detection.head_pose is not None
            and HeadPoseAnalyzer.is_looking_away(
                detection.head_pose, cfg.yaw_threshold, cfg.pitch_threshold
            )
        )

        if st._closed_counter >= cfg.drowsy_frame_count:
            st.current_state = AttentionState.DROWSY
            st.drowsy_frames += 1
            st.attention_score = max(0.0, st.attention_score - cfg.score_decay_drowsy)
        elif looking_away:
            st.current_state = AttentionState.DISTRACTED
            st.distracted_frames += 1
            st.attention_score = max(0.0, st.attention_score - cfg.score_decay_distracted)
        else:
            st.current_state = AttentionState.ATTENTIVE
            st.attentive_frames += 1
            st.attention_score = min(100.0, st.attention_score + cfg.score_recovery)

        return st

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def attention_percentage(self) -> float:
        total = max(1, self.stats.total_frames)
        return (self.stats.attentive_frames / total) * 100.0
