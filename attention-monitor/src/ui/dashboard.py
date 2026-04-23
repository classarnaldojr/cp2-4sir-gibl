from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ..detection.types import DetectionResult
from ..scoring.attention_scorer import AttentionState, SessionStats

# ─── Layout constants ────────────────────────────────────────────────────────

PANEL_W = 320
PAD     = 24

# ─── Color palette (BGR) ─────────────────────────────────────────────────────

_BG           = (22, 22, 22)
_DIVIDER      = (50, 50, 50)
_LABEL        = (120, 120, 120)
_VALUE        = (215, 215, 215)
_BAR_TRACK    = (45, 45, 45)

_STATE_COLOR: dict = {
    AttentionState.ATTENTIVE:  (55, 195, 55),
    AttentionState.DISTRACTED: (40, 155, 255),
    AttentionState.DROWSY:     (55, 55, 215),
    AttentionState.ABSENT:     (115, 115, 115),
}

_STATE_LABEL: dict = {
    AttentionState.ATTENTIVE:  "ATENTO",
    AttentionState.DISTRACTED: "DISTRAIDO",
    AttentionState.DROWSY:     "SONOLENTO",
    AttentionState.ABSENT:     "AUSENTE",
}


def _score_color(score: float) -> Tuple[int, int, int]:
    if score >= 70:
        return (55, 195, 55)
    if score >= 40:
        return (40, 155, 255)
    return (55, 55, 215)


def _text(img, text: str, pos: Tuple[int, int], scale: float, color, thickness: int = 1) -> None:
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _divider(img, y: int) -> None:
    cv2.line(img, (PAD, y), (PANEL_W - PAD, y), _DIVIDER, 1)


# ─── Dashboard ───────────────────────────────────────────────────────────────

class Dashboard:
    def render(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        stats: SessionStats,
        elapsed: float,
    ) -> np.ndarray:
        annotated = self._annotate_frame(frame.copy(), detection, stats)
        panel     = self._build_panel(frame.shape[0], stats, elapsed)
        return np.hstack([annotated, panel])

    # ── Frame overlay ─────────────────────────────────────────────────────────

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detection: DetectionResult,
        stats: SessionStats,
    ) -> np.ndarray:
        color = _STATE_COLOR[stats.current_state]

        if detection.face_detected:
            for eye_pts in detection.eye_points:
                for pt in eye_pts:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, color, -1, cv2.LINE_AA)

            if detection.nose_point:
                np_pt = (int(detection.nose_point[0]), int(detection.nose_point[1]))
                cv2.circle(frame, np_pt, 3, _LABEL, -1, cv2.LINE_AA)

            ear_text = f"EAR {detection.mean_ear:.3f}"
            _text(frame, ear_text, (8, 22), 0.50, _LABEL)

        if stats.current_state != AttentionState.ATTENTIVE:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

        return frame

    # ── Side panel ────────────────────────────────────────────────────────────

    def _build_panel(self, height: int, stats: SessionStats, elapsed: float) -> np.ndarray:
        panel = np.full((height, PANEL_W, 3), _BG, dtype=np.uint8)
        state = stats.current_state
        color = _STATE_COLOR[state]
        score = stats.attention_score

        y = 42
        _text(panel, "STATUS", (PAD, y), 0.40, _LABEL)
        y += 34
        _text(panel, _STATE_LABEL[state], (PAD, y), 1.0, color, 2)

        y += 26
        _divider(panel, y)

        y += 30
        _text(panel, "SCORE DE ATENCAO", (PAD, y), 0.40, _LABEL)
        y += 36
        sc = _score_color(score)
        _text(panel, f"{int(score)}%", (PAD, y), 1.6, sc, 2)

        y += 20
        bx  = PAD
        bw  = PANEL_W - PAD * 2
        bh  = 13
        cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), _BAR_TRACK, -1)
        filled = int(bw * max(0.0, min(1.0, score / 100.0)))
        if filled > 0:
            cv2.rectangle(panel, (bx, y), (bx + filled, y + bh), sc, -1)
        cv2.rectangle(panel, (bx, y), (bx + bw, y + bh), _DIVIDER, 1)

        y += bh + 24
        _divider(panel, y)

        y += 30
        mins, secs = divmod(int(elapsed), 60)
        _text(panel, "TEMPO DE SESSAO", (PAD, y), 0.40, _LABEL)
        y += 34
        _text(panel, f"{mins:02d}:{secs:02d}", (PAD, y), 1.10, _VALUE, 2)

        y += 28
        _divider(panel, y)

        y += 30
        _text(panel, "DISTRIBUICAO DE ESTADOS", (PAD, y), 0.40, _LABEL)

        total = max(1, stats.total_frames)
        rows = [
            ("Atento",    stats.attentive_frames,  _STATE_COLOR[AttentionState.ATTENTIVE]),
            ("Distraido", stats.distracted_frames,  _STATE_COLOR[AttentionState.DISTRACTED]),
            ("Sonolento", stats.drowsy_frames,      _STATE_COLOR[AttentionState.DROWSY]),
            ("Ausente",   stats.absent_frames,      _STATE_COLOR[AttentionState.ABSENT]),
        ]
        for label, frames, c in rows:
            y += 26
            pct = int(frames / total * 100)
            _text(panel, f"{label}:", (PAD, y), 0.42, _LABEL)
            _text(panel, f"{pct}%", (PAD + 118, y), 0.42, c, 1)

        y += 34
        _divider(panel, y)
        y += 24
        _text(panel, "[Q] Sair   [R] Reiniciar sessao", (PAD, y), 0.36, _LABEL)

        return panel
