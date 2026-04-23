from __future__ import annotations

from dataclasses import dataclass

from ..scoring.attention_scorer import SessionStats


@dataclass
class Report:
    duration_seconds: float
    attention_score: float
    attentive_pct: float
    distracted_pct: float
    drowsy_pct: float
    absent_pct: float
    total_frames: int


class SessionReport:
    def generate(self, stats: SessionStats, elapsed: float) -> Report:
        total = max(1, stats.total_frames)
        return Report(
            duration_seconds=elapsed,
            attention_score=round(stats.attention_score, 1),
            attentive_pct=round(stats.attentive_frames / total * 100, 1),
            distracted_pct=round(stats.distracted_frames / total * 100, 1),
            drowsy_pct=round(stats.drowsy_frames / total * 100, 1),
            absent_pct=round(stats.absent_frames / total * 100, 1),
            total_frames=total,
        )

    def print_summary(self, stats: SessionStats, elapsed: float) -> None:
        r = self.generate(stats, elapsed)
        mins, secs = divmod(int(r.duration_seconds), 60)
        sep = "=" * 52
        print(f"\n{sep}")
        print("  RELATORIO DE SESSAO — ATTENTION MONITOR")
        print(sep)
        print(f"  Duracao total    : {mins:02d}:{secs:02d}")
        print(f"  Score final      : {r.attention_score}%")
        print(f"  Tempo atento     : {r.attentive_pct}%")
        print(f"  Tempo distraido  : {r.distracted_pct}%")
        print(f"  Tempo sonolento  : {r.drowsy_pct}%")
        print(f"  Tempo ausente    : {r.absent_pct}%")
        print(f"  Total de frames  : {r.total_frames}")
        print(sep)
