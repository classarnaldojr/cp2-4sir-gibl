from __future__ import annotations

import sys

import cv2

from src.detection.face_detector import FaceDetector
from src.reporting.session_report import SessionReport
from src.scoring.attention_scorer import AttentionScorer
from src.ui.dashboard import Dashboard


def run(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Erro: webcam {camera_index} não encontrada.", file=sys.stderr)
        sys.exit(1)

    detector = FaceDetector()
    scorer   = AttentionScorer()
    dash     = Dashboard()
    report   = SessionReport()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            detection = detector.process(frame)
            stats     = scorer.update(detection)
            elapsed   = scorer.elapsed()

            output = dash.render(frame, detection, stats, elapsed)
            cv2.imshow("Attention Monitor", output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                scorer = AttentionScorer()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        report.print_summary(scorer.stats, scorer.elapsed())


if __name__ == "__main__":
    camera = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run(camera)
