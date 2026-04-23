from __future__ import annotations

from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np

from .eye_analyzer import EyeAnalyzer
from .head_pose_analyzer import HeadPoseAnalyzer
from .types import DetectionResult


class FaceDetector:
    """
    Wraps MediaPipe FaceMesh to expose only the signals needed for
    attention analysis: per-eye EAR values and normalized head pose.
    """

    # Six-point EAR landmark indices for each eye (MediaPipe FaceMesh).
    # Order: outer_corner, upper_inner, upper_outer, inner_corner, lower_outer, lower_inner
    _RIGHT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    _LEFT_EYE: Tuple[int, ...]  = (362, 385, 387, 263, 373, 380)

    # Structural landmarks for head pose estimation
    _NOSE_TIP:    int = 4
    _LEFT_CHEEK:  int = 234
    _RIGHT_CHEEK: int = 454
    _FOREHEAD:    int = 10
    _CHIN:        int = 152

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return DetectionResult(face_detected=False)

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def px(idx: int) -> Tuple[float, float]:
            return (lm[idx].x * w, lm[idx].y * h)

        right_eye_pts = [px(i) for i in self._RIGHT_EYE]
        left_eye_pts  = [px(i) for i in self._LEFT_EYE]

        right_ear = EyeAnalyzer.compute_ear(right_eye_pts)
        left_ear  = EyeAnalyzer.compute_ear(left_eye_pts)
        mean_ear  = (right_ear + left_ear) / 2.0

        head_pose = HeadPoseAnalyzer.compute(
            nose        = px(self._NOSE_TIP),
            left_cheek  = px(self._LEFT_CHEEK),
            right_cheek = px(self._RIGHT_CHEEK),
            forehead    = px(self._FOREHEAD),
            chin        = px(self._CHIN),
        )

        return DetectionResult(
            face_detected=True,
            left_ear=left_ear,
            right_ear=right_ear,
            mean_ear=mean_ear,
            head_pose=head_pose,
            eye_points=[right_eye_pts, left_eye_pts],
            nose_point=px(self._NOSE_TIP),
            raw_landmarks=results.multi_face_landmarks[0],
        )

    def close(self) -> None:
        self._face_mesh.close()
