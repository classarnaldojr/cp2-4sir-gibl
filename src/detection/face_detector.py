from __future__ import annotations

from typing import Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .eye_analyzer import EyeAnalyzer
from .head_pose_analyzer import HeadPoseAnalyzer
from .types import DetectionResult


class FaceDetector:
    _RIGHT_EYE: Tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    _LEFT_EYE:  Tuple[int, ...] = (362, 385, 387, 263, 373, 380)

    _NOSE_TIP:    int = 4
    _LEFT_CHEEK:  int = 234
    _RIGHT_CHEEK: int = 454
    _FOREHEAD:    int = 10
    _CHIN:        int = 152

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_path: str = "face_landmarker.task",
    ) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_faces=1,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> DetectionResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect(mp_image)

        if not results.face_landmarks:
            return DetectionResult(face_detected=False)

        lm = results.face_landmarks[0]
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

        # Reconstrói objeto compatível com o restante do código
        class _FakeLandmarks:
            landmark = lm

        return DetectionResult(
            face_detected=True,
            left_ear=left_ear,
            right_ear=right_ear,
            mean_ear=mean_ear,
            head_pose=head_pose,
            eye_points=[right_eye_pts, left_eye_pts],
            nose_point=px(self._NOSE_TIP),
            raw_landmarks=_FakeLandmarks(),
        )

    def close(self) -> None:
        self._landmarker.close()