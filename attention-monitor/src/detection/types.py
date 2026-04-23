from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from .head_pose_analyzer import HeadPose


@dataclass
class DetectionResult:
    face_detected: bool
    left_ear: float = 0.0
    right_ear: float = 0.0
    mean_ear: float = 0.0
    head_pose: Optional[HeadPose] = None
    eye_points: List[List[Tuple[float, float]]] = field(default_factory=list)
    nose_point: Optional[Tuple[float, float]] = None
    raw_landmarks: Any = None
