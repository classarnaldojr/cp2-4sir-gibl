from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


class EyeAnalyzer:
    """
    Eye Aspect Ratio (EAR) computation.

    Reference: Soukupova & Cech, "Real-Time Eye Blink Detection Using Facial Landmarks", 2016.
    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)

    The six landmark points must be ordered as:
      p0 = outer corner
      p1 = upper inner
      p2 = upper outer
      p3 = inner corner
      p4 = lower outer
      p5 = lower inner
    """

    CLOSED_THRESHOLD: float = 0.25

    @staticmethod
    def compute_ear(points: Sequence[Tuple[float, float]]) -> float:
        p = [np.array(pt, dtype=np.float64) for pt in points]
        vertical_a = np.linalg.norm(p[1] - p[5])
        vertical_b = np.linalg.norm(p[2] - p[4])
        horizontal = np.linalg.norm(p[0] - p[3])
        if horizontal < 1e-6:
            return 0.0
        return float((vertical_a + vertical_b) / (2.0 * horizontal))

    @staticmethod
    def is_closed(ear: float, threshold: float = 0.25) -> bool:
        return ear < threshold
