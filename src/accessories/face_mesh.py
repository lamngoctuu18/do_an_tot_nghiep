"""MediaPipe Face Mesh wrapper for accessory (sunglasses) anchoring.

Lazy import so the rest of the pipeline doesn't require mediapipe.face_mesh
when no glasses try-on is requested.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

_FACE_MESH = None
_INIT_FAILED = False


def _get_face_mesh():
    global _FACE_MESH, _INIT_FAILED
    if _INIT_FAILED:
        return None
    if _FACE_MESH is not None:
        return _FACE_MESH
    try:
        import mediapipe as mp  # type: ignore
        _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
        )
        return _FACE_MESH
    except Exception:
        _INIT_FAILED = True
        return None


# MediaPipe FaceMesh indices (refine_landmarks=True)
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_LEFT_EYE_INNER = 133
_RIGHT_EYE_INNER = 362
_NOSE_BRIDGE = 6
_NOSE_TIP = 4
_LEFT_TEMPLE = 234
_RIGHT_TEMPLE = 454


def detect_face_anchors(rgb: np.ndarray) -> Optional[dict]:
    """Return face anchors for glasses warp; None if no face found."""
    fm = _get_face_mesh()
    if fm is None or rgb is None or rgb.size == 0:
        return None
    try:
        res = fm.process(rgb)
    except Exception:
        return None
    if not res.multi_face_landmarks:
        return None

    h, w = rgb.shape[:2]
    lms = res.multi_face_landmarks[0].landmark

    def _px(idx: int) -> Tuple[float, float]:
        p = lms[idx]
        return (float(p.x * w), float(p.y * h))

    return {
        "left_eye_outer":  _px(_LEFT_EYE_OUTER),
        "right_eye_outer": _px(_RIGHT_EYE_OUTER),
        "left_eye_inner":  _px(_LEFT_EYE_INNER),
        "right_eye_inner": _px(_RIGHT_EYE_INNER),
        "nose_bridge":     _px(_NOSE_BRIDGE),
        "nose_tip":        _px(_NOSE_TIP),
        "left_temple":     _px(_LEFT_TEMPLE),
        "right_temple":    _px(_RIGHT_TEMPLE),
    }


__all__ = ["detect_face_anchors"]
