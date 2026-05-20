"""Pose → anchor-point helpers for each accessory subtype.

Each function reads from a `full_pose` dict (the same one used by the
pants pipeline) and returns a dict of pixel coordinates the warp module
will consume. All functions are fail-soft: missing keypoints → None.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

Point = Tuple[float, float]


def _pt(pose: Optional[dict], key: str) -> Optional[Point]:
    if pose is None:
        return None
    p = pose.get(key)
    if p is None:
        return None
    try:
        return (float(p[0]), float(p[1]))
    except Exception:
        return None


def _mid(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def shoe_anchors(pose: Optional[dict]) -> Dict[str, Optional[Point]]:
    return {
        "left_ankle":      _pt(pose, "left_ankle"),
        "right_ankle":     _pt(pose, "right_ankle"),
        "left_heel":       _pt(pose, "left_heel"),
        "right_heel":      _pt(pose, "right_heel"),
        "left_foot_index": _pt(pose, "left_foot_index"),
        "right_foot_index":_pt(pose, "right_foot_index"),
        "left_knee":       _pt(pose, "left_knee"),
        "right_knee":      _pt(pose, "right_knee"),
    }


def head_anchors(pose: Optional[dict]) -> Dict[str, Optional[Point]]:
    le = _pt(pose, "left_ear")
    re = _pt(pose, "right_ear")
    nose = _pt(pose, "nose")
    head_center = _mid(le, re) or nose
    head_top = None
    if head_center is not None and nose is not None:
        # crown ≈ head_center shifted up by inter-ear gap
        ear_gap = abs((le[0] - re[0])) if (le and re) else 60.0
        head_top = (head_center[0], head_center[1] - max(40.0, ear_gap * 1.05))
    return {
        "left_ear":    le,
        "right_ear":   re,
        "nose":        nose,
        "head_center": head_center,
        "head_top":    head_top,
    }


def eye_anchors(pose: Optional[dict]) -> Dict[str, Optional[Point]]:
    return {
        "left_eye":  _pt(pose, "left_eye"),
        "right_eye": _pt(pose, "right_eye"),
        "nose":      _pt(pose, "nose"),
        "left_ear":  _pt(pose, "left_ear"),
        "right_ear": _pt(pose, "right_ear"),
    }


def waist_anchors(pose: Optional[dict]) -> Dict[str, Optional[Point]]:
    lh = _pt(pose, "left_hip")
    rh = _pt(pose, "right_hip")
    return {
        "left_hip":   lh,
        "right_hip":  rh,
        "hip_center": _mid(lh, rh),
    }


def neck_anchors(pose: Optional[dict]) -> Dict[str, Optional[Point]]:
    ls = _pt(pose, "left_shoulder")
    rs = _pt(pose, "right_shoulder")
    neck_center = _mid(ls, rs)
    return {
        "left_shoulder":  ls,
        "right_shoulder": rs,
        "neck_center":    neck_center,
        "nose":           _pt(pose, "nose"),
    }


def shoulder_strap_anchors(pose: Optional[dict], side: str = "left") -> Dict[str, Optional[Point]]:
    ls = _pt(pose, "left_shoulder")
    rs = _pt(pose, "right_shoulder")
    lh = _pt(pose, "left_hip")
    rh = _pt(pose, "right_hip")
    if side == "right":
        carry_shoulder, opp_hip = rs, lh
    else:
        carry_shoulder, opp_hip = ls, rh
    return {
        "carry_shoulder": carry_shoulder,
        "opposite_hip":   opp_hip,
        "left_shoulder":  ls,
        "right_shoulder": rs,
        "left_hip":       lh,
        "right_hip":      rh,
    }


__all__ = [
    "shoe_anchors",
    "head_anchors",
    "eye_anchors",
    "waist_anchors",
    "neck_anchors",
    "shoulder_strap_anchors",
]
