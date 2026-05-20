"""Body landmark vocabulary.

`MEDIAPIPE_33` is the canonical MediaPipe Pose Landmarker landmark set. Use
the index constants when reading raw MediaPipe results; the name list is
what we attach to detector outputs so downstream code stays index-free.

`DERIVED` lists the geometric points we synthesise on top of the 33 raw
landmarks (waist_center, crotch_center, etc.) — those are computed by
`src.landmarks.derived` (added later) but the names live here so the
schema is in one place.
"""
from __future__ import annotations


MEDIAPIPE_33 = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)

# MediaPipe Pose landmark IDs in their canonical order.
MEDIAPIPE_INDEX = {name: i for i, name in enumerate(MEDIAPIPE_33)}

# Points computed from `MEDIAPIPE_33` — see derive_body_extras() in
# `src.landmarks.derived` for the geometric definitions.
DERIVED = (
    "neck_center",
    "shoulder_center",
    "chest_center",
    "under_bust_center",
    "waist_center",
    "hip_center",
    "crotch_center",
    "left_armpit", "right_armpit",
    "left_waist", "right_waist",
    "left_thigh_outer", "right_thigh_outer",
    "left_thigh_inner", "right_thigh_inner",
    "left_knee_outer", "right_knee_outer",
    "left_knee_inner", "right_knee_inner",
    "left_calf_outer", "right_calf_outer",
    "left_calf_inner", "right_calf_inner",
    "left_ankle_outer", "right_ankle_outer",
    "left_ankle_inner", "right_ankle_inner",
)

ALL_BODY = MEDIAPIPE_33 + DERIVED

__all__ = ["MEDIAPIPE_33", "MEDIAPIPE_INDEX", "DERIVED", "ALL_BODY"]
