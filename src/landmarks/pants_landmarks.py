"""Pants / shorts landmark vocabulary.

`PANTS_LANDMARKS` is the full long-pants vocabulary; `MIN_PANTS` is the
26-point subset to detect first when extending `detect_pants_landmarks` in
`tps_warp.py`. `SHORTS_LANDMARKS` is a separate shorter set because shorts
should NOT map to ankle/foot targets — doing so stretches a pair of shorts
into long pants.
"""
from __future__ import annotations


PANTS_LANDMARKS = (
    # Waist band
    "waist_left", "waist_right", "waist_center",
    "waist_top_left", "waist_top_right",
    "waist_bottom_left", "waist_bottom_right",
    # Hip
    "hip_left", "hip_right", "hip_center",
    "left_hip_outer", "right_hip_outer",
    # Crotch
    "crotch_center", "crotch_left", "crotch_right", "crotch_bottom",
    "front_rise_top", "front_rise_bottom",
    # Left leg
    "left_leg_outer_top", "left_leg_inner_top",
    "left_thigh_outer", "left_thigh_inner",
    "left_knee_outer", "left_knee_inner",
    "left_calf_outer", "left_calf_inner",
    "left_ankle_outer", "left_ankle_inner",
    "left_hem_outer", "left_hem_inner", "left_hem_center",
    # Right leg
    "right_leg_outer_top", "right_leg_inner_top",
    "right_thigh_outer", "right_thigh_inner",
    "right_knee_outer", "right_knee_inner",
    "right_calf_outer", "right_calf_inner",
    "right_ankle_outer", "right_ankle_inner",
    "right_hem_outer", "right_hem_inner", "right_hem_center",
    # Seams
    "left_side_seam_top", "left_side_seam_mid", "left_side_seam_bottom",
    "right_side_seam_top", "right_side_seam_mid", "right_side_seam_bottom",
    "inseam_top", "left_inseam_mid", "right_inseam_mid",
    "left_inseam_bottom", "right_inseam_bottom",
    # Belt / pocket / fly
    "belt_loop_left", "belt_loop_right",
    "pocket_left", "pocket_right",
    "fly_top", "fly_bottom",
)

MIN_PANTS = (
    "waist_left", "waist_right", "waist_center",
    "hip_left", "hip_right", "hip_center",
    "crotch_center", "crotch_bottom",
    "left_thigh_outer", "left_thigh_inner",
    "right_thigh_outer", "right_thigh_inner",
    "left_knee_outer", "left_knee_inner",
    "right_knee_outer", "right_knee_inner",
    "left_ankle_outer", "left_ankle_inner",
    "right_ankle_outer", "right_ankle_inner",
    "left_hem_outer", "left_hem_inner", "left_hem_center",
    "right_hem_outer", "right_hem_inner", "right_hem_center",
)

SHORTS_LANDMARKS = (
    "waist_left", "waist_right", "waist_center",
    "hip_left", "hip_right", "hip_center",
    "crotch_center", "crotch_bottom",
    "left_leg_opening_outer", "left_leg_opening_inner", "left_leg_opening_center",
    "right_leg_opening_outer", "right_leg_opening_inner", "right_leg_opening_center",
    "left_inseam_short", "right_inseam_short",
    "left_side_hem", "right_side_hem",
)

PANTS_BODY_TARGETS = {
    "waist_left": "left_waist",
    "waist_right": "right_waist",
    "hip_left": "left_hip",
    "hip_right": "right_hip",
    "crotch_center": "crotch_center",
    "left_knee_outer": "left_knee_outer",
    "left_knee_inner": "left_knee_inner",
    "right_knee_outer": "right_knee_outer",
    "right_knee_inner": "right_knee_inner",
    "left_hem_center": "left_ankle_or_left_foot_index",
    "right_hem_center": "right_ankle_or_right_foot_index",
}

# Shorts hem must stop at upper/mid thigh — never map to ankle/foot.
SHORTS_BODY_TARGETS = {
    "waist_left": "left_waist",
    "waist_right": "right_waist",
    "hip_left": "left_hip",
    "hip_right": "right_hip",
    "crotch_center": "crotch_center",
    "left_leg_opening_center": "left_upper_thigh",
    "right_leg_opening_center": "right_upper_thigh",
}

__all__ = [
    "PANTS_LANDMARKS", "MIN_PANTS", "SHORTS_LANDMARKS",
    "PANTS_BODY_TARGETS", "SHORTS_BODY_TARGETS",
]
