"""Skirt landmark vocabulary.

Skirt = dress lower half only. It should never replace the upper body, so
the body-target mapping excludes anything above the waist.
"""
from __future__ import annotations


SKIRT_LANDMARKS = (
    # Waist
    "waist_left", "waist_right", "waist_center",
    "waist_top_left", "waist_top_right",
    "waist_bottom_left", "waist_bottom_right",
    # Hip
    "hip_left", "hip_right", "hip_center",
    # Side seams
    "side_left_top", "side_left_mid", "side_left_bottom",
    "side_right_top", "side_right_mid", "side_right_bottom",
    # Front panel
    "front_panel_top", "front_panel_mid", "front_panel_bottom",
    # Pleats (up to 4)
    "pleat_1_top", "pleat_1_bottom",
    "pleat_2_top", "pleat_2_bottom",
    "pleat_3_top", "pleat_3_bottom",
    "pleat_4_top", "pleat_4_bottom",
    # Hem
    "hem_left", "hem_right", "hem_center",
    "hem_front_left", "hem_front_right",
    # Special details
    "slit_top", "slit_bottom",
    "zipper_top", "zipper_bottom",
)

SKIRT_BODY_TARGETS = {
    "waist_left": "left_waist",
    "waist_right": "right_waist",
    "hip_left": "left_hip",
    "hip_right": "right_hip",
    "hem_center": "thigh_or_knee_or_calf_center",
}

__all__ = ["SKIRT_LANDMARKS", "SKIRT_BODY_TARGETS"]
