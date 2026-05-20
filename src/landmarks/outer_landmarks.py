"""Outerwear (jacket / coat / blazer) landmark vocabulary.

Outer differs from `top_landmarks` in three places: front opening (zipper
or buttons), lapels/collar, and pocket anchors. ModaNet treats outer as a
separate meta-category for the same reason — jackets need distinct
control points to keep the placket aligned.
"""
from __future__ import annotations


OUTER_LANDMARKS = (
    # Base top
    "neck_left", "neck_right", "neck_center",
    "shoulder_left", "shoulder_right",
    "armpit_left", "armpit_right",
    "chest_left", "chest_right", "chest_center",
    "waist_left", "waist_right",
    "hem_left", "hem_right", "hem_center",
    # Collar / lapel
    "collar_tip_left", "collar_tip_right",
    "lapel_top_left", "lapel_top_right",
    "lapel_mid_left", "lapel_mid_right",
    "lapel_bottom_left", "lapel_bottom_right",
    # Front opening
    "front_opening_top", "front_opening_mid", "front_opening_bottom",
    "zipper_top", "zipper_bottom",
    "button_1", "button_2", "button_3", "button_4",
    # Sleeves
    "sleeve_root_left", "sleeve_root_right",
    "sleeve_elbow_left", "sleeve_elbow_right",
    "sleeve_cuff_left_outer", "sleeve_cuff_left_inner",
    "sleeve_cuff_right_outer", "sleeve_cuff_right_inner",
    # Pockets
    "pocket_left_top", "pocket_left_bottom",
    "pocket_right_top", "pocket_right_bottom",
)

BODY_TARGETS = {
    "neck_center": "neck_center",
    "shoulder_left": "left_shoulder",
    "shoulder_right": "right_shoulder",
    "armpit_left": "left_armpit",
    "armpit_right": "right_armpit",
    "waist_left": "left_waist",
    "waist_right": "right_waist",
    "hem_center": "hip_center_or_mid_thigh",
    "sleeve_cuff_left_outer": "left_wrist",
    "sleeve_cuff_right_outer": "right_wrist",
}

__all__ = ["OUTER_LANDMARKS", "BODY_TARGETS"]
