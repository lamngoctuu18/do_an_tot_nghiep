"""Jumpsuit / overall / romper landmark vocabulary.

Hybrid category — bodice landmarks from `top` plus leg landmarks from
`pants`. Keeping it as its own category prevents the pipeline from
splitting a jumpsuit into a separately-warped top and pants.
"""
from __future__ import annotations


JUMPSUIT_LANDMARKS = (
    # Upper body
    "neck_left", "neck_right", "neck_center",
    "shoulder_left", "shoulder_right",
    "armpit_left", "armpit_right",
    "chest_left", "chest_right", "chest_center",
    "waist_left", "waist_right", "waist_center",
    # Hip / crotch
    "hip_left", "hip_right", "hip_center",
    "crotch_center", "crotch_bottom",
    # Legs
    "left_thigh_outer", "left_thigh_inner",
    "right_thigh_outer", "right_thigh_inner",
    "left_knee_outer", "left_knee_inner",
    "right_knee_outer", "right_knee_inner",
    "left_hem_outer", "left_hem_inner",
    "right_hem_outer", "right_hem_inner",
    # Optional sleeves
    "sleeve_root_left", "sleeve_root_right",
    "sleeve_tip_left", "sleeve_tip_right",
    # Center seam
    "center_front_top", "center_front_mid", "center_front_crotch",
)

JUMPSUIT_BODY_TARGETS = {
    "neck_center": "neck_center",
    "shoulder_left": "left_shoulder",
    "shoulder_right": "right_shoulder",
    "waist_center": "waist_center",
    "hip_center": "hip_center",
    "crotch_center": "crotch_center",
    "left_knee_outer": "left_knee_outer",
    "right_knee_outer": "right_knee_outer",
    "left_hem_outer": "left_ankle_or_left_foot_index",
    "right_hem_outer": "right_ankle_or_right_foot_index",
}

__all__ = ["JUMPSUIT_LANDMARKS", "JUMPSUIT_BODY_TARGETS"]
