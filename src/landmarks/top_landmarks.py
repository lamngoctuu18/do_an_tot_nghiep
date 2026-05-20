"""Top / upper-body garment landmark vocabulary.

`TOP_LANDMARKS` is the full vocabulary, `CURRENT_TPS_24` mirrors the 24
control-point set already used by `src/tps_warp.py::detect_cloth_landmarks`
so callers can map the existing detector output without renaming keys.

`BODY_TARGETS` maps each top landmark to the body landmark it should warp
onto (see `src.landmarks.body_landmarks`).
"""
from __future__ import annotations


TOP_LANDMARKS = (
    # Neck / collar
    "neck_left", "neck_right", "neck_center",
    "collar_left", "collar_right", "collar_center",
    "collar_bottom_center",
    # Shoulder
    "shoulder_top_left", "shoulder_top_right",
    "shoulder_left", "shoulder_right",
    "shoulder_slope_left", "shoulder_slope_right",
    # Armhole / armpit
    "armpit_left", "armpit_right",
    "armhole_top_left", "armhole_top_right",
    "armhole_bottom_left", "armhole_bottom_right",
    # Chest / bust
    "chest_left", "chest_right", "chest_center",
    "bust_left", "bust_right",
    "under_bust_left", "under_bust_right", "under_bust_center",
    # Torso side
    "side_torso_left", "side_torso_right",
    "mid_left", "mid_right",
    # Waist / hem
    "waist_left", "waist_right", "waist_center",
    "hem_left", "hem_right", "hem_center",
    # Sleeves
    "sleeve_root_left", "sleeve_root_right",
    "sleeve_outer_left", "sleeve_outer_right",
    "sleeve_inner_left", "sleeve_inner_right",
    "sleeve_tip_left", "sleeve_tip_right",
    "sleeve_cuff_left_outer", "sleeve_cuff_left_inner",
    "sleeve_cuff_right_outer", "sleeve_cuff_right_inner",
    # Pattern / logo stabilisation
    "print_center", "logo_center",
    "front_placket_top", "front_placket_bottom",
)

# 24 points actually emitted by detect_cloth_landmarks() in tps_warp.py.
CURRENT_TPS_24 = (
    "collar",
    "shoulder_top_left", "shoulder_top_right",
    "shoulder_left", "shoulder_right",
    "chest_center", "chest_left", "chest_right",
    "side_left", "side_right",
    "waist_left", "waist_right",
    "mid_left", "mid_right",
    "hem_left", "hem_right",
    "armpit_left", "armpit_right",
    "under_bust_left", "under_bust_right",
    "sleeve_tip_left", "sleeve_tip_right",
    "sleeve_outer_left", "sleeve_outer_right",
)

BODY_TARGETS = {
    "neck_center": "neck_center",
    "shoulder_left": "left_shoulder",
    "shoulder_right": "right_shoulder",
    "armpit_left": "left_armpit",
    "armpit_right": "right_armpit",
    "chest_center": "chest_center",
    "waist_left": "left_waist",
    "waist_right": "right_waist",
    # hem can either sit at waist (cropped top) or hip (regular tee)
    "hem_center": "waist_or_hip_center",
    "sleeve_tip_left": "left_elbow_or_left_wrist",
    "sleeve_tip_right": "right_elbow_or_right_wrist",
}

__all__ = ["TOP_LANDMARKS", "CURRENT_TPS_24", "BODY_TARGETS"]
