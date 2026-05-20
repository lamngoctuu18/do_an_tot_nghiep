"""Footwear landmark vocabulary.

Two separate sets: low shoes (sneakers / loafers / heels) which anchor at
foot height, and boots which extend up to the calf or knee. Mapping shoes
to ankle landmarks is wrong — they should anchor at foot_index/heel; the
boot top maps to ankle/calf/knee depending on shaft height.
"""
from __future__ import annotations


SHOE_LANDMARKS = (
    # Left
    "left_shoe_toe", "left_shoe_tip",
    "left_shoe_inner_toe", "left_shoe_outer_toe",
    "left_shoe_vamp_center", "left_shoe_lace_center",
    "left_shoe_inner_side", "left_shoe_outer_side",
    "left_shoe_heel",
    "left_shoe_sole_front", "left_shoe_sole_back",
    # Right
    "right_shoe_toe", "right_shoe_tip",
    "right_shoe_inner_toe", "right_shoe_outer_toe",
    "right_shoe_vamp_center", "right_shoe_lace_center",
    "right_shoe_inner_side", "right_shoe_outer_side",
    "right_shoe_heel",
    "right_shoe_sole_front", "right_shoe_sole_back",
)

BOOTS_LANDMARKS = (
    # Left
    "left_boot_top_outer", "left_boot_top_inner",
    "left_boot_calf_outer", "left_boot_calf_inner",
    "left_boot_ankle_outer", "left_boot_ankle_inner",
    "left_boot_toe", "left_boot_heel",
    "left_boot_sole_front", "left_boot_sole_back",
    # Right
    "right_boot_top_outer", "right_boot_top_inner",
    "right_boot_calf_outer", "right_boot_calf_inner",
    "right_boot_ankle_outer", "right_boot_ankle_inner",
    "right_boot_toe", "right_boot_heel",
    "right_boot_sole_front", "right_boot_sole_back",
)

FOOTWEAR_BODY_TARGETS = {
    "left_shoe_heel": "left_heel",
    "right_shoe_heel": "right_heel",
    "left_shoe_toe": "left_foot_index",
    "right_shoe_toe": "right_foot_index",
    "left_shoe_inner_side": "left_ankle_inner",
    "right_shoe_inner_side": "right_ankle_inner",
    # Boot tops — choose by shaft height (ankle / calf / knee)
    "left_boot_top_outer": "left_ankle_or_calf_or_knee_outer",
    "right_boot_top_outer": "right_ankle_or_calf_or_knee_outer",
}

__all__ = ["SHOE_LANDMARKS", "BOOTS_LANDMARKS", "FOOTWEAR_BODY_TARGETS"]
