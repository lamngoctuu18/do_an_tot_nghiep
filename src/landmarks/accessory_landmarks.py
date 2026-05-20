"""Accessory landmark vocabulary.

One module covers the small accessories (belt, bag, scarf, tie, headwear,
sunglasses, gloves, socks) because their landmark counts are low. Each
group also gets its own body-target mapping — accessories anchor against
specific body landmarks (eyes for sunglasses, waist for belt, etc.), not
against the full pose envelope used for clothing.
"""
from __future__ import annotations


BAG_LANDMARKS = (
    "bag_top_left", "bag_top_right",
    "bag_bottom_left", "bag_bottom_right",
    "bag_center",
    "bag_left_mid", "bag_right_mid",
    "handle_left_base", "handle_right_base", "handle_top_center",
    "strap_top", "strap_mid", "strap_bottom",
    "strap_left_edge", "strap_right_edge",
    "flap_left", "flap_right", "flap_center",
    "zipper_left", "zipper_right", "zipper_center",
    "contact_shoulder", "contact_hand", "contact_hip",
)

# Different bag styles anchor differently. The pipeline must pick a style
# before mapping — there's no single "bag → body" rule.
BAG_BODY_TARGETS = {
    "shoulder_bag":  ("left_shoulder|right_shoulder", "opposite_hip"),
    "handbag":       ("left_wrist|right_wrist",),
    "crossbody":     ("shoulder_center", "opposite_hip"),
    "backpack":      ("left_shoulder", "right_shoulder", "torso_back_region"),
}

BELT_LANDMARKS = (
    "belt_left_end", "belt_right_end", "belt_center",
    "belt_top_left", "belt_top_right",
    "belt_bottom_left", "belt_bottom_right",
    "buckle_top_left", "buckle_top_right",
    "buckle_bottom_left", "buckle_bottom_right",
    "buckle_center",
    "belt_hole_1", "belt_hole_2", "belt_hole_3", "belt_hole_4",
)

BELT_BODY_TARGETS = {
    "belt_left_end": "left_waist",
    "belt_right_end": "right_waist",
    "belt_center": "waist_center",
    "buckle_center": "waist_center",
}

SCARF_LANDMARKS = (
    "scarf_neck_left", "scarf_neck_right", "scarf_neck_center",
    "scarf_knot_center", "scarf_knot_left", "scarf_knot_right",
    "scarf_tail_left_top", "scarf_tail_left_bottom",
    "scarf_tail_right_top", "scarf_tail_right_bottom",
    "scarf_outer_left", "scarf_outer_right",
    "scarf_bottom_center",
)

TIE_LANDMARKS = (
    "tie_knot_top", "tie_knot_bottom",
    "tie_knot_left", "tie_knot_right",
    "tie_center_top", "tie_center_mid",
    "tie_tip",
    "tie_left_edge_mid", "tie_right_edge_mid",
)

NECK_ACCESSORY_BODY_TARGETS = {
    "scarf_neck_left": "neck_left",
    "scarf_neck_right": "neck_right",
    "scarf_neck_center": "neck_center",
    "tie_tip": "chest_center",
    "scarf_bottom_center": "chest_or_waist_center",
}

HEADWEAR_LANDMARKS = (
    "hat_top",
    "hat_bottom_left", "hat_bottom_right", "hat_center",
    "hat_brim_left", "hat_brim_right", "hat_brim_center",
    "hat_crown_left", "hat_crown_right", "hat_crown_center",
    # Cap
    "cap_bill_left", "cap_bill_right", "cap_bill_tip",
    # Contact line — where the hat rim meets the head
    "head_contact_left", "head_contact_right",
    "forehead_contact_center",
)

HEADWEAR_BODY_TARGETS = {
    "hat_center": "head_center",
    "hat_bottom_left": "left_ear_or_left_head_side",
    "hat_bottom_right": "right_ear_or_right_head_side",
    "forehead_contact_center": "forehead_center",
}

SUNGLASSES_LANDMARKS = (
    "left_lens_left", "left_lens_right",
    "left_lens_top", "left_lens_bottom", "left_lens_center",
    "right_lens_left", "right_lens_right",
    "right_lens_top", "right_lens_bottom", "right_lens_center",
    "bridge_center",
    "left_temple", "right_temple",
)

SUNGLASSES_BODY_TARGETS = {
    "left_lens_center": "left_eye",
    "right_lens_center": "right_eye",
    "bridge_center": "nose",
    "left_temple": "left_ear",
    "right_temple": "right_ear",
}

GLOVE_LANDMARKS = (
    # Left
    "left_glove_wrist_left", "left_glove_wrist_right",
    "left_glove_palm_center",
    "left_glove_thumb_tip", "left_glove_index_tip",
    "left_glove_middle_tip", "left_glove_ring_tip", "left_glove_pinky_tip",
    # Right
    "right_glove_wrist_left", "right_glove_wrist_right",
    "right_glove_palm_center",
    "right_glove_thumb_tip", "right_glove_index_tip",
    "right_glove_middle_tip", "right_glove_ring_tip", "right_glove_pinky_tip",
)

GLOVE_BODY_TARGETS = {
    "left_glove_palm_center": "left_wrist",
    "right_glove_palm_center": "right_wrist",
    "left_glove_thumb_tip": "left_thumb",
    "right_glove_thumb_tip": "right_thumb",
    "left_glove_index_tip": "left_index",
    "right_glove_index_tip": "right_index",
    "left_glove_pinky_tip": "left_pinky",
    "right_glove_pinky_tip": "right_pinky",
}

SOCKS_LANDMARKS = (
    # Left
    "left_sock_top_outer", "left_sock_top_inner",
    "left_sock_calf_outer", "left_sock_calf_inner",
    "left_sock_ankle_outer", "left_sock_ankle_inner",
    "left_sock_toe", "left_sock_heel",
    # Right
    "right_sock_top_outer", "right_sock_top_inner",
    "right_sock_calf_outer", "right_sock_calf_inner",
    "right_sock_ankle_outer", "right_sock_ankle_inner",
    "right_sock_toe", "right_sock_heel",
)

SOCKS_BODY_TARGETS = {
    "left_sock_top_outer": "left_ankle_or_calf_or_knee_outer",
    "right_sock_top_outer": "right_ankle_or_calf_or_knee_outer",
    "left_sock_ankle_outer": "left_ankle_outer",
    "right_sock_ankle_outer": "right_ankle_outer",
    "left_sock_toe": "left_foot_index",
    "right_sock_toe": "right_foot_index",
    "left_sock_heel": "left_heel",
    "right_sock_heel": "right_heel",
}

__all__ = [
    "BAG_LANDMARKS", "BAG_BODY_TARGETS",
    "BELT_LANDMARKS", "BELT_BODY_TARGETS",
    "SCARF_LANDMARKS", "TIE_LANDMARKS", "NECK_ACCESSORY_BODY_TARGETS",
    "HEADWEAR_LANDMARKS", "HEADWEAR_BODY_TARGETS",
    "SUNGLASSES_LANDMARKS", "SUNGLASSES_BODY_TARGETS",
    "GLOVE_LANDMARKS", "GLOVE_BODY_TARGETS",
    "SOCKS_LANDMARKS", "SOCKS_BODY_TARGETS",
]
