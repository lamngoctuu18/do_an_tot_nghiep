"""Dress (one-piece) landmark vocabulary.

Combines bodice (top half) and skirt (lower half). `MIN_DRESS_EXTRA` is
the lower-half subset to detect first — bodice landmarks already overlap
with `top_landmarks.CURRENT_TPS_24`.

For sleeveless dresses, the sleeve-related entries must be EXCLUDED from
the warp control set; the mask builder also forbids painting on the arms
to avoid phantom sleeves.
"""
from __future__ import annotations


DRESS_LANDMARKS = (
    # Neck / collar
    "neck_left", "neck_right", "neck_center",
    "collar_left", "collar_right", "collar_center",
    "neckline_lowest",
    # Shoulder / straps
    "shoulder_left", "shoulder_right",
    "strap_left_top", "strap_right_top",
    "strap_left_bottom", "strap_right_bottom",
    # Armhole / sleeve
    "armpit_left", "armpit_right",
    "armhole_left", "armhole_right",
    "sleeve_root_left", "sleeve_root_right",
    "sleeve_tip_left", "sleeve_tip_right",
    "sleeve_cuff_left_outer", "sleeve_cuff_left_inner",
    "sleeve_cuff_right_outer", "sleeve_cuff_right_inner",
    # Bust / bodice
    "bust_left", "bust_right", "bust_center",
    "under_bust_left", "under_bust_right", "under_bust_center",
    "bodice_left", "bodice_right", "bodice_center",
    # Waist
    "waist_left", "waist_right", "waist_center",
    "waist_seam_left", "waist_seam_right", "waist_seam_center",
    # Hip / skirt upper
    "hip_left", "hip_right", "hip_center",
    "skirt_top_left", "skirt_top_right", "skirt_top_center",
    # Skirt body
    "skirt_mid_left", "skirt_mid_right", "skirt_mid_center",
    "skirt_lower_left", "skirt_lower_right", "skirt_lower_center",
    # Hem
    "hem_left", "hem_right", "hem_center",
    "hem_front_left", "hem_front_right",
    "hem_back_left", "hem_back_right",
    # Silhouette controls
    "a_line_left", "a_line_right",
    "flare_left", "flare_right",
    "mermaid_knee_left", "mermaid_knee_right",
    "slit_top", "slit_bottom",
)

# Lower half subset — combine with top_landmarks.CURRENT_TPS_24 for the
# bodice to get a full dress control set.
MIN_DRESS_EXTRA = (
    "bust_left", "bust_right", "bust_center",
    "waist_left", "waist_right", "waist_center",
    "hip_left", "hip_right", "hip_center",
    "skirt_mid_left", "skirt_mid_right", "skirt_mid_center",
    "hem_left", "hem_right", "hem_center",
    "flare_left", "flare_right",
    "slit_top", "slit_bottom",
)

# Hem mapping depends on dress length — pipeline must pick one based on
# detected silhouette (see `src/garment_silhouettes.py`).
DRESS_BODY_TARGETS = {
    "neck_center": "neck_center",
    "shoulder_left": "left_shoulder",
    "shoulder_right": "right_shoulder",
    "bust_center": "chest_center",
    "waist_center": "waist_center",
    "hip_center": "hip_center",
    "hem_center_mini": "mid_thigh_center",
    "hem_center_knee": "knee_center",
    "hem_center_midi": "calf_center",
    "hem_center_maxi": "ankle_center",
}

# When sleeve_type == "sleeveless", drop everything here from the warp set.
SLEEVE_KEYS_TO_DROP_WHEN_SLEEVELESS = (
    "sleeve_root_left", "sleeve_root_right",
    "sleeve_tip_left", "sleeve_tip_right",
    "sleeve_cuff_left_outer", "sleeve_cuff_left_inner",
    "sleeve_cuff_right_outer", "sleeve_cuff_right_inner",
)

__all__ = [
    "DRESS_LANDMARKS", "MIN_DRESS_EXTRA",
    "DRESS_BODY_TARGETS", "SLEEVE_KEYS_TO_DROP_WHEN_SLEEVELESS",
]
