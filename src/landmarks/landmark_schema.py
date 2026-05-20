"""Unified garment-landmark schema.

Aggregates every per-category landmark module so callers can look up the
vocabulary for a category with one import:

    from src.landmarks.landmark_schema import (
        GARMENT_LANDMARK_SCHEMA,
        BODY_TARGET_SCHEMA,
        PRIORITY_FOR_VTON,
    )

`GARMENT_LANDMARK_SCHEMA` maps each category name to its full landmark
tuple. `BODY_TARGET_SCHEMA` maps each category to its garment→body target
dict (where applicable). `PRIORITY_FOR_VTON` is the ordered rollout plan —
implement detection in this order to fix the most common VTON errors
first (pants hip slip, dress form, phantom sleeves, accessory bleed).
"""
from __future__ import annotations

from . import (
    body_landmarks,
    top_landmarks,
    outer_landmarks,
    pants_landmarks,
    dress_landmarks,
    skirt_landmarks,
    jumpsuit_landmarks,
    footwear_landmarks,
    accessory_landmarks,
)


GARMENT_LANDMARK_SCHEMA = {
    "top":         top_landmarks.TOP_LANDMARKS,
    "outer":       outer_landmarks.OUTER_LANDMARKS,
    "pants":       pants_landmarks.PANTS_LANDMARKS,
    "shorts":      pants_landmarks.SHORTS_LANDMARKS,
    "dress":       dress_landmarks.DRESS_LANDMARKS,
    "skirt":       skirt_landmarks.SKIRT_LANDMARKS,
    "jumpsuit":    jumpsuit_landmarks.JUMPSUIT_LANDMARKS,
    "shoes":       footwear_landmarks.SHOE_LANDMARKS,
    "boots":       footwear_landmarks.BOOTS_LANDMARKS,
    "bag":         accessory_landmarks.BAG_LANDMARKS,
    "belt":        accessory_landmarks.BELT_LANDMARKS,
    "scarf":       accessory_landmarks.SCARF_LANDMARKS,
    "tie":         accessory_landmarks.TIE_LANDMARKS,
    "headwear":    accessory_landmarks.HEADWEAR_LANDMARKS,
    "sunglasses":  accessory_landmarks.SUNGLASSES_LANDMARKS,
    "gloves":      accessory_landmarks.GLOVE_LANDMARKS,
    "socks":       accessory_landmarks.SOCKS_LANDMARKS,
}

BODY_TARGET_SCHEMA = {
    "top":         top_landmarks.BODY_TARGETS,
    "outer":       outer_landmarks.BODY_TARGETS,
    "pants":       pants_landmarks.PANTS_BODY_TARGETS,
    "shorts":      pants_landmarks.SHORTS_BODY_TARGETS,
    "dress":       dress_landmarks.DRESS_BODY_TARGETS,
    "skirt":       skirt_landmarks.SKIRT_BODY_TARGETS,
    "jumpsuit":    jumpsuit_landmarks.JUMPSUIT_BODY_TARGETS,
    "shoes":       footwear_landmarks.FOOTWEAR_BODY_TARGETS,
    "boots":       footwear_landmarks.FOOTWEAR_BODY_TARGETS,
    "bag":         accessory_landmarks.BAG_BODY_TARGETS,
    "belt":        accessory_landmarks.BELT_BODY_TARGETS,
    "scarf":       accessory_landmarks.NECK_ACCESSORY_BODY_TARGETS,
    "tie":         accessory_landmarks.NECK_ACCESSORY_BODY_TARGETS,
    "headwear":    accessory_landmarks.HEADWEAR_BODY_TARGETS,
    "sunglasses":  accessory_landmarks.SUNGLASSES_BODY_TARGETS,
    "gloves":      accessory_landmarks.GLOVE_BODY_TARGETS,
    "socks":       accessory_landmarks.SOCKS_BODY_TARGETS,
}

# Minimum subsets — implement detection for these first.
MIN_SCHEMA = {
    "top":   top_landmarks.CURRENT_TPS_24,
    "pants": pants_landmarks.MIN_PANTS,
    "dress": dress_landmarks.MIN_DRESS_EXTRA,
}

# Ordered rollout — each entry maps to a (category, detector_module) pair
# the team should implement next.
PRIORITY_FOR_VTON = (
    "body_33_mediapipe",      # extend detect_full_pose() — DONE
    "top_24_tps_current",     # already shipped in tps_warp.py
    "pants_26_min",           # extend detect_pants_landmarks()
    "dress_upper_plus_skirt", # combine top_24 + MIN_DRESS_EXTRA
    "belt_4",
    "bag_7",
    "shoes_4",
    "sunglasses_5",
    "headwear_5",
    "scarf_5",
)


def schema_for(category: str) -> tuple[str, ...]:
    """Return the full landmark vocabulary for `category`."""
    return GARMENT_LANDMARK_SCHEMA.get((category or "").lower(), ())


def body_targets_for(category: str) -> dict[str, object]:
    """Return the garment→body target map for `category`."""
    return BODY_TARGET_SCHEMA.get((category or "").lower(), {})


def all_body_landmark_names() -> tuple[str, ...]:
    return body_landmarks.ALL_BODY


__all__ = [
    "GARMENT_LANDMARK_SCHEMA",
    "BODY_TARGET_SCHEMA",
    "MIN_SCHEMA",
    "PRIORITY_FOR_VTON",
    "schema_for",
    "body_targets_for",
    "all_body_landmark_names",
]
