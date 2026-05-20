"""Accessory pipeline (Phase 0+).

Subtype-aware try-on for: shoes, boots, hat, sunglasses, belt, bag, scarf.
Each subtype owns its anchor extraction, warp, mask, and postprocess.
"""
from .subtype import (
    ACCESSORY_SUBTYPES,
    normalise_accessory_subtype,
    classify_accessory_subtype,
)

__all__ = [
    "ACCESSORY_SUBTYPES",
    "normalise_accessory_subtype",
    "classify_accessory_subtype",
]
