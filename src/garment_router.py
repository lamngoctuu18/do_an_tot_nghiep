"""Garment routing layer.

Centralises garment classification so app.py no longer scatters detection
calls across the pipeline. Wraps the existing classifiers in tps_warp.py
into a single `route_garment()` returning a structured result.

This is a thin facade — no detection logic changes here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .tps_warp import detect_garment_category, detect_pants_type, detect_pants_style
from .accessories import normalise_accessory_subtype, classify_accessory_subtype


@dataclass(frozen=True)
class GarmentRoute:
    category: str             # "top" | "pants" | "dress" | "accessory"
    subtype: str              # pants_type or sleeve hint
    pants_style: str          # "skinny" | "wide_leg" | "straight" | "regular"
    sleeve_type: str          # "short" | "long" | "sleeveless"
    confidence: float
    accessory_subtype: str = ""  # "shoes"|"boots"|"hat"|"sunglasses"|"belt"|"bag"|"scarf"|""


_CATEGORY_ALIASES = {
    "top": "top",
    "upper": "top",
    "shirt": "top",
    "tshirt": "top",
    "t-shirt": "top",
    "pants": "pants",
    "bottom": "pants",
    "trouser": "pants",
    "shorts": "pants",
    "dress": "dress",
    "skirt": "dress",         # treated as dress-lower in current pipeline
    "accessory": "accessory",
    "bag": "accessory",
    "hat": "accessory",
    "shoes": "accessory",
    "boots": "accessory",
    "belt": "accessory",
    "scarf": "accessory",
    "sunglasses": "accessory",
    "glasses": "accessory",
}


def normalise_category(value: Optional[str]) -> Optional[str]:
    """Map a user-supplied cloth_type string to internal category."""
    if not value:
        return None
    return _CATEGORY_ALIASES.get(str(value).lower().strip())


def route_garment(
    cloth_mask: np.ndarray,
    user_selected_type: Optional[str] = None,
    sleeve_type: str = "short",
) -> GarmentRoute:
    """Classify the garment using existing detectors.

    `user_selected_type` overrides automatic category when provided —
    callers (UI) often know better than the mask heuristic.
    `sleeve_type` is passed through (sleeve detection still lives in app.py
    where the parsing model is available).
    """
    override = normalise_category(user_selected_type)
    if override == "accessory":
        sub = normalise_accessory_subtype(user_selected_type)
        if not sub:
            sub = classify_accessory_subtype(cloth_mask)
        return GarmentRoute(
            category="accessory",
            subtype="generic",
            pants_style="regular",
            sleeve_type=sleeve_type,
            confidence=1.0,
            accessory_subtype=sub,
        )

    category = override or detect_garment_category(cloth_mask)
    pants_subtype = detect_pants_type(cloth_mask) if category == "pants" else "n/a"
    pants_style = detect_pants_style(cloth_mask) if category == "pants" else "regular"

    return GarmentRoute(
        category=category,
        subtype=pants_subtype,
        pants_style=pants_style,
        sleeve_type=sleeve_type,
        confidence=1.0 if override else 0.85,
    )


__all__ = ["GarmentRoute", "route_garment", "normalise_category"]
