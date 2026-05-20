"""Accessory subtype taxonomy + classification.

Subtype is a first-class field. UI strings flow in via category_lock; we
also offer a shape-based fallback in case the UI doesn't specify.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


ACCESSORY_SUBTYPES = (
    "shoes", "boots", "hat", "sunglasses", "belt", "bag", "scarf",
)

_SUBTYPE_ALIASES = {
    "shoes": "shoes", "shoe": "shoes", "sneaker": "shoes", "sneakers": "shoes",
    "sandal": "shoes", "sandals": "shoes", "footwear": "shoes",
    "boots": "boots", "boot": "boots",
    "hat": "hat", "cap": "hat", "mu": "hat", "non": "hat", "headwear": "hat",
    "sunglasses": "sunglasses", "glasses": "sunglasses", "kinh": "sunglasses",
    "belt": "belt", "that_lung": "belt",
    "bag": "bag", "handbag": "bag", "shoulder_bag": "bag", "crossbody": "bag",
    "backpack": "bag", "tui": "bag",
    "scarf": "scarf", "khan": "scarf", "tie": "scarf",
}


def normalise_accessory_subtype(value: Optional[str]) -> str:
    if not value:
        return ""
    return _SUBTYPE_ALIASES.get(str(value).lower().strip().replace(" ", "_"), "")


def classify_accessory_subtype(cloth_mask: Optional[np.ndarray]) -> str:
    """Shape-based fallback when UI doesn't supply a subtype.

    Heuristic only — UI selection is preferred. Returns "" if undecidable.
    """
    if cloth_mask is None:
        return ""
    m = (cloth_mask > 20).astype(np.uint8)
    if int(m.sum()) < 200:
        return ""
    ys, xs = np.where(m > 0)
    h = max(1, int(ys.max() - ys.min() + 1))
    w = max(1, int(xs.max() - xs.min() + 1))
    aspect = w / float(h)
    fill = float(m.sum()) / float(h * w)
    if aspect > 3.0 and fill > 0.45:
        return "belt"
    if aspect > 1.6 and fill > 0.30 and h < 0.45 * (h + w):
        return "sunglasses"
    if aspect > 1.4 and fill > 0.40:
        return "shoes"
    if aspect < 0.9 and fill > 0.45:
        return "bag"
    if 0.9 <= aspect <= 1.4:
        return "hat"
    return ""


__all__ = [
    "ACCESSORY_SUBTYPES",
    "normalise_accessory_subtype",
    "classify_accessory_subtype",
]
