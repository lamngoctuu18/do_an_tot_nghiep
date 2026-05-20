"""Strict category lock — hard allow/forbid registry.

Behaviour change vs. v18.x: after the category mask builder composes its
allow region (parsing ∪ envelope ∪ garment support − protect), we run a
SECOND pass that forcibly subtracts a stronger, generously dilated
"forbidden parts" mask. This stops cross-category bleed in three known
failure modes:

    top      → diffusion ever touching pants / legs / shoes
    pants    → diffusion ever touching torso shirt / arms / face
    dress    → diffusion ever touching shoes / face / hair
    accessory → diffusion ever touching anything outside the warped footprint

The original `_protect` set inside `category_mask_builder` already does a
softer version of this. The lock is stricter on purpose: larger dilation,
no overlap allowance, and explicit per-category forbid keys instead of an
implicit complement.

Public API:
    apply_category_lock(mask, category, parsing, *,
                        parsing_union_mask, fit_like,
                        garment_mask=None) -> np.ndarray
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


# Hard forbid lists per category. Anything here is subtracted from the
# diffusion mask with a fat dilation kernel after compose.
FORBID = {
    "top": (
        "pants", "skirt", "left_leg", "right_leg",
        "left_shoe", "right_shoe", "face", "hair", "hat", "sunglasses",
    ),
    "pants": (
        "upper_clothes", "dress", "left_arm", "right_arm",
        "face", "hair", "hat", "sunglasses", "scarf",
    ),
    "dress": (
        "face", "hair", "hat", "sunglasses",
        "left_shoe", "right_shoe",
    ),
    "accessory": (
        "face", "hair", "upper_clothes", "dress", "skirt", "pants",
        "left_arm", "right_arm", "left_leg", "right_leg",
        "left_shoe", "right_shoe",
    ),
}

# Dilation radius per category (pixels). Bigger = stricter lock.
DILATE = {
    "top": 9,
    "pants": 9,
    "dress": 7,
    "accessory": 3,
}


def apply_category_lock(
    mask: np.ndarray,
    category: str,
    parsing: Optional[dict],
    *,
    parsing_union_mask: Callable,
    fit_like: Callable,
    garment_mask: Optional[np.ndarray] = None,
    subtype: str = "",
) -> np.ndarray:
    """Subtract category-forbidden parts from `mask`.

    Falls back to a no-op when parsing is missing — the caller already
    has a softer protect-pass inside the mask builder so we don't want
    to over-erase.

    `subtype` lets the caller relax forbids per garment variant — e.g.
    a hoodie needs the hood to drape over the back of the hair, so we
    drop "hair" (and "hat") from FORBID["top"] when subtype == "hoodie".
    """
    cat = (category or "top").lower()
    sub = (subtype or "").lower()
    if cat not in FORBID or parsing is None:
        return mask

    forbid_keys = FORBID[cat]
    if cat == "top" and sub == "hoodie":
        forbid_keys = tuple(k for k in forbid_keys if k not in {"hair", "hat"})

    h, w = mask.shape[:2]
    forbid = parsing_union_mask(parsing, forbid_keys, (h, w))
    if int(cv2.countNonZero(forbid)) < 255 * 20:
        return mask

    radius = max(1, int(DILATE.get(cat, 5)))
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    forbid = cv2.dilate(forbid, kernel, iterations=1)

    # Accessory: also clamp to the warped garment footprint — diffusion
    # must never paint outside the accessory bounds.
    if cat == "accessory" and garment_mask is not None:
        gm = fit_like(garment_mask, mask, is_mask=True)
        gm = cv2.dilate((gm > 20).astype(np.uint8) * 255,
                        np.ones((5, 5), np.uint8), iterations=1)
        mask = cv2.bitwise_and(mask, gm)

    locked = cv2.subtract(mask, forbid)
    return (locked > 20).astype(np.uint8) * 255


__all__ = ["FORBID", "DILATE", "apply_category_lock"]
