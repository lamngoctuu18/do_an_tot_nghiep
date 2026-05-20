"""Paste hair_front / hands_outside / shoes back over the diffusion output.

Uses an erode-then-blur soft mask so edges are clean. Mirrors
`_apply_foreground_layer` in `app.py:544` but kept independent to avoid
importing from app.py (circular risk).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.masks.dress_mask_builder import DressMasks


def _soft_mask(mask: np.ndarray, blur_sigma: float = 1.5, erode_px: int = 2) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        m = cv2.erode(m, k, iterations=1)
    m_f = m.astype(np.float32) / 255.0
    if blur_sigma > 0:
        m_f = cv2.GaussianBlur(m_f, (0, 0), blur_sigma)
    return np.clip(m_f, 0.0, 1.0)


def _layer(base: np.ndarray, src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(mask.sum()) == 0:
        return base
    alpha = _soft_mask(mask)[..., None]
    out = base.astype(np.float32) * (1.0 - alpha) + src.astype(np.float32) * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def restore_occluders(
    diffusion_out: np.ndarray,
    person_rgb: np.ndarray,
    masks: DressMasks,
) -> np.ndarray:
    """Composite hair_front, hands_outside, and shoes back onto diffusion output."""
    out = diffusion_out
    out = _layer(out, person_rgb, masks.shoe_protect_mask)
    out = _layer(out, person_rgb, masks.hand_protect_mask)
    out = _layer(out, person_rgb, masks.hair_front_mask)
    return out
