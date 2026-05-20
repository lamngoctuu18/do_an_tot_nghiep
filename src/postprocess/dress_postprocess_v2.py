"""Dress v2 postprocess utilities.

Every blend is clipped by the dress target mask, so pattern reference and
color anchor cannot bleed outside the dress footprint (kills the rectangular
brown-block artifact at the root).
"""
from __future__ import annotations

import cv2
import numpy as np


def _safe_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _soft(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    m = (mask > 0).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), sigma)


def apply_color_anchor(
    out_rgb: np.ndarray,
    ref_rgb: np.ndarray,
    target_mask: np.ndarray,
    strength: float = 0.25,
    ref_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Match colour inside `target_mask` toward `ref_rgb` in LAB a/b channels.

    L channel is left untouched so vertical pleats / fold shading survive.
    Strength default lowered to 0.25 — higher values flatten the fabric.
    """
    bool_mask = target_mask > 64
    if int(bool_mask.sum()) < 50:
        return out_rgb
    out_lab = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if ref_lab.shape == out_lab.shape:
        ref_ab_mean = ref_lab[..., 1:][bool_mask].mean(axis=0)
    else:
        if ref_mask is not None and ref_mask.shape[:2] == ref_rgb.shape[:2] and int((ref_mask > 20).sum()) > 50:
            ref_valid = ref_mask > 20
        else:
            ref_i = ref_rgb.astype(np.int16)
            ref_valid = ~((ref_i[..., 0] > 232) & (ref_i[..., 1] > 232) & (ref_i[..., 2] > 232))
        if int(ref_valid.sum()) < 50:
            ref_valid = np.ones(ref_rgb.shape[:2], dtype=bool)
        ref_ab_mean = ref_lab[..., 1:][ref_valid].mean(axis=0)
    out_ab_mean = out_lab[..., 1:][bool_mask].mean(axis=0)
    shift = (ref_ab_mean - out_ab_mean) * float(np.clip(strength, 0.0, 1.0))
    alpha = _soft(target_mask, 2.0)[..., None]
    out_lab[..., 1:] = out_lab[..., 1:] + shift[None, None, :] * alpha
    out_lab = np.clip(out_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2RGB)


def remove_old_dress_ghost(
    out_rgb: np.ndarray,
    person_rgb: np.ndarray,
    old_clothes_mask: np.ndarray,
    target_mask: np.ndarray,
    chroma_thresh: float = 28.0,
) -> tuple[np.ndarray, bool]:
    """Erase old garment leftovers outside the new dress footprint.

    For pixels in (old_clothes_mask − target_mask) that still match the mean
    old-garment colour, run Telea inpaint from surrounding non-garment pixels.
    """
    spill_zone = cv2.subtract(old_clothes_mask, cv2.dilate(target_mask, np.ones((5, 5), np.uint8), iterations=1))
    if int(spill_zone.sum()) < 255 * 30:
        return out_rgb, False

    oc_bool = old_clothes_mask > 0
    if int(oc_bool.sum()) < 30:
        return out_rgb, False
    old_mean = person_rgb[oc_bool].astype(np.float32).mean(axis=0)
    diff = np.linalg.norm(out_rgb.astype(np.float32) - old_mean[None, None, :], axis=2)
    spill = ((diff < chroma_thresh) & (spill_zone > 0)).astype(np.uint8) * 255

    if int(spill.sum()) < 255 * 20:
        return out_rgb, False
    spill = cv2.morphologyEx(spill, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    spill_dilated = cv2.dilate(spill, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(out_rgb, spill_dilated, 3, cv2.INPAINT_TELEA)
    alpha = _soft(spill_dilated, 1.5)[..., None]
    blended = out_rgb.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
    return _safe_uint8(blended), True


def clean_hem(
    out_rgb: np.ndarray,
    person_rgb: np.ndarray,
    target_mask: np.ndarray,
    shoe_protect: np.ndarray,
) -> np.ndarray:
    """Feather hem and restore any pixels that drifted onto shoes."""
    if int(shoe_protect.sum()) < 50:
        return out_rgb
    bleed = cv2.bitwise_and(shoe_protect, cv2.dilate(target_mask, np.ones((11, 11), np.uint8), iterations=1))
    if int(bleed.sum()) < 30:
        return out_rgb
    alpha = _soft(bleed, 1.5)[..., None]
    restored = out_rgb.astype(np.float32) * (1.0 - alpha) + person_rgb.astype(np.float32) * alpha
    return _safe_uint8(restored)


def remove_rect_artifact(
    out_rgb: np.ndarray,
    fallback_rgb: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Kill brown rectangular pattern-reference leakage outside dress mask."""
    h, w = out_rgb.shape[:2]
    allowed = cv2.dilate(target_mask, np.ones((9, 9), np.uint8), iterations=1)
    allowed = cv2.GaussianBlur(allowed, (7, 7), 1.5)
    allowed = (allowed > 20).astype(np.uint8) * 255

    out = out_rgb.astype(np.float32)
    fb = fallback_rgb.astype(np.float32)
    diff = np.mean(np.abs(out - fb), axis=2)
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    brownish = (r > 90) & (g > 65) & (b > 45) & (r > b + 12) & (g > b + 5)
    outside = allowed < 20
    spill = ((diff > 12) & brownish & outside).astype(np.uint8) * 255
    if int(spill.sum()) < 20:
        return out_rgb, False
    spill = cv2.morphologyEx(spill, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    alpha = _soft(spill, 1.5)[..., None]
    cleaned = out * (1.0 - alpha) + fb * alpha
    return _safe_uint8(cleaned), True


def telea_inpaint_seed(
    person_rgb: np.ndarray,
    erase_mask: np.ndarray,
) -> np.ndarray:
    """Replace the legacy mean-skin/mean-bg flat fill with a Telea inpaint.

    This is the seed image that goes into diffusion: old-garment region is
    smoothly filled with surrounding pixels, no rectangular colour patch.
    """
    if int(erase_mask.sum()) < 30:
        return person_rgb.copy()
    m = (erase_mask > 0).astype(np.uint8) * 255
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(person_rgb, m, 5, cv2.INPAINT_TELEA)
