"""Accessory postprocess: per-subtype spill guards / face protect / ground guard.

Applied AFTER diffusion. Same shape contract as pants postprocess:
    output_rgb, init_tryon_rgb, mask, parsing, pose, *, fit_like, safe_uint8
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2
import numpy as np


def _protect_blend(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    protect_mask: np.ndarray,
    *,
    safe_uint8: Callable,
    dilate: int = 5,
    blur_sigma: float = 2.0,
) -> np.ndarray:
    if int(cv2.countNonZero(protect_mask)) == 0:
        return output_rgb
    if dilate > 0:
        k = np.ones((dilate, dilate), np.uint8)
        protect_mask = cv2.dilate(protect_mask, k, iterations=1)
    alpha = cv2.GaussianBlur(protect_mask.astype(np.float32) / 255.0,
                              (9, 9), blur_sigma)[..., None]
    blended = output_rgb.astype(np.float32) * (1.0 - alpha) + person_rgb.astype(np.float32) * alpha
    return safe_uint8(blended)


def _spill_guard(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    allowed: np.ndarray,
    gen_mask_soft: Optional[np.ndarray],
    *,
    safe_uint8: Callable,
    dilate: int = 5,
) -> np.ndarray:
    if gen_mask_soft is None:
        gen = ((np.abs(output_rgb.astype(np.float32) - init_tryon_rgb.astype(np.float32)).mean(axis=2)) > 12).astype(np.uint8) * 255
    else:
        gen = gen_mask_soft
    allowed_d = cv2.dilate(allowed, np.ones((dilate, dilate), np.uint8), iterations=1)
    spill = cv2.bitwise_and(gen, cv2.bitwise_not(allowed_d))
    if int(cv2.countNonZero(spill)) < 30:
        return output_rgb
    alpha = cv2.GaussianBlur((spill > 12).astype(np.float32), (9, 9), 2.0)[..., None]
    cleaned = output_rgb.astype(np.float32) * (1.0 - alpha) + init_tryon_rgb.astype(np.float32) * alpha
    return safe_uint8(cleaned)


def apply_shoes_pose_guard(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    gen_mask_soft: Optional[np.ndarray],
    parsing: Optional[dict],
    full_pose: Optional[dict],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
    is_boots: bool = False,
) -> np.ndarray:
    out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                        safe_uint8=safe_uint8, dilate=5)
    # protect legs above the boot/shoe upper cut line
    if parsing is not None and not is_boots:
        h, w = out.shape[:2]
        ref = np.zeros((h, w), dtype=np.uint8)
        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("left_leg", "right_leg", "pants", "skirt", "upper_clothes",
                     "dress", "left_arm", "right_arm", "face", "hair"):
            part = parsing.get(key)
            if part is not None:
                protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
        # only protect ABOVE the allowed footprint
        ys, _ = np.where(allowed > 20)
        if len(ys) > 50:
            top_y = max(0, int(ys.min()) - 4)
            protect[top_y:, :] = 0
        out = _protect_blend(out, person_rgb, protect,
                              safe_uint8=safe_uint8, dilate=5)
    return out


def apply_hat_face_protect(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    parsing: Optional[dict],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    if parsing is None:
        return output_rgb
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    protect = np.zeros((h, w), dtype=np.uint8)
    for key in ("face", "left_arm", "right_arm", "upper_clothes", "dress",
                 "sunglasses"):
        part = parsing.get(key)
        if part is not None:
            protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
    return _protect_blend(output_rgb, person_rgb, protect,
                           safe_uint8=safe_uint8, dilate=5)


def apply_glasses_face_protect(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    parsing: Optional[dict],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    """Glasses: hard-protect face except a narrow eye band (the lens area)."""
    if parsing is None:
        return output_rgb
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    face = parsing.get("face")
    if face is None:
        return output_rgb
    face = fit_like(face, ref, is_mask=True)
    # keep `allowed` (lens band) as the only "allow edit" inside face
    eye_band = cv2.dilate(allowed, np.ones((5, 5), np.uint8), iterations=1)
    protect = cv2.subtract(face, eye_band)
    return _protect_blend(output_rgb, person_rgb, protect,
                           safe_uint8=safe_uint8, dilate=3)


def apply_belt_protect(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    parsing: Optional[dict],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    if parsing is None:
        return output_rgb
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    protect = np.zeros((h, w), dtype=np.uint8)
    for key in ("upper_clothes", "pants", "skirt", "dress",
                 "left_arm", "right_arm", "face", "hair"):
        part = parsing.get(key)
        if part is not None:
            protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
    # remove the belt band itself from protect
    band = cv2.dilate(allowed, np.ones((3, 3), np.uint8), iterations=1)
    protect = cv2.subtract(protect, band)
    return _protect_blend(output_rgb, person_rgb, protect,
                           safe_uint8=safe_uint8, dilate=3)


def apply_bag_protect(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    parsing: Optional[dict],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    if parsing is None:
        return output_rgb
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    protect = np.zeros((h, w), dtype=np.uint8)
    for key in ("face", "hair", "hat"):
        part = parsing.get(key)
        if part is not None:
            protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
    return _protect_blend(output_rgb, person_rgb, protect,
                           safe_uint8=safe_uint8, dilate=5)


def apply_scarf_protect(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    parsing: Optional[dict],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    if parsing is None:
        return output_rgb
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    protect = np.zeros((h, w), dtype=np.uint8)
    for key in ("face", "hair", "hat"):
        part = parsing.get(key)
        if part is not None:
            protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
    return _protect_blend(output_rgb, person_rgb, protect,
                           safe_uint8=safe_uint8, dilate=5)


def apply_accessory_postprocess(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    person_rgb: np.ndarray,
    allowed: np.ndarray,
    gen_mask_soft: Optional[np.ndarray],
    parsing: Optional[dict],
    full_pose: Optional[dict],
    subtype: str,
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, str]:
    """Dispatcher. Returns (output, tag) for pipeline_info."""
    sub = (subtype or "").lower()
    if sub in {"shoes", "boots"}:
        out = apply_shoes_pose_guard(
            output_rgb, init_tryon_rgb, person_rgb, allowed, gen_mask_soft,
            parsing, full_pose, fit_like=fit_like, safe_uint8=safe_uint8,
            is_boots=(sub == "boots"),
        )
        return out, f"AccessoryPP:{sub}"
    if sub == "hat":
        out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                            safe_uint8=safe_uint8, dilate=5)
        out = apply_hat_face_protect(out, person_rgb, allowed, parsing,
                                       fit_like=fit_like, safe_uint8=safe_uint8)
        return out, "AccessoryPP:hat"
    if sub == "sunglasses":
        out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                            safe_uint8=safe_uint8, dilate=3)
        out = apply_glasses_face_protect(out, person_rgb, allowed, parsing,
                                           fit_like=fit_like, safe_uint8=safe_uint8)
        return out, "AccessoryPP:sunglasses"
    if sub == "belt":
        out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                            safe_uint8=safe_uint8, dilate=3)
        out = apply_belt_protect(out, person_rgb, allowed, parsing,
                                   fit_like=fit_like, safe_uint8=safe_uint8)
        return out, "AccessoryPP:belt"
    if sub == "bag":
        out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                            safe_uint8=safe_uint8, dilate=7)
        out = apply_bag_protect(out, person_rgb, allowed, parsing,
                                  fit_like=fit_like, safe_uint8=safe_uint8)
        return out, "AccessoryPP:bag"
    if sub == "scarf":
        out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                            safe_uint8=safe_uint8, dilate=5)
        out = apply_scarf_protect(out, person_rgb, allowed, parsing,
                                    fit_like=fit_like, safe_uint8=safe_uint8)
        return out, "AccessoryPP:scarf"

    out = _spill_guard(output_rgb, init_tryon_rgb, allowed, gen_mask_soft,
                        safe_uint8=safe_uint8, dilate=5)
    return out, "AccessoryPP:generic"


__all__ = [
    "apply_accessory_postprocess",
    "apply_shoes_pose_guard",
    "apply_hat_face_protect",
    "apply_glasses_face_protect",
    "apply_belt_protect",
    "apply_bag_protect",
    "apply_scarf_protect",
]
