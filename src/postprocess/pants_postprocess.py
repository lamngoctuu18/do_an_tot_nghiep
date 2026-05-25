"""Pants-specific postprocess steps.

Extracted 1:1 from `app.py` (v19.x). Helpers from `app.py` are injected as
callables to keep this module import-safe:

    fit_like(src, ref, *, is_mask)         -> np.ndarray
    safe_uint8(arr)                         -> np.ndarray (uint8)
    build_cloth_mask(rgb)                   -> np.ndarray (mask)

Public API:
    restore_upper_body_for_pants(output, person, parsing)
    build_shorts_edit_band(shape, full_pose)
    build_pants_shape_mask(shape, warped_mask, parsing, pants_type, *, fit_like)
    build_shorts_shape_mask(shape, warped_mask, parsing, full_pose, *, fit_like)
    build_shorts_wear_mask(shape, parsing, full_pose, edit_mask, *, fit_like)
    reference_garment_color(reference_cloth_rgb, *, safe_uint8, build_cloth_mask)
    render_reference_shorts(person_rgb, reference_cloth_rgb, shorts_mask, *, safe_uint8, build_cloth_mask)
    apply_pants_shape_guard(...)
    apply_shorts_shape_guard(...)
    build_pants_diffusion_seed(init_tryon, gen_mask_soft, reference_cloth_rgb, *, safe_uint8, build_cloth_mask)
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2
import numpy as np


def restore_upper_body_for_pants(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    parsing: Optional[dict],
) -> np.ndarray:
    if parsing is None:
        return output_rgb
    keep = np.zeros(output_rgb.shape[:2], dtype=np.uint8)
    h, w = keep.shape
    lower_top = None
    lower = np.zeros_like(keep)
    for key in ("pants", "skirt", "belt"):
        part = parsing.get(key)
        if part is not None:
            if part.shape[:2] != (h, w):
                part = cv2.resize(part, (w, h), interpolation=cv2.INTER_NEAREST)
            lower = cv2.bitwise_or(lower, (part > 20).astype(np.uint8) * 255)
    ys, _ = np.where(lower > 20)
    if len(ys) > 200:
        # Robust top: SegFormer sometimes has tiny shorts pixels above the
        # actual waistband. Keep upper-clothes restore above the real waist.
        lower_top = int(np.percentile(ys, 4))

    for key in ("upper_clothes", "left_arm", "right_arm", "face", "hair"):
        part = parsing.get(key)
        if part is not None:
            if part.shape[:2] != (h, w):
                part = cv2.resize(part, (w, h), interpolation=cv2.INTER_NEAREST)
            part = (part > 20).astype(np.uint8) * 255
            if key == "upper_clothes" and lower_top is not None:
                clip_y = max(0, lower_top - 2)
                part[clip_y:, :] = 0
                part = cv2.dilate(part, np.ones((5, 5), np.uint8), iterations=1)
                part[clip_y:, :] = 0
            else:
                part = cv2.dilate(part, np.ones((5, 5), np.uint8), iterations=1)
            keep = cv2.bitwise_or(keep, part)
    if int(cv2.countNonZero(keep)) == 0:
        return output_rgb
    alpha = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (9, 9), 2.0)[..., None]
    blended = output_rgb.astype(np.float32) * (1 - alpha) + person_rgb.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_shorts_edit_band(
    shape: Tuple[int, int],
    full_pose: Optional[dict],
) -> np.ndarray:
    h, w = shape
    band = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        band[:] = 255
        return band

    lh = full_pose.get("left_hip")
    rh = full_pose.get("right_hip")
    ls = full_pose.get("left_shoulder")
    rs = full_pose.get("right_shoulder")
    lk = full_pose.get("left_knee")
    rk = full_pose.get("right_knee")
    if not all(p is not None for p in (lh, rh, ls, rs)):
        band[:] = 255
        return band

    hip_y = float((lh[1] + rh[1]) * 0.5)
    sw = float(abs(rs[0] - ls[0]))
    hip_w = float(abs(rh[0] - lh[0]))
    ref = max(48.0, sw, hip_w * 2.0)

    # v22.12: tightened envelope. v22.11 was too generous — bottom reached
    # knee level and half-width flared 1.45× hip-width, producing a wide
    # trapezoid that SD repainted as flared knee-length shorts.
    # Target shape: just above the waistband down to mid-thigh, width ≈
    # natural hip line + small margin.
    top_y = max(0, int(hip_y - ref * 0.15))
    bot_y = int(hip_y + ref * 0.78)
    if lk is not None and rk is not None:
        knee_y = float((lk[1] + rk[1]) * 0.5)
        if knee_y > hip_y + ref * 0.35:
            # cap at mid-thigh (55% of hip→knee), never past it
            bot_y = min(bot_y, int(hip_y + (knee_y - hip_y) * 0.55))
    bot_y = min(h, max(top_y + 24, bot_y))

    # v22.12: half-width back to natural shorts cut. 1.10× hip-width is
    # enough margin for the warped reference without flaring sideways.
    hip_cx = float((lh[0] + rh[0]) * 0.5)
    half_w = max(sw * 0.62, hip_w * 1.10, 56.0)
    x_lo = max(0, int(hip_cx - half_w))
    x_hi = min(w, int(hip_cx + half_w))
    band[top_y:bot_y, x_lo:x_hi] = 255
    return band


def build_shorts_shape_mask(
    shape: Tuple[int, int],
    warped_mask: np.ndarray,
    parsing: Optional[dict],
    full_pose: Optional[dict],
    *,
    fit_like: Callable,
) -> np.ndarray:
    """v22.11: shorts shape mask = warped footprint ∪ full old shorts/skirt
    (within the pose-driven shorts envelope). The previous version only
    unioned the part of parsing.pants/skirt within 35px of the warp, which
    missed the wider/longer original beige shorts when the warp collapsed to
    a thin hammerhead — leaving SD with only the warped cloth's footprint
    and no leg openings to repaint.

    Pipeline:
      1. dilate warp footprint 9px (base seed)
      2. union FULL parsing.pants/skirt/belt
      3. subtract dilated upper-body protect (arms/face/hair/upper_clothes)
      4. clip to build_shorts_edit_band (pose-driven envelope)
    """
    h, w = shape
    ref = np.zeros((h, w), dtype=np.uint8)
    wm = fit_like(warped_mask, ref, is_mask=True)
    source = ((wm > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(source)) < 200:
        return source

    shape_mask = cv2.dilate(source, np.ones((9, 9), np.uint8), iterations=1)
    shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    if parsing:
        old_lower = np.zeros((h, w), dtype=np.uint8)
        for key in ("pants", "skirt", "belt"):
            part = parsing.get(key)
            if part is not None:
                old_lower = cv2.bitwise_or(old_lower, fit_like(part, ref, is_mask=True))
        if int(cv2.countNonZero(old_lower)) > 80:
            # v22.11: union the FULL old shorts/skirt (not just near-warp).
            # The edit_band clip below is the actual safety boundary.
            shape_mask = cv2.bitwise_or(shape_mask, old_lower)

        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("upper_clothes", "left_arm", "right_arm", "face", "hair"):
            # v22.11: do NOT protect "dress" — SegFormer often labels a
            # short outfit's lower hem as dress, which would eat the leg
            # area of the new shorts mask.
            part = parsing.get(key)
            if part is not None:
                protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
        if int(cv2.countNonZero(protect)) > 0:
            protect = cv2.dilate(protect, np.ones((5, 5), np.uint8), iterations=1)
            shape_mask = cv2.subtract(shape_mask, protect)

    # Clip to the tight hip→thigh band so any drift above the waistband or
    # sideways past the natural shorts width is removed.
    if full_pose is not None:
        edit_band = build_shorts_edit_band((h, w), full_pose)
        if int(cv2.countNonZero(edit_band)) > 0:
            shape_mask = cv2.bitwise_and(shape_mask, edit_band)

    shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    shape_mask = cv2.dilate(shape_mask, np.ones((3, 3), np.uint8), iterations=1)

    if int(cv2.countNonZero(shape_mask)) < 255:
        shape_mask = ((wm > 20).astype(np.uint8)) * 255
    return shape_mask


def build_shorts_wear_mask(
    shape: Tuple[int, int],
    parsing: Optional[dict],
    full_pose: Optional[dict],
    edit_mask: Optional[np.ndarray],
    *,
    fit_like: Callable,
) -> np.ndarray:
    """Pose-fitted final shorts silhouette.

    `build_shorts_shape_mask` is intentionally an edit/erase mask: it includes
    the old lower garment so diffusion can remove beige shorts. This mask is
    stricter and is used as the seed/final allowed shape, so diffusion cannot
    keep a skirt-like blob or a dangling vertical strip inside the broader edit
    region.
    """
    h, w = shape
    ref = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        return fit_like(edit_mask, ref, is_mask=True) if edit_mask is not None else ref

    lh = full_pose.get("left_hip")
    rh = full_pose.get("right_hip")
    ls = full_pose.get("left_shoulder")
    rs = full_pose.get("right_shoulder")
    lk = full_pose.get("left_knee")
    rk = full_pose.get("right_knee")
    if not all(p is not None for p in (lh, rh, ls, rs)):
        return fit_like(edit_mask, ref, is_mask=True) if edit_mask is not None else ref

    hip_a = np.array(lh, dtype=np.float32)
    hip_b = np.array(rh, dtype=np.float32)
    shoulder_a = np.array(ls, dtype=np.float32)
    shoulder_b = np.array(rs, dtype=np.float32)
    hip_c = (hip_a + hip_b) * 0.5
    sw = float(np.linalg.norm(shoulder_a - shoulder_b))
    hip_w_raw = float(np.linalg.norm(hip_a - hip_b))
    hip_w = max(24.0, hip_w_raw, sw * 0.62)
    ref_len = max(48.0, sw, hip_w * 2.0)

    knee_vals = [np.array(k, dtype=np.float32) for k in (lk, rk) if k is not None]
    knee_y = (
        max(float(k[1]) for k in knee_vals)
        if knee_vals else
        float(hip_c[1] + ref_len * 0.78)
    )
    knee_y = max(knee_y, float(hip_c[1] + ref_len * 0.68))
    hem_y = min(float(hip_c[1] + ref_len * 0.82), float(hip_c[1] + (knee_y - hip_c[1]) * 0.52))
    hem_y = max(hem_y, float(hip_c[1] + hip_w * 0.78))

    waist_top = float(hip_c[1] - ref_len * 0.16)
    waist_bot = float(hip_c[1] + ref_len * 0.10)
    hem_drop = max(24.0, hem_y - waist_bot)
    crotch_y = min(float(waist_bot + hem_drop * 0.48), float(hip_c[1] + hip_w * 0.46))
    waist_half = max(sw * 0.62, hip_w * 1.22, 54.0)
    hem_half = max(sw * 0.42, hip_w * 0.82, 40.0)
    leg_outer_half = max(sw * 0.50, hip_w * 0.98, 46.0)
    gap_top = max(5.0, hip_w * 0.08)
    gap_bot = max(12.0, hip_w * 0.22)

    if edit_mask is not None:
        edit = fit_like(edit_mask, ref, is_mask=True)
        edit = (edit > 20).astype(np.uint8) * 255
        ys_e, xs_e = np.where(edit > 0)
        if len(xs_e) > 100:
            band_top = int(max(0, hip_c[1] - ref_len * 0.12))
            band_bot = int(min(h - 1, hip_c[1] + ref_len * 0.42))
            waist_rows = (ys_e >= band_top) & (ys_e <= band_bot)
            if int(waist_rows.sum()) > 30:
                edit_half = max(
                    abs(float(xs_e[waist_rows].min()) - float(hip_c[0])),
                    abs(float(xs_e[waist_rows].max()) - float(hip_c[0])),
                )
                waist_half = max(waist_half, min(edit_half + 4.0, sw * 0.82, hip_w * 1.55))
                hem_half = max(hem_half, min(edit_half * 0.74, sw * 0.58, hip_w * 1.12))
                leg_outer_half = max(leg_outer_half, min(edit_half * 0.90, sw * 0.72, hip_w * 1.38))

    # v22.16: final wear mask is two leg panels plus a narrow waistband, not a
    # single box. The prior broad envelope removed the T artifact but forced
    # square corners and straight tube legs.
    wear = np.zeros((h, w), dtype=np.uint8)
    waist_rect_l = max(0, int(hip_c[0] - waist_half))
    waist_rect_r = min(w - 1, int(hip_c[0] + waist_half))
    waist_rect_t = max(0, int(waist_top))
    waist_rect_b = min(h - 1, int(waist_bot))
    cv2.rectangle(wear, (waist_rect_l, waist_rect_t), (waist_rect_r, waist_rect_b), 255, -1)
    waist_r = max(5, int((waist_rect_b - waist_rect_t) * 0.55))
    cv2.circle(wear, (waist_rect_l, (waist_rect_t + waist_rect_b) // 2), waist_r, 255, -1, lineType=cv2.LINE_AA)
    cv2.circle(wear, (waist_rect_r, (waist_rect_t + waist_rect_b) // 2), waist_r, 255, -1, lineType=cv2.LINE_AA)

    left_poly = np.array([
        [hip_c[0] - waist_half * 0.92, waist_bot - hem_drop * 0.05],
        [hip_c[0] - gap_top, crotch_y],
        [hip_c[0] - gap_bot, hem_y],
        [hip_c[0] - hem_half, hem_y + hem_drop * 0.04],
        [hip_c[0] - leg_outer_half, waist_bot + hem_drop * 0.10],
    ], dtype=np.int32)
    right_poly = np.array([
        [hip_c[0] + gap_top, crotch_y],
        [hip_c[0] + waist_half * 0.92, waist_bot - hem_drop * 0.05],
        [hip_c[0] + leg_outer_half, waist_bot + hem_drop * 0.10],
        [hip_c[0] + hem_half, hem_y + hem_drop * 0.04],
        [hip_c[0] + gap_bot, hem_y],
    ], dtype=np.int32)
    cv2.fillPoly(wear, [left_poly, right_poly], 255, lineType=cv2.LINE_AA)

    # Round lower outside corners and make the hem less ruler-straight.
    corner_r = max(8, int(hip_w * 0.16))
    for sx in (-1, 1):
        cx = int(hip_c[0] + sx * hem_half)
        cy = int(hem_y)
        cv2.ellipse(
            wear,
            (cx, cy),
            (corner_r, max(5, int(corner_r * 0.65))),
            0, 0, 360, 255, thickness=-1, lineType=cv2.LINE_AA,
        )

    # Explicit crotch gap so a broad inpaint area cannot settle into a skirt.
    gap_poly = np.array([
        [hip_c[0] - gap_top, crotch_y],
        [hip_c[0] + gap_top, crotch_y],
        [hip_c[0] + gap_bot, hem_y + max(6.0, hip_w * 0.10)],
        [hip_c[0] - gap_bot, hem_y + max(6.0, hip_w * 0.10)],
    ], dtype=np.int32)
    cv2.fillPoly(wear, [gap_poly], 0, lineType=cv2.LINE_AA)

    wear = cv2.morphologyEx(wear, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    wear = cv2.GaussianBlur(wear, (0, 0), 1.6)
    wear = (wear > 72).astype(np.uint8) * 255
    wear = cv2.morphologyEx(wear, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    edit_band = build_shorts_edit_band((h, w), full_pose)
    if int(cv2.countNonZero(edit_band)) > 100:
        wear = cv2.bitwise_and(wear, edit_band)

    if parsing:
        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("upper_clothes", "left_arm", "right_arm", "face", "hair"):
            part = parsing.get(key)
            if part is not None:
                protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
        if int(cv2.countNonZero(protect)) > 0:
            protect = cv2.dilate(protect, np.ones((5, 5), np.uint8), iterations=1)
            wear = cv2.subtract(wear, protect)

    if int(cv2.countNonZero(wear)) < 200 and edit_mask is not None:
        return fit_like(edit_mask, ref, is_mask=True)
    return (wear > 20).astype(np.uint8) * 255


def build_pants_shape_mask(
    shape: Tuple[int, int],
    warped_mask: np.ndarray,
    parsing: Optional[dict],
    pants_type: str,
    *,
    fit_like: Callable,
) -> np.ndarray:
    """Build a tight pants edit area from the warped pants footprint.

    The human-prior mask can become a broad lower-body blob on cropped photos.
    This mask keeps diffusion near the actual pants silhouette and only adds
    parsing pants pixels that are close to that silhouette.
    """
    h, w = shape
    ref = np.zeros((h, w), dtype=np.uint8)
    wm = fit_like(warped_mask, ref, is_mask=True)
    source = ((wm > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(source)) < 200:
        return source

    kernel_size = 9 if pants_type == "shorts" else 17
    source_shape = cv2.dilate(source, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    source_shape = cv2.morphologyEx(source_shape, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    if parsing:
        old_lower = np.zeros((h, w), dtype=np.uint8)
        for key in ("pants", "skirt", "belt"):
            part = parsing.get(key)
            if part is not None:
                old_lower = cv2.bitwise_or(old_lower, fit_like(part, ref, is_mask=True))
        if int(cv2.countNonZero(old_lower)) > 80:
            # v19.43: union only the part of old_lower NEAR the warp footprint.
            # Previously (v19.41) we unioned the FULL old pants silhouette to
            # ensure dark jeans get fully repainted, but that creates an
            # over-large repaint area which forces diffusion to invent the
            # whole pants → flat denim blob. A 35×35 dilation around the warp
            # still covers all original pants pixels for typical photos.
            near_source = cv2.dilate(source, np.ones((35, 35), np.uint8), iterations=1)
            source_shape = cv2.bitwise_or(source_shape, cv2.bitwise_and(old_lower, near_source))

        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("upper_clothes", "dress", "left_arm", "right_arm", "face", "hair"):
            part = parsing.get(key)
            if part is not None:
                protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
        if int(cv2.countNonZero(protect)) > 0:
            protect = cv2.dilate(protect, np.ones((9, 9), np.uint8), iterations=1)
            source_shape = cv2.subtract(source_shape, protect)

    return (source_shape > 20).astype(np.uint8) * 255


def reference_garment_color(
    reference_cloth_rgb: Optional[np.ndarray],
    *,
    safe_uint8: Callable,
    build_cloth_mask: Callable,
) -> np.ndarray:
    if reference_cloth_rgb is None:
        return np.array([24.0, 24.0, 24.0], dtype=np.float32)
    ref = safe_uint8(reference_cloth_rgb)
    try:
        ref_mask = build_cloth_mask(ref) > 127
    except Exception:
        ref_mask = np.ones(ref.shape[:2], dtype=bool)

    # Warped/debug cloth images may have a black canvas around the garment.
    # Drop only dark pixels connected to the image border so true black
    # garments in the middle of the reference remain valid.
    try:
        ref_lum = ref.astype(np.float32).mean(axis=2)
        border_dark = (ref_lum < 18.0).astype(np.uint8)
        num, labels = cv2.connectedComponents(border_dark, connectivity=8)
        if num > 1:
            edge_labels = np.concatenate((
                labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
            ))
            edge_labels = np.unique(edge_labels[edge_labels > 0])
            flood = np.isin(labels, edge_labels) if edge_labels.size else np.zeros_like(border_dark, dtype=bool)
        else:
            flood = np.zeros_like(border_dark, dtype=bool)
        cleaned_mask = ref_mask & (~flood)
        if int(cleaned_mask.sum()) >= 20:
            ref_mask = cleaned_mask
    except Exception:
        pass

    pixels = ref[ref_mask].reshape(-1, 3).astype(np.float32)
    if len(pixels) < 20:
        pixels = ref.reshape(-1, 3).astype(np.float32)
    lum = pixels.mean(axis=1)
    non_bg = pixels[lum < 235]
    if len(non_bg) >= 20:
        pixels = non_bg
        lum = pixels.mean(axis=1)
    order = np.argsort(lum)
    darkest = pixels[order[: max(20, int(len(order) * 0.45))]]
    return np.percentile(darkest, 45, axis=0).astype(np.float32)


def render_reference_shorts(
    person_rgb: np.ndarray,
    reference_cloth_rgb: Optional[np.ndarray],
    shorts_mask: np.ndarray,
    *,
    safe_uint8: Callable,
    build_cloth_mask: Callable,
) -> np.ndarray:
    base = reference_garment_color(
        reference_cloth_rgb, safe_uint8=safe_uint8, build_cloth_mask=build_cloth_mask
    )
    base_lum = float(np.mean(base))
    if base_lum < 28.0:
        base = base * (38.0 / max(base_lum, 1.0))
        base_lum = float(np.mean(base))
    hue = base / max(base_lum, 1.0)
    hue = np.clip(hue, 0.45, 1.80)

    gray = cv2.cvtColor(safe_uint8(person_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = shorts_mask > 20
    if int(mask.sum()) < 50:
        return person_rgb

    vals = gray[mask]
    med = float(np.median(vals))
    p10, p90 = np.percentile(vals, [10, 90])
    spread = max(18.0, float(p90 - p10))
    shade = np.clip((gray - med) / spread, -1.2, 1.4)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 3.0)
    luminance = np.clip(base_lum + shade * 38.0 + detail * 0.18, 10.0, 118.0)
    rendered = np.clip(luminance[..., None] * hue[None, None, :], 0.0, 255.0)
    return safe_uint8(rendered)


def apply_shorts_shape_guard(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    warped_mask: np.ndarray,
    gen_mask_soft: np.ndarray,
    parsing: Optional[dict],
    full_pose: Optional[dict],
    reference_cloth_rgb: Optional[np.ndarray],
    *,
    fit_like: Callable,
    safe_uint8: Callable,
    final_wear_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    src = fit_like(warped_mask, ref, is_mask=True)
    src = ((src > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(src)) < 200:
        return output_rgb, src

    # v22.13: the shorts edit area is intentionally wider than the warped
    # reference footprint. The old guard treated `src` as the only valid area
    # and restored every diffusion pixel outside it, so even a correct inpaint
    # mask was collapsed back to the narrow "T" shaped warp. Guard against
    # spill outside the stricter two-leg wear shape instead.
    if final_wear_mask is not None:
        wear_mask = fit_like(final_wear_mask, ref, is_mask=True)
        wear_mask = ((wear_mask > 20).astype(np.uint8)) * 255
    else:
        shape_mask = build_shorts_shape_mask(
            (h, w), src, parsing, full_pose, fit_like=fit_like
        )
        if int(cv2.countNonZero(shape_mask)) < 200:
            shape_mask = src
        wear_mask = build_shorts_wear_mask(
            (h, w), parsing, full_pose, shape_mask, fit_like=fit_like
        )
    if int(cv2.countNonZero(wear_mask)) < 200:
        wear_mask = src

    gen = fit_like(gen_mask_soft, ref, is_mask=True)
    allowed = cv2.dilate(wear_mask, np.ones((5, 5), np.uint8), iterations=1)
    spill = cv2.bitwise_and(gen, cv2.bitwise_not(allowed))
    if int(cv2.countNonZero(spill)) > 50:
        cleanup_alpha = cv2.GaussianBlur(
            (spill > 12).astype(np.float32), (9, 9), 2.0
        )
        cleanup_alpha = np.clip(cleanup_alpha, 0.0, 1.0)[..., None]
        output_rgb = safe_uint8(
            output_rgb.astype(np.float32) * (1.0 - cleanup_alpha)
            + init_tryon_rgb.astype(np.float32) * cleanup_alpha
        )
    return output_rgb, wear_mask


def apply_pants_shape_guard(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    warped_mask: np.ndarray,
    gen_mask_soft: np.ndarray,
    parsing: Optional[dict],
    pants_type: str,
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restore diffusion spill outside the warped pants silhouette."""
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    shape_mask = build_pants_shape_mask(
        (h, w), warped_mask, parsing, pants_type, fit_like=fit_like
    )
    if int(cv2.countNonZero(shape_mask)) < 200:
        return output_rgb, shape_mask

    gen = fit_like(gen_mask_soft, ref, is_mask=True)
    allowed = cv2.dilate(shape_mask, np.ones((5, 5), np.uint8), iterations=1)
    spill = cv2.bitwise_and(gen, cv2.bitwise_not(allowed))
    if int(cv2.countNonZero(spill)) < 50:
        return output_rgb, shape_mask

    alpha = cv2.GaussianBlur((spill > 12).astype(np.float32), (9, 9), 2.0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    cleaned = (
        output_rgb.astype(np.float32) * (1.0 - alpha)
        + init_tryon_rgb.astype(np.float32) * alpha
    )
    return safe_uint8(cleaned), shape_mask


def build_pants_diffusion_seed(
    init_tryon: np.ndarray,
    gen_mask_soft: np.ndarray,
    reference_cloth_rgb: Optional[np.ndarray],
    *,
    safe_uint8: Callable,
    build_cloth_mask: Callable,
    pants_type: str = "regular",
    warped_mask: Optional[np.ndarray] = None,
    cleanup_mask: Optional[np.ndarray] = None,
    cleanup_fill_rgb: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask_f = np.clip(gen_mask_soft.astype(np.float32) / 255.0, 0.0, 1.0)
    if float(mask_f.max()) < 0.05:
        return init_tryon

    base_color = reference_garment_color(
        reference_cloth_rgb, safe_uint8=safe_uint8, build_cloth_mask=build_cloth_mask
    )
    gray = cv2.cvtColor(safe_uint8(init_tryon), cv2.COLOR_RGB2GRAY).astype(np.float32)
    core = mask_f > 0.18
    center = float(np.median(gray[core])) if np.any(core) else float(np.median(gray))
    shade = np.clip((gray - center) * 0.18, -18.0, 24.0)
    fill = np.clip(base_color[None, None, :] + shade[..., None], 8.0, 245.0)

    # v19.42: build a "warped texture" mask = where init_tryon already contains
    # warped reference jeans (denim texture, waistband, pockets, seams). Keep
    # init_tryon as-is inside this area so diffusion has anchor for detail.
    # Only flat-fill the extension area (parsing.pants - warped footprint),
    # i.e. where original dark jeans would otherwise show through.
    if pants_type == "shorts":
        edit = ((mask_f > 0.04).astype(np.uint8)) * 255
        if warped_mask is not None:
            target = ((warped_mask > 20).astype(np.uint8)) * 255
            if target.shape[:2] != init_tryon.shape[:2]:
                target = cv2.resize(
                    target,
                    (init_tryon.shape[1], init_tryon.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
        else:
            target = ((mask_f > 0.18).astype(np.uint8)) * 255

        target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        target_f = cv2.GaussianBlur(target.astype(np.float32) / 255.0, (5, 5), 1.2)
        target_f = np.clip(target_f, 0.0, 1.0)
        target_f = np.minimum(target_f, np.clip(mask_f * 1.15, 0.0, 1.0))

        # v22.15: remove the CPU-warp "T" layer before SD sees the seed.
        # The edit mask is broader than the final shorts. Areas inside edit
        # but outside target should be clean body/background context, not the
        # black warped reference strip; otherwise diffusion refines that strip
        # and the shape guard can restore it.
        base_clean = safe_uint8(init_tryon)
        cleanup = edit.copy()
        if cleanup_mask is not None:
            clean_src = ((cleanup_mask > 20).astype(np.uint8)) * 255
            if clean_src.shape[:2] != init_tryon.shape[:2]:
                clean_src = cv2.resize(
                    clean_src,
                    (init_tryon.shape[1], init_tryon.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            cleanup = cv2.bitwise_or(
                cleanup,
                cv2.dilate(clean_src, np.ones((7, 7), np.uint8), iterations=1),
            )
        outside_target = cv2.subtract(
            cleanup,
            cv2.dilate(target, np.ones((3, 3), np.uint8), iterations=1),
        )
        if int(cv2.countNonZero(outside_target)) > 30:
            outside_target = cv2.morphologyEx(
                outside_target, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1,
            )
            outside_f = cv2.GaussianBlur(
                outside_target.astype(np.float32) / 255.0, (7, 7), 2.0,
            )[..., None]
            outside_f = np.clip(outside_f, 0.0, 1.0)
            if cleanup_fill_rgb is not None:
                fill_src = safe_uint8(cleanup_fill_rgb)
                if fill_src.shape[:2] != init_tryon.shape[:2]:
                    fill_src = cv2.resize(
                        fill_src,
                        (init_tryon.shape[1], init_tryon.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                base_clean = safe_uint8(
                    base_clean.astype(np.float32) * (1.0 - outside_f)
                    + fill_src.astype(np.float32) * outside_f
                )
            else:
                lum = base_clean.astype(np.float32).mean(axis=2)
                ctx_ring = cv2.subtract(
                    cv2.dilate(edit, np.ones((31, 31), np.uint8), iterations=1),
                    edit,
                )
                valid_ctx = (ctx_ring > 0) & (lum > 58.0)
                if int(valid_ctx.sum()) < 50:
                    valid_ctx = (target <= 20) & (lum > 58.0)
                if int(valid_ctx.sum()) >= 20:
                    ctx_color = np.median(base_clean[valid_ctx].reshape(-1, 3), axis=0).astype(np.float32)
                else:
                    ctx_color = np.median(base_clean.reshape(-1, 3), axis=0).astype(np.float32)
                ctx_fill = np.zeros_like(base_clean, dtype=np.float32)
                ctx_fill[:] = ctx_color
                base_clean = safe_uint8(
                    base_clean.astype(np.float32) * (1.0 - outside_f)
                    + ctx_fill * outside_f
                )
                try:
                    base_clean = cv2.inpaint(base_clean, outside_target, 5, cv2.INPAINT_TELEA)
                except Exception:
                    blur = cv2.GaussianBlur(base_clean, (21, 21), 6.0)
                    base_clean = safe_uint8(
                        base_clean.astype(np.float32) * (1.0 - outside_f)
                        + blur.astype(np.float32) * outside_f
                    )

        # Fill only the final two-leg shorts silhouette. The broader inpaint
        # mask remains editable, but it is not seeded with black fabric; this
        # prevents SD from reading the whole edit area as a skirt/oversized
        # blob.
        gray_clean = cv2.cvtColor(base_clean, cv2.COLOR_RGB2GRAY).astype(np.float32)
        smooth_gray = cv2.GaussianBlur(gray_clean, (0, 0), 10.0)
        target_core = target > 20
        center_short = (
            float(np.median(smooth_gray[target_core]))
            if np.any(target_core) else
            float(np.median(smooth_gray))
        )
        short_shade = np.clip((smooth_gray - center_short) * 0.07, -8.0, 14.0)
        short_fill = np.clip(base_color[None, None, :] + short_shade[..., None], 8.0, 245.0)
        alpha = np.clip(target_f * 0.96, 0.0, 0.96)[..., None]
        seed = base_clean.astype(np.float32) * (1.0 - alpha) + short_fill.astype(np.float32) * alpha
        if reference_cloth_rgb is not None and int(cv2.countNonZero(target)) > 300:
            try:
                ref_rgb = safe_uint8(reference_cloth_rgb)
                cloth_m = build_cloth_mask(ref_rgb)
                if cloth_m is not None and int(cv2.countNonZero(cloth_m)) > 200:
                    cloth_px = ref_rgb[cloth_m > 20].reshape(-1, 3).astype(np.float32)
                    lum = cloth_px.mean(axis=1)
                    base_lum = float(np.mean(base_color))
                    bright = lum > max(170.0, base_lum + 65.0)
                    bright_ratio = float(bright.mean()) if len(bright) else 0.0
                    if 0.003 <= bright_ratio <= 0.22 and base_lum < 150.0:
                        trim_color = np.median(cloth_px[bright], axis=0).astype(np.float32)
                        ys, _ = np.where(target > 20)
                        y1, y2 = int(ys.min()), int(ys.max())
                        xs = np.where(target > 20)[1]
                        x1, x2 = int(xs.min()), int(xs.max())
                        height = max(1, y2 - y1)
                        width = max(1, x2 - x1)
                        cx = int(np.median(xs))
                        yy, xx = np.indices(target.shape[:2])
                        side_zone = (xx <= int(x1 + width * 0.18)) | (xx >= int(x2 - width * 0.18))
                        hem_zone = yy >= int(y1 + height * 0.68)
                        center_exclude = np.abs(xx - cx) <= max(10, int(width * 0.11))
                        trim_zone = (
                            (yy >= int(y1 + height * 0.14))
                            & (side_zone | hem_zone)
                            & (~center_exclude)
                        )
                        boundary = cv2.subtract(
                            cv2.dilate(target, np.ones((3, 3), np.uint8), iterations=1),
                            cv2.erode(target, np.ones((5, 5), np.uint8), iterations=1),
                        )
                        trim_mask = np.where((boundary > 20) & trim_zone, 255, 0).astype(np.uint8)
                        trim_mask = cv2.dilate(trim_mask, np.ones((3, 3), np.uint8), iterations=1)
                        trim_mask[center_exclude] = 0
                        trim_alpha = cv2.GaussianBlur(
                            trim_mask.astype(np.float32) / 255.0, (5, 5), 1.2,
                        )
                        trim_alpha = np.clip(trim_alpha * 0.58, 0.0, 0.58)[..., None]
                        seed = seed * (1.0 - trim_alpha) + trim_color[None, None, :] * trim_alpha
            except Exception:
                pass
        return safe_uint8(seed)

    # long / cropped pants
    a_mask = np.clip((mask_f - 0.15) / 0.15, 0.0, 1.0)

    if warped_mask is not None:
        wm = (warped_mask > 20).astype(np.float32)
        wm_u8 = (wm * 255).astype(np.uint8)
        wm_core = cv2.erode(wm_u8, np.ones((3, 3), np.uint8), iterations=1)
        wm_soft = cv2.GaussianBlur(wm_core, (0, 0), 4.0).astype(np.float32) / 255.0
        # v19.43→v19.45: flat-fill ép xuống 0.48 để giữ shading/context từ
        # init_tryon nhiều hơn, không tạo nền xanh đều mà diffusion phải
        # refine lại từ đầu.
        # v19.49: keep enough reference color in extension areas to overwrite
        # dark source pants before diffusion starts.
        if pants_type == "regular":
            strength = np.full_like(mask_f, 0.90, dtype=np.float32)
            anchor_strength = np.full_like(mask_f, 0.88, dtype=np.float32)
            ys, _ = np.where(a_mask > 0.05)
            if len(ys) > 200:
                y1, y2 = int(ys.min()), int(ys.max())
                height = max(1, y2 - y1)
                waist_end = int(y1 + height * 0.16)
                lower_start = int(y1 + height * 0.72)
                strength[y1:waist_end, :] = np.maximum(strength[y1:waist_end, :], 0.94)
                anchor_strength[y1:waist_end, :] = np.maximum(anchor_strength[y1:waist_end, :], 0.93)
                strength[lower_start:, :] = np.maximum(strength[lower_start:, :], 0.98)
                anchor_strength[lower_start:, :] = np.maximum(anchor_strength[lower_start:, :], 0.96)
            fill_w_2d = a_mask * (
                wm_soft * anchor_strength + (1.0 - wm_soft) * strength
            )
        else:
            strength = np.full_like(mask_f, 0.70, dtype=np.float32)
            fill_w_2d = (1.0 - wm_soft) * a_mask * strength
        fill_w = np.clip(fill_w_2d, 0.0, 0.985)[..., None]
        keep_w = 1.0 - fill_w
        seed = init_tryon.astype(np.float32) * keep_w + fill.astype(np.float32) * fill_w
        return safe_uint8(np.clip(seed, 0.0, 255.0))

    # fallback: original behaviour
    alpha = a_mask[..., None]
    seed = init_tryon.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha
    return safe_uint8(seed)


__all__ = [
    "restore_upper_body_for_pants",
    "build_shorts_edit_band",
    "build_pants_shape_mask",
    "build_shorts_shape_mask",
    "build_shorts_wear_mask",
    "reference_garment_color",
    "render_reference_shorts",
    "apply_pants_shape_guard",
    "apply_shorts_shape_guard",
    "build_pants_diffusion_seed",
    "cleanup_pants_speckles",
    "cleanup_long_pants_denim_artifacts",
    "restore_long_pants_ankle_skin",
    "cleanup_shorts_external_spill",
    "cleanup_shorts_old_hem_bleed",
    "cleanup_shorts_upper_cloth_spill",
    "cleanup_shorts_center_trim_artifact",
    "recover_pants_texture_detail",
]


def recover_pants_texture_detail(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    pants_mask: np.ndarray,
    *,
    safe_uint8: Callable,
    detail_strength: float = 0.42,
    chroma_strength: float = 0.22,
    sharpen_strength: float = 0.22,
) -> np.ndarray:
    """v19.45: recover high-frequency denim detail (seams, pockets, waistband)
    from init_tryon (warped reference) inside the pants core, leaving the
    diffusion-refined drape/lighting intact.
    """
    h, w = output_rgb.shape[:2]
    if init_tryon_rgb.shape[:2] != (h, w):
        init_tryon_rgb = cv2.resize(init_tryon_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    if pants_mask.shape[:2] != (h, w):
        pants_mask = cv2.resize(pants_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask = pants_mask > 20
    if int(mask.sum()) < 400:
        return output_rgb

    core = cv2.erode(mask.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    if int(cv2.countNonZero(core)) < 300:
        core = mask.astype(np.uint8) * 255

    alpha = cv2.GaussianBlur((core > 20).astype(np.float32), (7, 7), 1.5)
    alpha = np.clip(alpha * detail_strength, 0.0, 0.78)

    out_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(init_tryon_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    out_l = out_lab[:, :, 0]
    src_l = src_lab[:, :, 0]
    sigma = 1.10
    src_detail = src_l - cv2.GaussianBlur(src_l, (0, 0), sigma)
    out_detail = out_l - cv2.GaussianBlur(out_l, (0, 0), sigma)
    detail_delta = np.clip(src_detail - out_detail, -26.0, 26.0)
    out_lab[:, :, 0] = np.clip(out_l + detail_delta * alpha, 0, 255)

    chroma_alpha = np.clip(alpha * chroma_strength, 0.0, 0.30)
    out_lab[:, :, 1] = out_lab[:, :, 1] * (1.0 - chroma_alpha) + src_lab[:, :, 1] * chroma_alpha
    out_lab[:, :, 2] = out_lab[:, :, 2] * (1.0 - chroma_alpha) + src_lab[:, :, 2] * chroma_alpha

    restored = cv2.cvtColor(safe_uint8(out_lab), cv2.COLOR_LAB2RGB)
    sharp = cv2.addWeighted(
        restored,
        1.0 + sharpen_strength,
        cv2.GaussianBlur(restored, (0, 0), 0.85),
        -sharpen_strength,
        0,
    )
    sharp_alpha = cv2.GaussianBlur((core > 20).astype(np.float32), (5, 5), 1.1)
    sharp_alpha = np.clip(sharp_alpha * 0.55, 0.0, 0.55)[..., None]
    return safe_uint8(
        restored.astype(np.float32) * (1.0 - sharp_alpha)
        + sharp.astype(np.float32) * sharp_alpha
    )


def cleanup_pants_speckles(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    pants_mask: np.ndarray,
    *,
    safe_uint8: Callable,
) -> np.ndarray:
    """v19.44: remove small denim speckles outside the pants silhouette.

    Connected components of color-change pixels (vs init_tryon) with area
    4–450 are blended back to init_tryon to erase faint bluish flecks the
    shape guard's main spill check is too coarse to catch.
    """
    h, w = output_rgb.shape[:2]
    mask = cv2.resize(pants_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    allowed = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)

    out = output_rgb.astype(np.float32)
    init = init_tryon_rgb.astype(np.float32)
    diff = np.mean(np.abs(out - init), axis=2)
    spill = ((diff > 18) & (allowed < 20)).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(spill, connectivity=8)
    small = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if 4 <= area <= 450:
            small[labels == i] = 255

    if int(cv2.countNonZero(small)) == 0:
        return output_rgb

    small = cv2.dilate(small, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(small.astype(np.float32) / 255.0, (7, 7), 1.5)[..., None]
    cleaned = out * (1.0 - alpha) + init * alpha
    return safe_uint8(cleaned)


def cleanup_long_pants_denim_artifacts(
    output_rgb: np.ndarray,
    pants_mask: np.ndarray,
    *,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress old belt/crotch/cuff artifacts left inside long jeans."""
    h, w = output_rgb.shape[:2]
    mask = pants_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(mask)) < 500:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 20)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cx = int(np.median(xs))

    out = safe_uint8(output_rgb)
    rgb = out.astype(np.float32)
    lum = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    garment_lum = lum[mask > 20]
    med_lum = float(np.median(garment_lum)) if garment_lum.size else float(np.median(lum))
    yy, xx = np.indices((h, w))

    sample_zone = (
        (mask > 20)
        & (yy >= int(y1 + height * 0.18))
        & (yy <= int(y1 + height * 0.78))
        & (np.abs(xx - cx) >= max(10, int(width * 0.10)))
    )
    sample_px = rgb[sample_zone]
    if sample_px.size < 60:
        sample_px = rgb[mask > 20]
    denim_color = (
        np.median(sample_px.reshape(-1, 3), axis=0).astype(np.float32)
        if sample_px.size
        else np.median(rgb.reshape(-1, 3), axis=0).astype(np.float32)
    )

    def _pick_components(
        src: np.ndarray,
        *,
        min_area: int,
        max_area: int,
        min_w: int = 1,
        max_h: Optional[int] = None,
        require_vertical: bool = False,
    ) -> np.ndarray:
        dst = np.zeros((h, w), dtype=np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
        for i in range(1, num):
            _x, _y, bw, bh, area = stats[i]
            if area < min_area or area > max_area:
                continue
            if bw < min_w:
                continue
            if max_h is not None and bh > max_h:
                continue
            if require_vertical and bh < bw * 1.25:
                continue
            dst[labels == i] = 255
        return dst

    mask_halo = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1) > 20
    waist_zone = (
        mask_halo
        & (yy >= max(0, int(y1 - height * 0.018)))
        & (yy <= int(y1 + height * 0.115))
        & (xx >= int(x1 + width * 0.08))
        & (xx <= int(x2 - width * 0.08))
    )
    dark_waist = waist_zone & (lum < max(130.0, med_lum - 20.0)) & (chroma < 190.0)
    dark_waist = cv2.morphologyEx(
        dark_waist.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((9, 3), np.uint8),
        iterations=1,
    )
    waist_rows = np.zeros((h, w), dtype=np.uint8)
    for row in range(max(0, int(y1 - height * 0.018)), min(h, int(y1 + height * 0.115) + 1)):
        row_xs = np.where((dark_waist[row, :] > 20) & waist_zone[row, :])[0]
        if len(row_xs) >= max(18, int(width * 0.10)):
            left = max(0, int(row_xs.min()) - 5)
            right = min(w - 1, int(row_xs.max()) + 5)
            waist_rows[row, left:right + 1] = 255
    waist_rows = cv2.morphologyEx(
        waist_rows,
        cv2.MORPH_CLOSE,
        np.ones((15, 5), np.uint8),
        iterations=1,
    )
    waist_art = _pick_components(
        dark_waist,
        min_area=35,
        max_area=max(7200, int(width * height * 0.028)),
        min_w=max(5, int(width * 0.04)),
        max_h=max(72, int(height * 0.090)),
    )
    waist_art = cv2.bitwise_or(waist_art, waist_rows)

    crotch_zone = (
        (mask > 20)
        & (yy >= int(y1 + height * 0.34))
        & (yy <= int(y1 + height * 0.56))
        & (np.abs(xx - cx) <= max(7, int(width * 0.052)))
    )
    dark_crotch = crotch_zone & (lum < max(104.0, med_lum - 34.0)) & (chroma < 165.0)
    dark_crotch = cv2.morphologyEx(
        dark_crotch.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 9), np.uint8),
        iterations=1,
    )
    crotch_art = _pick_components(
        dark_crotch,
        min_area=16,
        max_area=max(900, int(width * height * 0.0035)),
        max_h=max(92, int(height * 0.16)),
        require_vertical=True,
    )

    hem_env = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 19)),
        iterations=1,
    )
    bottom_by_col = np.full(w, -1, dtype=np.int32)
    for col in range(max(0, x1 - 4), min(w, x2 + 5)):
        col_ys = np.where(mask[:, col] > 20)[0]
        if len(col_ys):
            bottom_by_col[col] = int(col_ys.max())
    col_bottom = bottom_by_col[xx]
    hem_zone = (
        (hem_env > 20)
        & (col_bottom >= 0)
        & (yy >= np.maximum(int(y1 + height * 0.86), col_bottom - 5))
        & (yy <= np.minimum(h - 1, col_bottom + max(12, int(height * 0.025))))
    )
    inside_hem = mask > 20
    blue_old_cuff = (
        (rgb[:, :, 2] > rgb[:, :, 0] + 16.0)
        & (rgb[:, :, 1] > rgb[:, :, 0] + 4.0)
        & (lum < max(118.0, med_lum - 24.0))
    )
    old_cuff = hem_zone & (
        (inside_hem & (lum < max(78.0, med_lum - 62.0)) & (chroma < 115.0))
        | blue_old_cuff
    )
    old_cuff = cv2.morphologyEx(
        old_cuff.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    hem_art = _pick_components(
        old_cuff,
        min_area=18,
        max_area=max(1300, int(width * height * 0.006)),
        min_w=2,
    )

    if int(cv2.countNonZero(waist_art)) > max(12500, int(width * height * 0.045)):
        waist_art[:] = 0
    if int(cv2.countNonZero(crotch_art)) > max(850, int(width * height * 0.003)):
        crotch_art[:] = 0
    if int(cv2.countNonZero(hem_art)) > max(2200, int(width * height * 0.010)):
        hem_art[:] = 0

    artifact = cv2.bitwise_or(waist_art, cv2.bitwise_or(crotch_art, hem_art))
    artifact = cv2.dilate(artifact, np.ones((3, 3), np.uint8), iterations=1)
    if int(cv2.countNonZero(artifact)) < 20:
        return output_rgb, artifact

    blur = cv2.GaussianBlur(out, (31, 31), 8.0).astype(np.float32)
    fill = blur * 0.38 + denim_color[None, None, :] * 0.62
    cleaned = out.astype(np.float32)
    for region, strength in (
        (waist_art, 0.66),
        (crotch_art, 0.72),
        (hem_art, 0.88),
    ):
        if int(cv2.countNonZero(region)) <= 10:
            continue
        alpha = cv2.GaussianBlur(region.astype(np.float32) / 255.0, (7, 7), 1.5)
        alpha = np.clip(alpha * strength, 0.0, strength)[..., None]
        cleaned = cleaned * (1.0 - alpha) + fill * alpha

    return safe_uint8(cleaned), artifact


def restore_long_pants_ankle_skin(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    pants_mask: np.ndarray,
    parsing: Optional[dict],
    full_pose: Optional[dict],
    *,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace bluish old-cuff pixels around exposed ankles with skin tone."""
    h, w = output_rgb.shape[:2]
    out = safe_uint8(output_rgb)
    person = safe_uint8(person_rgb)
    if person.shape[:2] != (h, w):
        person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = pants_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(mask)) < 500:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 20)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    yy, xx = np.indices((h, w))
    shoe_mask = np.zeros((h, w), dtype=np.uint8)
    skin_ref = np.zeros((h, w), dtype=np.uint8)
    if parsing is not None:
        for key in ("left_shoe", "right_shoe"):
            part = parsing.get(key)
            if part is not None:
                if part.shape[:2] != (h, w):
                    part = cv2.resize(part, (w, h), interpolation=cv2.INTER_NEAREST)
                shoe_mask = cv2.bitwise_or(shoe_mask, (part > 20).astype(np.uint8) * 255)
        for key in ("face", "left_arm", "right_arm", "left_leg", "right_leg"):
            part = parsing.get(key)
            if part is not None:
                if part.shape[:2] != (h, w):
                    part = cv2.resize(part, (w, h), interpolation=cv2.INTER_NEAREST)
                skin_ref = cv2.bitwise_or(skin_ref, (part > 20).astype(np.uint8) * 255)
        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("upper_clothes", "pants", "skirt", "dress", "hair"):
            part = parsing.get(key)
            if part is not None:
                if part.shape[:2] != (h, w):
                    part = cv2.resize(part, (w, h), interpolation=cv2.INTER_NEAREST)
                protect = cv2.bitwise_or(protect, (part > 20).astype(np.uint8) * 255)
        if int(cv2.countNonZero(protect)) > 0:
            skin_ref = cv2.subtract(skin_ref, cv2.dilate(protect, np.ones((3, 3), np.uint8), iterations=1))

    if int(cv2.countNonZero(skin_ref)) < 80 and full_pose is not None:
        face_r = max(10, int(width * 0.055))
        nose = full_pose.get("nose")
        if nose is not None:
            cv2.circle(skin_ref, (int(nose[0]), int(nose[1])), face_r, 255, -1, lineType=cv2.LINE_AA)
        left_eye = full_pose.get("left_eye")
        right_eye = full_pose.get("right_eye")
        if nose is not None and left_eye is not None and right_eye is not None:
            for eye in (left_eye, right_eye):
                cheek = (
                    int((float(eye[0]) + float(nose[0])) * 0.5),
                    int((float(eye[1]) + float(nose[1])) * 0.5 + face_r * 0.35),
                )
                cv2.circle(skin_ref, cheek, max(7, int(face_r * 0.58)), 255, -1, lineType=cv2.LINE_AA)

    if full_pose is not None:
        for side in ("left", "right"):
            for key in (f"{side}_heel", f"{side}_foot_index"):
                pt = full_pose.get(key)
                if pt is not None:
                    cv2.circle(shoe_mask, (int(pt[0]), int(pt[1])), 10, 255, -1, lineType=cv2.LINE_AA)

    person_f = person.astype(np.float32)
    ref = skin_ref > 20
    r, g, b = person_f[:, :, 0], person_f[:, :, 1], person_f[:, :, 2]
    lum_person = person_f.mean(axis=2)
    warm_skin = (
        (r > b + 8.0)
        & (g > b - 18.0)
        & (lum_person > 55.0)
        & (lum_person < 238.0)
    )
    ref = ref & warm_skin
    if int(ref.sum()) < 80:
        ref = warm_skin
    if int(ref.sum()) < 40:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)
    skin_color = np.median(person_f[ref].reshape(-1, 3), axis=0).astype(np.float32)
    skin_lum = max(1.0, float(np.mean(skin_color)))

    ankle_zone = np.zeros((h, w), dtype=np.uint8)
    hip_w = max(24.0, float(width) * 0.46)
    if full_pose is not None:
        for side in ("left", "right"):
            ankle = full_pose.get(f"{side}_ankle")
            heel = full_pose.get(f"{side}_heel")
            if ankle is None:
                continue
            ax, ay = float(ankle[0]), float(ankle[1])
            fy = max(ay + hip_w * 0.16, float(heel[1]) if heel is not None else ay)
            top = max(0, int(ay - hip_w * 0.26))
            bottom = min(h - 1, int(fy + hip_w * 0.03))
            half = max(14, int(hip_w * 0.18))
            cv2.ellipse(
                ankle_zone,
                (int(ax), int((top + bottom) * 0.5)),
                (half, max(12, int((bottom - top) * 0.55))),
                0, 0, 360, 255, -1, lineType=cv2.LINE_AA,
            )

    if int(cv2.countNonZero(ankle_zone)) < 80:
        ankle_zone[
            int(y1 + height * 0.86):min(h, y2 + max(8, int(height * 0.035))),
            max(0, x1 - int(width * 0.08)):min(w, x2 + int(width * 0.08)),
        ] = 255

    out_f = out.astype(np.float32)
    lum = out_f.mean(axis=2)
    shoe_protect = ((shoe_mask > 20) & (lum < 56.0)).astype(np.uint8) * 255
    shoe_protect = cv2.dilate(shoe_protect, np.ones((5, 5), np.uint8), iterations=1)
    ankle_zone = cv2.subtract(ankle_zone, shoe_protect)

    blue_cuff = (
        (out_f[:, :, 2] > out_f[:, :, 0] + 8.0)
        & (out_f[:, :, 1] > out_f[:, :, 0] - 10.0)
        & (lum > 35.0)
        & (lum < 205.0)
    )
    repair = ((ankle_zone > 20) & blue_cuff).astype(np.uint8) * 255
    repair = cv2.morphologyEx(repair, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    repair = cv2.morphologyEx(repair, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if int(cv2.countNonZero(repair)) < 20:
        return output_rgb, repair

    if int(cv2.countNonZero(repair)) > max(9000, int(width * height * 0.026)):
        repair = cv2.bitwise_and(repair, ankle_zone)
        if int(cv2.countNonZero(repair)) > max(13000, int(width * height * 0.036)):
            return output_rgb, np.zeros((h, w), dtype=np.uint8)

    shade = np.clip(lum / skin_lum, 0.88, 1.15)
    skin_fill = np.clip(skin_color[None, None, :] * shade[..., None], 0, 255)
    local_blur = cv2.GaussianBlur(out, (17, 17), 4.0).astype(np.float32)
    fill = skin_fill * 0.90 + local_blur * 0.10
    alpha = cv2.GaussianBlur(repair.astype(np.float32) / 255.0, (7, 7), 1.6)
    alpha = np.clip(alpha * 0.92, 0.0, 0.92)[..., None]
    cleaned = out_f * (1.0 - alpha) + fill * alpha
    return safe_uint8(cleaned), repair


def cleanup_shorts_external_spill(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    shorts_mask: np.ndarray,
    *,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restore non-garment artifacts around shorts back to the person image."""
    h, w = output_rgb.shape[:2]
    mask = shorts_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(mask)) < 200:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 20)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    margin_x = max(18, int(width * 0.22))
    margin_y = max(28, int(height * 0.70))
    zone = np.zeros((h, w), dtype=np.uint8)
    zone[
        max(0, y1 - int(height * 0.06)):min(h, y2 + margin_y),
        max(0, x1 - margin_x):min(w, x2 + margin_x),
    ] = 255
    allowed = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    zone = cv2.bitwise_and(zone, cv2.bitwise_not(allowed))

    out = safe_uint8(output_rgb)
    person = safe_uint8(person_rgb)
    if person.shape[:2] != (h, w):
        person = cv2.resize(person, (w, h), interpolation=cv2.INTER_LINEAR)
    diff = np.mean(np.abs(out.astype(np.float32) - person.astype(np.float32)), axis=2)
    spill = ((zone > 20) & (diff > 12.0)).astype(np.uint8) * 255
    spill = cv2.morphologyEx(spill, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    spill = cv2.morphologyEx(spill, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if int(cv2.countNonZero(spill)) < 20:
        return output_rgb, spill

    alpha = cv2.GaussianBlur(spill.astype(np.float32) / 255.0, (9, 9), 2.2)
    alpha = np.clip(alpha * 0.95, 0.0, 0.95)[..., None]
    cleaned = out.astype(np.float32) * (1.0 - alpha) + person.astype(np.float32) * alpha
    return safe_uint8(cleaned), spill


def cleanup_shorts_old_hem_bleed(
    output_rgb: np.ndarray,
    shorts_mask: np.ndarray,
    *,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove warm pixels from the previous shorts at the new shorts hem/gap."""
    h, w = output_rgb.shape[:2]
    mask = shorts_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(mask)) < 200:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 20)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    yy, xx = np.indices((h, w))
    lower_zone = (
        (yy >= int(y1 + height * 0.54))
        & (yy <= min(h - 1, y2 + max(8, int(height * 0.14))))
        & (xx >= max(0, x1 - max(8, int(width * 0.05))))
        & (xx <= min(w - 1, x2 + max(8, int(width * 0.05))))
    )
    inner_edge = cv2.subtract(
        mask,
        cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1),
    )
    outer_edge = cv2.subtract(
        cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1),
        mask,
    )
    cx = int(np.median(xs))
    center_gap_zone = (
        (np.abs(xx - cx) <= max(14, int(width * 0.17)))
        & (yy >= int(y1 + height * 0.50))
        & (yy <= min(h - 1, y2 + max(6, int(height * 0.09))))
    )
    edge = cv2.bitwise_or(
        inner_edge,
        cv2.bitwise_and(outer_edge, center_gap_zone.astype(np.uint8) * 255),
    )

    out = safe_uint8(output_rgb)
    rgb = out.astype(np.int16)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    warm_peach = (
        (r > 130)
        & (g > 82)
        & (r > b + 16)
        & (r > g + 3)
    )
    pale_old_fringe = (
        (r > 170)
        & (g > 130)
        & (b > 105)
        & (r > b + 12)
        & (g > b + 4)
    )
    bleed = np.where((edge > 20) & lower_zone & (warm_peach | pale_old_fringe), 255, 0).astype(np.uint8)
    bleed = cv2.morphologyEx(bleed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    bleed = cv2.morphologyEx(bleed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    bleed = cv2.dilate(bleed, np.ones((3, 3), np.uint8), iterations=1)
    if int(cv2.countNonZero(bleed)) < 10:
        return output_rgb, bleed

    inpainted = cv2.inpaint(out, bleed, 3, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(bleed.astype(np.float32) / 255.0, (5, 5), 1.2)
    alpha = np.clip(alpha * 0.90, 0.0, 0.90)[..., None]
    cleaned = out.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
    return safe_uint8(cleaned), bleed


def cleanup_shorts_upper_cloth_spill(
    output_rgb: np.ndarray,
    shorts_mask: np.ndarray,
    *,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove bright shirt/scarf strips generated over denim shorts."""
    h, w = output_rgb.shape[:2]
    mask = shorts_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(mask)) < 300:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 20)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cx = int(np.median(xs))

    out = safe_uint8(output_rgb)
    rgb = out.astype(np.float32)
    lum = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    garment_lum = lum[mask > 20]
    med_lum = float(np.median(garment_lum)) if garment_lum.size else float(np.median(lum))
    yy, xx = np.indices((h, w))
    core = (
        (mask > 20)
        & (yy >= int(y1 + height * 0.08))
        & (yy <= int(y1 + height * 0.78))
        & (np.abs(xx - cx) <= max(18, int(width * 0.31)))
    )
    bright_neutral = core & (lum > max(165.0, med_lum + 58.0)) & (chroma < 62.0)
    bright_neutral = cv2.morphologyEx(
        bright_neutral.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((3, 7), np.uint8),
        iterations=1,
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bright_neutral, connectivity=8)
    spill = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < 45:
            continue
        if bh < max(18, int(height * 0.18)):
            continue
        if bw > max(28, int(width * 0.28)):
            continue
        if y > y1 + height * 0.60:
            continue
        aspect = float(bh) / max(1.0, float(bw))
        if aspect < 1.8:
            continue
        spill[labels == i] = 255

    if int(cv2.countNonZero(spill)) < 20:
        return output_rgb, spill

    spill = cv2.dilate(spill, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(out, spill, 3, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(spill.astype(np.float32) / 255.0, (5, 5), 1.1)
    alpha = np.clip(alpha * 0.95, 0.0, 0.95)[..., None]
    cleaned = out.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
    return safe_uint8(cleaned), spill


def cleanup_shorts_center_trim_artifact(
    output_rgb: np.ndarray,
    shorts_mask: np.ndarray,
    *,
    safe_uint8: Callable,
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress accidental bright vertical trim/drawstring at the crotch."""
    h, w = output_rgb.shape[:2]
    mask = shorts_mask
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = ((mask > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(mask)) < 200:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    ys, xs = np.where(mask > 20)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    cx = int(np.median(xs))
    yy, xx = np.indices(mask.shape[:2])
    center_band = np.abs(xx - cx) <= max(8, int(width * 0.075))
    vertical = (yy >= int(y1 + height * 0.10)) & (yy <= min(h - 1, y2 + 2))
    region = (mask > 20) & center_band & vertical
    if int(region.sum()) < 20:
        return output_rgb, np.zeros((h, w), dtype=np.uint8)

    out = safe_uint8(output_rgb)
    lum = out.astype(np.float32).mean(axis=2)
    garment = mask > 20
    garment_lum = lum[garment]
    med_lum = float(np.median(garment_lum)) if garment_lum.size else float(np.median(lum))
    bright = region & (lum > max(105.0, med_lum + 34.0))
    artifact = bright.astype(np.uint8) * 255
    artifact = cv2.morphologyEx(artifact, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8), iterations=1)
    artifact = cv2.dilate(artifact, np.ones((3, 3), np.uint8), iterations=1)
    if int(cv2.countNonZero(artifact)) < 8:
        return output_rgb, artifact

    dark_pool = garment & (~center_band) & (lum < max(115.0, med_lum + 22.0))
    if int(dark_pool.sum()) < 50:
        dark_pool = garment & (~center_band)
    if int(dark_pool.sum()) >= 20:
        fill_color = np.median(out[dark_pool].reshape(-1, 3), axis=0).astype(np.float32)
    else:
        fill_color = np.array([18.0, 18.0, 18.0], dtype=np.float32)
    alpha = cv2.GaussianBlur(artifact.astype(np.float32) / 255.0, (5, 5), 1.2)
    alpha = np.clip(alpha * 0.88, 0.0, 0.88)[..., None]
    cleaned = out.astype(np.float32) * (1.0 - alpha) + fill_color[None, None, :] * alpha
    return safe_uint8(cleaned), artifact
