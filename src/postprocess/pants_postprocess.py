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
    for key in ("upper_clothes", "left_arm", "right_arm", "face", "hair"):
        part = parsing.get(key)
        if part is not None:
            keep = cv2.bitwise_or(keep, part)
    if int(cv2.countNonZero(keep)) == 0:
        return output_rgb
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=1)
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

    top_y = max(0, int(hip_y - ref * 0.30))
    bot_y = int(hip_y + ref * 0.95)
    if lk is not None and rk is not None:
        knee_y = float((lk[1] + rk[1]) * 0.5)
        if knee_y > hip_y + ref * 0.35:
            bot_y = max(bot_y, int(hip_y + (knee_y - hip_y) * 0.72))
    bot_y = min(h, max(top_y + 24, bot_y))
    band[top_y:bot_y, :] = 255
    return band


def build_shorts_shape_mask(
    shape: Tuple[int, int],
    warped_mask: np.ndarray,
    parsing: Optional[dict],
    full_pose: Optional[dict],
    *,
    fit_like: Callable,
) -> np.ndarray:
    h, w = shape
    ref = np.zeros((h, w), dtype=np.uint8)
    wm = fit_like(warped_mask, ref, is_mask=True)
    shape_mask = ((wm > 20).astype(np.uint8)) * 255

    band = np.zeros((h, w), dtype=np.uint8)
    if full_pose is not None:
        lh = full_pose.get("left_hip")
        rh = full_pose.get("right_hip")
        ls = full_pose.get("left_shoulder")
        rs = full_pose.get("right_shoulder")
        if all(p is not None for p in (lh, rh, ls, rs)):
            hip_y = float((lh[1] + rh[1]) * 0.5)
            hip_cx = float((lh[0] + rh[0]) * 0.5)
            sw = max(48.0, float(abs(rs[0] - ls[0])))
            hip_w = max(float(abs(rh[0] - lh[0])), sw * 0.62)
            top = max(0, int(hip_y - sw * 0.30))
            bottom = min(h, int(hip_y + max(sw * 1.18, hip_w * 1.70, 72.0)))
            left = max(0, int(hip_cx - max(sw * 0.86, hip_w * 1.45)))
            right = min(w, int(hip_cx + max(sw * 0.86, hip_w * 1.45)))
            band[top:bottom, left:right] = 255
    if int(cv2.countNonZero(band)) == 0:
        band[:] = 255

    if parsing:
        old_shorts = np.zeros((h, w), dtype=np.uint8)
        for key in ("pants", "skirt", "belt"):
            part = parsing.get(key)
            if part is not None:
                old_shorts = cv2.bitwise_or(old_shorts, fit_like(part, ref, is_mask=True))
        old_shorts = cv2.bitwise_and(old_shorts, band)
        if int(cv2.countNonZero(old_shorts)) > 80:
            shape_mask = cv2.bitwise_or(shape_mask, old_shorts)

        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("upper_clothes", "dress", "left_arm", "right_arm", "face", "hair"):
            part = parsing.get(key)
            if part is not None:
                protect = cv2.bitwise_or(protect, fit_like(part, ref, is_mask=True))
        if int(cv2.countNonZero(protect)) > 0:
            protect = cv2.dilate(protect, np.ones((5, 5), np.uint8), iterations=1)
            shape_mask = cv2.subtract(shape_mask, protect)

    shape_mask = cv2.bitwise_and(shape_mask, band)
    shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    shape_mask = cv2.dilate(shape_mask, np.ones((3, 3), np.uint8), iterations=1)

    if int(cv2.countNonZero(shape_mask)) < 255:
        shape_mask = ((wm > 20).astype(np.uint8)) * 255
    return shape_mask


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
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = output_rgb.shape[:2]
    ref = np.zeros((h, w), dtype=np.uint8)
    src = fit_like(warped_mask, ref, is_mask=True)
    src = ((src > 20).astype(np.uint8)) * 255
    if int(cv2.countNonZero(src)) < 200:
        return output_rgb, src

    gen = fit_like(gen_mask_soft, ref, is_mask=True)
    src_dil = cv2.dilate(src, np.ones((5, 5), np.uint8), iterations=1)
    outside_src = cv2.bitwise_and(gen, cv2.bitwise_not(src_dil))
    if int(cv2.countNonZero(outside_src)) > 50:
        cleanup_alpha = cv2.GaussianBlur(
            (outside_src > 12).astype(np.float32), (9, 9), 2.0
        )
        cleanup_alpha = np.clip(cleanup_alpha, 0.0, 1.0)[..., None]
        output_rgb = safe_uint8(
            output_rgb.astype(np.float32) * (1.0 - cleanup_alpha)
            + init_tryon_rgb.astype(np.float32) * cleanup_alpha
        )
    return output_rgb, src


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
        alpha = np.clip(mask_f ** 0.72 * 0.96, 0.0, 0.96)[..., None]
        seed = init_tryon.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha
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
        flat_fill_strength = 0.70
        fill_w = ((1.0 - wm_soft) * a_mask * flat_fill_strength)[..., None]
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
    "reference_garment_color",
    "render_reference_shorts",
    "apply_pants_shape_guard",
    "apply_shorts_shape_guard",
    "build_pants_diffusion_seed",
    "cleanup_pants_speckles",
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
