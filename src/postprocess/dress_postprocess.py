"""Dress-specific postprocess steps.

Extracted 1:1 from `app.py` (v18.x). Helpers injected:

    fit_like(src, ref, *, is_mask) -> np.ndarray
    safe_uint8(arr)                 -> np.ndarray uint8
    garment_texture_valid_mask(rgb, mask) -> np.ndarray bool/uint8

Public API:
    build_dress_diffusion_seed(init_tryon, gen_mask, garment_mask, *, fit_like, safe_uint8)
    preserve_dress_pattern_gpu(generated, source, garment_mask, *, fit_like, safe_uint8, **opts)
    lock_dress_source_pattern_final(output, source, garment_mask, raw_diffusion=None, *,
                                     fit_like, safe_uint8, garment_texture_valid_mask, **opts)
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


def build_dress_diffusion_seed(
    init_tryon_rgb: np.ndarray,
    gen_mask: np.ndarray,
    garment_mask: np.ndarray,
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    gen_mask = fit_like(gen_mask, init_tryon_rgb, is_mask=True)
    garment_mask = fit_like(garment_mask, init_tryon_rgb, is_mask=True)
    active = gen_mask > 20
    if int(active.sum()) < 500:
        return init_tryon_rgb

    soft_alpha = cv2.GaussianBlur(active.astype(np.float32), (11, 11), 2.6)
    soft_alpha = np.clip(soft_alpha * 0.42, 0.0, 0.42)[..., None]

    init_lab = cv2.cvtColor(init_tryon_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    guide_lab = init_lab.copy()
    light_low = cv2.GaussianBlur(init_lab[:, :, 0], (0, 0), 3.0)
    chroma_a = cv2.GaussianBlur(init_lab[:, :, 1], (0, 0), 1.2)
    chroma_b = cv2.GaussianBlur(init_lab[:, :, 2], (0, 0), 1.2)
    guide_lab[:, :, 0] = init_lab[:, :, 0] * 0.82 + light_low * 0.18
    guide_lab[:, :, 1] = init_lab[:, :, 1] * 0.96 + chroma_a * 0.04
    guide_lab[:, :, 2] = init_lab[:, :, 2] * 0.96 + chroma_b * 0.04
    guide = cv2.cvtColor(safe_uint8(guide_lab), cv2.COLOR_LAB2RGB).astype(np.float32)

    garment_active = (garment_mask > 20) & active
    if int(garment_active.sum()) > 200:
        median_rgb = np.median(init_tryon_rgb[garment_active], axis=0).astype(np.float32)
        guide[active] = guide[active] * 0.99 + median_rgb * 0.01

        ys, xs = np.where(garment_active)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        height = max(1, y2 - y1)
        width = max(1, x2 - x1)
        yy, xx = np.indices(garment_mask.shape[:2], dtype=np.float32)
        xn = (xx - (x1 + x2) * 0.5) / max(1.0, width * 0.5)
        yn = (yy - y1) / max(1.0, float(height))
        vertical_folds = np.sin((xn * 4.8 + yn * 1.1) * np.pi) * 11.0
        side_shadow = -15.0 * np.exp(-((np.abs(xn) - 0.76) ** 2) / 0.050)
        waist_shadow = -10.0 * np.exp(-((yn - 0.42) ** 2) / 0.018) * np.exp(-(xn ** 2) / 0.65)
        center_highlight = 7.0 * np.exp(-(xn ** 2) / 0.18) * np.clip((yn - 0.18) / 0.52, 0.0, 1.0)
        fold_delta = cv2.GaussianBlur(
            (vertical_folds + side_shadow + waist_shadow + center_highlight).astype(np.float32),
            (0, 0),
            2.6,
        )

        lab = cv2.cvtColor(safe_uint8(guide), cv2.COLOR_RGB2LAB).astype(np.float32)
        fold_mask = cv2.GaussianBlur(garment_active.astype(np.float32), (17, 17), 5.0)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + fold_delta * fold_mask, 0, 255)
        guide = cv2.cvtColor(safe_uint8(lab), cv2.COLOR_LAB2RGB).astype(np.float32)

    seed = (
        init_tryon_rgb.astype(np.float32) * (1.0 - soft_alpha)
        + guide.astype(np.float32) * soft_alpha
    )
    return safe_uint8(seed)


def preserve_dress_pattern_gpu(
    generated_rgb: np.ndarray,
    source_rgb: np.ndarray,
    garment_mask: np.ndarray,
    core_mask: Optional[np.ndarray] = None,
    detail_strength: float = 0.30,
    chroma_strength: float = 0.42,
    low_freq_from_generated: float = 0.94,
    edge_relax: float = 0.55,
    *,
    fit_like: Callable,
    safe_uint8: Callable,
) -> np.ndarray:
    source_rgb = fit_like(source_rgb, generated_rgb, is_mask=False)
    garment_mask = fit_like(garment_mask, generated_rgb, is_mask=True)

    mask = garment_mask > 20
    if int(mask.sum()) < 500:
        return generated_rgb

    if core_mask is None:
        core_u8 = cv2.erode(
            mask.astype(np.uint8) * 255,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
    else:
        core_u8 = fit_like(core_mask, generated_rgb, is_mask=True)
        core_u8 = cv2.erode(core_u8, np.ones((5, 5), np.uint8), iterations=1)

    core_f = cv2.GaussianBlur((core_u8 > 20).astype(np.float32), (9, 9), 2.2)
    full_f = cv2.GaussianBlur(mask.astype(np.float32), (17, 17), 4.0)
    edge_f = np.clip(full_f - core_f, 0.0, 1.0)

    detail_core = float(np.clip(detail_strength, 0.0, 1.0))
    detail_edge = float(np.clip(detail_strength * edge_relax, 0.0, 1.0))
    chroma_core = float(np.clip(chroma_strength, 0.0, 1.0))
    chroma_edge = float(np.clip(chroma_strength * edge_relax, 0.0, 1.0))
    low_mix = float(np.clip(low_freq_from_generated, 0.70, 1.0))

    detail_alpha = np.clip(core_f * detail_core + edge_f * detail_edge, 0.0, 1.0)
    chroma_alpha = np.clip(core_f * chroma_core + edge_f * chroma_edge, 0.0, 1.0)

    gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    gen_l = gen_lab[:, :, 0]
    src_l = src_lab[:, :, 0]

    gen_low = cv2.GaussianBlur(gen_l, (0, 0), 4.6)
    src_low = cv2.GaussianBlur(src_l, (0, 0), 4.6)
    target_low = gen_low * low_mix + src_low * (1.0 - low_mix)

    fine_sigma = 1.45
    gen_detail = gen_l - cv2.GaussianBlur(gen_l, (0, 0), fine_sigma)
    src_detail = src_l - cv2.GaussianBlur(src_l, (0, 0), fine_sigma)
    detail_delta = np.clip(src_detail - gen_detail, -26.0, 26.0)

    rebuilt_l = target_low + gen_detail + detail_delta * detail_alpha
    l_alpha = np.clip(full_f, 0.0, 1.0)
    gen_lab[:, :, 0] = np.clip(
        gen_l * (1.0 - l_alpha) + rebuilt_l * l_alpha,
        0,
        255,
    )

    gen_lab[:, :, 1] = (
        gen_lab[:, :, 1] * (1.0 - chroma_alpha)
        + src_lab[:, :, 1] * chroma_alpha
    )
    gen_lab[:, :, 2] = (
        gen_lab[:, :, 2] * (1.0 - chroma_alpha)
        + src_lab[:, :, 2] * chroma_alpha
    )

    return cv2.cvtColor(safe_uint8(gen_lab), cv2.COLOR_LAB2RGB)


def lock_dress_source_pattern_final(
    output_rgb: np.ndarray,
    source_rgb: np.ndarray,
    garment_mask: np.ndarray,
    raw_diffusion_rgb: Optional[np.ndarray] = None,
    detail_strength: float = 0.92,
    chroma_strength: float = 0.88,
    source_low_mix: float = 0.30,
    edge_relax: float = 0.50,
    *,
    fit_like: Callable,
    safe_uint8: Callable,
    garment_texture_valid_mask: Callable,
) -> np.ndarray:
    source_rgb = fit_like(source_rgb, output_rgb, is_mask=False)
    garment_mask = fit_like(garment_mask, output_rgb, is_mask=True)
    if raw_diffusion_rgb is not None:
        raw_diffusion_rgb = fit_like(raw_diffusion_rgb, output_rgb, is_mask=False)

    mask = garment_mask > 20
    if int(mask.sum()) < 500:
        return output_rgb

    ys, xs = np.where(mask)
    if ys.size:
        y_top, y_bot = int(ys.min()), int(ys.max())
        strip_h = max(2, int(round((y_bot - y_top + 1) * 0.08)))
        top_block = np.zeros_like(mask)
        top_block[y_top : y_top + strip_h, :] = True
        mask_no_top = mask & (~top_block)
        if int(mask_no_top.sum()) >= 500:
            mask = mask_no_top

    mask_u8 = mask.astype(np.uint8) * 255
    core_u8 = cv2.erode(mask_u8, np.ones((5, 5), np.uint8), iterations=1)
    if int(core_u8.sum()) < 255 * 350:
        core_u8 = mask_u8

    valid = garment_texture_valid_mask(source_rgb, garment_mask)
    valid_f = cv2.GaussianBlur(valid.astype(np.float32), (9, 9), 2.0)
    valid_f = np.clip(valid_f, 0.0, 1.0)
    if float(valid_f[mask].mean()) < 0.08:
        valid_f = mask.astype(np.float32)

    core_f = cv2.GaussianBlur((core_u8 > 20).astype(np.float32), (9, 9), 2.0)
    full_f = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 3.6)
    edge_f = np.clip(full_f - core_f, 0.0, 1.0)
    detail_alpha = np.clip(
        (core_f * (0.58 + valid_f * 0.42) + edge_f * edge_relax * (0.35 + valid_f * 0.65))
        * float(np.clip(detail_strength, 0.0, 1.0)),
        0.0,
        0.94,
    )

    out_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    out_l = out_lab[:, :, 0]
    src_l = src_lab[:, :, 0]

    low_sigma = 14.0
    out_low = cv2.GaussianBlur(out_l, (0, 0), low_sigma)
    if raw_diffusion_rgb is not None:
        raw_l = cv2.cvtColor(raw_diffusion_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0]
        raw_low = cv2.GaussianBlur(raw_l, (0, 0), low_sigma)
        out_low = out_low * 0.70 + raw_low * 0.30

    src_low = cv2.GaussianBlur(src_l, (0, 0), low_sigma)
    low_mix = float(np.clip(source_low_mix, 0.0, 0.55))
    target_low = out_low * (1.0 - low_mix) + src_low * low_mix
    src_detail = np.clip(src_l - src_low, -86.0, 86.0)
    rebuilt_l = np.clip(target_low + src_detail, 0, 255)

    out_lab[:, :, 0] = np.clip(
        out_l * (1.0 - detail_alpha) + rebuilt_l * detail_alpha,
        0,
        255,
    )

    chroma_alpha = np.clip(
        detail_alpha * valid_f * float(np.clip(chroma_strength, 0.0, 1.0)),
        0.0,
        0.92,
    )
    out_lab[:, :, 1] = out_lab[:, :, 1] * (1.0 - chroma_alpha) + src_lab[:, :, 1] * chroma_alpha
    out_lab[:, :, 2] = out_lab[:, :, 2] * (1.0 - chroma_alpha) + src_lab[:, :, 2] * chroma_alpha

    locked = cv2.cvtColor(safe_uint8(out_lab), cv2.COLOR_LAB2RGB)
    sharp = cv2.addWeighted(locked, 1.18, cv2.GaussianBlur(locked, (0, 0), 0.75), -0.18, 0)
    sharp_alpha = np.clip(cv2.GaussianBlur((core_u8 > 20).astype(np.float32), (7, 7), 1.4) * 0.34, 0.0, 0.34)
    return safe_uint8(locked.astype(np.float32) * (1.0 - sharp_alpha[..., None]) + sharp.astype(np.float32) * sharp_alpha[..., None])


__all__ = [
    "build_dress_diffusion_seed",
    "preserve_dress_pattern_gpu",
    "lock_dress_source_pattern_final",
]
