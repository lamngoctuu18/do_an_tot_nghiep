from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import random
import re

import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv

from src.image_ops import (
    blend_tryon,
    build_cloth_mask,
    build_skeleton_erase_mask,
    compute_body_measurements,
    compute_leg_measurements,
    detect_full_pose,
    detect_upper_body_box,
    erase_clothing_region,
    full_pose_to_box,
    prefit_scale_cloth,
    read_image_rgb,
    save_image_rgb,
    segment_cloth_ensemble,
    segment_cloth_u2net,
    smooth_pose_landmarks,
    warp_cloth_to_torso,
)
from src.human_parsing import (
    parse_human,
    get_arm_mask,
    get_clothing_mask,
    get_foreground_keep_mask,
    get_neck_mask,
    get_pants_mask,
    get_skin_mask,
)
from src.tps_warp import tps_warp_cloth, warp_sleeves_to_arms, classify_garment_type, simple_affine_warp_cloth, detect_garment_category, detect_pants_landmarks, detect_pants_type, detect_pants_style, piecewise_warp_pants_cloth
from src.gen_tryon import GenConfig, generate_tryon_image
from src.cloud_vton_router import CloudVTONUnavailableError, generate_with_cloud_router
try:
    from src.gemini_prompt import (
        analyze_garment_prompt_with_gemini,
        fallback_describe_garment,
        GeminiPromptUnavailableError,
    )
except Exception:  # pragma: no cover — Gemini is optional
    analyze_garment_prompt_with_gemini = None
    fallback_describe_garment = None
    GeminiPromptUnavailableError = RuntimeError
from src.garment_router import route_garment
from src.masks.category_mask_builder import build_category_mask
from src.prompts.category_prompts import build_category_negative
from src.postprocess.pants_postprocess import (
    restore_upper_body_for_pants as _pp_restore_upper_body,
    build_shorts_edit_band as _pp_build_shorts_edit_band,
    build_pants_shape_mask as _pp_build_pants_shape_mask,
    build_shorts_shape_mask as _pp_build_shorts_shape_mask,
    build_shorts_wear_mask as _pp_build_shorts_wear_mask,
    reference_garment_color as _pp_reference_garment_color,
    render_reference_shorts as _pp_render_reference_shorts,
    apply_pants_shape_guard as _pp_apply_pants_shape_guard,
    apply_shorts_shape_guard as _pp_apply_shorts_shape_guard,
    build_pants_diffusion_seed as _pp_build_pants_diffusion_seed,
    cleanup_pants_speckles as _pp_cleanup_pants_speckles,
    cleanup_long_pants_denim_artifacts as _pp_cleanup_long_pants_denim_artifacts,
    restore_long_pants_ankle_skin as _pp_restore_long_pants_ankle_skin,
    cleanup_shorts_external_spill as _pp_cleanup_shorts_external_spill,
    cleanup_shorts_old_hem_bleed as _pp_cleanup_shorts_old_hem_bleed,
    cleanup_shorts_upper_cloth_spill as _pp_cleanup_shorts_upper_cloth_spill,
    cleanup_shorts_center_trim_artifact as _pp_cleanup_shorts_center_trim_artifact,
    recover_pants_texture_detail as _pp_recover_pants_texture_detail,
)
from src.postprocess.dress_postprocess import (
    build_dress_diffusion_seed as _pp_build_dress_diffusion_seed,
)
from src.garment_silhouettes import detect_dress_silhouette, build_dress_width_curve
from src.storage import resolve_storage_config


load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)


storage = resolve_storage_config()

# Debug: set VTON_DEBUG=1 to save intermediate images for diagnostics
_DEBUG = os.getenv("VTON_DEBUG", "0").strip() in ("1", "true", "yes")


def _debug_save(name: str, image: np.ndarray, is_mask: bool = False) -> None:
    """Save intermediate image for debugging when VTON_DEBUG=1."""
    if not _DEBUG:
        return
    debug_dir = storage.base_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%H%M%S")
    path = debug_dir / f"{ts}_{name}.png"
    if is_mask:
        cv2.imwrite(str(path), image)
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# ═══════════════════════════════════════════════════════════════════
#  Helper functions (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════════

def _build_upper_body_prior_mask(image_shape: tuple[int, int], pose_box) -> np.ndarray:
    height, width = image_shape

    left_shoulder = np.array(pose_box.left_shoulder)
    right_shoulder = np.array(pose_box.right_shoulder)
    left_hip = np.array(pose_box.left_hip)
    right_hip = np.array(pose_box.right_hip)

    shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
    hip_width = float(np.linalg.norm(left_hip - right_hip))
    torso_height = float(np.mean([abs(left_hip[1] - left_shoulder[1]), abs(right_hip[1] - right_shoulder[1])]))

    center_x = int((left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4)
    top_y = int(min(left_shoulder[1], right_shoulder[1]) - 0.08 * torso_height)
    bottom_y = int(max(left_hip[1], right_hip[1]) + 0.35 * torso_height)
    half_w = int(max(shoulder_width * 0.95, hip_width * 1.0))

    left_x = max(0, center_x - half_w)
    right_x = min(width - 1, center_x + half_w)
    top_y = max(0, top_y)
    bottom_y = min(height - 1, bottom_y)

    prior = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(prior, (left_x, top_y), (right_x, bottom_y), 255, thickness=-1)

    sleeve_radius = max(10, int(0.26 * shoulder_width))
    cv2.circle(prior, tuple(left_shoulder.astype(int)), sleeve_radius, 255, thickness=-1)
    cv2.circle(prior, tuple(right_shoulder.astype(int)), sleeve_radius, 255, thickness=-1)

    kernel = np.ones((17, 17), np.uint8)
    prior = cv2.morphologyEx(prior, cv2.MORPH_CLOSE, kernel, iterations=1)
    return prior


def _build_masks_for_garment_preservation(
    binary_mask: np.ndarray,
    image_shape: tuple[int, int],
    pose_box,
) -> tuple[np.ndarray, np.ndarray]:
    prior_mask = _build_upper_body_prior_mask(image_shape, pose_box)

    dilate_kernel = np.ones((15, 15), np.uint8)  # v16.5: from 25→15, prevent shoulder bleed
    dilated_garment = cv2.dilate(binary_mask, dilate_kernel, iterations=1)
    gen_mask = cv2.bitwise_or(dilated_garment, prior_mask)

    erode_kernel = np.ones((11, 11), np.uint8)
    core_mask = cv2.erode(binary_mask, erode_kernel, iterations=1)

    if int(gen_mask.sum()) < 2550:
        gen_mask = cv2.bitwise_or(binary_mask, prior_mask)

    return core_mask, gen_mask


def _apply_color_consistency(
    generated_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    garment_mask: np.ndarray,
    strength: float = 0.6,
) -> np.ndarray:
    mask = garment_mask > 0
    if mask.sum() < 200:
        return generated_rgb

    gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    blended = gen_lab.copy()
    eps = 1e-6
    for channel in range(3):
        gen_vals = gen_lab[..., channel][mask]
        ref_vals = ref_lab[..., channel][mask]

        gen_mean, gen_std = float(gen_vals.mean()), float(gen_vals.std() + eps)
        ref_mean, ref_std = float(ref_vals.mean()), float(ref_vals.std() + eps)

        adjusted = (gen_vals - gen_mean) * (ref_std / gen_std) + ref_mean
        blended_vals = gen_vals * (1.0 - strength) + adjusted * strength
        blended[..., channel][mask] = np.clip(blended_vals, 0, 255)

    return cv2.cvtColor(_safe_uint8(blended), cv2.COLOR_LAB2RGB)


def _smooth_hoodie_lower_torso_tone(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    garment_mask: np.ndarray,
    full_pose: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce lower-belly colour patches on fitted hoodie outputs."""
    h, w = output_rgb.shape[:2]
    mask = _fit_like(garment_mask, output_rgb, is_mask=True)
    mask = (mask > 20).astype(np.uint8) * 255
    ys, xs = np.where(mask > 20)
    empty = np.zeros((h, w), dtype=np.uint8)
    if len(xs) < 300:
        return output_rgb, empty

    y1, y2 = int(ys.min()), int(ys.max())
    height = max(1, y2 - y1)
    lower_y1 = y1 + int(height * 0.54)
    lower_y2 = y1 + int(height * 0.88)
    upper_y1 = y1 + int(height * 0.18)
    upper_y2 = y1 + int(height * 0.46)

    lower_band = np.zeros((h, w), dtype=np.uint8)
    upper_band = np.zeros((h, w), dtype=np.uint8)
    lower_band[max(0, lower_y1):min(h, lower_y2 + 1), :] = 255
    upper_band[max(0, upper_y1):min(h, upper_y2 + 1), :] = 255
    lower = cv2.bitwise_and(mask, lower_band)
    upper = cv2.bitwise_and(mask, upper_band)

    if full_pose is not None:
        required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        if all(full_pose.get(k) is not None for k in required):
            ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
            rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
            lh = np.array(full_pose["left_hip"], dtype=np.float32)
            rh = np.array(full_pose["right_hip"], dtype=np.float32)
            sw = max(24.0, float(np.linalg.norm(ls - rs)))
            hip_w = max(18.0, float(np.linalg.norm(lh - rh)))
            sh_c = (ls + rs) * 0.5
            hip_c = (lh + rh) * 0.5
            center_keep = np.zeros((h, w), dtype=np.uint8)
            for y in range(max(0, lower_y1), min(h, lower_y2 + 1)):
                f = float(np.clip((y - sh_c[1]) / max(1.0, hip_c[1] - sh_c[1]), 0.0, 1.0))
                cx = float(sh_c[0] * (1.0 - f) + hip_c[0] * f)
                half = max(sw * 0.46, hip_w * 0.62)
                x1 = max(0, int(round(cx - half)))
                x2 = min(w - 1, int(round(cx + half)))
                center_keep[y, x1:x2 + 1] = 255
            lower = cv2.bitwise_and(lower, center_keep)

    lower = cv2.morphologyEx(lower, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    upper = cv2.erode(upper, np.ones((3, 3), np.uint8), iterations=1)
    if int(cv2.countNonZero(lower)) < 200 or int(cv2.countNonZero(upper)) < 200:
        return output_rgb, empty

    out_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    init_lab = cv2.cvtColor(init_tryon_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lower_bool = lower > 20
    upper_bool = upper > 20

    target = (
        np.median(out_lab[upper_bool], axis=0) * 0.65
        + np.median(init_lab[upper_bool], axis=0) * 0.35
    )
    current = np.median(out_lab[lower_bool], axis=0)
    delta = target - current
    delta[0] = float(np.clip(delta[0], -8.0, 8.0))
    delta[1] = float(np.clip(delta[1], -4.0, 4.0))
    delta[2] = float(np.clip(delta[2], -4.0, 4.0))

    alpha = cv2.GaussianBlur(lower.astype(np.float32) / 255.0, (15, 15), 4.0)
    alpha = np.clip(alpha * 0.72, 0.0, 0.72)
    adjusted = out_lab.copy()
    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] + delta[0] * alpha, 0, 255)
    adjusted[:, :, 1] = np.clip(adjusted[:, :, 1] + delta[1] * alpha, 0, 255)
    adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] + delta[2] * alpha, 0, 255)
    return cv2.cvtColor(_safe_uint8(adjusted), cv2.COLOR_LAB2RGB), lower


def _seal_hoodie_hem_notches(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill accidental vertical gaps in the lower hoodie hem."""
    fixed = (mask > 20).astype(np.uint8) * 255
    original = fixed.copy()
    repair = np.zeros_like(fixed)
    ys, xs = np.where(fixed > 20)
    if len(xs) < 300:
        return fixed, repair

    y1, y2 = int(ys.min()), int(ys.max())
    height = max(1, y2 - y1)
    start_y = y1 + int(height * 0.55)
    rows = []
    for y in range(start_y, y2 + 1):
        row_xs = np.where(fixed[y] > 20)[0]
        if len(row_xs) < 12:
            continue
        rows.append((y, int(row_xs.min()), int(row_xs.max())))
    if len(rows) < 6:
        return fixed, repair

    rights = np.array([r for _, _, r in rows], dtype=np.float32)
    lefts = np.array([l for _, l, _ in rows], dtype=np.float32)
    widths = rights - lefts + 1.0
    target_right = int(np.percentile(rights, 88))
    target_left = int(np.percentile(lefts, 12))
    median_width = float(np.median(widths))

    max_side_fill = max(8, int(median_width * 0.16))
    for y, left, right in rows:
        # Hoodie hems should be continuous. Fill only obvious bite-outs so the
        # silhouette stays fitted but does not develop a skirt-like slit.
        if right < target_right - 10:
            fill_to = min(fixed.shape[1] - 1, right + max_side_fill, target_right)
            fixed[y, right:fill_to + 1] = 255
        if left > target_left + 10:
            fill_from = max(0, left - max_side_fill, target_left)
            fixed[y, fill_from:left + 1] = 255

    repair = cv2.bitwise_and(fixed, cv2.bitwise_not(original))
    repair[:start_y, :] = 0
    return fixed, repair


def _build_hoodie_structure_anchor_mask(
    init_tryon_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> np.ndarray:
    """Detect thin hoodie details that diffusion should not repaint."""
    h, w = init_tryon_rgb.shape[:2]
    mask = _fit_like(garment_mask, init_tryon_rgb, is_mask=True)
    mask = (mask > 20).astype(np.uint8) * 255
    empty = np.zeros((h, w), dtype=np.uint8)
    mask_area = int(cv2.countNonZero(mask))
    if mask_area < 300:
        return empty

    ys, _ = np.where(mask > 20)
    y1, y2 = int(ys.min()), int(ys.max())
    height = max(1, y2 - y1)

    bands = np.zeros((h, w), dtype=np.uint8)
    # Neck/drawstrings, kangaroo pocket seam, and ribbed waist/hem.
    for a, b in ((0.00, 0.34), (0.42, 0.68), (0.74, 0.98)):
        ya = max(0, y1 + int(height * a))
        yb = min(h - 1, y1 + int(height * b))
        bands[ya:yb + 1, :] = 255
    banded_mask = cv2.bitwise_and(mask, bands)
    if int(cv2.countNonZero(banded_mask)) < 80:
        return empty

    gray = cv2.cvtColor(init_tryon_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.6)
    edges = cv2.Canny(gray_blur, 28, 82)

    low = cv2.GaussianBlur(gray, (0, 0), 2.2)
    detail = cv2.absdiff(gray, low)
    vals = detail[banded_mask > 20]
    detail_threshold = 10.0
    if vals.size:
        detail_threshold = max(10.0, float(np.percentile(vals, 88)) * 0.90)
    detail_mask = ((detail >= detail_threshold) & (banded_mask > 20)).astype(np.uint8) * 255

    mask_edge = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    structure = cv2.bitwise_or(edges, detail_mask)
    structure = cv2.bitwise_or(structure, cv2.bitwise_and(mask_edge, bands))
    structure = cv2.bitwise_and(structure, banded_mask)
    structure = cv2.morphologyEx(
        structure,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    structure = cv2.dilate(
        structure,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    max_area = max(80, int(mask_area * 0.18))
    if int(cv2.countNonZero(structure)) > max_area:
        strong_edges = cv2.Canny(gray_blur, 52, 132)
        strong_threshold = max(
            13.0,
            float(np.percentile(vals, 94)) if vals.size else 13.0,
        )
        strong_detail = ((detail >= strong_threshold) & (banded_mask > 20)).astype(np.uint8) * 255
        structure = cv2.bitwise_or(strong_edges, strong_detail)
        structure = cv2.bitwise_or(structure, cv2.bitwise_and(mask_edge, bands))
        structure = cv2.bitwise_and(structure, banded_mask)
        structure = cv2.dilate(
            structure,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    return (structure > 20).astype(np.uint8) * 255


def _build_hoodie_sleeve_torso_seam_mask(
    shape: tuple[int, int],
    garment_mask: np.ndarray,
    full_pose: dict | None,
) -> np.ndarray:
    """Pose-guided side seam where hoodie sleeve must separate from torso."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        return out
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(full_pose.get(k) is None for k in required):
        return out

    mask = _fit_like(garment_mask, out, is_mask=True)
    mask = (mask > 20).astype(np.uint8) * 255
    ys, _ = np.where(mask > 20)
    if len(ys) < 300:
        return out

    ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
    rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
    lh = np.array(full_pose["left_hip"], dtype=np.float32)
    rh = np.array(full_pose["right_hip"], dtype=np.float32)
    sw = max(24.0, float(np.linalg.norm(ls - rs)))
    sh_c = (ls + rs) * 0.5
    hip_c = (lh + rh) * 0.5
    torso_h = max(30.0, float(hip_c[1] - sh_c[1]))
    garment_bottom = int(ys.max())
    thickness = max(2, int(round(sw * 0.025)))

    for side in ("left", "right"):
        sh = np.array(full_pose.get(f"{side}_shoulder"), dtype=np.float32)
        hip = np.array(full_pose.get(f"{side}_hip"), dtype=np.float32)
        if sh.size != 2 or hip.size != 2:
            continue
        outer = -1.0 if sh[0] < sh_c[0] else 1.0
        inward = -outer
        start = np.array(
            [sh[0] + inward * sw * 0.13, sh[1] + sw * 0.20],
            dtype=np.float32,
        )
        mid = np.array(
            [
                sh[0] * 0.38 + hip[0] * 0.62 + outer * sw * 0.035,
                sh[1] + torso_h * 0.55,
            ],
            dtype=np.float32,
        )
        end_y = min(float(garment_bottom - sw * 0.08), hip[1] + torso_h * 0.42)
        end = np.array(
            [hip[0] + outer * sw * 0.08, end_y],
            dtype=np.float32,
        )
        pts = np.array([start, mid, end], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [pts], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
        # Short armpit crease keeps the sleeve from visually melting into the chest.
        crease_start = np.array(
            [sh[0] + outer * sw * 0.05, sh[1] + sw * 0.12],
            dtype=np.float32,
        )
        crease_end = start + np.array([outer * sw * 0.04, sw * 0.03], dtype=np.float32)
        cv2.line(
            out,
            tuple(np.round(crease_start).astype(int)),
            tuple(np.round(crease_end).astype(int)),
            255,
            thickness=max(2, thickness),
            lineType=cv2.LINE_AA,
        )

    out = cv2.bitwise_and(out, mask)
    out = cv2.dilate(
        out,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return (out > 20).astype(np.uint8) * 255


def _add_hoodie_sleeve_torso_separation(
    output_rgb: np.ndarray,
    garment_mask: np.ndarray,
    full_pose: dict | None,
    *,
    strength: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """Darken a subtle underarm/side seam so sleeves read apart from torso."""
    seam = _build_hoodie_sleeve_torso_seam_mask(output_rgb.shape[:2], garment_mask, full_pose)
    if int(cv2.countNonZero(seam)) < 30:
        return output_rgb, seam

    strength = float(np.clip(strength, 0.0, 0.45))
    alpha = cv2.GaussianBlur(seam.astype(np.float32) / 255.0, (5, 5), 1.2)
    alpha = np.clip(alpha * strength, 0.0, strength)

    lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] - alpha * 34.0, 0, 255)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + alpha * 1.5, 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2] + alpha * 1.5, 0, 255)
    return cv2.cvtColor(_safe_uint8(lab), cv2.COLOR_LAB2RGB), seam


def _build_hoodie_pose_sleeve_mask(
    shape: tuple[int, int],
    full_pose: dict | None,
    parsing: dict | None = None,
) -> np.ndarray:
    """Pose sleeves for hoodie seed/mask when the TPS torso warp loses sleeves."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        return out
    ls = full_pose.get("left_shoulder")
    rs = full_pose.get("right_shoulder")
    if ls is None or rs is None:
        return out

    ls_np = np.array(ls, dtype=np.float32)
    rs_np = np.array(rs, dtype=np.float32)
    sw = max(24.0, float(np.linalg.norm(ls_np - rs_np)))
    upper_r = max(7, int(round(sw * 0.125)))
    lower_r = max(5, int(round(sw * 0.085)))

    for side in ("left", "right"):
        sh = full_pose.get(f"{side}_shoulder")
        el = full_pose.get(f"{side}_elbow")
        wr = full_pose.get(f"{side}_wrist")
        if sh is None or el is None:
            continue
        if wr is None:
            wr = el
        sh_p = np.array(sh, dtype=np.float32)
        el_p = np.array(el, dtype=np.float32)
        wr_p = np.array(wr, dtype=np.float32)
        cuff_p = el_p + (wr_p - el_p) * 0.80

        sleeve = np.zeros((h, w), dtype=np.uint8)
        cv2.line(
            sleeve,
            tuple(np.round(sh_p).astype(int)),
            tuple(np.round(el_p).astype(int)),
            255,
            thickness=upper_r * 2,
            lineType=cv2.LINE_AA,
        )
        cv2.line(
            sleeve,
            tuple(np.round(el_p).astype(int)),
            tuple(np.round(cuff_p).astype(int)),
            255,
            thickness=lower_r * 2,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(sleeve, tuple(np.round(sh_p).astype(int)), max(5, int(upper_r * 0.65)), 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(sleeve, tuple(np.round(el_p).astype(int)), upper_r, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(sleeve, tuple(np.round(cuff_p).astype(int)), lower_r, 255, -1, lineType=cv2.LINE_AA)

        if parsing is not None:
            arm = parsing.get(f"{side}_arm")
            if arm is not None:
                arm_u8 = (_fit_like(arm, sleeve, is_mask=True) > 20).astype(np.uint8) * 255
                arm_pad_k = max(5, int(round(sw * 0.07))) | 1
                arm_u8 = cv2.dilate(
                    arm_u8,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (arm_pad_k, arm_pad_k)),
                    iterations=1,
                )
                corridor = cv2.dilate(
                    sleeve,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (arm_pad_k, arm_pad_k)),
                    iterations=1,
                )
                sleeve = cv2.bitwise_or(sleeve, cv2.bitwise_and(arm_u8, corridor))

        top_limit = max(0, int(round(min(sh_p[1], el_p[1]) - sw * 0.22)))
        bottom_limit = min(h, int(round(max(cuff_p[1], el_p[1]) + sw * 0.08)))
        sleeve[:top_limit, :] = 0
        sleeve[bottom_limit:, :] = 0
        sleeve = cv2.morphologyEx(
            sleeve,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1,
        )
        out = cv2.bitwise_or(out, sleeve)

    return (out > 20).astype(np.uint8) * 255


def _paint_hoodie_pose_sleeves(
    warped_cloth: np.ndarray,
    source_mask: np.ndarray,
    sleeve_mask: np.ndarray,
    full_pose: dict | None = None,
) -> np.ndarray:
    """Fill pose sleeve areas with neutral garment texture before diffusion."""
    sleeve_mask = _fit_like(sleeve_mask, warped_cloth, is_mask=True)
    source_mask = _fit_like(source_mask, warped_cloth, is_mask=True)
    active = sleeve_mask > 20
    if int(active.sum()) < 80:
        return warped_cloth

    valid = (source_mask > 20) & (warped_cloth.sum(axis=2) > 20)
    if int(valid.sum()) < 100:
        return warped_cloth

    median_rgb = np.median(warped_cloth[valid], axis=0).astype(np.uint8)
    fabric_base = warped_cloth.copy()
    fabric_base[~valid] = median_rgb
    fabric_soft = cv2.GaussianBlur(fabric_base, (0, 0), 5.0)
    fill = _safe_uint8(fabric_soft.astype(np.float32) * 0.55 + median_rgb.astype(np.float32) * 0.45)

    alpha = cv2.GaussianBlur(active.astype(np.float32), (9, 9), 2.0)
    alpha = np.clip(alpha * 0.92, 0.0, 0.92)[..., None]
    seeded = _safe_uint8(warped_cloth.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha)

    detail = np.zeros(sleeve_mask.shape[:2], dtype=np.float32)
    yy, xx = np.indices(sleeve_mask.shape[:2], dtype=np.float32)
    sleeve_alpha = cv2.GaussianBlur(active.astype(np.float32), (9, 9), 2.0)
    sleeve_alpha = np.clip(sleeve_alpha, 0.0, 1.0)
    micro = (
        np.sin(xx * 0.55 + yy * 0.17) * 1.4
        + np.sin(xx * 0.13 - yy * 0.42) * 1.1
    )
    detail += micro * sleeve_alpha

    if full_pose is not None:
        ls = full_pose.get("left_shoulder")
        rs = full_pose.get("right_shoulder")
        if ls is not None and rs is not None:
            ls_np = np.array(ls, dtype=np.float32)
            rs_np = np.array(rs, dtype=np.float32)
            sw = max(24.0, float(np.linalg.norm(ls_np - rs_np)))
            body_cx = float((ls_np[0] + rs_np[0]) * 0.5)
            for side in ("left", "right"):
                sh = full_pose.get(f"{side}_shoulder")
                el = full_pose.get(f"{side}_elbow")
                wr = full_pose.get(f"{side}_wrist")
                if sh is None or el is None:
                    continue
                if wr is None:
                    wr = el
                sh_p = np.array(sh, dtype=np.float32)
                el_p = np.array(el, dtype=np.float32)
                wr_p = np.array(wr, dtype=np.float32)
                cuff_p = el_p + (wr_p - el_p) * 0.80
                outer = -1.0 if sh_p[0] < body_cx else 1.0

                shadow = np.zeros_like(sleeve_mask)
                highlight = np.zeros_like(sleeve_mask)
                cuff = np.zeros_like(sleeve_mask)
                shadow_offset = np.array([outer * sw * 0.055, 0.0], dtype=np.float32)
                highlight_offset = np.array([-outer * sw * 0.035, 0.0], dtype=np.float32)
                pts_shadow = [sh_p + shadow_offset, el_p + shadow_offset, cuff_p + shadow_offset]
                pts_highlight = [sh_p + highlight_offset, el_p + highlight_offset, cuff_p + highlight_offset]
                for p0, p1 in zip(pts_shadow[:-1], pts_shadow[1:]):
                    cv2.line(
                        shadow,
                        tuple(np.round(p0).astype(int)),
                        tuple(np.round(p1).astype(int)),
                        255,
                        max(2, int(sw * 0.018)),
                        lineType=cv2.LINE_AA,
                    )
                for p0, p1 in zip(pts_highlight[:-1], pts_highlight[1:]):
                    cv2.line(
                        highlight,
                        tuple(np.round(p0).astype(int)),
                        tuple(np.round(p1).astype(int)),
                        255,
                        max(2, int(sw * 0.014)),
                        lineType=cv2.LINE_AA,
                    )
                cv2.circle(
                    cuff,
                    tuple(np.round(cuff_p).astype(int)),
                    max(4, int(sw * 0.045)),
                    255,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                shadow_f = cv2.GaussianBlur(shadow.astype(np.float32) / 255.0, (7, 7), 1.5)
                highlight_f = cv2.GaussianBlur(highlight.astype(np.float32) / 255.0, (7, 7), 1.5)
                cuff_f = cv2.GaussianBlur(cuff.astype(np.float32) / 255.0, (5, 5), 1.1)
                sleeve_side = sleeve_alpha
                detail += (-8.0 * shadow_f + 4.5 * highlight_f - 5.5 * cuff_f) * sleeve_side

    lab = cv2.cvtColor(seeded, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + detail, 0, 255)
    return cv2.cvtColor(_safe_uint8(lab), cv2.COLOR_LAB2RGB)


def _remove_hoodie_edge_spill(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    garment_mask: np.ndarray,
    parsing: dict | None,
    full_pose: dict | None,
    gen_mask_soft: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restore tiny gray hoodie spill outside the clean pose silhouette."""
    h, w = output_rgb.shape[:2]
    base = _fit_like(garment_mask, output_rgb, is_mask=True)
    base = (base > 20).astype(np.uint8) * 255
    empty = np.zeros((h, w), dtype=np.uint8)
    if int(cv2.countNonZero(base)) < 300:
        return output_rgb, empty

    clean_base = cv2.morphologyEx(
        base,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    clean_base = cv2.morphologyEx(
        clean_base,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    allowed = clean_base
    if full_pose is not None:
        allowed = _build_hoodie_pose_fit_mask((h, w), full_pose, parsing, clean_base)
        sleeve_allow = _build_hoodie_pose_sleeve_mask((h, w), full_pose, parsing)
        allowed = cv2.bitwise_or(allowed, sleeve_allow)
    allowed = cv2.morphologyEx(
        allowed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    allowed = cv2.dilate(
        allowed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    near = cv2.dilate(
        allowed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        iterations=1,
    )
    spill_zone = cv2.subtract(near, allowed)
    if gen_mask_soft is not None:
        gm = _fit_like(gen_mask_soft, output_rgb, is_mask=True)
        spill_zone = cv2.bitwise_or(spill_zone, cv2.bitwise_and((gm > 12).astype(np.uint8) * 255, cv2.bitwise_not(allowed)))

    lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB)
    grayish = (
        (lab[:, :, 0] > 72)
        & (lab[:, :, 0] < 230)
        & (np.abs(lab[:, :, 1].astype(np.int16) - 128) < 9)
        & (np.abs(lab[:, :, 2].astype(np.int16) - 128) < 11)
    )
    diff = np.mean(np.abs(output_rgb.astype(np.int16) - person_rgb.astype(np.int16)), axis=2)
    spill = ((spill_zone > 20) & grayish & (diff > 5.0)).astype(np.uint8) * 255
    spill = cv2.morphologyEx(spill, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    if int(cv2.countNonZero(spill)) < 15:
        return output_rgb, empty

    alpha = cv2.GaussianBlur(spill.astype(np.float32) / 255.0, (7, 7), 1.8)
    alpha = np.clip(alpha * 0.95, 0.0, 0.95)[..., None]
    cleaned = _safe_uint8(
        output_rgb.astype(np.float32) * (1.0 - alpha)
        + person_rgb.astype(np.float32) * alpha
    )
    return cleaned, spill


def _sharpen_hoodie_output(
    output_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> np.ndarray:
    """Light LAB unsharp mask inside hoodie only."""
    mask = _fit_like(garment_mask, output_rgb, is_mask=True)
    mask = (mask > 20).astype(np.uint8) * 255
    if int(cv2.countNonZero(mask)) < 300:
        return output_rgb
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    alpha = cv2.GaussianBlur(core.astype(np.float32) / 255.0, (7, 7), 1.6)
    alpha = np.clip(alpha * 0.58, 0.0, 0.58)

    lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_blur = cv2.GaussianBlur(lab[:, :, 0], (0, 0), 1.1)
    detail = lab[:, :, 0] - l_blur
    lab[:, :, 0] = np.clip(lab[:, :, 0] + detail * alpha * 1.25, 0, 255)
    sharpened = cv2.cvtColor(_safe_uint8(lab), cv2.COLOR_LAB2RGB)
    return _safe_uint8(
        output_rgb.astype(np.float32) * (1.0 - alpha[..., None])
        + sharpened.astype(np.float32) * alpha[..., None]
    )


def _restore_hoodie_structure_detail(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend back crisp hoodie collar/shoulder/waist/hem detail from the seed."""
    h, w = output_rgb.shape[:2]
    mask = _fit_like(garment_mask, output_rgb, is_mask=True)
    mask = (mask > 20).astype(np.uint8) * 255
    ys, xs = np.where(mask > 20)
    empty = np.zeros((h, w), dtype=np.uint8)
    if len(xs) < 300:
        return output_rgb, empty

    structure = _build_hoodie_structure_anchor_mask(init_tryon_rgb, mask)

    structure = cv2.morphologyEx(
        structure,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    structure = cv2.bitwise_and(structure, mask)
    if int(cv2.countNonZero(structure)) < 50:
        return output_rgb, empty

    alpha = cv2.GaussianBlur(structure.astype(np.float32) / 255.0, (7, 7), 1.8)
    alpha = np.clip(alpha * 0.68, 0.0, 0.68)[..., None]
    restored = (
        output_rgb.astype(np.float32) * (1.0 - alpha)
        + init_tryon_rgb.astype(np.float32) * alpha
    )
    return _safe_uint8(restored), structure


def _build_hoodie_diffusion_seed(
    init_tryon_rgb: np.ndarray,
    gen_mask: np.ndarray,
    garment_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Give diffusion room to redraw hoodie folds without losing anchors."""
    h, w = init_tryon_rgb.shape[:2]
    gen_mask = _fit_like(gen_mask, init_tryon_rgb, is_mask=True)
    garment_mask = _fit_like(garment_mask, init_tryon_rgb, is_mask=True)
    editable = ((gen_mask > 20) & (garment_mask > 20)).astype(np.uint8) * 255
    if int(cv2.countNonZero(editable)) < 500:
        return init_tryon_rgb, np.zeros((h, w), dtype=np.uint8)

    core = cv2.erode(editable, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    core_bool = core > 20
    if int(core_bool.sum()) < 300:
        core_bool = editable > 20

    median_rgb = np.median(init_tryon_rgb[core_bool], axis=0).astype(np.float32)
    smooth_rgb = cv2.GaussianBlur(init_tryon_rgb, (0, 0), 5.0).astype(np.float32)
    guide = smooth_rgb * 0.38 + median_rgb[None, None, :] * 0.62

    anchor = _build_hoodie_structure_anchor_mask(init_tryon_rgb, editable)
    anchor_alpha = cv2.GaussianBlur(anchor.astype(np.float32) / 255.0, (5, 5), 1.2)
    anchor_alpha = np.clip(anchor_alpha * 0.82, 0.0, 0.82)[..., None]
    guide = guide * (1.0 - anchor_alpha) + init_tryon_rgb.astype(np.float32) * anchor_alpha

    edit_alpha = cv2.GaussianBlur(editable.astype(np.float32) / 255.0, (11, 11), 3.2)
    edit_alpha = np.clip(edit_alpha * 0.72, 0.0, 0.72)[..., None]
    seed = init_tryon_rgb.astype(np.float32) * (1.0 - edit_alpha) + guide * edit_alpha
    return _safe_uint8(seed), anchor


def _build_dress_diffusion_seed(
    init_tryon_rgb: np.ndarray,
    gen_mask: np.ndarray,
    garment_mask: np.ndarray,
) -> np.ndarray:
    return _pp_build_dress_diffusion_seed(
        init_tryon_rgb, gen_mask, garment_mask,
        fit_like=_fit_like, safe_uint8=_safe_uint8,
    )


def _parsing_union_mask(
    parsing: dict | None,
    keys: tuple[str, ...],
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not parsing:
        return mask
    for key in keys:
        part = parsing.get(key)
        if part is None:
            continue
        if part.shape[:2] != shape:
            part = cv2.resize(part, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_or(mask, (part > 20).astype(np.uint8) * 255)
    return mask


def _build_top_hair_underlap_mask(
    shape: tuple[int, int],
    parsing: dict | None,
    full_pose: dict | None,
    base_torso_mask: np.ndarray | None = None,
) -> np.ndarray:
    """v19.47: build từ torso/shoulder envelope (KHÔNG dùng warped_mask, vì
    nó đang bị thủng ở vùng tóc). underlap = hair ∩ upper_torso_band ∪
    shoulder_bridge — vùng cần được vẽ áo PHÍA SAU tóc."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if parsing is None or full_pose is None:
        return out
    hair = parsing.get("hair")
    if hair is None:
        return out
    hair = _fit_like(hair, out, is_mask=True)

    if base_torso_mask is None:
        base_torso_mask = np.zeros((h, w), dtype=np.uint8)
    else:
        base_torso_mask = _fit_like(base_torso_mask, out, is_mask=True)

    torso = _build_pose_coarse_body_envelope((h, w), full_pose, "top", base_torso_mask)
    if int(cv2.countNonZero(torso)) == 0:
        return out

    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(full_pose.get(k) is None for k in required):
        return out
    ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
    rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
    lh = np.array(full_pose["left_hip"], dtype=np.float32)
    rh = np.array(full_pose["right_hip"], dtype=np.float32)
    shoulder_w = max(24.0, float(np.linalg.norm(ls - rs)))
    torso_h = max(30.0, float(((lh[1] + rh[1]) - (ls[1] + rs[1])) * 0.5))
    center_sh = (ls + rs) * 0.5

    # v19.50: top_y chỉ đẩy lên rất nhẹ (5% shoulder_w) — đủ để bao tóc
    # rủ TRÊN vai, không lấn sâu vào vùng cổ.
    top_y = int(max(0, min(ls[1], rs[1]) - shoulder_w * 0.05))
    chest_bottom = int(min(h - 1, center_sh[1] + torso_h * 0.42))
    chest_band = np.zeros((h, w), dtype=np.uint8)
    chest_band[top_y:chest_bottom, :] = 255
    torso_band = cv2.bitwise_and(torso, chest_band)

    shoulder_bridge = np.zeros((h, w), dtype=np.uint8)
    shoulder_bridge_parts: list[np.ndarray] = []
    bridge_len = int(torso_h * 0.40)
    bridge_th = max(12, int(shoulder_w * 0.20))
    for sh in (ls, rs):
        bridge_part = np.zeros((h, w), dtype=np.uint8)
        p0 = (int(sh[0]), int(sh[1]))
        p1 = (int(sh[0]), int(min(h - 1, sh[1] + bridge_len)))
        cv2.line(bridge_part, p0, p1, 255, bridge_th, lineType=cv2.LINE_AA)
        cv2.circle(bridge_part, p0, max(8, bridge_th // 2), 255, -1, lineType=cv2.LINE_AA)
        shoulder_bridge = cv2.bitwise_or(shoulder_bridge, bridge_part)
        shoulder_bridge_parts.append(bridge_part)

    torso_hint = cv2.bitwise_or(torso_band, shoulder_bridge)

    # v19.50: tách 2 nguồn underlap:
    #   (a) hair_overlap = hair_dilate ∩ chest_band (tóc rủ trên vai/ngực)
    #   (b) shoulder_band_under_hair = shoulder_bridge touched by hair_dilate
    #       — giữ nguyên chiều dài bridge nếu vai bị hair label che
    hair_d = cv2.dilate(hair, np.ones((21, 21), np.uint8), iterations=1)
    torso_d = cv2.dilate(torso_hint, np.ones((19, 19), np.uint8), iterations=1)

    hair_overlap = cv2.bitwise_and(hair_d, chest_band)
    shoulder_band_under_hair = np.zeros((h, w), dtype=np.uint8)
    min_bridge_touch = max(8, bridge_th)
    for bridge_part in shoulder_bridge_parts:
        if int(cv2.countNonZero(cv2.bitwise_and(bridge_part, hair_d))) >= min_bridge_touch:
            shoulder_band_under_hair = cv2.bitwise_or(shoulder_band_under_hair, bridge_part)
    overlap = cv2.bitwise_or(hair_overlap, shoulder_band_under_hair)

    # Vẫn intersect với torso_d để chắc chắn không tràn ra ngoài silhouette
    overlap = cv2.bitwise_and(overlap, torso_d)

    overlap = cv2.morphologyEx(overlap, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    overlap = cv2.dilate(overlap, np.ones((7, 7), np.uint8), iterations=1)
    overlap = cv2.GaussianBlur(overlap, (9, 9), 2.0)
    overlap = (overlap > 16).astype(np.uint8) * 255

    # v19.50: chỉ trừ vùng SAU CỔ — hộp HẸP giữa 2 vai, CHỈ ở phần TRÊN
    # đường vai (không lan xuống chest_band → giữ collar/vai trước).
    inset = int(shoulder_w * 0.20)
    x_lo = int(min(ls[0], rs[0])) + inset
    x_hi = int(max(ls[0], rs[0])) - inset
    if x_hi > x_lo:
        neck_zone = np.zeros((h, w), dtype=np.uint8)
        neck_zone[:top_y, x_lo:x_hi] = 255
        overlap = cv2.subtract(overlap, neck_zone)

    # Trừ face dilated (an toàn — không vẽ áo lên mặt)
    _face_m = parsing.get("face")
    if _face_m is not None:
        _face_m = _fit_like(_face_m, out, is_mask=True)
        _face_d = cv2.dilate(_face_m, np.ones((5, 5), np.uint8), iterations=1)
        overlap = cv2.subtract(overlap, _face_d)

    return overlap


def _build_pose_coarse_body_envelope(
    shape: tuple[int, int],
    full_pose: dict | None,
    garment_category: str,
    garment_mask: np.ndarray,
) -> np.ndarray:
    """Build a body/pose envelope instead of a tight garment contour."""
    h, w = shape
    envelope = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        return envelope
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(key not in full_pose for key in required):
        return envelope

    ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
    rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
    lh = np.array(full_pose["left_hip"], dtype=np.float32)
    rh = np.array(full_pose["right_hip"], dtype=np.float32)
    shoulder_w = max(24.0, float(np.linalg.norm(ls - rs)))
    hip_w = max(18.0, float(np.linalg.norm(lh - rh)))
    torso_h = max(30.0, float(((lh[1] + rh[1]) - (ls[1] + rs[1])) * 0.5))
    center_sh = (ls + rs) * 0.5
    center_hip = (lh + rh) * 0.5

    if garment_category == "pants":
        top_y = int(max(0, center_hip[1] - torso_h * 0.25))
        bottom_y = h - 1
        ys, _ = np.where(garment_mask > 20)
        if len(ys) > 100:
            bottom_y = min(bottom_y, int(ys.max() + max(12, shoulder_w * 0.12)))
        width_top = max(hip_w * 0.85, shoulder_w * 0.42)
        width_bottom = max(width_top * 0.62, shoulder_w * 0.34)
        for y in range(top_y, bottom_y + 1):
            f = (y - top_y) / max(1.0, float(bottom_y - top_y))
            half_w = (1.0 - f) * width_top + f * width_bottom
            x1 = max(0, int(round(center_hip[0] - half_w)))
            x2 = min(w - 1, int(round(center_hip[0] + half_w)))
            envelope[y, x1:x2 + 1] = 255
    else:
        top_y = int(max(0, center_sh[1] - shoulder_w * 0.35))
        if garment_category == "dress":
            ys, _ = np.where(garment_mask > 20)
            garment_bottom = int(ys.max()) if len(ys) > 100 else int(center_hip[1] + torso_h * 1.9)
            warped_bottom = garment_bottom + int(shoulder_w * 0.08)
            # v18.7: Trust the warped garment extent for hem position. The old
            # `max(garment_bottom, hip + torso*1.35)` floor extended every dress
            # to past-knee — mid-thigh sources came out knee/shin length. Only
            # apply a torso-relative floor when warp clearly failed (ended above
            # hip), and cap at mid-shin to prevent ankle-length over-painting.
            if warped_bottom < center_hip[1]:
                bottom_y = int(min(h - 1, center_hip[1] + int(torso_h * 0.6)))
            else:
                hard_cap = int(center_hip[1] + torso_h * 1.55)
                bottom_y = int(min(h - 1, min(hard_cap, warped_bottom)))
            width_curve = (
                (0.00, shoulder_w * 0.78),
                (0.32, max(shoulder_w * 0.56, hip_w * 0.72)),
                (0.68, max(shoulder_w * 0.62, hip_w * 0.86)),
                (1.00, max(shoulder_w * 0.70, hip_w * 0.96)),
            )
        else:
            bottom_y = int(min(h - 1, center_hip[1] + torso_h * 0.42))
            width_curve = (
                (0.00, shoulder_w * 0.72),
                (0.45, max(shoulder_w * 0.58, hip_w * 0.70)),
                (1.00, max(shoulder_w * 0.54, hip_w * 0.78)),
            )

        curve = np.array(width_curve, dtype=np.float32)
        for y in range(top_y, bottom_y + 1):
            f = (y - top_y) / max(1.0, float(bottom_y - top_y))
            hip_mix = min(1.0, max(0.0, (y - center_sh[1]) / max(1.0, center_hip[1] - center_sh[1])))
            cx = float(center_sh[0] * (1.0 - hip_mix) + center_hip[0] * hip_mix)
            half_w = float(np.interp(f, curve[:, 0], curve[:, 1]))
            x1 = max(0, int(round(cx - half_w)))
            x2 = min(w - 1, int(round(cx + half_w)))
            envelope[y, x1:x2 + 1] = 255

        if garment_category in {"top", "dress"}:
            arm_thickness = max(10, int(round(shoulder_w * (0.16 if garment_category == "dress" else 0.13))))
            for side in ("left", "right"):
                sh = full_pose.get(f"{side}_shoulder")
                el = full_pose.get(f"{side}_elbow")
                wr = full_pose.get(f"{side}_wrist")
                if sh is None:
                    continue
                points = [tuple(map(int, sh))]
                if el is not None:
                    points.append(tuple(map(int, el)))
                if wr is not None:
                    points.append(tuple(map(int, wr)))
                if len(points) >= 2:
                    for p0, p1 in zip(points[:-1], points[1:]):
                        cv2.line(envelope, p0, p1, 255, arm_thickness)
                    for point in points:
                        cv2.circle(envelope, point, max(arm_thickness // 2, 5), 255, -1)

    return cv2.morphologyEx(envelope, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)


def _build_hoodie_pose_fit_mask(
    shape: tuple[int, int],
    full_pose: dict | None,
    parsing: dict | None,
    base_mask: np.ndarray,
) -> np.ndarray:
    """Tight hoodie silhouette for post-TPS correction.

    TPS preserves too much of the flat-lay hoodie rectangle. This mask uses the
    model's current upper-clothes parsing plus pose sleeves, then clips to the
    warped hoodie neighbourhood so the seed is body-shaped before diffusion.
    """
    h, w = shape
    base = (base_mask > 20).astype(np.uint8) * 255
    out = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        return base
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(full_pose.get(k) is None for k in required):
        return base

    ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
    rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
    lh = np.array(full_pose["left_hip"], dtype=np.float32)
    rh = np.array(full_pose["right_hip"], dtype=np.float32)
    sw = max(24.0, float(np.linalg.norm(ls - rs)))
    hip_w = max(18.0, float(np.linalg.norm(lh - rh)))
    sh_c = (ls + rs) * 0.5
    hip_c = (lh + rh) * 0.5
    torso_h = max(30.0, float(((lh[1] + rh[1]) - (ls[1] + rs[1])) * 0.5))

    ys, xs = np.where(base > 20)
    if len(xs) < 80:
        return base
    base_y1, base_y2 = int(ys.min()), int(ys.max())

    top_y = max(0, min(base_y1, int(min(ls[1], rs[1]) - sw * 0.18)))
    bottom_y = min(h - 1, int(max(lh[1], rh[1]) + torso_h * 0.36))
    bottom_y = max(bottom_y, min(h - 1, base_y2 - int(sw * 0.10)))

    for y in range(top_y, bottom_y + 1):
        shoulder_half = max(sw * 0.58, hip_w * 0.74)
        waist_half = max(sw * 0.48, hip_w * 0.66)
        hem_half = max(sw * 0.52, hip_w * 0.72)
        if y <= sh_c[1]:
            top_f = float(np.clip((y - top_y) / max(1.0, sh_c[1] - top_y), 0.0, 1.0))
            neck_half = max(sw * 0.34, hip_w * 0.46)
            half = neck_half * (1.0 - top_f) + shoulder_half * top_f
        else:
            f = float(np.clip((y - sh_c[1]) / max(1.0, hip_c[1] - sh_c[1]), 0.0, 1.0))
            if f < 0.55:
                half = shoulder_half * (1.0 - f / 0.55) + waist_half * (f / 0.55)
            else:
                ff = (f - 0.55) / 0.45
                half = waist_half * (1.0 - ff) + hem_half * ff
        f_c = 0.0 if y <= sh_c[1] else f
        cx = float(sh_c[0] * (1.0 - f_c) + hip_c[0] * f_c)
        x1 = max(0, int(round(cx - half)))
        x2 = min(w - 1, int(round(cx + half)))
        out[y, x1:x2 + 1] = 255

    # Keep the existing parsed upper garment silhouette when reliable. Drop
    # left_arm/right_arm here — including them re-introduces the wide bare-arm
    # band the category mask already stripped out, producing visible "extra
    # marks" along the shoulder/sleeve outline in the final composite.
    parsed_top = np.zeros((h, w), dtype=np.uint8)
    if parsing is not None:
        for key in ("upper_clothes", "dress"):
            part = parsing.get(key)
            if part is None:
                continue
            if part.shape[:2] != (h, w):
                part = cv2.resize(part, (w, h), interpolation=cv2.INTER_NEAREST)
            parsed_top = cv2.bitwise_or(parsed_top, (part > 20).astype(np.uint8) * 255)
        if int(cv2.countNonZero(parsed_top)) > 80:
            parsed_top = cv2.morphologyEx(
                parsed_top,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            parsed_top = cv2.dilate(
                parsed_top,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            out = cv2.bitwise_or(out, parsed_top)

    # Pose sleeves: enough room for sleeves to follow arms, not a rectangle.
    for side in ("left", "right"):
        sh = full_pose.get(f"{side}_shoulder")
        el = full_pose.get(f"{side}_elbow")
        wr = full_pose.get(f"{side}_wrist")
        if sh is None or el is None:
            continue
        if wr is None:
            wr = el
        sh_i = tuple(map(int, sh))
        el_i = tuple(map(int, el))
        wr_i = tuple(map(int, wr))
        upper_r = max(7, int(sw * 0.11))
        lower_r = max(6, int(sw * 0.09))
        cv2.line(out, sh_i, el_i, 255, upper_r * 2, lineType=cv2.LINE_AA)
        cv2.line(out, el_i, wr_i, 255, lower_r * 2, lineType=cv2.LINE_AA)
        cv2.circle(out, sh_i, max(5, int(upper_r * 0.65)), 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, el_i, upper_r, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, wr_i, lower_r, 255, -1, lineType=cv2.LINE_AA)

    # Preserve hood/neck area from the warped hoodie, but do not OR back the
    # wide flat-lay shoulder bar that TPS often creates.
    hood_band = np.zeros((h, w), dtype=np.uint8)
    hood_bottom = min(h - 1, int(sh_c[1] + sw * 0.08))
    hood_left = max(0, int(sh_c[0] - sw * 0.48))
    hood_right = min(w - 1, int(sh_c[0] + sw * 0.48))
    hood_band[:hood_bottom + 1, hood_left:hood_right + 1] = 255
    hood_keep = cv2.bitwise_and(base, hood_band)
    out = cv2.bitwise_or(out, hood_keep)

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    near_base = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    if int(cv2.countNonZero(parsed_top)) > 80:
        near_base = cv2.bitwise_or(near_base, parsed_top)
    out = cv2.bitwise_and(out, near_base)
    if int(cv2.countNonZero(out)) < max(500, int(cv2.countNonZero(base) * 0.55)):
        return base
    return (out > 20).astype(np.uint8) * 255


def _build_human_tryon_prior_mask(
    garment_mask: np.ndarray,
    parsing: dict | None,
    full_pose: dict | None,
    garment_category: str,
    sleeve_type: str = "long",
    accessory_subtype: str = "",
    top_subtype: str = "",
) -> tuple[np.ndarray, bool]:
    """TripVVT-style diffusion mask: coarse human prior + AI parsing + pose.

    Logic moved to `src/masks/category_mask_builder.py` (per-category rules).
    This function now dispatches; behaviour is preserved exactly.
    """
    return build_category_mask(
        garment_mask,
        parsing,
        full_pose,
        category=garment_category,
        sleeve_type=sleeve_type,
        accessory_subtype=accessory_subtype,
        top_subtype=top_subtype,
        parsing_union_mask=_parsing_union_mask,
        pose_envelope_fn=_build_pose_coarse_body_envelope,
        neck_mask_fn=get_neck_mask,
        fit_like=_fit_like,
    )


def _build_dress_drape_delta(garment_mask: np.ndarray) -> np.ndarray:
    mask = garment_mask > 20
    delta = np.zeros(garment_mask.shape[:2], dtype=np.float32)
    if int(mask.sum()) < 500:
        return delta

    ys, xs = np.where(mask)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    height = max(1.0, float(y2 - y1))
    half_width = max(1.0, float(x2 - x1) * 0.5)
    yy, xx = np.indices(garment_mask.shape[:2], dtype=np.float32)
    xn = (xx - (x1 + x2) * 0.5) / half_width
    yn = (yy - y1) / height

    body_fade = np.clip((yn - 0.08) / 0.62, 0.0, 1.0)
    skirt_fade = np.clip((yn - 0.25) / 0.58, 0.0, 1.0)
    vertical_folds = np.sin((xn * 4.2 + yn * 0.42) * np.pi) * 8.5 * body_fade
    fine_folds = np.sin((xn * 7.5 - yn * 0.30) * np.pi) * 4.0 * skirt_fade
    waist_shadow = -15.0 * np.exp(-((yn - 0.39) ** 2) / 0.018) * np.exp(-(xn ** 2) / 0.70)
    side_shadow = -9.0 * np.exp(-((np.abs(xn) - 0.78) ** 2) / 0.040)
    center_highlight = 4.0 * np.exp(-(xn ** 2) / 0.20) * np.clip((yn - 0.18) / 0.55, 0.0, 1.0)
    lower_ripple = np.sin((xn * 5.8 + yn * 0.18) * np.pi) * 5.5 * np.clip((yn - 0.22) / 0.55, 0.0, 1.0)
    hip_shadow = -9.0 * np.exp(-((yn - 0.47) ** 2) / 0.030) * np.exp(-(xn ** 2) / 0.90)

    delta = vertical_folds + fine_folds + waist_shadow + side_shadow + center_highlight + lower_ripple + hip_shadow
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 6.0)
    delta = cv2.GaussianBlur(delta.astype(np.float32), (0, 0), 3.0) * np.clip(mask_f, 0.0, 1.0)
    return np.clip(delta, -28.0, 24.0)


def _restore_core_garment(
    generated_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    core_mask: np.ndarray,
    preserve_strength: float,
) -> np.ndarray:
    target_h, target_w = init_tryon_rgb.shape[:2]

    if generated_rgb.shape[:2] != (target_h, target_w):
        generated_rgb = cv2.resize(generated_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if core_mask.shape[:2] != (target_h, target_w):
        core_mask = cv2.resize(core_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    core_alpha = (core_mask.astype(np.float32) / 255.0) * np.clip(preserve_strength, 0.0, 1.0)
    core_alpha = core_alpha[..., None]

    output = generated_rgb.astype(np.float32) * (1.0 - core_alpha) + init_tryon_rgb.astype(np.float32) * core_alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _enforce_garment_identity(
    generated_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    garment_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Force output to stay close to original warped garment on the full garment area."""
    target_h, target_w = init_tryon_rgb.shape[:2]
    if generated_rgb.shape[:2] != (target_h, target_w):
        generated_rgb = cv2.resize(generated_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if garment_mask.shape[:2] != (target_h, target_w):
        garment_mask = cv2.resize(garment_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    full_alpha = (garment_mask.astype(np.float32) / 255.0) * np.clip(strength, 0.0, 1.0)
    full_alpha = full_alpha[..., None]
    output = generated_rgb.astype(np.float32) * (1.0 - full_alpha) + init_tryon_rgb.astype(np.float32) * full_alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _is_blackout_artifact(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> bool:
    """Detect catastrophic dark outputs while allowing dark garment colors."""
    if candidate_rgb.shape[:2] != reference_rgb.shape[:2]:
        candidate_rgb = cv2.resize(
            candidate_rgb,
            (reference_rgb.shape[1], reference_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    if garment_mask.shape[:2] != reference_rgb.shape[:2]:
        garment_mask = cv2.resize(
            garment_mask,
            (reference_rgb.shape[1], reference_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    outside = garment_mask <= 0
    if int(outside.sum()) < 200:
        return False

    cand_out = candidate_rgb[outside].astype(np.float32)
    ref_out = reference_rgb[outside].astype(np.float32)
    cand_mean = float(cand_out.mean())
    ref_mean = float(ref_out.mean())

    return cand_mean < max(5.0, ref_mean * 0.15)


def _apply_foreground_layer(
    base_rgb: np.ndarray,
    person_rgb: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Composite person's arms/face/hair on top of the try-on result.

    Uses erode-then-blur strategy (same as _soft_mask) for clean edges.
    """
    fg_f = _soft_mask(fg_mask, blur_sigma=1.5, erode_px=2)[..., None]
    result = base_rgb.astype(np.float32) * (1.0 - fg_f) + person_rgb.astype(np.float32) * fg_f
    return _safe_uint8(result)


def _remove_dress_rect_artifact(
    output_rgb: np.ndarray,
    fallback_rgb: np.ndarray,
    dress_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Remove brown rectangular pattern-reference leakage outside dress mask.

    Pattern reference is composed inside a rectangle; if any later step blends
    it without clipping by the dress mask, a brown rectangular block can
    appear (most visibly below the feet). This restores fallback pixels in
    any "dress-coloured" region that sits clearly outside the dress mask.
    """
    h, w = output_rgb.shape[:2]
    if dress_mask.shape[:2] != (h, w):
        dress_mask = cv2.resize(dress_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    allowed = cv2.dilate(dress_mask, np.ones((9, 9), np.uint8), iterations=1)
    allowed = cv2.GaussianBlur(allowed, (7, 7), 1.5)
    allowed = (allowed > 20).astype(np.uint8) * 255

    out = output_rgb.astype(np.float32)
    fb = fallback_rgb.astype(np.float32)
    diff = np.mean(np.abs(out - fb), axis=2)
    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    brownish = (r > 90) & (g > 65) & (b > 45) & (r > b + 12) & (g > b + 5)
    outside = allowed < 20
    spill = ((diff > 12) & brownish & outside).astype(np.uint8) * 255
    if int(cv2.countNonZero(spill)) < 20:
        return output_rgb, False
    spill = cv2.morphologyEx(spill, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(spill.astype(np.float32) / 255.0, (7, 7), 1.5)[..., None]
    cleaned = out * (1.0 - alpha) + fb * alpha
    return _safe_uint8(cleaned), True


def _dress_row_width(mask: np.ndarray, row: int) -> float:
    row = max(0, min(mask.shape[0] - 1, int(row)))
    nz = np.where(mask[row] > 0)[0]
    return float(nz.max() - nz.min()) if len(nz) >= 4 else 0.0


def _dress_reference_ab_stats(
    reference_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> tuple[np.ndarray, float, float, float] | None:
    if reference_rgb is None or reference_rgb.size == 0:
        return None
    if garment_mask.shape[:2] != reference_rgb.shape[:2]:
        garment_mask = cv2.resize(
            garment_mask,
            (reference_rgb.shape[1], reference_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    valid = garment_mask > 20
    if int(valid.sum()) < 50:
        return None
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = ref_lab[valid]
    ab = lab[:, 1:3]
    center = np.median(ab, axis=0)
    spread = float(np.median(np.linalg.norm(ab - center[None, :], axis=1)))
    l_center = float(np.median(lab[:, 0]))
    l_spread = float(np.median(np.abs(lab[:, 0] - l_center)))
    return center.astype(np.float32), spread, l_center, l_spread


def _dress_candidate_color_mask(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> np.ndarray:
    garment_mask = _fit_like(garment_mask, candidate_rgb, is_mask=True)
    stats = _dress_reference_ab_stats(reference_rgb, garment_mask)
    target = garment_mask > 20
    if stats is None or int(target.sum()) < 50:
        return target.astype(np.uint8) * 255
    center, spread, l_center, l_spread = stats
    cand_lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dist = np.linalg.norm(cand_lab[..., 1:3] - center[None, None, :], axis=2)
    ab_threshold = float(np.clip(18.0 + spread * 2.2, 20.0, 46.0))
    l_threshold = float(np.clip(24.0 + l_spread * 2.4, 30.0, 78.0))
    l_ok = np.abs(cand_lab[..., 0] - l_center) <= l_threshold
    mask = ((dist <= ab_threshold) & l_ok & target).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def _dress_prompt_length_hint(prompt: str | None) -> str:
    text = (prompt or "").lower()
    if "maxi" in text or "ankle" in text or "floor" in text:
        return "maxi"
    if "midi" in text or "calf" in text:
        return "midi"
    if "knee" in text:
        return "knee"
    if "mini" in text or "short dress" in text or "above knee" in text:
        return "mini"
    return "midi"


def _legacy_dress_diffusion_shape_guard(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    garment_mask: np.ndarray,
    full_pose: dict | None,
    parsing: dict | None,
    sleeve_type: str,
    length_hint: str,
) -> tuple[bool, list[str], np.ndarray]:
    """Check whether dress diffusion still matches the intended geometry."""
    h, w = candidate_rgb.shape[:2]
    target = _fit_like(garment_mask, candidate_rgb, is_mask=True)
    target = (target > 20).astype(np.uint8) * 255
    ys, xs = np.where(target > 0)
    if len(xs) < 300:
        return True, [], target

    dress_like = _dress_candidate_color_mask(candidate_rgb, reference_rgb, target)
    core = cv2.erode(target, np.ones((5, 5), np.uint8), iterations=1)
    coverage = float(cv2.countNonZero(cv2.bitwise_and(dress_like, core))) / max(1.0, float(cv2.countNonZero(core)))

    reasons: list[str] = []
    if coverage < 0.36:
        reasons.append(f"coverage={coverage:.2f}")

    ty1, ty2 = int(ys.min()), int(ys.max())
    th = max(1, ty2 - ty1)
    cys, cxs = np.where(dress_like > 0)
    if len(cxs) < 100:
        reasons.append("no_dress_like_pixels")
        return False, reasons, dress_like

    cy1, cy2 = int(cys.min()), int(cys.max())
    if cy1 < ty1 - max(12, int(th * 0.08)):
        reasons.append("neckline_too_high")
    if length_hint in {"midi", "maxi"} and cy2 < ty1 + int(th * 0.72):
        reasons.append("hem_too_short")

    if parsing:
        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("face", "hair"):
            region = parsing.get(key)
            if region is not None:
                protect = cv2.bitwise_or(
                    protect,
                    (_fit_like(region, candidate_rgb, is_mask=True) > 20).astype(np.uint8) * 255,
                )
        if int(cv2.countNonZero(protect)) > 0:
            overlap = cv2.countNonZero(cv2.bitwise_and(dress_like, protect))
            if overlap > max(20, int(cv2.countNonZero(protect) * 0.035)):
                reasons.append("dress_over_face_or_hair")

    if full_pose:
        ls = full_pose.get("left_shoulder")
        rs = full_pose.get("right_shoulder")
        lh = full_pose.get("left_hip")
        rh = full_pose.get("right_hip")
        lk = full_pose.get("left_knee")
        rk = full_pose.get("right_knee")
        la = full_pose.get("left_ankle")
        ra = full_pose.get("right_ankle")
        if ls is not None and rs is not None and lh is not None and rh is not None:
            ls_arr = np.array(ls, dtype=np.float32)
            rs_arr = np.array(rs, dtype=np.float32)
            lh_arr = np.array(lh, dtype=np.float32)
            rh_arr = np.array(rh, dtype=np.float32)
            shoulder_w = max(24.0, float(np.linalg.norm(ls_arr - rs_arr)))
            waist_y = int(round((min(ls_arr[1], rs_arr[1]) * 0.35) + (max(lh_arr[1], rh_arr[1]) * 0.65)))
            expected_waist = _dress_row_width(target, waist_y)
            actual_waist = _dress_row_width(dress_like, waist_y)
            if expected_waist > 12 and actual_waist > expected_waist * 1.28:
                reasons.append(f"waist_too_wide={actual_waist:.0f}/{expected_waist:.0f}")
            if expected_waist > 12 and actual_waist < expected_waist * 0.42:
                reasons.append(f"waist_missing={actual_waist:.0f}/{expected_waist:.0f}")

            if sleeve_type == "long":
                sleeve_count = 0
                sleeve_hit = 0
                for side in ("left", "right"):
                    env = _build_arm_pose_envelope((h, w), full_pose, parsing, side)
                    if env is None:
                        continue
                    env = cv2.bitwise_and(env, target)
                    n = int(cv2.countNonZero(env))
                    if n < 40:
                        continue
                    sleeve_count += n
                    sleeve_hit += int(cv2.countNonZero(cv2.bitwise_and(dress_like, env)))
                if sleeve_count > 80:
                    sleeve_cov = float(sleeve_hit) / float(sleeve_count)
                    if sleeve_cov < 0.20:
                        reasons.append(f"sleeve_missing={sleeve_cov:.2f}")

            if (
                length_hint in {"midi", "maxi"}
                and lk is not None and rk is not None
                and la is not None and ra is not None
            ):
                knee_y = float((float(lk[1]) + float(rk[1])) * 0.5)
                ankle_y = float((float(la[1]) + float(ra[1])) * 0.5)
                midi_min = knee_y + (ankle_y - knee_y) * 0.22
                if cy2 < midi_min:
                    reasons.append("hem_above_midi_band")
                if cy2 > ankle_y + shoulder_w * 0.18:
                    reasons.append("hem_hits_shoes")

    return len(reasons) == 0, reasons, dress_like


def _blend_dress_luminance_detail(
    base_rgb: np.ndarray,
    detail_rgb: np.ndarray,
    garment_mask: np.ndarray,
    strength: float = 0.18,
) -> np.ndarray:
    garment_mask = _fit_like(garment_mask, base_rgb, is_mask=True)
    if int(cv2.countNonZero(garment_mask)) < 50:
        return base_rgb
    if detail_rgb.shape[:2] != base_rgb.shape[:2]:
        detail_rgb = cv2.resize(detail_rgb, (base_rgb.shape[1], base_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    detail_lab = cv2.cvtColor(detail_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    alpha = cv2.GaussianBlur((garment_mask > 20).astype(np.float32), (9, 9), 2.0)
    alpha = np.clip(alpha * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0)
    base_lab[..., 0] = base_lab[..., 0] * (1.0 - alpha) + detail_lab[..., 0] * alpha
    return cv2.cvtColor(np.clip(base_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _restore_exposed_arm_skin(
    init_tryon: np.ndarray,
    person_rgb: np.ndarray,
    parsing: dict | None,
    warped_mask: np.ndarray,
    sleeve_type: str,
) -> np.ndarray:
    """Restore visible arm skin from the original person image for short sleeves.

    For short-sleeve garments, the area below the sleeve hem should show
    the person's real skin rather than whatever the blend produced.  This
    makes the try-on look far more natural without any diffusion cost.
    """
    if parsing is None or sleeve_type != "short":
        return init_tryon

    arm_mask = get_arm_mask(parsing)
    skin_mask = get_skin_mask(parsing)
    if arm_mask is None or skin_mask is None:
        return init_tryon

    # Only restore skin where the arm is visible AND not covered by garment
    exposed_arm = cv2.bitwise_and(arm_mask, skin_mask)
    garment_region = (warped_mask > 20).astype(np.uint8) * 255
    exposed_arm = cv2.bitwise_and(exposed_arm, cv2.bitwise_not(garment_region))

    if int(exposed_arm.sum()) < 255 * 50:
        return init_tryon

    # Slight expansion + soft feather for natural transition at sleeve hem
    exposed_arm = cv2.dilate(exposed_arm, np.ones((3, 3), np.uint8), iterations=1)
    exposed_arm_f = cv2.GaussianBlur(
        (exposed_arm > 0).astype(np.float32),
        (7, 7),
        2.5,
    )[..., None]

    mixed = (
        init_tryon.astype(np.float32) * (1.0 - exposed_arm_f)
        + person_rgb.astype(np.float32) * exposed_arm_f
    )
    return _safe_uint8(mixed)


def _build_sleeve_protect_mask(
    warped_mask: np.ndarray,
    arm_mask: np.ndarray | None,
) -> np.ndarray | None:
    """Areas where garment sleeves should stay in front of arm foreground layer."""
    if arm_mask is None:
        return None
    wm = (warped_mask > 20).astype(np.uint8) * 255
    wm = cv2.dilate(wm, np.ones((7, 7), np.uint8), iterations=1)
    arm_d = cv2.dilate(arm_mask, np.ones((9, 9), np.uint8), iterations=1)
    protect = cv2.bitwise_and(wm, arm_d)
    if int(protect.sum()) < 255 * 40:
        return None
    return protect


def _classify_dress_sleeve_type(cloth_mask: np.ndarray) -> str:
    """Dress-specific sleeve detection (v18.21).

    v18.15 only looked at upper rows and missed long-sleeve sheath shots
    where the arms hang straight down by the body — shoulder cap width ≈
    bust width because the arms are merged with the torso.  v18.21 adds a
    hem-flare cue: a narrow hem (sheath/mermaid silhouette) strongly
    implies the sleeves are tucked alongside the body, so default to
    long-sleeve unless there is a clear strap-only top.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        return "sleeveless"

    y1, y2 = int(ys.min()), int(ys.max())
    ch = max(1, y2 - y1)
    h_mask = cloth_mask.shape[0]

    def row_width(rel: float) -> float:
        row = max(0, min(h_mask - 1, y1 + int(ch * rel)))
        nz = np.where(cloth_mask[row] > 0)[0]
        return float(nz.max() - nz.min()) if len(nz) > 4 else 0.0

    bust_widths = [row_width(f) for f in (0.18, 0.22, 0.26, 0.30)]
    bust_widths = [w for w in bust_widths if w > 0]
    if not bust_widths:
        return "sleeveless"
    bust_w = max(bust_widths)

    upper_widths = [row_width(f) for f in (0.04, 0.06, 0.08, 0.10, 0.12)]
    upper_widths_pos = [w for w in upper_widths if w > 0]
    max_upper = max(upper_widths_pos) if upper_widths_pos else 0.0
    shoulder_top = row_width(0.02)

    hem_w = row_width(0.92)
    mid_skirt = row_width(0.65)
    waist_w = row_width(0.45)
    side_mid_peak = max(row_width(f) for f in (0.36, 0.42, 0.48, 0.54))
    lower_body_widths = [row_width(f) for f in (0.60, 0.65, 0.72, 0.78)]
    lower_body_widths = [w for w in lower_body_widths if w > 0]
    lower_body_w = float(np.median(lower_body_widths)) if lower_body_widths else 0.0

    # Long sleeves hanging beside the dress create a local horizontal "spike"
    # around the waist/hip rows. A-line dresses often have narrow upper rows,
    # so the old upper-row-only test read this exact source as sleeveless.
    if (
        lower_body_w > 10
        and side_mid_peak > max(bust_w, lower_body_w) * 1.10
        and waist_w > lower_body_w * 1.12
    ):
        return "long"

    # Strap-only top: shoulder very narrow AND hem clearly flares = sleeveless.
    if (
        shoulder_top > 0
        and shoulder_top < bust_w * 0.55
        and max_upper < bust_w * 0.95
    ):
        return "sleeveless"

    # Flared / A-line / skater silhouette: wide hem with narrow upper = sleeveless.
    if hem_w > bust_w * 1.28 and max_upper < bust_w * 1.10:
        return "sleeveless"

    # Sheath / pencil / mermaid: hem ≈ bust → arms typically hang by sides.
    # If upper rows hold ~bust width down to the waist, sleeves are long.
    if hem_w < bust_w * 1.12 and max_upper >= bust_w * 0.92:
        if waist_w >= bust_w * 0.85 and mid_skirt <= bust_w * 1.10:
            return "long"
        return "short"

    # Moderate flare with clear upper-row protrusion.
    if max_upper >= bust_w * 1.05:
        mid_lower = row_width(0.33)
        if mid_lower > bust_w * 1.05 and max_upper > bust_w * 1.12:
            return "long"
        return "short"

    return "sleeveless"


def _build_arm_pose_envelope(
    image_shape: tuple[int, int],
    pose: dict[str, tuple[int, int]] | None,
    parsing: dict[str, np.ndarray] | None,
    side: str,
) -> np.ndarray | None:
    """Build a tight sleeve-allowed area from the model arm pose."""
    if pose is None:
        return None

    h, w = image_shape
    sh = pose.get(f"{side}_shoulder")
    el = pose.get(f"{side}_elbow")
    wr = pose.get(f"{side}_wrist")
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    if sh is None or el is None or ls is None or rs is None:
        return None

    sh_p = tuple(map(int, sh))
    el_p = tuple(map(int, el))
    wr_p = tuple(map(int, wr if wr is not None else el))
    sw = max(20.0, float(np.linalg.norm(np.array(ls, dtype=np.float32) - np.array(rs, dtype=np.float32))))
    # v18.13: partial revert of v18.12 over-tightening (sleeves became strand-thin
    # and detached). Restore moderate width while staying below the original 0.13.
    upper_r = max(5, int(sw * 0.115))
    lower_r = max(4, int(sw * 0.10))

    env = np.zeros((h, w), dtype=np.uint8)
    cv2.line(env, sh_p, el_p, 255, thickness=upper_r * 2, lineType=cv2.LINE_AA)
    cv2.line(env, el_p, wr_p, 255, thickness=lower_r * 2, lineType=cv2.LINE_AA)
    cv2.circle(env, sh_p, int(upper_r * 1.15), 255, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(env, el_p, upper_r, 255, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(env, wr_p, lower_r, 255, thickness=-1, lineType=cv2.LINE_AA)

    if parsing:
        arm_region = parsing.get(f"{side}_arm")
        if arm_region is not None:
            arm_u8 = (_fit_like(arm_region, env, is_mask=True) > 20).astype(np.uint8) * 255
            # v18.13: 0.08 → 0.09 — slightly more parsing dilation so the
            # envelope hugs the actual arm contour, not just the pose line.
            k = (max(5, int(sw * 0.09)) | 1)
            arm_u8 = cv2.dilate(
                arm_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                iterations=1,
            )
            env = cv2.bitwise_or(env, arm_u8)

    top_limit = max(0, int(sh_p[1] - sw * 0.18))
    env[:top_limit, :] = 0
    # v18.13: hard bottom clamp — sleeve envelope must not reach below the
    # wrist (plus a small slack). This was the root cause of "sleeve sticking
    # to dress hip": when parsing arm region was noisy or pose wrist was high,
    # the envelope+dilate could extend down across the hip band.
    bottom_y_ref = int(max(sh_p[1], el_p[1], wr_p[1]))
    bottom_limit = min(h, bottom_y_ref + int(sw * 0.18))
    env[bottom_limit:, :] = 0
    env = cv2.morphologyEx(env, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return env


def _clip_sleeve_to_arm_pose(
    sleeve_rgb: np.ndarray,
    sleeve_mask_f: np.ndarray,
    pose: dict[str, tuple[int, int]] | None,
    parsing: dict[str, np.ndarray] | None,
    side: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Constrain a warped long sleeve to the model arm size and direction."""
    envelope = _build_arm_pose_envelope(sleeve_mask_f.shape[:2], pose, parsing, side)
    if envelope is None:
        return sleeve_rgb, sleeve_mask_f, False

    allowed = cv2.GaussianBlur(envelope.astype(np.float32) / 255.0, (9, 9), 2.0)
    # v18.13: 1.03 → 1.06 — small inflation so sleeve isn't a thin strand,
    # but well below v18.11's 1.12 which leaked into the hip band.
    allowed = np.clip(allowed * 1.06, 0.0, 1.0)
    clipped = np.clip(sleeve_mask_f.astype(np.float32) * allowed, 0.0, 1.0)
    if int((clipped > 0.05).sum()) < 50:
        return sleeve_rgb, sleeve_mask_f, False

    old_area = max(1, int((sleeve_mask_f > 0.05).sum()))
    new_area = int((clipped > 0.05).sum())
    if new_area < old_area * 0.35:
        return sleeve_rgb, sleeve_mask_f, False

    return sleeve_rgb, clipped, True


def _shape_sleeve_to_arm_contour(
    warped_cloth: np.ndarray,
    warped_mask: np.ndarray,
    parsing: dict[str, np.ndarray],
    pose: dict[str, tuple[int, int]],
    sleeve_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-TPS arm-contour-guided sleeve shaping (v16.8b).

    GP-VTON insight: instead of warping sleeves separately with TPS control points,
    reshape the warped mask to follow the arm contour from human parsing.
    This makes sleeves naturally conform to the person's arm shape.

    Only processes short sleeves. Long sleeves use separate sleeve warp.
    """
    if sleeve_type != "short" or parsing is None or pose is None:
        return warped_cloth, warped_mask

    arm_mask_L = parsing.get("left_arm")
    arm_mask_R = parsing.get("right_arm")
    if arm_mask_L is None and arm_mask_R is None:
        return warped_cloth, warped_mask

    h, w = warped_mask.shape[:2]

    # ── Body geometry from pose ──
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    lh = pose.get("left_hip")
    rh = pose.get("right_hip")
    if not all([ls, rs, lh, rh]):
        return warped_cloth, warped_mask

    body_cx = (ls[0] + rs[0]) / 2.0
    shoulder_w = abs(rs[0] - ls[0])
    if shoulder_w < 20:
        return warped_cloth, warped_mask
    torso_top_y = min(ls[1], rs[1])
    torso_bot_y = max(lh[1], rh[1])
    torso_h = max(1.0, torso_bot_y - torso_top_y)

    # Torso center column (excluded from sleeve shaping)
    torso_half_w = shoulder_w * 0.32

    # ── Sleeve zone: upper arm area outside torso center ──
    zone_y_top = max(0, int(torso_top_y - torso_h * 0.05))
    zone_y_bot = min(h, int(torso_top_y + torso_h * 0.45))

    rows = np.arange(h)
    cols = np.arange(w)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    y_in_zone = (row_grid >= zone_y_top) & (row_grid <= zone_y_bot)
    x_outside_torso = np.abs(col_grid - body_cx) > torso_half_w
    sleeve_zone = y_in_zone & x_outside_torso

    if sleeve_zone.sum() < 100:
        return warped_cloth, warped_mask

    # ── Arm envelope: where sleeve IS ALLOWED to exist ──
    arm_combined = np.zeros((h, w), dtype=np.uint8)
    if arm_mask_L is not None:
        arm_combined = cv2.bitwise_or(arm_combined, arm_mask_L)
    if arm_mask_R is not None:
        arm_combined = cv2.bitwise_or(arm_combined, arm_mask_R)

    # Dilate arm mask outward — sleeve fabric extends beyond visible skin
    dilate_px = max(7, int(shoulder_w * 0.12)) | 1
    k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    arm_envelope = cv2.dilate(arm_combined, k_ellipse, iterations=1)

    # Union with upper_clothes in sleeve zone — existing garment on arm is valid
    upper_clothes = parsing.get("upper_clothes")
    if upper_clothes is not None:
        arm_cloth = cv2.bitwise_and(
            upper_clothes,
            (sleeve_zone.astype(np.uint8) * 255),
        )
        arm_envelope = cv2.bitwise_or(arm_envelope, arm_cloth)

    # Also include the current warped garment's torso column as always-allowed
    # (we never clip the central torso, only the sleeve zone)
    torso_column = np.abs(col_grid - body_cx) <= torso_half_w
    arm_envelope[torso_column] = 255

    # ── Signed distance field from arm envelope boundary ──
    env_u8 = (arm_envelope > 0).astype(np.uint8) * 255
    dist_inside = cv2.distanceTransform(env_u8, cv2.DIST_L2, 5).astype(np.float32)
    dist_outside = cv2.distanceTransform(255 - env_u8, cv2.DIST_L2, 5).astype(np.float32)
    signed_dist = dist_inside - dist_outside

    # ── Alpha modulation in sleeve zone ──
    # v16.9: Wider margin (10%), higher floor (0.25), reduced ramp (0.35).
    # Previous values (0.08/0.15/0.55) were too aggressive → ate sleeve shape.
    # With diffusion now handling texture, contour shaping only needs light nudges.
    margin = max(8.0, shoulder_w * 0.10)
    alpha_mod = np.clip((signed_dist + margin) / (2.0 * margin), 0.25, 1.0)

    # Only modify the sleeve zone — torso center is untouched
    # v16.8c: Wider ramp (18px) and reduced strength (0.55) for gentle blending
    torso_dist = np.abs(col_grid.astype(np.float32) - float(body_cx)) - float(torso_half_w)
    ramp = np.clip(torso_dist / 18.0, 0.0, 1.0)

    old_mask = warped_mask.copy()
    mask_f = warped_mask.astype(np.float32)

    # v16.9: Reduced strength (0.35) — let diffusion handle the rest
    effective_alpha = 1.0 - ramp * 0.35 * (1.0 - alpha_mod)
    # Only apply in the Y-band of the sleeve zone
    y_mask = y_in_zone.astype(np.float32)
    effective_alpha = 1.0 - y_mask * (1.0 - effective_alpha)

    mask_f *= effective_alpha
    warped_mask = mask_f.clip(0, 255).astype(np.uint8)

    # v16.8c: Bilateral filter DISABLED — it flattens sleeve shape
    # The soft margin + reduced strength above already produce clean edges.

    # ── Handle clipped pixels: use blurred interior (not black) ──
    newly_clipped = (old_mask > 20) & (warped_mask < 5)
    if newly_clipped.sum() > 0:
        interior_blur = cv2.GaussianBlur(warped_cloth, (15, 15), 4.0)
        warped_cloth = warped_cloth.copy()
        warped_cloth[newly_clipped] = interior_blur[newly_clipped]

    return warped_cloth, warped_mask


# ═══════════════════════════════════════════════════════════════════
#  Category Lock — let user override auto-detect when garment shape
#  fools detect_garment_category (e.g. crewneck tee read as dress).
# ═══════════════════════════════════════════════════════════════════
CATEGORY_LOCK_CHOICES = [
    "auto", "top", "tshirt", "shirt", "hoodie", "jacket", "outer",
    "pants", "jeans", "shorts", "dress", "skirt",
    "belt", "bag", "scarf", "hat", "sunglasses", "shoes", "boots",
    "generic",
]


def _normalize_category_lock(category_lock: str | None) -> str:
    value = (category_lock or "auto").strip().lower()
    if value in {"", "none", "default"}:
        return "auto"
    return value if value in CATEGORY_LOCK_CHOICES else "auto"


def _locked_garment_category(category_lock: str, cloth_mask: np.ndarray | None = None) -> str:
    """Map UI lock → internal pipeline category (top/pants/dress/skirt/accessory)."""
    lock = _normalize_category_lock(category_lock)
    if lock == "auto":
        return detect_garment_category(cloth_mask) if cloth_mask is not None else "top"
    if lock in {"top", "tshirt", "shirt", "hoodie", "jacket", "outer", "generic"}:
        return "top"
    if lock in {"pants", "jeans", "shorts"}:
        return "pants"
    if lock == "dress":
        return "dress"
    if lock == "skirt":
        return "skirt"
    if lock in {"belt", "bag", "scarf", "hat", "sunglasses", "shoes", "boots"}:
        return "accessory"
    return "top"


def _locked_accessory_subtype(category_lock: str) -> str:
    """UI lock → accessory subtype string (or "" when not an accessory)."""
    lock = _normalize_category_lock(category_lock)
    if lock in {"shoes", "boots", "hat", "sunglasses", "belt", "bag", "scarf"}:
        return lock
    if lock in {"glasses"}:
        return "sunglasses"
    return ""


def _locked_top_subtype(category_lock: str) -> str:
    """UI lock → top subtype (hoodie/tshirt/shirt/jacket/outer) or "" for plain top."""
    lock = _normalize_category_lock(category_lock)
    if lock in {"hoodie", "tshirt", "shirt", "jacket", "outer"}:
        return lock
    return ""


def _cloud_type_from_category_lock(category_lock: str, fallback_category: str | None = None) -> str:
    """Cloud VTON only knows upper/lower/overall."""
    lock = _normalize_category_lock(category_lock)
    if lock == "auto":
        if fallback_category == "dress":
            return "overall"
        if fallback_category in {"pants", "skirt"}:
            return "lower"
        return "upper"
    if lock == "dress":
        return "overall"
    if lock in {"pants", "jeans", "shorts", "skirt", "shoes", "boots"}:
        return "lower"
    return "upper"


def _category_prompt_from_lock(category_lock: str) -> str:
    lock = _normalize_category_lock(category_lock)
    return {
        "top": "a realistic upper-body top garment matching the reference",
        "tshirt": "a realistic t-shirt matching the reference",
        "shirt": "a realistic long-sleeve button-up shirt matching the reference, pointed collar and visible front button placket",
        "hoodie": "a realistic hoodie matching the reference",
        "jacket": "a realistic jacket matching the reference",
        "outer": "realistic outerwear matching the reference",
        "pants": "realistic lower-body pants matching the reference",
        "jeans": "realistic jeans matching the reference",
        "shorts": "realistic shorts matching the reference",
        "dress": "a realistic full-body dress matching the garment reference",
        "skirt": "a realistic skirt matching the reference",
        "belt": "a realistic belt accessory matching the reference",
        "bag": "a realistic bag accessory matching the reference",
        "scarf": "a realistic scarf accessory matching the reference",
        "hat": "a realistic hat accessory matching the reference",
        "sunglasses": "realistic sunglasses matching the reference",
        "shoes": "realistic shoes matching the reference",
        "boots": "realistic boots matching the reference",
    }.get(lock, "a realistic garment matching the reference")


def _save_temp_input(image: np.ndarray, prefix: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = storage.inputs_dir / f"{prefix}_{timestamp}.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def _apply_quality_preset(
    preset: str,
    fit_scale: float,
    alpha: float,
    gen_steps: int,
    gen_guidance: float,
    preserve_strength: float,
    refiner_mode: str,
) -> tuple[float, float, int, float, float, str]:
    p = (preset or "balanced").strip().lower()

    if p == "fast":
        return (
            float(np.clip(fit_scale, 0.95, 1.12)),
            float(np.clip(alpha, 0.86, 0.98)),
            int(np.clip(gen_steps, 4, 12)),
            float(np.clip(gen_guidance, 0.8, 2.0)),
            float(np.clip(preserve_strength, 0.75, 0.95)),
            "lcm",
        )
    if p == "hq":
        mode = "dpm++" if refiner_mode == "base" else refiner_mode
        return (
            float(np.clip(fit_scale, 1.0, 1.20)),
            float(np.clip(alpha, 0.55, 0.75)),
            int(np.clip(max(gen_steps, 24), 24, 28)),
            float(np.clip(gen_guidance, 4.8, 5.4)),
            float(np.clip(preserve_strength, 0.70, 0.95)),
            mode,
        )

    # balanced — relaxed clamps so category auto-presets (pants/jeans/dress)
    # can pass through. Previously alpha was forced ≥0.88 which washed out
    # preserved garment texture for jeans (preset 0.64–0.68).
    return (
        float(np.clip(fit_scale, 0.90, 1.25)),
        float(np.clip(alpha, 0.55, 1.0)),
        int(np.clip(max(gen_steps, 20), 8, 28)),
        float(np.clip(gen_guidance, 2.0, 5.5)),
        float(np.clip(preserve_strength, 0.30, 0.95)),
        "lcm" if refiner_mode == "base" else refiner_mode,
    )


# ═══════════════════════════════════════════════════════════════════
#  Phase A: Cloud post-processing (identity + color preservation)
# ═══════════════════════════════════════════════════════════════════

def _postprocess_cloud_result(
    cloud_rgb: np.ndarray,
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Post-process cloud VTON result for identity and color consistency.

    1. Resize to match person dimensions
    2. Blackout artifact detection
    3. Face/hair identity preservation (SegFormer parsing)
    4. Color consistency on garment region
    """
    info = []
    h, w = person_rgb.shape[:2]

    # Resize cloud output to match person
    if cloud_rgb.shape[:2] != (h, w):
        cloud_rgb = cv2.resize(cloud_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    # Blackout check
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    if _is_blackout_artifact(cloud_rgb, person_rgb, dummy_mask):
        raise RuntimeError("Cloud output has blackout artifact")

    # Face/hair identity preservation
    person_parsing = parse_human(person_rgb)
    if person_parsing:
        face_hair_mask = np.zeros((h, w), dtype=np.uint8)
        for key in ("face", "hair", "hat", "sunglasses", "left_arm", "right_arm"):
            if key in person_parsing:
                face_hair_mask = cv2.bitwise_or(face_hair_mask, person_parsing[key])

        if face_hair_mask.sum() > 255 * 100:
            # Feathered composite: person's face/hair onto cloud result
            face_hair_mask = cv2.GaussianBlur(face_hair_mask, (15, 15), 0)
            alpha = (face_hair_mask.astype(np.float32) / 255.0)[..., None]
            cloud_rgb = _safe_uint8(
                cloud_rgb.astype(np.float32) * (1.0 - alpha)
                + person_rgb.astype(np.float32) * alpha
            )
            info.append("FacePreserve")

    # Color consistency on garment region
    cloud_parsing = parse_human(cloud_rgb)
    if cloud_parsing:
        garment_mask = get_clothing_mask(cloud_parsing)
        if garment_mask is not None and garment_mask.sum() > 255 * 100:
            cloud_rgb = _apply_color_consistency(
                generated_rgb=cloud_rgb,
                reference_rgb=cloth_rgb,
                garment_mask=garment_mask,
                strength=0.3,
            )
            info.append("ColorMatch")

    return cloud_rgb, info


# ═══════════════════════════════════════════════════════════════════
#  Phase B: CPU Geometric Pipeline (fallback)
# ═══════════════════════════════════════════════════════════════════

def _safe_uint8(arr: np.ndarray) -> np.ndarray:
    """Clamp NaN/Inf and convert to uint8 safely — prevents 'invalid value in cast' warnings."""
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def _fit_like(img: np.ndarray, ref: np.ndarray, is_mask: bool = False) -> np.ndarray:
    """Resize img to match ref's spatial dimensions if they differ."""
    if img.shape[:2] != ref.shape[:2]:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=interp)
    return img


def _sanitize_rgb_output(img: np.ndarray) -> np.ndarray:
    """Convert any diffusion output (float 0-1, float 0-255, uint8) to clean uint8."""
    img = np.nan_to_num(img.astype(np.float64), nan=0.0, posinf=255.0, neginf=0.0)
    if img.dtype != np.uint8:
        if float(np.nanmax(img)) <= 1.5:
            img = img * 255.0
        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return img


def _enforce_garment_identity(
    diffusion_rgb: np.ndarray,
    cpu_rgb: np.ndarray,
    garment_mask: np.ndarray,
    blend_ratio: float = 0.80,
) -> np.ndarray:
    """Blend CPU garment back into diffusion output to recover from darkening."""
    mask_f = (garment_mask > 127).astype(np.float32)[..., None] * blend_ratio
    result = (
        diffusion_rgb.astype(np.float32) * (1.0 - mask_f)
        + cpu_rgb.astype(np.float32) * mask_f
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def _suppress_dress_edge_halo(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> np.ndarray:
    """Remove bright/grey fringe just outside the dress footprint.

    The dress edge should end at the garment mask.  Any low-saturation or
    overly-bright residue in the outside ring is usually CPU fill/alpha bleed,
    not valid garment texture.
    """
    output_rgb = _fit_like(output_rgb, person_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, person_rgb, is_mask=True)

    mask = (garment_mask > 20).astype(np.uint8) * 255
    if int(mask.sum()) < 255 * 500:
        return output_rgb

    outer = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    inner = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    ring = cv2.subtract(outer, inner)
    if int(ring.sum()) < 255 * 50:
        return output_rgb

    out_lab = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    person_lab = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_l = out_lab[:, :, 0]
    person_l = person_lab[:, :, 0]

    out_f = output_rgb.astype(np.float32)
    person_f = person_rgb.astype(np.float32)
    chroma = out_f.max(axis=2) - out_f.min(axis=2)
    color_delta = np.mean(np.abs(out_f - person_f), axis=2)

    ring_bool = ring > 0
    halo = ring_bool & (
        (out_l > person_l + 10.0) |
        ((chroma < 34.0) & (color_delta > 14.0))
    )
    if int(halo.sum()) < 30:
        return output_rgb

    halo_u8 = halo.astype(np.uint8) * 255
    halo_u8 = cv2.dilate(halo_u8, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(halo_u8.astype(np.float32) / 255.0, (9, 9), 2.5)
    alpha = np.clip(alpha * (ring > 0).astype(np.float32), 0.0, 1.0)[..., None]

    result = (
        output_rgb.astype(np.float32) * (1.0 - alpha)
        + person_rgb.astype(np.float32) * alpha
    )
    return _safe_uint8(result)


def _remove_old_collar_bleed(
    output_rgb: np.ndarray,
    source_rgb: np.ndarray,
    garment_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Replace red old-shirt collar pixels that diffusion can preserve."""
    source_rgb = _fit_like(source_rgb, output_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, output_rgb, is_mask=True)
    mask = garment_mask > 20
    ys, xs = np.where(mask)
    if len(xs) < 500:
        return output_rgb, False

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1)
    band = np.zeros(mask.shape, dtype=bool)
    band[y1:min(mask.shape[0], y1 + int(h * 0.24)), max(0, x1 - 8):min(mask.shape[1], x2 + 9)] = True
    band &= cv2.dilate((mask.astype(np.uint8) * 255), np.ones((13, 13), np.uint8), iterations=1) > 0

    out_i = output_rgb.astype(np.int16)
    src_i = source_rgb.astype(np.int16)
    out_red = (
        (out_i[:, :, 0] > out_i[:, :, 1] + 28)
        & (out_i[:, :, 0] > out_i[:, :, 2] + 24)
        & (out_i[:, :, 0] > 88)
    )
    src_red = (
        (src_i[:, :, 0] > src_i[:, :, 1] + 22)
        & (src_i[:, :, 0] > src_i[:, :, 2] + 18)
        & (src_i[:, :, 0] > 82)
    )
    bleed = band & out_red & ~src_red
    if int(bleed.sum()) < 12:
        return output_rgb, False

    bleed_u8 = cv2.dilate(bleed.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(bleed_u8.astype(np.float32) / 255.0, (7, 7), 2.0)[..., None]
    alpha = np.clip(alpha, 0.0, 1.0)
    result = (
        output_rgb.astype(np.float32) * (1.0 - alpha)
        + source_rgb.astype(np.float32) * alpha
    )
    return _safe_uint8(result), True


def _inpaint_old_light_collar_bleed(
    output_rgb: np.ndarray,
    garment_mask: np.ndarray,
    parsing: dict | None = None,
) -> tuple[np.ndarray, bool]:
    """Inpaint bright/white old-shirt collar remnants near a dress neckline."""
    garment_mask = _fit_like(garment_mask, output_rgb, is_mask=True)
    mask = garment_mask > 20
    ys, xs = np.where(mask)
    if len(xs) < 500:
        return output_rgb, False

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1)
    band = np.zeros(mask.shape, dtype=bool)
    band[
        max(0, y1 - int(h * 0.03)):min(mask.shape[0], y1 + int(h * 0.24)),
        max(0, x1 - 14):min(mask.shape[1], x2 + 15),
    ] = True
    near = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((17, 17), np.uint8), iterations=1) > 0

    protect = np.zeros(mask.shape, dtype=np.uint8)
    if parsing:
        for key in ("face", "hair", "hat", "sunglasses"):
            region = parsing.get(key)
            if region is not None:
                protect = cv2.bitwise_or(
                    protect,
                    (_fit_like(region, output_rgb, is_mask=True) > 20).astype(np.uint8) * 255,
                )
        protect = cv2.dilate(protect, np.ones((5, 5), np.uint8), iterations=1)

    rgb_i = output_rgb.astype(np.int16)
    chroma = rgb_i.max(axis=2) - rgb_i.min(axis=2)
    mean_rgb = rgb_i.mean(axis=2)
    bright_collar = (
        (mean_rgb > 168)
        & (rgb_i.min(axis=2) > 138)
        & (chroma < 58)
        & band
        & near
        & (protect == 0)
    )
    if int(bright_collar.sum()) < 12:
        return output_rgb, False

    collar_u8 = cv2.dilate(bright_collar.astype(np.uint8) * 255, np.ones((5, 5), np.uint8), iterations=1)
    collar_u8 = cv2.bitwise_and(collar_u8, cv2.bitwise_not(protect))
    bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, collar_u8, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    fixed_rgb = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
    alpha = cv2.GaussianBlur((collar_u8 > 0).astype(np.float32), (7, 7), 2.0)[..., None]
    alpha = np.clip(alpha, 0.0, 1.0)
    return _safe_uint8(output_rgb.astype(np.float32) * (1.0 - alpha) + fixed_rgb.astype(np.float32) * alpha), True


def _inpaint_old_red_bleed(
    output_rgb: np.ndarray,
    garment_mask: np.ndarray,
    parsing: dict | None = None,
    protect_hair: bool = False,
    allow_large_upper: bool = True,
) -> tuple[np.ndarray, bool]:
    """Inpaint old red shirt/sleeve bleed inside the generated dress layer."""
    garment_mask = _fit_like(garment_mask, output_rgb, is_mask=True)
    mask = garment_mask > 20
    if int(mask.sum()) < 500:
        return output_rgb, False

    protect = np.zeros(mask.shape, dtype=np.uint8)
    if parsing:
        # Hair is intentionally not protected before HairOverlay: try_on()
        # restores hair later, and protecting it lets the old red collar stay
        # connected to the hair component. For post-HairOverlay cleanup callers
        # can protect hair explicitly.
        protect_keys = ("face", "hat", "sunglasses", "hair") if protect_hair else ("face", "hat", "sunglasses")
        for key in protect_keys:
            region = parsing.get(key)
            if region is not None:
                protect = cv2.bitwise_or(
                    protect,
                    (_fit_like(region, output_rgb, is_mask=True) > 20).astype(np.uint8) * 255,
                )
        protect_k = 17 if protect_hair else 7
        protect = cv2.dilate(protect, np.ones((protect_k, protect_k), np.uint8), iterations=1)

    rgb_i = output_rgb.astype(np.int16)
    red_bleed = (
        (rgb_i[:, :, 0] > rgb_i[:, :, 1] + 30)
        & (rgb_i[:, :, 0] > rgb_i[:, :, 2] + 24)
        & (rgb_i[:, :, 0] > 88)
        & (rgb_i[:, :, 1] < 150)
        & (rgb_i[:, :, 2] < 145)
        & mask
        & (protect == 0)
    )
    if int(red_bleed.sum()) < 20:
        return output_rgb, False

    red_u8 = red_bleed.astype(np.uint8) * 255
    ys, _xs = np.where(mask)
    top_limit = int(ys.min() + max(1, ys.max() - ys.min()) * 0.28) if len(ys) else 0
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(red_u8, 8)
    filtered = np.zeros_like(red_u8)
    max_component = max(120, int(mask.sum() * 0.025))
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        centroid_y = float(centroids[idx][1])
        is_upper_collar_bleed = allow_large_upper and parsing is not None and centroid_y <= top_limit
        if 12 <= area and (area <= max_component or is_upper_collar_bleed):
            filtered[labels == idx] = 255

    if int((filtered > 0).sum()) < 20:
        return output_rgb, False

    red_u8 = cv2.dilate(filtered, np.ones((5, 5), np.uint8), iterations=1)
    red_u8 = cv2.bitwise_and(red_u8, (mask.astype(np.uint8) * 255))
    red_u8 = cv2.bitwise_and(red_u8, cv2.bitwise_not(protect))
    bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    fixed = cv2.inpaint(bgr, red_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    fixed_rgb = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
    alpha = cv2.GaussianBlur((red_u8 > 0).astype(np.float32), (7, 7), 2.0)[..., None]
    alpha = np.clip(alpha, 0.0, 1.0)
    result = output_rgb.astype(np.float32) * (1.0 - alpha) + fixed_rgb.astype(np.float32) * alpha
    return _safe_uint8(result), True


def _paste_original_hair_layer(
    base_rgb: np.ndarray,
    person_rgb: np.ndarray,
    parsing: dict | None,
) -> tuple[np.ndarray, bool]:
    """Paste the original hair back after garment cleanup/refinement."""
    if not parsing or "hair" not in parsing:
        return base_rgb, False

    hair_raw = (_fit_like(parsing["hair"], base_rgb, is_mask=True) > 20).astype(np.uint8) * 255
    if int(hair_raw.sum()) < 255 * 40:
        return base_rgb, False

    hair_mask = cv2.morphologyEx(hair_raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    old_upper = get_clothing_mask(parsing)
    if old_upper is not None:
        old_upper = (_fit_like(old_upper, base_rgb, is_mask=True) > 20).astype(np.uint8) * 255
        old_upper = cv2.dilate(old_upper, np.ones((5, 5), np.uint8), iterations=1)
        person_i = person_rgb.astype(np.int16)
        shirt_red = (
            (person_i[:, :, 0] > person_i[:, :, 1] + 36)
            & (person_i[:, :, 0] > person_i[:, :, 2] + 30)
            & (person_i[:, :, 0] > 105)
            & (person_i[:, :, 1] < 130)
            & (person_i[:, :, 2] < 125)
        ).astype(np.uint8) * 255
        hair_mask = cv2.subtract(hair_mask, cv2.bitwise_and(shirt_red, old_upper))

    if int(hair_mask.sum()) < 255 * 40:
        return base_rgb, False

    hair_alpha = cv2.GaussianBlur(
        hair_mask.astype(np.float32) / 255.0,
        (7, 7),
        1.8,
    )[..., None]
    hair_alpha = np.clip(hair_alpha, 0.0, 1.0)
    output = _safe_uint8(
        base_rgb.astype(np.float32) * (1.0 - hair_alpha)
        + person_rgb.astype(np.float32) * hair_alpha
    )
    return output, True


def _complete_dress_body_mask(
    body_mask: np.ndarray,
    full_pose: dict | None,
) -> np.ndarray:
    """Make the lower dress footprint continuous before diffusion.

    Affine bodycon masks often taper too aggressively at the hem, leaving shorts
    visible and giving diffusion an incomplete skirt seed.  This preserves the
    existing upper body but keeps hip-to-hem rows at a reasonable dress width.
    """
    mask = (body_mask > 20).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return mask

    y1, y2 = int(ys.min()), int(ys.max())
    dress_h = max(1, y2 - y1)
    row_widths = []
    for frac in (0.35, 0.42, 0.50, 0.58):
        row = max(0, min(mask.shape[0] - 1, y1 + int(dress_h * frac)))
        nz = np.where(mask[row] > 0)[0]
        if len(nz) > 4:
            row_widths.append(int(nz.max() - nz.min()))
    if not row_widths:
        return mask

    ref_width = int(np.median(row_widths))
    # v18.14: cho phép hem nở rộng (A-line / fit-and-flare). Lower-half
    # progression: waist (ref) → mid-skirt (ref*1.10) → hem (ref*1.30).
    # Sheath/column dresses không bị ảnh hưởng vì TPS-mask gốc đã giới hạn
    # bằng bitwise_and với env trong _fit_dress_body_mask_to_pose downstream;
    # nhưng nếu TPS sụp về cột, hàm này sẽ thêm flare trở lại.
    min_waist_width = max(54, int(ref_width * 1.00))
    min_mid_skirt = int(ref_width * 1.10)
    min_hem_width = int(ref_width * 1.30)

    body_center_x = float(np.median(xs))
    if full_pose is not None and "left_hip" in full_pose and "right_hip" in full_pose:
        body_center_x = float((full_pose["left_hip"][0] + full_pose["right_hip"][0]) * 0.5)

    result = mask.copy()
    start_y = y1 + int(dress_h * 0.35)  # bắt đầu từ waist (~35%) để bao gồm chiết eo + flare
    mid_y = y1 + int(dress_h * 0.55)    # mid-skirt
    end_y = y2
    for row in range(start_y, end_y + 1):
        if row <= mid_y:
            progress = (row - start_y) / max(1, mid_y - start_y)
            target_w = int((1.0 - progress) * min_waist_width + progress * min_mid_skirt)
        else:
            progress = (row - mid_y) / max(1, end_y - mid_y)
            target_w = int((1.0 - progress) * min_mid_skirt + progress * min_hem_width)
        nz = np.where(result[row] > 0)[0]
        if len(nz) > 4:
            cx = float((nz.min() + nz.max()) * 0.5)
            cur_w = int(nz.max() - nz.min())
        else:
            cx = body_center_x
            cur_w = 0
        if cur_w >= target_w:
            continue
        x1 = max(0, int(round(cx - target_w * 0.5)))
        x2 = min(result.shape[1] - 1, int(round(cx + target_w * 0.5)))
        result[row, x1:x2 + 1] = 255

    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    result = cv2.GaussianBlur(result, (3, 3), 0.6)
    return np.clip(result, 0, 255).astype(np.uint8)


def _fit_dress_body_mask_to_pose(
    body_mask: np.ndarray,
    full_pose: dict | None,
    silhouette: str | None = None,
    source_bust_half: float | None = None,
    fit_scale: float = 1.0,
) -> tuple[np.ndarray, bool]:
    """Keep the dress footprint close to the model shoulder/waist/hip size.

    v18.19: when ``silhouette`` is supplied, replace the hardcoded width
    heuristic with a template from ``garment_silhouettes`` anchored to
    the source cloth bust half-width.  Falls back to the v18.16 pose
    curve otherwise.
    """
    if full_pose is None:
        return body_mask, False
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(k not in full_pose for k in required):
        return body_mask, False

    mask = (body_mask > 20).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return body_mask, False

    h, w = mask.shape[:2]
    ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
    rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
    lh = np.array(full_pose["left_hip"], dtype=np.float32)
    rh = np.array(full_pose["right_hip"], dtype=np.float32)
    sw = max(24.0, float(np.linalg.norm(ls - rs)))
    hip_w = max(18.0, float(abs(lh[0] - rh[0])))
    y1, y2 = int(ys.min()), int(ys.max())
    shoulder_y = float(min(ls[1], rs[1]))
    hip_y = float(max(lh[1], rh[1]))
    top_y = max(y1, int(shoulder_y - sw * 0.22))
    bot_y = y2
    if bot_y <= top_y + 20:
        return body_mask, False

    shoulder_cx = float((ls[0] + rs[0]) * 0.5)
    hip_cx = float((lh[0] + rh[0]) * 0.5)
    env = np.zeros_like(mask)

    if silhouette and source_bust_half and source_bust_half > 6.0:
        # Template-driven curve (silhouette-aware, ASTM-anchored).
        width_curve = build_dress_width_curve(silhouette, source_bust_half, sw, hip_w)
    else:
        # Fallback: v18.16 hardcoded pose curve.
        width_curve = np.array([
            [0.00, sw * 0.62],
            [0.12, sw * 0.60],
            [0.30, sw * 0.50],
            [0.50, max(sw * 0.56, hip_w * 0.78)],
            [0.75, max(sw * 0.66, hip_w * 0.95)],
            [1.00, max(sw * 0.82, hip_w * 1.22)],
        ], dtype=np.float32)

    # UI dress preset historically sends ~1.16 as the neutral value. Normalize
    # around that so an auto-detected A-line does not inflate the whole mask.
    fit_factor = float(np.clip(float(fit_scale) / 1.16, 0.78, 1.08))
    width_curve = width_curve.copy()
    width_curve[:, 1] *= fit_factor

    # Sample actual TPS-warped silhouette half-widths at the same fractions so a
    # flared/A-line dress that TPS preserved is kept; pose curve only fills in
    # where TPS collapsed too narrow.
    sampled_half = {}
    for frac in (0.00, 0.12, 0.30, 0.50, 0.75, 1.00):
        row = int(round(top_y + frac * (bot_y - top_y)))
        row = max(0, min(h - 1, row))
        nz = np.where(mask[row] > 0)[0]
        if len(nz) >= 4:
            sampled_half[frac] = float(nz.max() - nz.min()) * 0.5
        else:
            sampled_half[frac] = 0.0

    for y in range(top_y, min(h, bot_y + 1)):
        f = (y - top_y) / max(1.0, float(bot_y - top_y))
        cx_f = min(1.0, max(0.0, (y - shoulder_y) / max(1.0, hip_y - shoulder_y)))
        cx = shoulder_cx * (1.0 - cx_f) + hip_cx * cx_f
        half_w = float(np.interp(f, width_curve[:, 0], width_curve[:, 1]))
        # Use the pose/silhouette curve as an upper bound. The old max()
        # preserved oversized affine masks, so the final dress ignored the
        # model's shoulder/waist/hip size and looked like a flat wide panel.
        frac_keys = np.array(list(sampled_half.keys()), dtype=np.float32)
        frac_vals = np.array([sampled_half[k] for k in sampled_half], dtype=np.float32)
        sampled_w = float(np.interp(f, frac_keys, frac_vals))
        if sampled_w > 2.0:
            half_w = min(half_w, sampled_w * 1.02)
        x_l = max(0, int(round(cx - half_w)))
        x_r = min(w - 1, int(round(cx + half_w)))
        env[y, x_l:x_r + 1] = 255

    env = cv2.GaussianBlur(env, (9, 9), 2.0)
    env = (env > 24).astype(np.uint8) * 255
    clipped = cv2.bitwise_and(mask, env)

    old_area = int(cv2.countNonZero(mask))
    new_area = int(cv2.countNonZero(clipped))
    if new_area < max(500, int(old_area * 0.42)):
        return body_mask, False

    clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return clipped, True


def _propagate_texture_into_mask(
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
    max_iter: int = 80,
) -> np.ndarray:
    """Fill target pixels from nearest valid garment texture, avoiding grey BG."""
    target = target_mask > 20
    valid = (valid_mask > 0) & target
    if int(target.sum()) < 50 or int(valid.sum()) < 30:
        return image_rgb

    result = image_rgb.copy()
    current = valid.copy()
    missing = target & ~current
    h, w = target.shape[:2]
    shifts = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )

    for _ in range(max_iter):
        if not missing.any():
            break
        filled_any = False
        filled = np.zeros_like(current, dtype=bool)

        for dy, dx in shifts:
            src_y1 = max(0, -dy)
            src_y2 = min(h, h - dy)
            src_x1 = max(0, -dx)
            src_x2 = min(w, w - dx)
            dst_y1 = max(0, dy)
            dst_y2 = min(h, h + dy)
            dst_x1 = max(0, dx)
            dst_x2 = min(w, w + dx)

            neighbor = np.zeros_like(current, dtype=bool)
            neighbor[dst_y1:dst_y2, dst_x1:dst_x2] = current[src_y1:src_y2, src_x1:src_x2]
            fill = missing & neighbor & ~filled
            if not fill.any():
                continue

            shifted_rgb = np.zeros_like(result)
            shifted_rgb[dst_y1:dst_y2, dst_x1:dst_x2] = result[src_y1:src_y2, src_x1:src_x2]
            result[fill] = shifted_rgb[fill]
            filled[fill] = True
            filled_any = True

        if not filled_any:
            break
        current[filled] = True
        missing = target & ~current

    if missing.any():
        median_rgb = np.median(result[current], axis=0).astype(np.uint8)
        result[missing] = median_rgb

    return result


def _is_plain_dress_texture(image_rgb: np.ndarray, base_mask: np.ndarray) -> bool:
    """Detect solid/near-solid dress fabric before pattern propagation."""
    base = base_mask > 20
    if int(base.sum()) < 300:
        return False

    lab = cv2.cvtColor(_safe_uint8(image_rgb), cv2.COLOR_RGB2LAB).astype(np.float32)
    ab_std = float(lab[:, :, 1:3][base].std())
    l_std = float(lab[:, :, 0][base].std())
    return ab_std < 8.0 and l_std < 32.0


def _extend_plain_dress_texture_to_mask(
    image_rgb: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Extend a plain dress with smooth row colours, not repeated texture."""
    target = target_mask > 20
    source = (source_mask > 20) & target
    if int(target.sum()) < 50 or int(source.sum()) < 30:
        return image_rgb

    result = image_rgb.copy()
    missing = target & ~source
    if not bool(missing.any()):
        result[~target] = 0
        return _safe_uint8(result)

    h, _w = target.shape[:2]
    row_color = np.zeros((h, 3), dtype=np.float32)
    row_valid = np.zeros(h, dtype=bool)
    for y in range(h):
        xs = np.where(source[y])[0]
        if len(xs) >= 3:
            row_color[y] = np.median(image_rgb[y, xs], axis=0).astype(np.float32)
            row_valid[y] = True

    valid_rows = np.where(row_valid)[0]
    if len(valid_rows) == 0:
        row_color[:] = np.median(image_rgb[source], axis=0).astype(np.float32)
    else:
        all_rows = np.arange(h)
        for c in range(3):
            row_color[:, c] = np.interp(all_rows, valid_rows, row_color[valid_rows, c])
        row_color = cv2.GaussianBlur(row_color.reshape(h, 1, 3), (1, 31), 0).reshape(h, 3)

    rows = np.where(target.any(axis=1))[0]
    for y in rows:
        missing_x = np.where(missing[y])[0]
        if len(missing_x) == 0:
            continue

        source_x = np.where(source[y])[0]
        base = row_color[y]
        if len(source_x) >= 3:
            first = int(source_x[0])
            last = int(source_x[-1])
            left = image_rgb[y, first].astype(np.float32) * 0.70 + base * 0.30
            right = image_rgb[y, last].astype(np.float32) * 0.70 + base * 0.30
            for x in missing_x:
                if int(x) < first:
                    result[y, int(x)] = _safe_uint8(left)
                elif int(x) > last:
                    result[y, int(x)] = _safe_uint8(right)
                else:
                    result[y, int(x)] = _safe_uint8(base)
        else:
            result[y, missing_x] = _safe_uint8(base)

    smooth = cv2.GaussianBlur(result, (7, 7), 1.6)
    result[missing] = smooth[missing]
    result[~target] = 0
    return _safe_uint8(result)


def _garment_texture_valid_mask(image_rgb: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """Select real garment texture pixels, excluding flat neutral warp fill."""
    base = base_mask > 20
    if int(base.sum()) < 50:
        return base

    # Compute local texture only from the garment support.  Outside-support
    # black pixels otherwise inflate local_std at edges and make neutral fill
    # look like valid patterned fabric.
    base_pixels = image_rgb[base]
    median_rgb = np.median(base_pixels, axis=0).astype(np.float32) if len(base_pixels) else np.array([128, 128, 128], dtype=np.float32)
    texture_rgb = image_rgb.copy()
    texture_rgb[~base] = np.clip(median_rgb, 0, 255).astype(np.uint8)

    img_i = texture_rgb.astype(np.int16)
    rgb_sum = img_i.sum(axis=2)
    chroma = img_i.max(axis=2) - img_i.min(axis=2)
    min_rgb = img_i.min(axis=2)
    gray = cv2.cvtColor(texture_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    local_sq = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    color_dist = np.mean(np.abs(texture_rgb.astype(np.float32) - median_rgb[None, None, :]), axis=2)

    neutral_fill = (rgb_sum > 330) & (rgb_sum < 505) & (chroma < 28) & (local_std < 9.0)
    flat_fill = (rgb_sum > 250) & (chroma < 34) & (local_std < 4.5)
    bright_cpu_fill = (min_rgb > 218) & (chroma < 42) & (local_std < 18.0)
    smooth_neutral_fill = (rgb_sum > 210) & (rgb_sum < 620) & (chroma < 24) & (local_std < 12.0)
    too_dark = rgb_sum < 45
    pattern_detail = (color_dist > 26.0) | (chroma > 32) | ((rgb_sum < 330) & (local_std > 8.0))
    valid = base & pattern_detail & ~neutral_fill & ~flat_fill & ~bright_cpu_fill & ~smooth_neutral_fill & ~too_dark

    if int(valid.sum()) < max(30, int(base.sum() * 0.08)):
        valid = base & ~neutral_fill & ~flat_fill & ~bright_cpu_fill & ~smooth_neutral_fill & ~too_dark
    if int(valid.sum()) < max(30, int(base.sum() * 0.08)):
        valid = base & ~too_dark & (chroma > 8)
    return valid


def _repeat_row_texture_into_mask(
    image_rgb: np.ndarray,
    target_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend patterned garment rows sideways instead of flat edge colors."""
    target = target_mask > 20
    valid = (valid_mask > 0) & target
    result = image_rgb.copy()
    filled = np.zeros(target.shape[:2], dtype=bool)
    if int(target.sum()) < 50 or int(valid.sum()) < 30:
        return result, filled

    rows = np.where(target.any(axis=1))[0]
    for y in rows:
        source_x = np.where(valid[y])[0]
        missing_x = np.where(target[y] & ~valid[y])[0]
        if len(source_x) < 4 or len(missing_x) == 0:
            continue

        first = int(source_x[0])
        last = int(source_x[-1])
        n_src = len(source_x)
        for x in missing_x:
            x_i = int(x)
            if x_i < first:
                src_idx = (first - x_i) % n_src
            elif x_i > last:
                src_idx = n_src - 1 - ((x_i - last) % n_src)
            else:
                insert_at = int(np.searchsorted(source_x, x_i))
                if insert_at <= 0:
                    src_idx = 0
                elif insert_at >= n_src:
                    src_idx = n_src - 1
                else:
                    left_i = insert_at - 1
                    right_i = insert_at
                    src_idx = left_i if x_i - source_x[left_i] <= source_x[right_i] - x_i else right_i
            result[y, x_i] = image_rgb[y, source_x[src_idx]]
            filled[y, x_i] = True

    return result, filled


def _extend_dress_texture_to_mask(
    warped_cloth: np.ndarray,
    warped_mask: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Inpaint source dress texture into support pixels added for skirt coverage."""
    support = (support_mask > 20).astype(np.uint8) * 255
    existing = warped_mask > 20
    if int((support > 0).sum()) < 50 or int(existing.sum()) < 300:
        return warped_cloth

    if _is_plain_dress_texture(warped_cloth, warped_mask):
        return _extend_plain_dress_texture_to_mask(warped_cloth, warped_mask, support)

    valid = _garment_texture_valid_mask(warped_cloth, existing.astype(np.uint8) * 255)
    row_filled, row_fill_mask = _repeat_row_texture_into_mask(warped_cloth, support, valid)

    return _propagate_texture_into_mask(
        row_filled,
        support,
        valid | row_fill_mask,
        max_iter=120,
    )


def _build_clean_dress_pattern_reference(
    warped_cloth: np.ndarray,
    support_mask: np.ndarray,
    source_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build garment-only pattern reference without CPU composite artifacts."""
    support = (support_mask > 20).astype(np.uint8) * 255
    if int(cv2.countNonZero(support)) < 300:
        return warped_cloth

    source = support
    if source_mask is not None:
        source = cv2.bitwise_and(_fit_like(source_mask, support, is_mask=True), support)
        if int(cv2.countNonZero(source)) < 300:
            source = support

    if _is_plain_dress_texture(warped_cloth, source):
        reference = _extend_plain_dress_texture_to_mask(warped_cloth, source, support)
        reference[support <= 20] = 0
        return _safe_uint8(reference)

    valid = _garment_texture_valid_mask(warped_cloth, source)
    reference = np.zeros_like(warped_cloth)
    reference[valid] = warped_cloth[valid]
    reference, row_filled = _repeat_row_texture_into_mask(reference, support, valid)
    reference = _propagate_texture_into_mask(
        reference,
        support,
        valid | row_filled,
        max_iter=160,
    )

    # Keep reference garment-only. Outside-mask pixels must never become a
    # second CPU-shaped garment source during final pattern locking.
    outside = support <= 20
    reference = reference.copy()
    reference[outside] = 0
    return _safe_uint8(reference)


def _merge_clean_dress_pattern_reference(
    primary_rgb: np.ndarray | None,
    fallback_rgb: np.ndarray,
    support_mask: np.ndarray,
) -> np.ndarray:
    """Merge garment reference texture with filtered CPU fallback coverage."""
    support = (support_mask > 20).astype(np.uint8) * 255
    fallback_rgb = _fit_like(fallback_rgb, support, is_mask=False)
    if primary_rgb is None:
        return _build_clean_dress_pattern_reference(fallback_rgb, support)

    primary_rgb = _fit_like(primary_rgb, fallback_rgb, is_mask=False)
    if _is_plain_dress_texture(primary_rgb, support):
        merged = _extend_plain_dress_texture_to_mask(primary_rgb, support, support)
        merged[support <= 20] = 0
        return _safe_uint8(merged)

    primary_valid = _garment_texture_valid_mask(primary_rgb, support)
    fallback_valid = _garment_texture_valid_mask(fallback_rgb, support)

    merged = np.zeros_like(fallback_rgb)
    merged[fallback_valid] = fallback_rgb[fallback_valid]
    merged[primary_valid] = primary_rgb[primary_valid]
    merged_valid = primary_valid | fallback_valid
    if int(merged_valid.sum()) < max(30, int((support > 20).sum() * 0.04)):
        merged = primary_rgb.copy()
        merged_valid = primary_valid
    merged = _propagate_texture_into_mask(
        merged,
        support,
        merged_valid,
        max_iter=160,
    )
    merged[support <= 20] = 0
    return _safe_uint8(merged)


def _match_cloth_brightness(
    warped_cloth: np.ndarray,
    person_rgb: np.ndarray,
    warped_mask: np.ndarray,
) -> np.ndarray:
    """Match warped cloth lighting to person's skin/background.

    Simple mean-brightness scaling in the garment region so the cloth
    doesn't look "pasted on" with different exposure/lighting.
    Only adjusts the L channel (lightness) to preserve cloth color.
    """
    mask_bool = warped_mask > 25
    if mask_bool.sum() < 500:
        return warped_cloth

    cloth_lab = cv2.cvtColor(warped_cloth, cv2.COLOR_RGB2LAB).astype(np.float32)
    person_lab = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Compare lightness of cloth vs person in the garment region
    cloth_L = cloth_lab[:, :, 0][mask_bool]
    person_L = person_lab[:, :, 0][mask_bool]

    cloth_mean = float(cloth_L.mean()) + 1e-6
    person_mean = float(person_L.mean()) + 1e-6

    # Gentle scaling (strength 0.35) — don't overdo it
    scale = person_mean / cloth_mean
    scale = 1.0 + (scale - 1.0) * 0.35  # dampen
    scale = float(np.clip(scale, 0.7, 1.4))  # safety bounds

    cloth_lab[:, :, 0][mask_bool] = np.clip(cloth_L * scale, 0, 255)

    result = cv2.cvtColor(_safe_uint8(cloth_lab), cv2.COLOR_LAB2RGB)
    # Only apply in mask region, keep rest as-is
    out = warped_cloth.copy()
    out[mask_bool] = result[mask_bool]
    return out


def _soft_mask(mask: np.ndarray, blur_sigma: float = 0.5, erode_px: int = 1) -> np.ndarray:
    """Convert mask to soft [0,1] float with HARD CORE + THIN SOFT EDGE.

    Erode inward to create hard core at 1.0, then thin Gaussian edge.
    Keeps transition zone VERY narrow to avoid dark border artifacts.
    v16.7e: erode_px=1, clamp<0.03, subtract 0.03 — minimal halo.
    """
    binary = (mask > 25).astype(np.uint8) * 255

    # ERODE to create the hard core
    if erode_px > 0:
        k_erode = np.ones((erode_px, erode_px), np.uint8)
        core = cv2.erode(binary, k_erode, iterations=1)
    else:
        core = binary.copy()

    # Soft edge: blur the original binary (thin transition)
    mask_f = binary.astype(np.float32) / 255.0
    ksize = int(blur_sigma * 4) | 1
    soft = cv2.GaussianBlur(mask_f, (ksize, ksize), blur_sigma)

    # Combine: CORE at 1.0, SOFT at edges only
    core_f = core.astype(np.float32) / 255.0
    result = np.maximum(core_f, soft)

    # v16.7g: Minimal clamp — kill only ghost fringe pixels at 0.02 threshold.
    # Previous 0.05 was pulling mask edge too far inside → white gap at sides.
    result[result < 0.02] = 0.0
    result = np.clip(result - 0.02, 0.0, 1.0)
    # Re-normalize so core is back to 1.0
    rmax = result.max()
    if rmax > 0.01:
        result = result / rmax

    return np.clip(result, 0.0, 1.0)


def _run_cpu_geometric_pipeline(
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    fit_scale: float,
    alpha: float,
    y_offset: float,
    category_lock: str = "auto",
) -> tuple[np.ndarray, np.ndarray, dict | None, object, dict | None, list[str], bool, str, np.ndarray | None]:
    """Simplified CPU geometric pipeline: Parse → Pose → Segment → TPS → SoftBlend.

    Key changes from previous version:
    - Removed Poisson blend (was double-blending with alpha, causing dark edges)
    - Removed SkinRestore step (was fighting with erase step)
    - Single soft-mask blend instead of TightFeather+Erode+Dilate chain
    - NaN/Inf clamping at every compositing boundary
    - Foreground layer (face/hair/arms) still preserved on top

    Returns: (output_rgb, warped_mask, parsing, pose_box, full_pose, pipeline_info)
    """
    pipeline_info = []
    h_out, w_out = person_rgb.shape[:2]

    # ── Step 1: Human Parsing (SegFormer) ──
    parsing = parse_human(person_rgb)
    if parsing:
        pipeline_info.append("Parsing")

    # ── Step 2: Pose Estimation ──
    full_pose = None
    try:
        full_pose = detect_full_pose(person_rgb)
        full_pose = smooth_pose_landmarks(full_pose, person_rgb.shape[:2])
        pose_box = full_pose_to_box(full_pose)
        pipeline_info.append("Pose")
        # Debug: draw keypoints on person image
        if _DEBUG and full_pose is not None:
            kp_img = person_rgb.copy()
            colors = {
                "nose": (255, 0, 0), "left_shoulder": (0, 255, 0), "right_shoulder": (0, 255, 0),
                "left_elbow": (0, 0, 255), "right_elbow": (0, 0, 255),
                "left_wrist": (255, 255, 0), "right_wrist": (255, 255, 0),
                "left_hip": (255, 0, 255), "right_hip": (255, 0, 255),
            }
            for name, pt in full_pose.items():
                c = colors.get(name, (200, 200, 200))
                cv2.circle(kp_img, (int(pt[0]), int(pt[1])), 6, c, -1)
                cv2.putText(kp_img, name.split("_")[-1], (int(pt[0])+8, int(pt[1])-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
            # Draw arm lines
            for sh, el, wr in [("left_shoulder", "left_elbow", "left_wrist"),
                                ("right_shoulder", "right_elbow", "right_wrist")]:
                if sh in full_pose and el in full_pose:
                    cv2.line(kp_img, tuple(map(int, full_pose[sh])), tuple(map(int, full_pose[el])), (0, 255, 255), 2)
                if el in full_pose and wr in full_pose:
                    cv2.line(kp_img, tuple(map(int, full_pose[el])), tuple(map(int, full_pose[wr])), (0, 255, 255), 2)
            _debug_save("02_pose_keypoints", kp_img)
    except Exception:
        pose_box = detect_upper_body_box(person_rgb)
        pipeline_info.append("Pose(basic)")

    # ── Step 3: Cloth Segmentation ──
    # The product mask is only for extracting/warping the reference garment.
    # Diffusion uses a human parsing + pose mask later, so RMBG-2.0 garment
    # ensemble is opt-in instead of the default control mask.
    mask_source = "fallback"
    use_garment_mask_ensemble = os.getenv("VTON_USE_GARMENT_MASK_ENSEMBLE", "0").strip() == "1"
    if use_garment_mask_ensemble:
        try:
            cloth_mask = segment_cloth_ensemble(cloth_rgb)
            mask_source = "ensemble"
            pipeline_info.append("GarmentMaskEnsemble")
        except Exception:
            try:
                cloth_mask = segment_cloth_u2net(cloth_rgb)
                mask_source = "u2net"
                pipeline_info.append("U2Net")
            except Exception:
                cloth_mask = build_cloth_mask(cloth_rgb)
                pipeline_info.append("MaskFallback")
    else:
        try:
            cloth_mask = segment_cloth_u2net(cloth_rgb)
            mask_source = "u2net"
            pipeline_info.append("U2Net")
        except Exception:
            cloth_mask = build_cloth_mask(cloth_rgb)
            pipeline_info.append("MaskFallback")

    # SegFormer merge on the product image is opt-in.  The AI semantic mask is
    # now applied on the human/person side, not the garment product side.
    if os.getenv("VTON_GARMENT_PARSE_MERGE", "0").strip() == "1":
        try:
            cloth_parsing = parse_human(cloth_rgb)
            if cloth_parsing:
                mask_segformer = get_clothing_mask(cloth_parsing)
                if mask_segformer is not None and int(mask_segformer.sum()) > 255 * 100:
                    merged_f = 0.7 * cloth_mask.astype(np.float32) + 0.3 * mask_segformer.astype(np.float32)
                    cloth_mask = (merged_f > 127).astype(np.uint8) * 255
                    cloth_mask = cv2.morphologyEx(
                        cloth_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2,
                    )
                    pipeline_info.append("GarmentParseMerge")
        except Exception:
            pass

    # v16.7f: Erode cloth mask by 3px BEFORE any further processing.
    # U2Net mask edges contain semi-transparent pixels that blend garment with
    # white background. Stronger erode removes this fringe at source.
    if mask_source == "ensemble":
        cloth_mask = cv2.erode(cloth_mask, np.ones((3, 3), np.uint8), iterations=1)
        pipeline_info.append("MaskEdgeTrim:RMBG2")
    else:
        cloth_mask = cv2.erode(cloth_mask, np.ones((5, 5), np.uint8), iterations=2)
    # Re-threshold to binary after erode (remove any partial values)
    cloth_mask = ((cloth_mask > 127).astype(np.uint8)) * 255
    _debug_save("03a_cloth_mask", cloth_mask, is_mask=True)

    # v16.7f: Replace background with median garment color BEFORE dilation.
    # This ensures that when dilation re-expands the mask, the pixels it covers
    # are already garment-colored, not white.
    _inner_mask_bool = cloth_mask > 127
    if _inner_mask_bool.sum() > 100:
        _median_bg = np.median(cloth_rgb[_inner_mask_bool], axis=0).astype(np.uint8)
        _bg_fill = np.full_like(cloth_rgb, _median_bg)
        cloth_rgb = np.where(_inner_mask_bool[..., None], cloth_rgb, _bg_fill)

    # Dilate cloth mask for over-coverage (like IDM-VTON).
    h_cloth = cloth_mask.shape[0]
    dilate_k = max(5, int(h_cloth * 0.025))  # ~2.5% of image height
    dilate_k = dilate_k | 1  # ensure odd
    cloth_mask = cv2.dilate(cloth_mask, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)
    # Close small holes that may appear after dilation
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    # Second BG fill pass after dilation — covers any remaining white pixels
    # at dilated mask edges
    _cloth_mask_bool = cloth_mask > 127
    if _cloth_mask_bool.sum() > 100:
        _outside_garment = _cloth_mask_bool & ~_inner_mask_bool
        if _outside_garment.sum() > 10:
            cloth_rgb[_outside_garment] = _median_bg

    # Pre-fit scale
    scaled_cloth, scaled_mask = cloth_rgb, cloth_mask
    if full_pose is not None:
        measurements = compute_body_measurements(full_pose)
        scaled_cloth, scaled_mask = prefit_scale_cloth(
            cloth_rgb, cloth_mask, measurements,
        )
        pipeline_info.append("PreFit")

    # ── Step 3a: Detect garment category (top/pants/dress) ──
    category_lock = _normalize_category_lock(category_lock)
    if category_lock == "auto":
        garment_category = detect_garment_category(cloth_mask)
        pipeline_info.append(f"Category:auto:{garment_category}")
        print(f"[GARMENT] Category auto-detected: {garment_category}")
    else:
        garment_category = _locked_garment_category(category_lock, cloth_mask)
        pipeline_info.append(f"CategoryLock:{category_lock}->{garment_category}")
        print(f"[GARMENT] Category locked: {category_lock} -> {garment_category}")

    # Accessory subtype (shoes/boots/hat/sunglasses/belt/bag/scarf)
    accessory_subtype = ""
    if garment_category == "accessory":
        accessory_subtype = _locked_accessory_subtype(category_lock)
        if not accessory_subtype:
            try:
                from src.accessories import classify_accessory_subtype
                accessory_subtype = classify_accessory_subtype(cloth_mask)
            except Exception:
                accessory_subtype = ""
        pipeline_info.append(f"AccessorySubtype:{accessory_subtype or 'unknown'}")
        print(f"[GARMENT] Accessory subtype: {accessory_subtype}")
        # Phụ kiện không cần prefit theo body (prefit scale theo shoulder/torso
        # sẽ kéo ảnh giày vào vùng ngực). Khôi phục cloth gốc trước khi warp.
        scaled_cloth, scaled_mask = cloth_rgb, cloth_mask
        pipeline_info.append("AccessoryPreFitSkip")

    # Top subtype (hoodie/tshirt/jacket/outer) — drives a separate prompt
    # tail, an extended hood region in the diffusion mask, and a relaxed
    # category lock so the hood can drape over the back of the hair.
    top_subtype = ""
    if garment_category == "top":
        top_subtype = _locked_top_subtype(category_lock)
        if top_subtype:
            pipeline_info.append(f"TopSubtype:{top_subtype}")

    pants_type = "regular"
    pants_style = "regular"
    if garment_category == "pants" and full_pose is not None:
        pants_type = detect_pants_type(cloth_mask)
        pants_style = detect_pants_style(cloth_mask)
        pants_landmarks = detect_pants_landmarks(cloth_mask)
        leg_meas = compute_leg_measurements(full_pose)
        print(f"[GARMENT] Pants type: {pants_type} / style: {pants_style}")
        print(f"[GARMENT] Pants landmarks: { {k: f'({v[0]:.0f},{v[1]:.0f})' for k, v in pants_landmarks.items()} }")
        print(f"[GARMENT] Leg measurements: hip_w={leg_meas['hip_width']:.0f}px leg_len={leg_meas['leg_length']:.0f}px")
        pipeline_info.append(f"PantsType:{pants_type}")
        pipeline_info.append(f"PantsStyle:{pants_style}")
    elif garment_category == "dress":
        print(f"[GARMENT] Dress detected — full-body coverage mode")

    # ── Step 3b: Classify garment type (short/long sleeve, loose/tight) ──
    garment_info = classify_garment_type(cloth_mask)
    new_sleeve_type = garment_info["sleeve_type"]  # "short", "long", "sleeveless"
    # v18.15: classify_garment_type was tuned for TOPS. For dresses with A-line
    # / fit-and-flare skirts, the skirt flare past 50% looks like a long sleeve
    # extending down, and the bodice (wider than waist) looks like a sleeve at
    # the top → false-positive "long". Re-classify dresses using only the upper
    # bodice region (top 35%) so the skirt cannot influence sleeve detection.
    if garment_category == "dress":
        new_sleeve_type = _classify_dress_sleeve_type(cloth_mask)
        pipeline_info.append(f"DressSleeveType:{new_sleeve_type}:v18.15")
    _debug_save("03b_garment_type", cloth_rgb)  # for reference

    # ── Detect OLD garment's sleeve type from human parsing ──
    # v16.11c: Skip for pants (no sleeves)
    if garment_category != "top":
        old_sleeve_type = None  # Not applicable for pants/dress
    else:
        # v16 FIX: Previous logic compared clothing mask vs arm mask, but SegFormer
        # labels arms as "left_arm"/"right_arm" (skin) — clothing ON the arm is still
        # labeled "upper_clothes", NOT "arm". So arm_coverage was always ~0 for
        # short-sleeve shirts → detected as "sleeveless" → wrong transition.
        # NEW: Use clothing mask WIDTH relative to shoulder width at different heights.
        # If clothing extends wide beyond shoulders at upper arm level → has sleeves.
        old_sleeve_type = "short"  # safe default: assume same type, no arm processing
        if parsing and full_pose is not None:
            old_clothes_for_type = get_clothing_mask(parsing)
            if old_clothes_for_type is not None and int(old_clothes_for_type.sum()) > 255 * 200:
                # Measure clothing width at shoulder level and at elbow level
                ls_x = full_pose["left_shoulder"][0]
                rs_x = full_pose["right_shoulder"][0]
                sh_y = int((full_pose["left_shoulder"][1] + full_pose["right_shoulder"][1]) / 2)
                shoulder_w = abs(rs_x - ls_x)

                # Check clothing width at 3 vertical slices
                ys_old, xs_old = np.where(old_clothes_for_type > 0)
                if len(xs_old) > 100:
                    y_min_old, y_max_old = int(ys_old.min()), int(ys_old.max())
                    old_h = max(1, y_max_old - y_min_old)

                    # Width at shoulder level (upper 20%)
                    band_sh = (ys_old >= y_min_old) & (ys_old < y_min_old + int(old_h * 0.25))
                    w_at_shoulder = (int(xs_old[band_sh].max()) - int(xs_old[band_sh].min())) if band_sh.sum() > 20 else 0

                    # Width at mid-arm level (40%-60%)
                    band_mid = (ys_old >= y_min_old + int(old_h * 0.40)) & (ys_old < y_min_old + int(old_h * 0.60))
                    w_at_mid = (int(xs_old[band_mid].max()) - int(xs_old[band_mid].min())) if band_mid.sum() > 20 else 0

                    # Ratio: if clothing is wider than shoulder span → has sleeves
                    width_ratio_sh = w_at_shoulder / max(1, shoulder_w)
                    width_ratio_mid = w_at_mid / max(1, shoulder_w)

                    if width_ratio_mid > 1.3:
                        old_sleeve_type = "long"  # clothing extends far beyond shoulders at mid
                    elif width_ratio_sh > 1.05:
                        old_sleeve_type = "short"  # clothing extends slightly beyond shoulders
                    else:
                        old_sleeve_type = "sleeveless"  # clothing narrower than shoulders

    # v16: Simplified transition — for same-type or upgrade, NO arm processing.
    # Only DOWNGRADE (e.g., long→short) needs arm erase. Default to NO processing.
    # v16.11c: Skip for pants (old_sleeve_type is None)
    if garment_category == "top" and old_sleeve_type is not None:
        _SLEEVE_RANK = {"sleeveless": 0, "short": 1, "long": 2}
        old_rank = _SLEEVE_RANK.get(old_sleeve_type, 1)
        new_rank = _SLEEVE_RANK.get(new_sleeve_type, 1)
        needs_arm_erase = old_rank > new_rank
        transition = f"{old_sleeve_type}_to_{new_sleeve_type}"
        pipeline_info.append(f"Type:{new_sleeve_type}")
        pipeline_info.append(f"Trans:{transition}")
    else:
        needs_arm_erase = False
        transition = "N/A"
        if garment_category != "top":
            pipeline_info.append(f"Type:{garment_category}")

    # ── Step 4: TPS/Affine Warp ──
    # v16.3: PRE-ALIGNMENT — scale and translate cloth to match person's body.
    # For tops: align to shoulder width/center.
    # For pants (v16.11c): align to hip width/center instead.
    if full_pose is not None and garment_category == "pants":
        # ── PANTS: Hip-based pre-alignment ──
        lh_pose = np.array(full_pose.get("left_hip", [0, 0]), dtype=np.float64)
        rh_pose = np.array(full_pose.get("right_hip", [0, 0]), dtype=np.float64)
        if lh_pose[0] > 0 and rh_pose[0] > 0:
            hip_cx = float((lh_pose[0] + rh_pose[0]) / 2.0)
            hip_cy = float((lh_pose[1] + rh_pose[1]) / 2.0)
            hip_w = float(np.linalg.norm(rh_pose - lh_pose))

            ys_sc, xs_sc = np.where(scaled_mask > 0)
            if len(xs_sc) > 100:
                cloth_x1, cloth_x2 = int(xs_sc.min()), int(xs_sc.max())
                cloth_y1 = int(ys_sc.min())
                cloth_cx = (cloth_x1 + cloth_x2) / 2.0
                cloth_w = max(1, cloth_x2 - cloth_x1)

                # Scale pants to hip width. Bumped 1.1 → 1.22 so shorts cover
                # the hand-on-hip pose where the hand sits outside hip width.
                scale_ratio = float(np.clip((hip_w * 1.22) / cloth_w, 0.4, 2.5))
                if abs(scale_ratio - 1.0) > 0.05:
                    h_sc, w_sc = scaled_cloth.shape[:2]
                    new_w = max(10, int(w_sc * scale_ratio))
                    new_h = max(10, int(h_sc * scale_ratio))
                    scaled_cloth = cv2.resize(scaled_cloth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    scaled_mask = cv2.resize(scaled_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    ys_sc, xs_sc = np.where(scaled_mask > 0)
                    cloth_cx = (int(xs_sc.min()) + int(xs_sc.max())) / 2.0
                    cloth_y1 = int(ys_sc.min())

                # Translate: waistband top → hip level
                shift_x = hip_cx - cloth_cx
                # Waist top of pants aligns ~10px above hip center
                shift_y = (hip_cy - int(h_out * 0.05)) - cloth_y1
                M_pants = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                scaled_cloth = cv2.warpAffine(
                    scaled_cloth, M_pants, (w_out, h_out),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
                scaled_mask = cv2.warpAffine(
                    scaled_mask, M_pants, (w_out, h_out),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                pipeline_info.append("HipAlign")
    elif full_pose is not None and garment_category not in ("dress", "accessory"):
        # v16.19: Dress skips ShoulderAlign — simple_affine_warp_cloth handles
        # full shoulder→ankle positioning. Running ShoulderAlign first centers
        # the dress at shoulder_Y (wrong for a long garment), clips the top
        # off-canvas, and then Affine_dress computes wrong scale_h from the
        # truncated visible height. Skip for dress; let Affine_dress do it all.
        ls = np.array(full_pose["left_shoulder"], dtype=np.float64)
        rs = np.array(full_pose["right_shoulder"], dtype=np.float64)
        target_center = (ls + rs) / 2.0
        target_width = float(np.linalg.norm(rs - ls))

        ys_sc, xs_sc = np.where(scaled_mask > 0)
        if len(xs_sc) > 100:
            cloth_y1, cloth_y2 = int(ys_sc.min()), int(ys_sc.max())
            cloth_h_range = max(1, cloth_y2 - cloth_y1)
            cloth_x1, cloth_x2 = int(xs_sc.min()), int(xs_sc.max())
            cloth_w_actual = max(1, cloth_x2 - cloth_x1)

            # v16.3 FIX 1: DETECT REAL COLLAR Y — use mass center of top 15%
            # instead of fixed 0.20 ratio. Different garment types have different
            # collar positions (crew neck ~15%, hoodie ~10%, tank ~25%).
            top_cutoff = cloth_y1 + int(cloth_h_range * 0.15)
            top_region = ys_sc < top_cutoff
            if top_region.sum() > 20:
                cloth_collar_y = float(ys_sc[top_region].mean())
            else:
                cloth_collar_y = float(cloth_y1 + cloth_h_range * 0.18)

            # v16.6 FIX: CENTER X via BBOX CENTER — mean() is wrong when
            # garment has asymmetric print/text (e.g. "HUSTLE" shifted).
            cloth_cx = (cloth_x1 + cloth_x2) / 2.0

            # v16.8c: Scale — width-based, reduced from 1.28→1.20 for short sleeves.
            # 1.28 was too large → garment wider than shoulders → TPS/contour had to
            # pull it back in → caused sleeve deformation at edges.
            _top_sub = (top_subtype or "").lower()
            _width_factor = 1.20
            if garment_category == "top" and _top_sub == "hoodie":
                _width_factor = float(os.getenv("VTON_HOODIE_WIDTH_FACTOR", "1.02").strip() or 1.02)
                _width_factor = float(np.clip(_width_factor, 0.88, 1.12))
                pipeline_info.append(f"HoodieScaleFit:v22.3:w{_width_factor:.2f}")
            scale_x = (target_width * _width_factor) / cloth_w_actual
            if garment_category == "dress":
                # v20.5: Dress length — default to KNEE-length unless source
                # aspect ratio clearly indicates a maxi/long dress. The old
                # shoulder→ankle target stretched mid-calf and knee-length
                # references all the way to the ankle, which looks unnatural
                # because most dresses end above the ankle in real life.
                # Aspect (cloth_h / cloth_w):
                #   < 2.10 → above-knee / knee / midi → target shoulder→knee
                #   2.10-2.55 → midi/maxi → target knee + half-shin
                #   > 2.55 → maxi/floor → target shoulder→ankle
                lk_y = full_pose.get("left_knee", [0, h_out * 0.72])[1]
                rk_y = full_pose.get("right_knee", [0, h_out * 0.72])[1]
                la_y = full_pose.get("left_ankle", full_pose.get("left_knee", [0, h_out * 0.92]))[1]
                ra_y = full_pose.get("right_ankle", full_pose.get("right_knee", [0, h_out * 0.92]))[1]
                shoulder_y = (ls[1] + rs[1]) * 0.5
                knee_y = max(lk_y, rk_y)
                ankle_y = max(la_y, ra_y)
                body_h_knee = max(1.0, abs(knee_y - shoulder_y))
                body_h_mid = max(1.0, abs((knee_y + ankle_y) * 0.5 - shoulder_y))
                body_h_ankle = max(1.0, abs(ankle_y - shoulder_y))
                cloth_h_actual = max(1, cloth_y2 - cloth_y1)
                cloth_aspect = cloth_h_actual / max(1, cloth_w_actual)
                if cloth_aspect > 2.55:
                    _target_h = body_h_ankle * 0.95
                    pipeline_info.append("DressLength:v20.5:maxi")
                elif cloth_aspect > 2.10:
                    _target_h = body_h_mid * 0.98
                    pipeline_info.append("DressLength:v20.5:midi")
                else:
                    _target_h = body_h_knee * 1.08
                    pipeline_info.append("DressLength:v20.5:knee")
                body_h = max(_target_h, target_width * 2.0)
                scale_y = body_h / cloth_h_actual
                scale_ratio = min(scale_x * 1.05, scale_y)
            elif new_sleeve_type == "long":
                # Estimate torso height from shoulders to hips
                if "left_hip" in full_pose and "right_hip" in full_pose:
                    _lh_y = full_pose["left_hip"][1]
                    _rh_y = full_pose["right_hip"][1]
                    torso_h = max(abs(_lh_y - ls[1]), abs(_rh_y - rs[1]))
                else:
                    torso_h = target_width * 1.4
                cloth_h_actual = max(1, cloth_y2 - cloth_y1)
                _height_factor = 1.10
                if garment_category == "top" and _top_sub == "hoodie":
                    _height_factor = float(os.getenv("VTON_HOODIE_HEIGHT_FACTOR", "1.02").strip() or 1.02)
                    _height_factor = float(np.clip(_height_factor, 0.90, 1.12))
                    pipeline_info.append(f"HoodieScaleFit:v22.3:h{_height_factor:.2f}")
                scale_y = (torso_h * _height_factor) / cloth_h_actual
                scale_ratio = min(scale_x, scale_y)
            else:
                # Short/sleeveless: width-based, but add soft height constraint
                # to prevent garment hanging below torso
                if "left_hip" in full_pose and "right_hip" in full_pose:
                    _lh_y = full_pose["left_hip"][1]
                    _rh_y = full_pose["right_hip"][1]
                    torso_h = max(abs(_lh_y - ls[1]), abs(_rh_y - rs[1]))
                    cloth_h_actual_s = max(1, cloth_y2 - cloth_y1)
                    scale_y_soft = (torso_h * 1.10) / cloth_h_actual_s
                    # v16.8c: Reduced from 1.08→1.03 to prevent stiff vertical stretch
                    scale_ratio = min(scale_x, scale_y_soft * 1.03)
                else:
                    scale_ratio = scale_x
            scale_ratio = float(np.clip(scale_ratio, 0.5, 2.5))

            if abs(scale_ratio - 1.0) > 0.05:
                h_sc, w_sc = scaled_cloth.shape[:2]
                new_w = max(10, int(w_sc * scale_ratio))
                new_h = max(10, int(h_sc * scale_ratio))
                scaled_cloth = cv2.resize(scaled_cloth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scaled_mask = cv2.resize(scaled_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Recompute after resize
                ys_sc, xs_sc = np.where(scaled_mask > 0)
                if len(xs_sc) > 50:
                    cloth_y1 = int(ys_sc.min())
                    cloth_y2 = int(ys_sc.max())
                    cloth_h_range = max(1, cloth_y2 - cloth_y1)
                    top_cutoff = cloth_y1 + int(cloth_h_range * 0.15)
                    top_region = ys_sc < top_cutoff
                    if top_region.sum() > 20:
                        cloth_collar_y = float(ys_sc[top_region].mean())
                    else:
                        cloth_collar_y = float(cloth_y1 + cloth_h_range * 0.18)
                    cloth_cx = (int(xs_sc.min()) + int(xs_sc.max())) / 2.0

            # v16.7: ROBUST ALIGN — use body center (not target_center which is
            # shoulder midpoint), and mix collar Y with bbox center Y for stability.
            body_cx = (ls[0] + rs[0]) / 2.0
            body_cy = (ls[1] + rs[1]) / 2.0
            cloth_cy = (cloth_y1 + cloth_y2) / 2.0

            # Vertical anchor: use bbox center Y — simpler and more stable
            anchor_y = cloth_cy

            shift_x = body_cx - cloth_cx
            shift_y = body_cy - anchor_y

            # v16.7: Adaptive anti-drift (soft, not hard bias)
            shift_x -= np.clip(shift_x * 0.05, -8, 8)

            # v16.7c: ROTATION ALIGN — rotate garment to match shoulder slope.
            # Without this, TPS has to compensate for shoulder tilt → causes warp artifacts.
            # This is the most critical missing step that caused garment to appear tilted.
            dx_shoulder = rs[0] - ls[0]
            dy_shoulder = rs[1] - ls[1]
            shoulder_angle = np.degrees(np.arctan2(dy_shoulder, dx_shoulder))
            # Only rotate if shoulder slope is significant (>1 degree)
            if abs(shoulder_angle) > 1.0:
                # v16.9: Clamp rotation to ±8° (was ±15°). 15° was too aggressive —
                # caused heavy garment deformation especially at sleeves/shoulders.
                # Most natural shoulder tilt is <5°, pose noise can push higher.
                rot_angle = float(np.clip(shoulder_angle, -8, 8))
                # Rotate around garment center, then translate
                M_rot = cv2.getRotationMatrix2D((cloth_cx, anchor_y), rot_angle, 1.0)
                # Add translation into the rotation matrix
                M_rot[0, 2] += shift_x
                M_rot[1, 2] += shift_y
                scaled_cloth = cv2.warpAffine(
                    scaled_cloth, M_rot, (w_out, h_out),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                scaled_mask = cv2.warpAffine(
                    scaled_mask, M_rot, (w_out, h_out),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                pipeline_info.append(f"RotAlign({rot_angle:.1f}°)")
            else:
                M_align = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                scaled_cloth = cv2.warpAffine(
                    scaled_cloth, M_align, (w_out, h_out),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                scaled_mask = cv2.warpAffine(
                    scaled_mask, M_align, (w_out, h_out),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            pipeline_info.append("ShoulderAlign")
            _debug_save("03c_shoulder_aligned", scaled_cloth)

    tps_ok = False
    # v16.11c: For pants, skip TPS — use affine until full pants TPS is implemented
    # v16.18: For dress, ALSO skip TPS. TPS is tuned for top garments (torso
    # body-follow, graduated sleeve); applying it to a long bodycon dress
    # creates deformed silhouettes. Affine gives a clean shoulder→ankle scale.
    if full_pose is not None and garment_category not in ("pants", "dress", "accessory"):
        try:
            warped_cloth, warped_mask = tps_warp_cloth(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=(h_out, w_out),
                fit_scale=fit_scale,
                y_offset_ratio=y_offset,
                sleeve_type=new_sleeve_type,
                garment_category=garment_category,
            )
            pipeline_info.append("TPS")
            tps_ok = True

            # v16.2: CENTER-OF-MASS SANITY CHECK — if warped garment center is
            # too far from body center, TPS produced a bad warp → fallback.
            wm_ys, wm_xs = np.where(warped_mask > 20)
            if len(wm_xs) > 100:
                com_x = float(wm_xs.mean())
                body_cx = (full_pose["left_shoulder"][0] + full_pose["right_shoulder"][0]) / 2.0
                if abs(com_x - body_cx) > w_out * 0.03:
                    pipeline_info.append(f"TPS_OFFCENTER({com_x:.0f}vs{body_cx:.0f})")
                    tps_ok = False
        except Exception as e:
            _debug_save("04_tps_failed", scaled_cloth)
            pipeline_info.append(f"TPS_FAIL({e})")
            # Better fallback: shoulder-aligned affine warp (much better than Persp)
            try:
                warped_cloth, warped_mask = simple_affine_warp_cloth(
                    cloth_rgb=scaled_cloth,
                    cloth_mask=scaled_mask,
                    pose=full_pose,
                    output_shape=(h_out, w_out),
                    garment_category=garment_category,
                )
                pipeline_info.append(f"Affine({garment_category})")
            except Exception:
                warped_cloth, warped_mask = warp_cloth_to_torso(
                    person_rgb=person_rgb, cloth_rgb=cloth_rgb,
                    cloth_mask=cloth_mask, box=pose_box,
                    fit_scale=fit_scale, y_offset_ratio=y_offset,
                )
                pipeline_info.append("Persp")
    elif full_pose is not None and garment_category == "pants":
        # v19.26: pants need the piecewise hip+legs warper. simple_affine_warp_cloth
        # has no pants branch — it scales to shoulder width and places the
        # garment at the shoulder line, which collapses to nothing after the
        # ShortsRegionClip (hip→knee) is applied downstream.
        try:
            warped_cloth, warped_mask = piecewise_warp_pants_cloth(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=(h_out, w_out),
                pants_type=pants_type,
                pants_style=pants_style,
                fit_scale=fit_scale,
            )
            pipeline_info.append(f"PantsPiecewise:{pants_type}:{pants_style}:v19.31")
            if int(cv2.countNonZero(warped_mask)) < 200:
                # Piecewise produced an empty/tiny result — try the affine fallback
                raise RuntimeError("piecewise_pants_empty")
        except Exception as e:
            pipeline_info.append(f"PantsPiecewise_fail({e})")
            try:
                warped_cloth, warped_mask = simple_affine_warp_cloth(
                    cloth_rgb=scaled_cloth,
                    cloth_mask=scaled_mask,
                    pose=full_pose,
                    output_shape=(h_out, w_out),
                    garment_category="pants",
                )
                pipeline_info.append("Affine_pants_fallback")
            except Exception as e2:
                pipeline_info.append(f"Affine_pants_fail({e2})")
                warped_cloth, warped_mask = warp_cloth_to_torso(
                    person_rgb=person_rgb, cloth_rgb=cloth_rgb,
                    cloth_mask=cloth_mask, box=pose_box,
                    fit_scale=fit_scale, y_offset_ratio=y_offset,
                )
                pipeline_info.append("Persp")
    elif full_pose is not None and garment_category == "dress":
        # v16.18: Dress uses AFFINE warp (shoulder->ankle) instead of TPS.
        # TPS over-constrains long bodycon dresses; affine keeps the original
        # silhouette of the dress and just scales it onto the person.
        try:
            warped_cloth, warped_mask = simple_affine_warp_cloth(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=(h_out, w_out),
                garment_category="dress",
            )
            pipeline_info.append("Affine_dress")
        except Exception as e:
            pipeline_info.append(f"Affine_dress_fail({e})")
            warped_cloth, warped_mask = warp_cloth_to_torso(
                person_rgb=person_rgb, cloth_rgb=cloth_rgb,
                cloth_mask=cloth_mask, box=pose_box,
                fit_scale=fit_scale, y_offset_ratio=y_offset,
            )
            pipeline_info.append("Persp")
    elif full_pose is not None and garment_category == "accessory":
        # ── Nhánh accessory: dispatch per-subtype warp ──
        try:
            from src.warps.accessory_warp import warp_accessory as _warp_accessory
            from src.accessories.anchors import (
                shoe_anchors as _shoe_anchors,
                head_anchors as _head_anchors,
                eye_anchors as _eye_anchors,
                waist_anchors as _waist_anchors,
                neck_anchors as _neck_anchors,
                shoulder_strap_anchors as _strap_anchors,
            )
            _sub = locals().get("accessory_subtype", "") or ""
            # Chọn bộ anchor đúng cho từng loại phụ kiện
            if _sub in {"shoes", "boots"}:
                _pose_anchors = _shoe_anchors(full_pose)
            elif _sub == "hat":
                _pose_anchors = _head_anchors(full_pose)
            elif _sub == "sunglasses":
                _pose_anchors = _eye_anchors(full_pose)
                # Tinh chỉnh bằng MediaPipe Face Mesh nếu có (eye_outer + nose_bridge chính xác hơn)
                _face_anchors = None
                try:
                    from src.accessories.face_mesh import detect_face_anchors as _detect_face
                    _face_anchors = _detect_face(person_rgb)
                except Exception:
                    _face_anchors = None
            elif _sub == "belt":
                _pose_anchors = _waist_anchors(full_pose)
            elif _sub == "scarf":
                _pose_anchors = _neck_anchors(full_pose)
            elif _sub == "bag":
                _pose_anchors = _strap_anchors(full_pose, side="left")
            else:
                _pose_anchors = {}

            _face_anchors = locals().get("_face_anchors", None)

            warped_cloth, warped_mask = _warp_accessory(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                subtype=_sub,
                out_shape=(h_out, w_out),
                pose_anchors=_pose_anchors,
                face_anchors=_face_anchors,
            )
            pipeline_info.append(f"AccessoryWarp:{_sub or 'unknown'}:v1")
            if int(cv2.countNonZero(warped_mask)) < 100:
                raise RuntimeError("accessory_warp_empty")
        except Exception as e:
            pipeline_info.append(f"AccessoryWarp_fail({e})")
            # Fallback: simple_affine để pipeline không sập
            try:
                warped_cloth, warped_mask = simple_affine_warp_cloth(
                    cloth_rgb=scaled_cloth,
                    cloth_mask=scaled_mask,
                    pose=full_pose,
                    output_shape=(h_out, w_out),
                    garment_category="top",
                )
                pipeline_info.append("Affine_accessory_fallback")
            except Exception:
                warped_cloth, warped_mask = warp_cloth_to_torso(
                    person_rgb=person_rgb, cloth_rgb=cloth_rgb,
                    cloth_mask=cloth_mask, box=pose_box,
                    fit_scale=fit_scale, y_offset_ratio=y_offset,
                )
                pipeline_info.append("Persp_accessory")
    else:
        warped_cloth, warped_mask = warp_cloth_to_torso(
            person_rgb=person_rgb, cloth_rgb=cloth_rgb,
            cloth_mask=cloth_mask, box=pose_box,
            fit_scale=fit_scale, y_offset_ratio=y_offset,
        )
        pipeline_info.append("Persp")

    # Clamp warped outputs
    warped_cloth = _safe_uint8(warped_cloth)
    warped_mask = _safe_uint8(warped_mask)

    # v16.3: SHOULDER BOX CLAMP — prevent garment from extending beyond
    # reasonable bounds around the shoulder axis. TPS can push pixels far
    # outside the body → garment "flies off" to the side.
    if full_pose is not None and garment_category != "accessory":
        _ls = full_pose["left_shoulder"]
        _rs = full_pose["right_shoulder"]
        _lh = full_pose.get("left_hip", _ls)
        _rh = full_pose.get("right_hip", _rs)
        _sw = abs(_rs[0] - _ls[0])
        if garment_category == "pants":
            # v19.23: PANTS HIP/THIGH CLAMP — pants must be anchored at the
            # hip line, NOT the shoulder line. Using shoulder bounds caused
            # shorts to be rendered at the torso.
            _hw = max(1, abs(_rh[0] - _lh[0]))
            # v19.25: pose hip width is often <50px; widen with shoulder fallback
            _hw = max(_hw, int(_sw * 0.62))
            _hip_y = int((_lh[1] + _rh[1]) / 2)
            left_bound = max(0, int(min(_lh[0], _rh[0]) - _hw * 0.85))
            right_bound = min(w_out, int(max(_lh[0], _rh[0]) + _hw * 0.85))
            # Top of pants clamp = a bit above hip (waistband), never above
            # the shoulder midpoint.
            top_bound = max(0, int(_hip_y - _sw * 0.25))
            _la = full_pose.get("left_ankle")
            _ra = full_pose.get("right_ankle")
            _pants_type = locals().get("pants_type", "regular")
            if _pants_type == "shorts":
                bottom_bound = min(h_out, int(_hip_y + _sw * 1.25))
            elif _la is not None and _ra is not None:
                _ankle_y = int(max(_la[1], _ra[1]))
                bottom_bound = min(h_out, int(_ankle_y + _sw * 0.15))
            else:
                bottom_bound = h_out
        else:
            # v16.10e: Widened 0.28→0.35 — don't clip shoulders/sleeves
            left_bound = max(0, int(min(_ls[0], _rs[0]) - _sw * 0.35))
            right_bound = min(w_out, int(max(_ls[0], _rs[0]) + _sw * 0.35))
            top_bound = max(0, int(min(_ls[1], _rs[1]) - _sw * 0.40))
            # v16.11d: For dress, extend bottom to full frame height (skirt coverage)
            if garment_category == "dress":
                bottom_bound = h_out
            else:
                bottom_bound = min(h_out, int(max(_lh[1], _rh[1]) + _sw * 0.50))
        warped_mask[:top_bound, :] = 0
        warped_mask[bottom_bound:, :] = 0
        warped_mask[:, :left_bound] = 0
        warped_mask[:, right_bound:] = 0
        warped_cloth[:top_bound, :] = 0
        warped_cloth[bottom_bound:, :] = 0
        warped_cloth[:, :left_bound] = 0
        warped_cloth[:, right_bound:] = 0

        # v16.6: POST-TPS DRIFT CORRECTION — if garment COM drifted from body
        # center after TPS, nudge it back. This catches subtle TPS drift that
        # passes the 8% threshold but still looks off-center.
        _wm_ys, _wm_xs = np.where(warped_mask > 20)
        if len(_wm_xs) > 50:
            _com_x = float(_wm_xs.mean())
            _body_cx = (_ls[0] + _rs[0]) / 2.0
            _drift = _com_x - _body_cx
            if abs(_drift) > 2:
                M_fix = np.float32([[1, 0, -_drift], [0, 1, 0]])
                warped_cloth = cv2.warpAffine(warped_cloth, M_fix, (w_out, h_out),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                warped_mask = cv2.warpAffine(warped_mask, M_fix, (w_out, h_out),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pipeline_info.append(f"DriftFix({_drift:.0f}px)")

        # v16.9: Hair subtract REMOVED.
        # Previously subtracted hair from warped_mask+warped_cloth → garment erased
        # under hair → no garment data for diffusion or compositing under hair.
        # SOTA approach: garment extends FULLY under hair in the CPU composite.
        # HairOverlay at the end pastes original hair on top for clean layering.

    if (
        garment_category == "top"
        and (top_subtype or "").lower() == "hoodie"
        and full_pose is not None
        and os.getenv("VTON_HOODIE_POSE_FIT", "1").strip().lower() not in {"0", "false", "no", "off"}
    ):
        hoodie_fit_mask = _build_hoodie_pose_fit_mask(
            (h_out, w_out), full_pose, parsing, warped_mask,
        )
        fit_area = int(cv2.countNonZero(hoodie_fit_mask))
        old_area = int(cv2.countNonZero((warped_mask > 20).astype(np.uint8)))
        if fit_area > 500 and old_area > 500:
            # TPS keeps flat-lay hoodies too rectangular. Clip the seed to a
            # pose-fitted silhouette before every downstream mask and diffusion.
            pre_fit_mask = (warped_mask > 20).astype(np.uint8) * 255
            clipped_mask = cv2.bitwise_and(warped_mask, hoodie_fit_mask)
            hoodie_pose_sleeves = _build_hoodie_pose_sleeve_mask(
                (h_out, w_out), full_pose, parsing,
            )
            hoodie_pose_sleeves = cv2.bitwise_and(hoodie_pose_sleeves, hoodie_fit_mask)
            if int(cv2.countNonZero(hoodie_pose_sleeves)) > 80:
                clipped_mask = cv2.bitwise_or(clipped_mask, hoodie_pose_sleeves)
                warped_cloth = _paint_hoodie_pose_sleeves(
                    warped_cloth, pre_fit_mask, hoodie_pose_sleeves, full_pose,
                )
                _debug_save("04c_hoodie_pose_sleeve_mask", hoodie_pose_sleeves, is_mask=True)
                pipeline_info.append("HoodiePoseSleeveSeed:v23.2")
            if os.getenv("VTON_HOODIE_HEM_REPAIR", "0").strip().lower() in {"1", "true", "yes", "on"}:
                sealed_mask, hem_repair = _seal_hoodie_hem_notches(clipped_mask)
            else:
                sealed_mask = clipped_mask
                hem_repair = np.zeros_like(clipped_mask)
                pipeline_info.append("HoodieHemSealSkip:v23.3")
            hoodie_soft = cv2.GaussianBlur(
                (sealed_mask > 20).astype(np.float32), (5, 5), 1.0,
            )
            hoodie_soft = np.clip(hoodie_soft, 0.0, 1.0)
            warped_cloth = _safe_uint8(warped_cloth.astype(np.float32) * hoodie_soft[..., None])
            if int(cv2.countNonZero(hem_repair)) > 20:
                valid = (pre_fit_mask > 20) & (warped_cloth.sum(axis=2) > 20)
                if int(valid.sum()) > 100:
                    fill_base = warped_cloth.copy()
                    fill_base[~valid] = np.median(warped_cloth[valid], axis=0).astype(np.uint8)
                    fill_blur = cv2.GaussianBlur(fill_base, (21, 21), 8.0)
                    warped_cloth[hem_repair > 20] = fill_blur[hem_repair > 20]
                _debug_save("04b_hoodie_hem_repair", hem_repair, is_mask=True)
                pipeline_info.append("HoodieHemSeal:v22.6")
            warped_mask = sealed_mask
            final_area = int(cv2.countNonZero((warped_mask > 20).astype(np.uint8)))
            _debug_save("04a_hoodie_pose_fit_mask", hoodie_fit_mask, is_mask=True)
            pipeline_info.append(f"HoodiePoseFit:v22.6:{old_area}->{final_area}")

    _debug_save("04_warped_cloth", warped_cloth)
    _debug_save("04_warped_mask", warped_mask, is_mask=True)

    # v16.10d: EdgeRepair DISABLED — diffusion handles edge texture.
    # EdgeRepair made edges "safe" but flattened texture. With GPU diffusion
    # at strength 0.82, let diffusion regenerate edges naturally.

    # ── v16.8b: Arm-contour-guided sleeve shaping (short sleeves only) ──
    # GP-VTON insight: reshape warped mask to follow arm contour from parsing.
    # TPS warps the full garment → sleeves often deformed. The arm parsing mask
    # provides the ground-truth arm boundary → clip sleeve beyond arm contour.
    # v16.8c: Only trigger when sleeve significantly overlaps arm area.
    # If overlap is small, TPS output is fine — no need for extra correction.
    # v16.10f: Lightweight arm-boundary clip for short sleeves.
    # Instead of full ArmContourShape (too aggressive), simply clip garment pixels
    # that extend beyond arm+torso parsing boundary. Prevents sleeve excess.
    # ── Step 4b: SleeveClip (ONLY for TOP garments with SHORT sleeves) ──
    # v16.11c: Skip for pants/dress
    if garment_category == "top" and new_sleeve_type == "short" and parsing is not None:
        # v16.11: Build body region from parsing ONLY (not warped_mask).
        # Including warped_mask defeats the purpose — excess TPS sleeve pixels
        # would be included in the allowed region, so nothing gets clipped.
        body_region = np.zeros((h_out, w_out), dtype=np.uint8)
        for bkey in ("left_arm", "right_arm", "upper_clothes", "dress", "torso"):
            if bkey in parsing:
                body_region = cv2.bitwise_or(body_region, parsing[bkey])
        # Tighter dilation (5px, was 9) — allow slight overflow but clip real excess
        body_region = cv2.dilate(body_region, np.ones((5, 5), np.uint8), iterations=1)
        # Soft edge for natural transition
        body_edge = cv2.GaussianBlur(body_region, (7, 7), 0)
        body_f = (body_edge.astype(np.float32) / 255.0)

        # Fade garment to zero outside body region
        warped_mask_f = warped_mask.astype(np.float32) * body_f
        warped_mask = np.clip(warped_mask_f, 0, 255).astype(np.uint8)
        for c in range(3):
            warped_cloth[:, :, c] = (warped_cloth[:, :, c].astype(np.float32) * body_f).astype(np.uint8)
        pipeline_info.append("SleeveClip")

    # ── Step 4c: Sleeve warp (TOP/DRESS garments with LONG sleeves) ──
    # v16.68: Dresses use the same sleeve warp as tops, but the old
    # Dress-only SleeveArmCover expansion is removed. That expansion caused
    # large round shoulder caps and a second visible sleeve layer.
    sleeve_data = {}  # side -> (rgb, mask_float)
    if garment_category in ("top", "dress") and full_pose is not None and new_sleeve_type == "long":
        try:
            sleeve_result = warp_sleeves_to_arms(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=(h_out, w_out),
                sleeve_type=new_sleeve_type,
            )
            if sleeve_result is not None:
                sleeve_data = sleeve_result
                for side, (s_rgb, s_mask_f) in sleeve_data.items():
                    s_rgb = _safe_uint8(s_rgb)
                    if garment_category == "dress":
                        s_rgb, s_mask_f, _sleeve_clipped = _clip_sleeve_to_arm_pose(
                            s_rgb,
                            s_mask_f,
                            full_pose,
                            parsing,
                            side,
                        )
                        if _sleeve_clipped:
                            pipeline_info.append(f"DressSleevePoseClip:{side}:v16.72")
                    sleeve_data[side] = (s_rgb, s_mask_f)
                    _debug_save(f"04c_sleeve_{side}_rgb", s_rgb)
                    _debug_save(f"04c_sleeve_{side}_mask", (s_mask_f * 255).astype(np.uint8), is_mask=True)
        except Exception:
            pass  # Sleeve warp failed, keep torso-only result

    # Unified garment support used by dress erase/diffusion.  The base affine
    # mask can be narrower than the cleaned old-clothes area; include
    # independently warped sleeves so later stages do not preserve a grey
    # body-shaped underlayer around the visible dress.
    dress_body_support_mask = (warped_mask > 20).astype(np.uint8) * 255
    if garment_category == "dress":
        completed_body = _complete_dress_body_mask(dress_body_support_mask, full_pose)
        if int(cv2.countNonZero(completed_body)) > int(cv2.countNonZero(dress_body_support_mask)) + 80:
            dress_body_support_mask = completed_body
            pipeline_info.append("DressSkirtComplete:v16.60")

        # v18.19: detect source silhouette from cloth_mask so the pose-fit
        # envelope follows the actual dress shape (a-line/sheath/fit-and-flare
        # etc.) instead of a single hardcoded curve.
        _silhouette_name: str | None = None
        _source_bust_half: float | None = None
        try:
            _sil_name, _sil_conf = detect_dress_silhouette(cloth_mask)
            # v18.21: commit even at low confidence — fallback hardcoded
            # curve was producing wrong shapes for sheath / mermaid sources.
            # Any template match is better than a generic a_line curve.
            if _sil_conf > 0.0:
                _silhouette_name = _sil_name
                ys_c, xs_c = np.where(cloth_mask > 0)
                if len(xs_c) > 100:
                    y1c, y2c = int(ys_c.min()), int(ys_c.max())
                    bust_row = max(0, min(cloth_mask.shape[0] - 1,
                                           y1c + int((y2c - y1c) * 0.12)))
                    nz = np.where(cloth_mask[bust_row] > 0)[0]
                    if len(nz) >= 4:
                        _source_bust_half = float(nz.max() - nz.min()) * 0.5
                if new_sleeve_type == "long" and _silhouette_name in {"empire", "ball_gown"}:
                    # Long sleeves hanging down the sides inflate mid-body rows
                    # and make this source look like an empire/ball-gown mask.
                    # For body fitting, treat it as an A-line dress and let the
                    # separate sleeve warp handle arms.
                    pipeline_info.append(f"DressSilhouetteOverride:{_silhouette_name}->a_line:v20.3")
                    _silhouette_name = "a_line"
                pipeline_info.append(
                    f"DressSilhouette:{_silhouette_name}:{_sil_conf:.2f}:v18.21"
                )
        except Exception as _sil_exc:
            print(f"[DRESS] silhouette detection failed: {_sil_exc}")

        pose_fit_body, _pose_fit_applied = _fit_dress_body_mask_to_pose(
            dress_body_support_mask,
            full_pose,
            silhouette=_silhouette_name,
            source_bust_half=_source_bust_half,
            fit_scale=fit_scale,
        )
        if _pose_fit_applied:
            dress_body_support_mask = pose_fit_body
            pipeline_info.append("DressBodyPoseFit:v16.72")

        # Keep the body panel off the arms for every dress. Long sleeves are
        # added back from `sleeve_data`; without this carve, a wide body mask
        # paints a flat dress slab over the raised hand/forearm.
        if parsing is not None:
            _arms = np.zeros_like(dress_body_support_mask)
            for _ak in ("left_arm", "right_arm"):
                if _ak in parsing:
                    _arms = cv2.bitwise_or(_arms, parsing[_ak])
            if int(cv2.countNonZero(_arms)) > 100:
                # Erode arms slightly so the dress-arm seam still has overlap
                # for natural shoulder-strap blending, but the bulk of the arm
                # stays clear.
                _arm_kernel = 3 if new_sleeve_type == "long" else 5
                _arms_core = cv2.erode(_arms, np.ones((_arm_kernel, _arm_kernel), np.uint8), iterations=1)
                dress_body_support_mask = cv2.subtract(dress_body_support_mask, _arms_core)
                pipeline_info.append(f"DressArmCarve:{new_sleeve_type}:v20.3")

    garment_support_mask = dress_body_support_mask.copy()
    for _s_rgb, _s_mask_f in sleeve_data.values():
        _s_u8 = (_s_mask_f > 0.04).astype(np.uint8) * 255
        garment_support_mask = cv2.bitwise_or(garment_support_mask, _s_u8)
    if garment_category == "dress":
        if parsing and full_pose is not None:
            _old_upper = get_clothing_mask(parsing)
            if _old_upper is not None and "left_shoulder" in full_pose and "right_shoulder" in full_pose:
                _ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
                _rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
                _sw = max(20.0, float(np.linalg.norm(_ls - _rs)))
                _sh_y = float((_ls[1] + _rs[1]) * 0.5)
                yy, xx = np.indices((h_out, w_out))
                _band = (
                    (yy >= int(max(0, _sh_y - _sw * 0.24)))
                    & (yy <= int(min(h_out - 1, _sh_y + _sw * 0.50)))
                    & (xx >= int(max(0, min(_ls[0], _rs[0]) - _sw * 0.28)))
                    & (xx <= int(min(w_out - 1, max(_ls[0], _rs[0]) + _sw * 0.28)))
                ).astype(np.uint8) * 255
                _near_dress = cv2.dilate(
                    garment_support_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
                    iterations=1,
                )
                _person_i = person_rgb.astype(np.int16)
                _old_red = (
                    (_person_i[:, :, 0] > _person_i[:, :, 1] + 28)
                    & (_person_i[:, :, 0] > _person_i[:, :, 2] + 22)
                    & (_person_i[:, :, 0] > 90)
                    & (_person_i[:, :, 1] < 145)
                ).astype(np.uint8) * 255
                _seal_source = cv2.bitwise_or(_old_upper, _old_red)
                _seal = cv2.bitwise_and(_seal_source, _band)
                _seal = cv2.bitwise_and(_seal, _near_dress)
                if int(_seal.sum()) > 255 * 20:
                    dress_body_support_mask = cv2.bitwise_or(dress_body_support_mask, _seal)
                    garment_support_mask = cv2.bitwise_or(garment_support_mask, _seal)
                    pipeline_info.append("DressShoulderSeal:v16.70")
        garment_support_mask = cv2.morphologyEx(
            garment_support_mask, cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8), iterations=1,
        )
    _debug_save("04d_garment_support_mask", garment_support_mask, is_mask=True)

    # PRE-FILL warped cloth's transparent area with blurred cloth colors.
    # v16.7d: Use GaussianBlur instead of median color fill.
    # Median creates a FLAT single color that doesn't match local edge colors,
    # causing visible color-jump halo at edges. Blur provides spatially-varying
    # fill that matches the garment's local color at each edge point.
    # SAFE now because white background was replaced with median garment color
    # BEFORE warping (earlier in pipeline), so blur won't spread white pixels.
    # v16.7c: Lower threshold from 25→8 so EdgeRepair-cleaned outer pixels
    # are preserved from warped_cloth instead of being replaced by blur (halo fix).
    cloth_mask_bool = warped_mask > 8
    cloth_fill_bg = cv2.GaussianBlur(warped_cloth, (21, 21), 8.0)
    warped_cloth_prefilled = cloth_fill_bg.copy()
    warped_cloth_prefilled[cloth_mask_bool] = warped_cloth[cloth_mask_bool]

    # Fix black leak — any pixel inside mask that is very dark (sum < 30)
    # is a TPS boundary artifact. Replace with blurred garment.
    if cloth_mask_bool.sum() > 100:
        wc_bright = warped_cloth_prefilled.sum(axis=2)
        black_leak = cloth_mask_bool & (wc_bright < 30)
        black_leak_count = int(black_leak.sum())
        if black_leak_count > 0 and black_leak_count < cloth_mask_bool.sum() * 0.3:
            warped_cloth_prefilled[black_leak] = cloth_fill_bg[black_leak]

    # ── Step 5: Erase old clothing from person ──
    skin_mask = get_skin_mask(parsing) if parsing else None

    # v16 FIX: PRESERVE = face + hair ONLY.
    # Previous: included arms + wide neck band in preserve → blocked garment
    # at shoulders, kept old shirt pixels visible through gaps.
    # Now: only face and hair are sacred. Arms are NOT preserved (garment
    # renders over arm boundary naturally). Hair overlays AFTER compositing.
    preserve_mask = np.zeros((h_out, w_out), dtype=np.uint8)
    if parsing:
        # Face + hat + sunglasses — these must NEVER be erased
        for pkey in ("face", "hat", "sunglasses"):
            if pkey in parsing:
                preserve_mask = cv2.bitwise_or(preserve_mask, parsing[pkey])
        # v16.10: Hair is NOT in preserve_mask. Old clothing under hair must be
        # erased so new garment extends under hair. HairOverlay pastes hair on top later.

        # v16: Minimal neck band — just 5px below chin to prevent
        # garment from overlapping onto chin. Previous 15x24 ellipse was
        # way too large → blocked garment at neckline.
        face_m = parsing.get("face")
        if face_m is not None and garment_category != "dress":
            neck_skin_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 10))
            neck_skin = cv2.dilate(face_m, neck_skin_k, iterations=1)
            face_ys = np.where(face_m > 0)[0]
            if len(face_ys) > 5:
                neck_skin[:int(face_ys.max() * 0.97), :] = 0
            preserve_mask = cv2.bitwise_or(preserve_mask, neck_skin)
        elif garment_category == "dress":
            pass
        # Minimal dilation — just 3px safety margin
        preserve_mask = cv2.dilate(preserve_mask, np.ones((3, 3), np.uint8), iterations=1)
    _debug_save("05a_preserve_mask", preserve_mask, is_mask=True)

    # Build erase_mask: prefer PARSING clothing mask over skeleton rectangle.
    if parsing:
        old_clothes = get_clothing_mask(parsing)
        if old_clothes is not None and int(old_clothes.sum()) > 255 * 200:
            erase_mask = old_clothes.copy()
            # v16: Moderate dilation (5x5 iter=1) — covers old garment edges
            # without over-expanding into arm/neck regions.
            erase_mask = cv2.dilate(erase_mask, np.ones((5, 5), np.uint8), iterations=1)

            # v16.11f: For dress, also erase shorts/legs where skirt will cover
            if garment_category == "dress":
                for _dk in ("pants", "skirt", "left_leg", "right_leg"):
                    _dp = parsing.get(_dk)
                    if _dp is not None:
                        erase_mask = cv2.bitwise_or(erase_mask, _dp)
                # v16.19: For dress, only erase the OLD CLOTHING on arms (intersection),
                # NOT the full arm skin. Full arm erase created gray holes where the
                # dress sleeve (from affine warp) didn't reach → gray rectangle artifact.
                # Intersect arm_mask with old_clothes to get only the sleeve overlap area.
                _arm_any = get_arm_mask(parsing)
                if _arm_any is not None:
                    _arm_sleeve_only = cv2.bitwise_and(_arm_any, erase_mask)
                    if int(_arm_sleeve_only.sum()) > 255 * 50:
                        _arm_dil = cv2.dilate(_arm_sleeve_only, np.ones((5, 5), np.uint8), iterations=1)
                        erase_mask = cv2.bitwise_or(erase_mask, _arm_dil)
                        pipeline_info.append("ArmErase")
                # v16.13: Add neckline-band erase — old top collars often bleed
                # through at the dress neckline (e.g. red tank-top collar visible
                # above the dress). Build a horizontal strip from chin to "upper
                # chest" using pose shoulder-y and force it into erase_mask so
                # SoftErase cleans those collar pixels before TPS/diffusion.
                try:
                    if full_pose is not None and "left_shoulder" in full_pose and "right_shoulder" in full_pose:
                        _ls = np.array(full_pose["left_shoulder"], dtype=np.float64)
                        _rs = np.array(full_pose["right_shoulder"], dtype=np.float64)
                        _sh_y = float((_ls[1] + _rs[1]) / 2)
                        _sw = max(40.0, float(np.linalg.norm(_ls - _rs)))
                        # Band goes from shoulder-y - 0.25*shoulder_w (just under chin)
                        # down to shoulder-y + 0.75*shoulder_w (upper chest).
                        _band_top = max(0, int(_sh_y - _sw * 0.25))
                        _band_bot = min(h_out, int(_sh_y + _sw * 0.75))
                        _band_cx = int((_ls[0] + _rs[0]) / 2)
                        _band_hw = int(_sw * 1.10)
                        _band_l = max(0, _band_cx - _band_hw)
                        _band_r = min(w_out, _band_cx + _band_hw)
                        _nb = np.zeros_like(erase_mask)
                        _nb[_band_top:_band_bot, _band_l:_band_r] = 255
                        erase_mask = cv2.bitwise_or(erase_mask, _nb)
                        pipeline_info.append("NeckErase")
                except Exception:
                    pass

            # SLEEVE TRANSITION: Only erase arm clothing when DOWNGRADING
            # sleeve coverage (e.g., long→short). For same-type (short→short)
            # or upgrades (short→long), do NOT touch arms.
            if needs_arm_erase:
                arm_m = get_arm_mask(parsing)
                if arm_m is not None:
                    # Include arm regions covered by old clothing
                    arm_clothing = cv2.bitwise_and(old_clothes, arm_m)
                    # Dilate to catch edges of old sleeve on arms
                    arm_clothing = cv2.dilate(arm_clothing, np.ones((7, 7), np.uint8), iterations=2)
                    erase_mask = cv2.bitwise_or(erase_mask, arm_clothing)
                    pipeline_info.append("SleeveErase")
        elif full_pose is not None:
            erase_mask = build_skeleton_erase_mask(full_pose, (h_out, w_out))
        else:
            erase_mask = (warped_mask > 20).astype(np.uint8) * 255
    elif full_pose is not None:
        erase_mask = build_skeleton_erase_mask(full_pose, (h_out, w_out))
    else:
        erase_mask = (warped_mask > 20).astype(np.uint8) * 255

    # Also erase where the new garment will go.
    wm_bin = (
        garment_support_mask.copy()
        if garment_category == "dress"
        else (warped_mask > 20).astype(np.uint8) * 255
    )
    erase_mask = cv2.bitwise_or(erase_mask, wm_bin)

    # Subtract preserve areas — face/hair/neck/arms must NOT be erased
    erase_mask = cv2.subtract(erase_mask, preserve_mask)
    if garment_category == "dress":
        # Keep dress erase local to the new garment footprint. The broad
        # neckline/old-clothes erase is useful for red-collar bleed, but if it
        # extends outside the generated dress it creates visible grey slabs
        # that diffusion then treats as part of the seed.
        _dress_erase_support = cv2.dilate(
            garment_support_mask, np.ones((13, 13), np.uint8), iterations=1,
        )
        # v18.18: for sleeveless dress, the new neckline is lower than the
        # original dress collar/drape. Without expanding the erase support
        # upward, the old drape (above new neckline) survives in CPU output
        # and bleeds through diffusion as a dark patch at the chest. Extend
        # support to include the neck region + decolletage band above the
        # new dress, so old garment pixels there are cleaned to skin.
        if new_sleeve_type == "sleeveless" and parsing is not None:
            _neck_extra = parsing.get("neck") if "neck" in parsing else None
            face_p = parsing.get("face")
            if face_p is not None:
                _chest_band = cv2.dilate(
                    face_p,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 65)),
                    iterations=2,
                )
                _dress_erase_support = cv2.bitwise_or(_dress_erase_support, _chest_band)
            if _neck_extra is not None:
                _dress_erase_support = cv2.bitwise_or(_dress_erase_support, _neck_extra)
            pipeline_info.append("DressSleevelessNeckErase:v18.18")
        erase_mask = cv2.bitwise_and(erase_mask, _dress_erase_support)
        pipeline_info.append("DressEraseClip:v16.58")
    _debug_save("05b_erase_mask", erase_mask, is_mask=True)

    # v13 FIX: SOFT BODY REPLACEMENT instead of hard erase.
    # Previous: erase_clothing_region() used Telea inpaint → dark/smeared patches.
    # New: Replace erased area with BLURRED body pixels (never black).
    # The garment covers this area anyway — we just need a clean non-black
    # background where the mask edge might bleed slightly.
    # v16: Tighter erase feather (9x9 sigma=3, was 15x15 sigma=5).
    # Previous was too wide → blurred old garment into surrounding area.
    erase_f = cv2.GaussianBlur(
        (erase_mask > 30).astype(np.float32), (9, 9), 3.0
    )[..., None]
    # Blurred version of person as fill (smooth skin-like background)
    body_blurred = cv2.GaussianBlur(person_rgb, (25, 25), 8.0)
    person_cleaned = _safe_uint8(
        person_rgb.astype(np.float32) * (1.0 - erase_f)
        + body_blurred.astype(np.float32) * erase_f
    )
    # v16.17: For DRESS, follow-up with a HARD neutral-grey fill inside the
    # erase_mask so NO old-garment colour (red shirt, denim shorts) survives
    # to contaminate the TPS init or the diffusion seed. Neutral grey is the
    # ideal starting colour for diffusion — no colour bias.
    # v16.18: Use the BACKGROUND median colour (sampled from image corners)
    # instead of flat 128 grey. This blends more naturally with the scene
    # edges and avoids the "grey blob" look in the composite.
    if garment_category == "dress":
        _hard_erase = (erase_mask > 30).astype(np.float32)
        _hard_erase = cv2.GaussianBlur(_hard_erase, (5, 5), 1.5)[..., None]
        # v16.22: Restrict the fill to PERSON SILHOUETTE ONLY so we never
        # paint onto the real background.
        _person_sil = np.zeros(person_cleaned.shape[:2], dtype=np.uint8)
        if parsing:
            for _sk in (
                "face", "hair", "hat", "sunglasses",
                "upper_clothes", "dress", "coat", "scarf",
                "left_arm", "right_arm", "neck",
                "pants", "skirt", "left_leg", "right_leg",
                "left_shoe", "right_shoe",
            ):
                _sv = parsing.get(_sk)
                if _sv is not None:
                    _person_sil = cv2.bitwise_or(_person_sil, _sv)
        if int(_person_sil.sum()) > 255 * 500:
            _person_sil = cv2.morphologyEx(_person_sil, cv2.MORPH_CLOSE,
                                           np.ones((15, 15), np.uint8))
            _person_sil = cv2.dilate(_person_sil, np.ones((5, 5), np.uint8),
                                     iterations=1)
            _sil_f = (_person_sil > 0).astype(np.float32)[..., None]
            _hard_erase = _hard_erase * _sil_f
        # v16.54: Split the fill by WHETHER the pixel lies inside the warped
        # dress or not:
        #   - Inside warped_mask -> dress mean colour, so alpha-edge feathering
        #     blends invisibly.
        #   - Outside warped_mask -> BACKGROUND colour, not skin colour.
        # The old skin fill created a wide brown/grey body-shaped silhouette
        # around the real dress. When diffusion/clip preserved that seed, it
        # looked exactly like a second dress layer behind the garment.
        _wm_bool = (warped_mask > 64)
        if int(_wm_bool.sum()) > 200:
            _dress_rgb = warped_cloth[_wm_bool].astype(np.float32).mean(axis=0)
        else:
            _dress_rgb = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        # Sample skin from face parsing (or neck as fallback).
        _skin_rgb = None
        if parsing:
            for _face_key in ("face", "neck"):
                _fm = parsing.get(_face_key)
                if _fm is not None and int(_fm.sum()) > 255 * 50:
                    _fb = (_fm > 0)
                    _skin_rgb = person_rgb[_fb].astype(np.float32).mean(axis=0)
                    break
        if _skin_rgb is None:
            _skin_rgb = _dress_rgb.copy()
        # Sample background from image corners. For catalog/product images this
        # is usually white/light grey and removes the old shirt/shorts without
        # leaving a person-shaped underlayer.
        _ch = max(6, h_out // 12)
        _cw = max(6, w_out // 12)
        _corner_pixels = np.concatenate([
            person_rgb[:_ch, :_cw].reshape(-1, 3),
            person_rgb[:_ch, -_cw:].reshape(-1, 3),
            person_rgb[-_ch:, :_cw].reshape(-1, 3),
            person_rgb[-_ch:, -_cw:].reshape(-1, 3),
        ], axis=0).astype(np.float32)
        _bg_rgb = np.median(_corner_pixels, axis=0)
        # Guard against dark UI/card pixels if input was already cropped oddly.
        if float(_bg_rgb.mean()) < 120:
            _bg_rgb = np.array([245.0, 245.0, 245.0], dtype=np.float32)

        # v20.3 (P1): erase the FULL old-garment footprint (old clothes ∪
        # pants ∪ skirt ∪ legs ∪ dress), not just pixels inside the new
        # dress silhouette. Then split the fill into 3 regions so the
        # uncovered "spill" outside the new dress doesn't show as a grey
        # body-shaped slab:
        #   - inside support  → new dress mean colour (covered anyway)
        #   - on legs/skirt, outside support → skin tone (continues thigh)
        #   - elsewhere outside support → background colour
        # Without this, a wide flared original skirt leaves its outline
        # bleeding past the narrower new-dress silhouette.
        _full_old_mask = np.zeros(person_cleaned.shape[:2], dtype=np.uint8)
        if old_clothes is not None:
            _full_old_mask = cv2.bitwise_or(_full_old_mask, old_clothes)
        if parsing:
            for _k in ("dress", "pants", "skirt", "left_leg", "right_leg"):
                _p = parsing.get(_k)
                if _p is not None:
                    _full_old_mask = cv2.bitwise_or(_full_old_mask, _p)
        _full_old_mask = cv2.subtract(_full_old_mask, preserve_mask)
        _full_old_mask = cv2.dilate(_full_old_mask, np.ones((5, 5), np.uint8), iterations=1)

        # v20.4: chroma-based extension. Parsing often misses lateral flare
        # of wide skirts — pixels are clearly "old dress" by colour but get
        # labelled as background. Sample the mean colour of confirmed old
        # clothing, then mark any pixel inside the person silhouette with
        # similar colour as also "old garment" so it gets erased.
        if old_clothes is not None and int(old_clothes.sum()) > 255 * 200:
            _oc_bool = old_clothes > 0
            _old_mean = person_rgb[_oc_bool].astype(np.float32).mean(axis=0)
            _diff = np.linalg.norm(
                person_rgb.astype(np.float32) - _old_mean[None, None, :],
                axis=2,
            )
            _color_close = (_diff < 35.0).astype(np.uint8) * 255
            # Build a "search band" = person silhouette minus preserve (face/
            # hair/arms) so chroma match never touches skin tones.
            _search_band = cv2.subtract(_person_sil, preserve_mask)
            _color_close = cv2.bitwise_and(_color_close, _search_band)
            _color_close = cv2.morphologyEx(
                _color_close, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
            )
            _color_close = cv2.morphologyEx(
                _color_close, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
            )
            _full_old_mask = cv2.bitwise_or(_full_old_mask, _color_close)
            pipeline_info.append("DressOldColourGapFill:v20.4")

        # Lateral dilation to absorb thin parsing gaps along skirt sides;
        # clamp to person silhouette so we never paint real background.
        _full_old_mask = cv2.dilate(_full_old_mask, np.ones((5, 25), np.uint8), iterations=1)
        _full_old_mask = cv2.bitwise_and(_full_old_mask, _person_sil)
        _debug_save("05c_full_old_mask", _full_old_mask, is_mask=True)

        _full_erase_f = cv2.GaussianBlur(
            (_full_old_mask > 30).astype(np.float32), (7, 7), 2.0
        )[..., None]
        _hard_erase = np.maximum(_hard_erase, _full_erase_f)

        _wm_inside = cv2.dilate(
            (garment_support_mask > 20).astype(np.uint8) * 255,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        _inside_f = (_wm_inside > 0).astype(np.float32)[..., None]

        _leg_mask = np.zeros(person_cleaned.shape[:2], dtype=np.uint8)
        if parsing:
            for _lk in ("left_leg", "right_leg", "pants", "skirt", "dress"):
                _lv = parsing.get(_lk)
                if _lv is not None:
                    _leg_mask = cv2.bitwise_or(_leg_mask, _lv)
        _leg_mask = cv2.dilate(_leg_mask, np.ones((5, 5), np.uint8), iterations=1)
        _leg_f = ((_leg_mask > 0).astype(np.float32)[..., None]) * (1.0 - _inside_f)

        _fill_rgb = (
            _inside_f * _dress_rgb[None, None, :]
            + _leg_f * _skin_rgb[None, None, :]
            + (1.0 - _inside_f - _leg_f) * _bg_rgb[None, None, :]
        )
        person_cleaned = _safe_uint8(
            person_cleaned.astype(np.float32) * (1.0 - _hard_erase)
            + _fill_rgb * _hard_erase
        )
        _debug_save("05c_dress_full_erase", person_cleaned)
        pipeline_info.append("DressFullErase:v20.4")
    # ── Step 6: Color match cloth to person lighting ──
    # Match warped cloth brightness to person BEFORE blending.
    # This prevents the garment looking "pasted on" with different lighting.
    warped_cloth_matched = _match_cloth_brightness(warped_cloth_prefilled, person_cleaned, warped_mask)
    if garment_category == "dress":
        warped_cloth_matched = _extend_dress_texture_to_mask(
            warped_cloth_matched,
            warped_mask,
            dress_body_support_mask,
        )
        dress_pattern_reference = _build_clean_dress_pattern_reference(
            warped_cloth_matched,
            garment_support_mask,
            source_mask=warped_mask,
        )
        _debug_save("05c_dress_pattern_reference", dress_pattern_reference)
        pipeline_info.append("DressTextureFill:v16.61")
        pipeline_info.append("DressPatternRef:v18.2")
    else:
        dress_pattern_reference = None

    # ── Step 6b: Body Curve Map + DrapeHint ──
    # v16.9: DISABLED when diffusion is primary (strength 0.75).
    # These CPU-side shading hacks fight with diffusion's own fold/shadow generation.
    # Diffusion at 0.75 strength regenerates garment texture including natural folds,
    # body curvature shading, and drape. CPU curve/drape only adds noise to the
    # init_image that diffusion then has to correct.
    # Only kept as light hint — reduced from previous strengths.
    if full_pose is not None and garment_category not in ("dress", "accessory"):
        wm_bool_drape = warped_mask > 30
        if wm_bool_drape.sum() > 500:
            # v16.11b: Restrict drape to torso-center only (exclude sleeves).
            # Sobel-based texture on sleeves creates artificial gathered look.
            ys_d, xs_d = np.where(wm_bool_drape)
            y1_d, y2_d = int(ys_d.min()), int(ys_d.max())
            x1_d, x2_d = int(xs_d.min()), int(xs_d.max())
            cloth_w_d = max(1, x2_d - x1_d)
            cloth_h_d = max(1, y2_d - y1_d)
            # Torso core: inset 20% from sides (skip sleeves), top 30% down (skip shoulders)
            torso_mask = np.zeros_like(warped_mask, dtype=np.float32)
            tx1 = x1_d + int(cloth_w_d * 0.20)
            tx2 = x2_d - int(cloth_w_d * 0.20)
            ty1 = y1_d + int(cloth_h_d * 0.30)
            ty2 = y2_d
            torso_mask[ty1:ty2, tx1:tx2] = 1.0
            torso_mask = cv2.GaussianBlur(torso_mask, (21, 21), 7.0)
            wm_bool_drape = wm_bool_drape & (torso_mask > 0.3)
            person_gray = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # v16.9: Minimal curve (1.5%) — just enough spatial hint for diffusion init
            body_depth = cv2.GaussianBlur(person_gray, (61, 61), 20.0)
            d_min = float(body_depth.min())
            d_range = max(1.0, float(body_depth.max()) - d_min)
            body_depth_norm = (body_depth - d_min) / d_range
            body_curve = cv2.Sobel(body_depth_norm, cv2.CV_32F, 1, 0, ksize=5)
            body_curve = cv2.GaussianBlur(body_curve, (31, 31), 10.0)
            bc_max = max(1e-6, float(np.abs(body_curve).max()))
            body_curve = body_curve / bc_max

            curve_strength = 0.015  # v16.9: minimal hint, diffusion handles real shading
            curve_mask = wm_bool_drape.astype(np.float32)
            curve_mask = cv2.GaussianBlur(curve_mask, (15, 15), 5.0)
            curve_factor = 1.0 + body_curve * curve_strength * curve_mask
            warped_cloth_matched = _safe_uint8(
                warped_cloth_matched.astype(np.float32) * curve_factor[..., None]
            )

            # v16.9: DrapeHint reduced to minimal spatial hint (was 0.15 → 0.06)
            body_curvature = cv2.GaussianBlur(person_gray, (51, 51), 20.0)
            curvature_in_mask = body_curvature[wm_bool_drape]
            c_min = float(np.percentile(curvature_in_mask, 5))
            c_max = float(np.percentile(curvature_in_mask, 95))
            c_range = max(1.0, c_max - c_min)
            drape_map = 0.96 + 0.08 * (body_curvature - c_min) / c_range
            drape_map = np.clip(drape_map, 0.94, 1.06)

            x_grad = cv2.Sobel(person_gray, cv2.CV_32F, 1, 0, ksize=5)
            x_grad = cv2.GaussianBlur(x_grad, (31, 31), 10.0)
            x_max = max(1.0, float(np.abs(x_grad[wm_bool_drape]).max()))
            x_grad_norm = x_grad / x_max
            drape_map = drape_map + x_grad_norm * 0.02

            y_grad = cv2.Sobel(person_gray, cv2.CV_32F, 0, 1, ksize=5)
            y_grad = cv2.GaussianBlur(y_grad, (31, 31), 10.0)
            y_max = max(1.0, float(np.abs(y_grad[wm_bool_drape]).max()))
            y_grad_norm = y_grad / y_max
            drape_map = drape_map + y_grad_norm * 0.01

            drape_map = np.clip(drape_map, 0.94, 1.06)

            drape_mask_f = cv2.GaussianBlur(
                wm_bool_drape.astype(np.float32), (15, 15), 5.0
            )
            drape_strength = drape_mask_f * 0.06  # v16.9: minimal — diffusion does the real work
            drape_factor = 1.0 + (drape_map - 1.0) * drape_strength
            warped_cloth_matched = _safe_uint8(
                warped_cloth_matched.astype(np.float32) * drape_factor[..., None]
            )
            pipeline_info.append("DrapeHint")

    # v16.7d: Edge smoothing DISABLED — with upstream BG replacement + tight
    # _soft_mask (blur 0.5, erode 2, clamp 0.15), edge smoothing is no longer
    # needed and only creates blur/halo artifacts at garment boundaries.
    _debug_save("05b_cloth_color_matched", warped_cloth_matched)

    # ── Step 7: 3-LAYER COMPOSITE (torso → left sleeve → right sleeve) ──
    _debug_save("05_person_cleaned", person_cleaned)

    # v16.9: Save FULL warped_mask (including under-hair) for downstream use.
    # HairOverlay and diffusion need to know the full garment extent.
    warped_mask_full = garment_support_mask.copy()

    # v16.9: Only subtract FACE (not hair) from compositing mask.
    # Garment must not overlap face, but SHOULD extend under hair.
    # Hair overlay at the end handles the layering.
    face_only_mask = np.zeros((h_out, w_out), dtype=np.uint8)
    if parsing:
        for pkey in ("face", "hat", "sunglasses"):
            if pkey in parsing:
                face_only_mask = cv2.bitwise_or(face_only_mask, parsing[pkey])
        # Minimal neck band
        face_m = parsing.get("face")
        if face_m is not None and garment_category != "dress":
            face_ys = np.where(face_m > 0)[0]
            if len(face_ys) > 5:
                neck_skin_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 10))
                neck_skin = cv2.dilate(face_m, neck_skin_k, iterations=1)
                neck_skin[:int(face_ys.max() * 0.97), :] = 0
                face_only_mask = cv2.bitwise_or(face_only_mask, neck_skin)
        face_only_mask = cv2.dilate(face_only_mask, np.ones((3, 3), np.uint8), iterations=1)

    composite_mask = dress_body_support_mask.copy() if garment_category == "dress" else warped_mask.copy()
    if face_only_mask.sum() > 0:
        composite_mask = cv2.subtract(composite_mask, face_only_mask)
        if garment_category != "dress":
            warped_mask = composite_mask

    # v19.61: TOP — nếu áo gen ra (warped) hẹp hơn parsing.upper_clothes của
    # người mẫu, vùng vai/sườn vẫn lộ màu áo cũ. Mở rộng composite_mask để phủ
    # toàn bộ parsing.upper_clothes và fill phần mở rộng bằng màu trung bình
    # của áo (sampled trong warped_mask) → không còn rò màu đỏ.
    if garment_category == "top" and parsing is not None:
        _orig_top_raw = parsing.get("upper_clothes")
        if _orig_top_raw is None:
            _orig_top_raw = parsing.get("dress")
        if _orig_top_raw is not None and int(_orig_top_raw.sum()) > 255 * 200:
            _orig_top_raw = _fit_like(_orig_top_raw, composite_mask, is_mask=True)
            _orig_top_raw = (_orig_top_raw > 20).astype(np.uint8) * 255
            # v19.65: 3 sửa lỗi:
            # (a) Vùng tóc-trên-vai: lấy hair pixels nằm DƯỚI face_y_max → là
            #     tóc xõa lên vai, cần phủ áo bên dưới. Hair overlay sẽ vẽ lại
            #     tóc lên trên sau.
            # (b) Viền cổ đỏ: erode _face_only 3px trước khi subtract.
            # (c) Tay áo lệch lộ đỏ: giao của parsing.{arms} ∩ parsing.upper_clothes
            #     (vùng tay áo cũ phủ lên cánh tay) — giữ lại để mở rộng phủ.
            _orig_top = cv2.dilate(_orig_top_raw, np.ones((5, 5), np.uint8), iterations=1)

            # (a) Hair-on-shoulder: hair pixels dưới đáy mặt
            _hair = parsing.get("hair")
            _face = parsing.get("face")
            if _hair is not None and _face is not None:
                _hair = _fit_like(_hair, _orig_top, is_mask=True)
                _hair_bin = (_hair > 20).astype(np.uint8) * 255
                _face_fit = _fit_like(_face, _orig_top, is_mask=True)
                _face_ys = np.where(_face_fit > 20)[0]
                if len(_face_ys) > 5:
                    _face_y_max = int(_face_ys.max())
                    _hair_shoulder = _hair_bin.copy()
                    _hair_shoulder[:_face_y_max, :] = 0
                    _orig_top = cv2.bitwise_or(_orig_top, _hair_shoulder)

            # (c) Arm-old-shirt intersection: giữ phần áo cũ phủ trên cánh tay
            _arm_under_shirt = np.zeros_like(_orig_top)
            for _ak in ("left_arm", "right_arm"):
                _av = parsing.get(_ak)
                if _av is not None:
                    _af = _fit_like(_av, _orig_top, is_mask=True)
                    _ab = (_af > 20).astype(np.uint8) * 255
                    _inter = cv2.bitwise_and(_ab, _orig_top_raw)
                    _arm_under_shirt = cv2.bitwise_or(_arm_under_shirt, _inter)
            _arm_under_shirt = cv2.dilate(_arm_under_shirt, np.ones((3, 3), np.uint8), iterations=1)

            # Person silhouette (clip extension)
            _person_sil = np.zeros_like(_orig_top)
            for _sk in (
                "face", "hair", "hat", "sunglasses", "upper_clothes", "dress",
                "left_arm", "right_arm", "neck", "pants", "skirt",
                "left_leg", "right_leg",
            ):
                _sv = parsing.get(_sk)
                if _sv is not None:
                    _sf = _fit_like(_sv, _orig_top, is_mask=True)
                    _person_sil = cv2.bitwise_or(_person_sil, (_sf > 20).astype(np.uint8) * 255)
            if int(_person_sil.sum()) > 255 * 200:
                _orig_top = cv2.bitwise_and(_orig_top, _person_sil)

            # Trừ tay/mũ — sau đó CỘNG LẠI _arm_under_shirt (vùng tay áo cũ).
            _exclude = np.zeros_like(_orig_top)
            for _xk in ("hat", "sunglasses", "left_arm", "right_arm"):
                _xv = parsing.get(_xk)
                if _xv is not None:
                    _xf = _fit_like(_xv, _orig_top, is_mask=True)
                    _exclude = cv2.bitwise_or(_exclude, (_xf > 20).astype(np.uint8) * 255)
            if int(_exclude.sum()) > 0:
                _exclude_eroded = cv2.erode(_exclude, np.ones((3, 3), np.uint8), iterations=1)
                _orig_top = cv2.subtract(_orig_top, _exclude_eroded)
            # Restore arm-under-shirt sleeve coverage
            if int(_arm_under_shirt.sum()) > 0:
                _orig_top = cv2.bitwise_or(_orig_top, _arm_under_shirt)
            # (b) Trừ FACE đã erode 3px → collar được phủ thêm
            _face_only = parsing.get("face")
            if _face_only is not None:
                _face_only = _fit_like(_face_only, _orig_top, is_mask=True)
                _face_only = (_face_only > 20).astype(np.uint8) * 255
                _face_only = cv2.erode(_face_only, np.ones((3, 3), np.uint8), iterations=1)
                _orig_top = cv2.subtract(_orig_top, _face_only)
            _wm_bin = (warped_mask > 20).astype(np.uint8) * 255
            if (top_subtype or "").lower() == "hoodie":
                _hoodie_cover_cap = cv2.dilate(
                    _wm_bin,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    iterations=1,
                )
                _orig_top = cv2.bitwise_and(_orig_top, _hoodie_cover_cap)
                pipeline_info.append("HoodieTopCoverCap:v22.4")
            _extension = cv2.subtract(_orig_top, _wm_bin)
            if int(_extension.sum()) > 255 * 50:
                # Sample màu trung bình của áo trong vùng warped
                _wm_bool = _wm_bin > 20
                if int(_wm_bool.sum()) > 200:
                    _mean_rgb = warped_cloth_matched[_wm_bool].mean(axis=0)
                else:
                    _mean_rgb = np.array([128.0, 128.0, 128.0], dtype=np.float32)
                _ext_f = cv2.GaussianBlur(
                    (_extension > 20).astype(np.float32), (7, 7), 2.0,
                )[..., None]
                _fill = np.full_like(warped_cloth_matched, 0)
                _fill[:] = _mean_rgb.astype(np.uint8)
                warped_cloth_matched = _safe_uint8(
                    warped_cloth_matched.astype(np.float32) * (1.0 - _ext_f)
                    + _fill.astype(np.float32) * _ext_f
                )
                composite_mask = cv2.bitwise_or(composite_mask, _extension)
                warped_mask = composite_mask
                pipeline_info.append("TopCoverOldShirt:v19.65")

    _debug_save("06a_warped_mask_after_protect", composite_mask, is_mask=True)

    # v16.7e: Edge blur DISABLED — was creating halo band.
    # Diffusion will handle edge smoothing naturally.
    # edge_band = cv2.morphologyEx(
    #     (warped_mask > 20).astype(np.uint8) * 255,
    #     cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8),
    # )
    # bad_edge = edge_band > 0
    # if bad_edge.sum() > 50:
    #     cloth_edge_blur = cv2.GaussianBlur(warped_cloth_matched, (3, 3), 0.5)
    #     warped_cloth_matched[bad_edge] = cloth_edge_blur[bad_edge]

    # Layer 1: TORSO (warped cloth, no sleeves mixed in)
    # v16.8: Softer feather (sigma=1.2, erode=2) — previous 0.5/1 was too sharp,
    # creating an obvious "cut-out" look at garment edges. Wider transition zone
    # lets the garment blend more naturally into skin.
    # v16.18: For DRESS, use a HARD mask (tiny 1px feather only) so the dress
    # appears SOLID on the person, not ghost/translucent. Tops still need a
    # soft feather for natural sleeve edge.
    if garment_category == "dress":
        dress_core = (composite_mask > 48).astype(np.uint8) * 255
        dress_core = cv2.morphologyEx(
            dress_core, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1,
        )
        torso_soft = cv2.GaussianBlur(
            dress_core.astype(np.float32) / 255.0, (3, 3), 0.6,
        )
        torso_soft[torso_soft < 0.35] = 0.0
        torso_soft = np.clip(torso_soft, 0.0, 1.0)
        pipeline_info.append("HardDressMask")
    else:
        torso_soft = _soft_mask(composite_mask, blur_sigma=1.2, erode_px=2)
    _debug_save("06_torso_mask", (torso_soft * 255).astype(np.uint8), is_mask=True)
    torso_alpha = torso_soft[..., None]

    init_tryon = (
        person_cleaned.astype(np.float32) * (1.0 - torso_alpha)
        + warped_cloth_matched.astype(np.float32) * torso_alpha
    )
    init_tryon = _safe_uint8(init_tryon)
    visible_alpha = torso_soft.copy()

    # Layer 2+3: SLEEVES (v16: tighter seam ring)
    # v16: 3x3 dilate (was 5x5) and (5,5) blur (was 7,7) — narrower seam band
    torso_u8 = (torso_soft * 255).clip(0, 255).astype(np.uint8)
    torso_dilated = cv2.dilate(torso_u8, np.ones((3, 3), np.uint8), iterations=1)
    seam_ring = cv2.subtract(torso_dilated, torso_u8)  # thin border ring
    seam_f = cv2.GaussianBlur(
        seam_ring.astype(np.float32) / 255.0, (5, 5), 1.0,
    )
    dress_body_texture_valid = None
    if garment_category == "dress":
        dress_body_texture_valid = _garment_texture_valid_mask(
            warped_cloth_matched,
            dress_body_support_mask,
        )

    for side in ("left", "right"):
        if side not in sleeve_data:
            continue
        s_rgb, s_mask_f = sleeve_data[side]

        # Sleeve warp uses neutral gray outside the source sleeve.  When the
        # target sleeve mask is wider than the narrow source strip, that gray
        # gets composited as "background stuck to the dress".  Fill every
        # masked sleeve pixel from nearest real sleeve texture first.
        s_target = (s_mask_f > 0.05).astype(np.uint8) * 255
        s_valid = _garment_texture_valid_mask(s_rgb, s_target)
        s_rgb, s_row_fill = _repeat_row_texture_into_mask(s_rgb, s_target, s_valid)
        s_rgb = _propagate_texture_into_mask(s_rgb, s_target, s_valid | s_row_fill, max_iter=90)
        if garment_category == "dress" and dress_body_texture_valid is not None:
            body_sleeve_rgb, body_sleeve_fill = _repeat_row_texture_into_mask(
                warped_cloth_matched,
                s_target,
                dress_body_texture_valid,
            )
            body_sleeve_rgb = _propagate_texture_into_mask(
                body_sleeve_rgb,
                s_target,
                dress_body_texture_valid | body_sleeve_fill,
                max_iter=90,
            )
            s_detail = _garment_texture_valid_mask(s_rgb, s_target)
            replace_flat = (s_target > 0) & ~s_detail
            if int(replace_flat.sum()) > 20:
                s_rgb[replace_flat] = body_sleeve_rgb[replace_flat]

        # Remove torso overlap from sleeve mask (sleeve only outside torso)
        s_exclusive = np.clip(s_mask_f - torso_soft, 0.0, 1.0)

        # In the seam ring: cross-fade sleeve at 60% opacity for smooth join
        s_in_seam = np.minimum(s_mask_f, seam_f) * 0.6
        s_final = np.clip(s_exclusive + s_in_seam, 0.0, 1.0)

        # Soft edge: gaussian blur the final sleeve alpha for anti-aliasing.
        # Dresses need a harder sleeve alpha because the underlayer has already
        # erased old sleeve pixels; low alpha shows that neutral fill as grey.
        s_final_u8 = (s_final * 255).clip(0, 255).astype(np.uint8)
        if garment_category == "dress":
            s_final = cv2.GaussianBlur(
                s_final_u8.astype(np.float32) / 255.0, (3, 3), 0.6,
            )
            s_final[s_final < 0.12] = 0.0
            s_final = np.clip(s_final * 1.18, 0.0, 0.98)
        else:
            s_final = cv2.GaussianBlur(
                s_final_u8.astype(np.float32) / 255.0, (5, 5), 1.0,
            )

        s_alpha = s_final[..., None]
        init_tryon = (
            init_tryon.astype(np.float32) * (1.0 - s_alpha)
            + s_rgb.astype(np.float32) * s_alpha
        )
        init_tryon = _safe_uint8(init_tryon)
        visible_alpha = np.maximum(visible_alpha, s_final)
        _debug_save(f"06_after_sleeve_{side}", init_tryon)

    pipeline_info.append("3LayerBlend")

    if garment_category == "dress":
        visible_u8 = (visible_alpha > 0.08).astype(np.uint8) * 255
        visible_guard = cv2.dilate(visible_u8, np.ones((5, 5), np.uint8), iterations=1)
        restore_bg = cv2.subtract(erase_mask, visible_guard)
        if int(restore_bg.sum()) > 255 * 50:
            restore_f = cv2.GaussianBlur(
                (restore_bg > 20).astype(np.float32), (7, 7), 2.0
            )[..., None]
            restore_f = np.clip(restore_f, 0.0, 1.0)
            init_tryon = _safe_uint8(
                init_tryon.astype(np.float32) * (1.0 - restore_f)
                + person_rgb.astype(np.float32) * restore_f
            )
            pipeline_info.append("EraseRestore:v16.61")

    # ── Step 7b: Shoulder-zone soft blend — DISABLED v16.7d ──
    # ShoulderSmooth was blurring the garment at shoulder boundaries.
    # With rotation align + tighter soft_mask, it's no longer needed.
    # v16: 5% at 0.35w × 0.10h — DISABLED v16.7d to prevent blur artifacts.
    # if full_pose is not None and new_sleeve_type != "long":
    #     ... ShoulderSmooth skipped ...

    _debug_save("07_after_blend", init_tryon)

    # v14: Color safety clamp — prevent extreme black/white from blend math
    init_tryon = np.clip(init_tryon, 10, 245).astype(np.uint8)

    # Update warped_mask to include sleeve coverage for downstream steps
    wm_garment = (composite_mask > 20).astype(np.uint8) * 255
    for side in sleeve_data:
        s_rgb, s_mask_f = sleeve_data[side]
        sleeve_uint8 = (s_mask_f * 255).clip(0, 255).astype(np.uint8)
        wm_garment = cv2.bitwise_or(wm_garment, sleeve_uint8)
    warped_mask = wm_garment
    warped_mask_full = cv2.bitwise_or(warped_mask_full, wm_garment)

    # ── Step 8: Layer foreground on top ──
    # Two-pass overlay:
    #   Pass 1: Arms + face (with sleeve_protect subtraction for arm-sleeve overlap)
    #   Pass 2: Hair overlay (ALWAYS on top — garment renders UNDER hair)
    # This split is critical: if sleeve_protect subtracts from hair mask,
    # hair disappears where it overlaps garment → unnatural.
    if parsing:
        # Pass 1: Arms + face (NOT hair)
        # v16.24: For DRESS, DO NOT paste arms back — dress sleeves must cover
        # the upper arms. Pasting person_rgb arms on top of the 3LayerBlend
        # result re-introduces bare skin + old-shirt pixels, which diffusion
        # then only partially overwrites (strength 0.55) → ghost/translucent
        # dress with skin bleed. Only keep face/hat/sunglasses for dress.
        # v18.16: For sleeveless dresses, arms MUST be restored — otherwise
        # the wide dress envelope (1.40×hip_w hem) leaks pink color onto the
        # arms, producing the "dress bleeding outside" effect. Long-sleeve
        # dresses keep the v16.24 behaviour (no arm restore) since sleeves
        # should cover the arms.
        if garment_category == "dress":
            if new_sleeve_type == "sleeveless":
                _fg_keys = ("left_arm", "right_arm", "face", "hat", "sunglasses")
            else:
                _fg_keys = ("face", "hat", "sunglasses")
        else:
            _fg_keys = ("left_arm", "right_arm", "face", "hat", "sunglasses")
        arm_face_mask = np.zeros((h_out, w_out), dtype=np.uint8)
        for pkey in _fg_keys:
            if pkey in parsing:
                arm_face_mask = cv2.bitwise_or(arm_face_mask, parsing[pkey])
        arm_face_mask = cv2.dilate(arm_face_mask, np.ones((3, 3), np.uint8), iterations=1)

        arm_mask = get_arm_mask(parsing)
        sleeve_protect = _build_sleeve_protect_mask(warped_mask, arm_mask)
        if sleeve_protect is not None:
            arm_face_mask = cv2.subtract(arm_face_mask, sleeve_protect)

        if arm_face_mask.sum() > 0:
            init_tryon = _apply_foreground_layer(init_tryon, person_rgb, arm_face_mask)

        # Pass 2: Hair overlay — ALWAYS on top, no sleeve_protect subtraction.
        # v16.9c: Erode hair mask to only paste core hair, avoid old-shirt bleed at edges.
        init_tryon, _ = _paste_original_hair_layer(init_tryon, person_rgb, parsing)

    # ── Restore exposed arm skin for short sleeves ──
    # v16.24: SKIP for dress — dress sleeves cover the arms, we don't want to
    # repaint skin there. This was another layer pasting person_rgb on top of
    # the dress blend, causing "chồng lớp" (ghost/translucent dress).
    if garment_category != "dress":
        init_tryon = _restore_exposed_arm_skin(
            init_tryon=init_tryon,
            person_rgb=person_rgb,
            parsing=parsing,
            warped_mask=warped_mask,
            sleeve_type=new_sleeve_type,
        )
        if new_sleeve_type == "short":
            pipeline_info.append("ArmSkin")

    init_tryon = _safe_uint8(init_tryon)
    _debug_save("08_final_cpu", init_tryon)
    # v16.11c: Also return garment_category for downstream diffusion mask building
    return init_tryon, warped_mask_full, parsing, pose_box, full_pose, pipeline_info, tps_ok, garment_category, dress_pattern_reference, pants_type


# ═══════════════════════════════════════════════════════════════════
#  Phase C: Local Diffusion Refinement (fallback refinement)
# ═══════════════════════════════════════════════════════════════════


def _build_sleeve_preserve_mask(warped_mask: np.ndarray) -> np.ndarray | None:
    """Giữ lại một dải mỏng ở vùng tay áo để diffusion không làm sai silhouette."""
    ys, xs = np.where(warped_mask > 20)
    if len(xs) < 100:
        return None

    y1, y2 = int(ys.min()), int(ys.max())
    cloth_h = max(1, y2 - y1)

    # Chỉ giữ phần trên của áo, nơi sleeve/cap-sleeve nằm
    sleeve_rows = ys < y1 + int(cloth_h * 0.38)
    if sleeve_rows.sum() < 50:
        return None

    sleeve_mask = np.zeros_like(warped_mask, dtype=np.uint8)
    sleeve_mask[ys[sleeve_rows], xs[sleeve_rows]] = 255

    # Nới nhẹ và feather để blend mượt
    sleeve_mask = cv2.dilate(sleeve_mask, np.ones((7, 7), np.uint8), iterations=1)
    sleeve_mask = cv2.GaussianBlur(sleeve_mask, (7, 7), 0)
    return sleeve_mask


def _restore_upper_body_for_pants(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    parsing: dict | None,
) -> np.ndarray:
    return _pp_restore_upper_body(output_rgb, person_rgb, parsing)


def _build_shorts_edit_band(
    shape: tuple[int, int],
    full_pose: dict | None,
) -> np.ndarray:
    return _pp_build_shorts_edit_band(shape, full_pose)


def _build_shorts_shape_mask(
    shape: tuple[int, int],
    warped_mask: np.ndarray,
    parsing: dict | None,
    full_pose: dict | None,
) -> np.ndarray:
    return _pp_build_shorts_shape_mask(
        shape, warped_mask, parsing, full_pose, fit_like=_fit_like,
    )


def _build_shorts_wear_mask(
    shape: tuple[int, int],
    parsing: dict | None,
    full_pose: dict | None,
    edit_mask: np.ndarray | None,
) -> np.ndarray:
    return _pp_build_shorts_wear_mask(
        shape, parsing, full_pose, edit_mask, fit_like=_fit_like,
    )


def _build_pants_shape_mask(
    shape: tuple[int, int],
    warped_mask: np.ndarray,
    parsing: dict | None,
    pants_type: str,
) -> np.ndarray:
    return _pp_build_pants_shape_mask(
        shape, warped_mask, parsing, pants_type, fit_like=_fit_like,
    )


def _reference_garment_color(reference_cloth_rgb: np.ndarray | None) -> np.ndarray:
    return _pp_reference_garment_color(
        reference_cloth_rgb,
        safe_uint8=_safe_uint8,
        build_cloth_mask=build_cloth_mask,
    )


def _render_reference_shorts(
    person_rgb: np.ndarray,
    reference_cloth_rgb: np.ndarray | None,
    shorts_mask: np.ndarray,
) -> np.ndarray:
    return _pp_render_reference_shorts(
        person_rgb, reference_cloth_rgb, shorts_mask,
        safe_uint8=_safe_uint8, build_cloth_mask=build_cloth_mask,
    )


def _apply_shorts_shape_guard(
    output_rgb: np.ndarray,
    person_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    warped_mask: np.ndarray,
    gen_mask_soft: np.ndarray,
    parsing: dict | None,
    full_pose: dict | None,
    reference_cloth_rgb: np.ndarray | None,
    final_wear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _pp_apply_shorts_shape_guard(
        output_rgb, person_rgb, init_tryon_rgb,
        warped_mask, gen_mask_soft, parsing, full_pose, reference_cloth_rgb,
        fit_like=_fit_like, safe_uint8=_safe_uint8, final_wear_mask=final_wear_mask,
    )


def _apply_pants_shape_guard(
    output_rgb: np.ndarray,
    init_tryon_rgb: np.ndarray,
    warped_mask: np.ndarray,
    gen_mask_soft: np.ndarray,
    parsing: dict | None,
    pants_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    return _pp_apply_pants_shape_guard(
        output_rgb, init_tryon_rgb, warped_mask, gen_mask_soft,
        parsing, pants_type, fit_like=_fit_like, safe_uint8=_safe_uint8,
    )


def _build_pants_lower_leg_fit_mask(
    shape: tuple[int, int],
    full_pose: dict | None,
    parsing: dict | None = None,
    pants_style: str = "regular",
) -> np.ndarray:
    """v19.46: bóp ống quần từ gối → mắt cá theo trục knee→ankle.
    Vùng trên gối giữ trắng (không siết) để bảo toàn hông/đùi đang tốt.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if full_pose is None:
        mask[:] = 255
        return mask
    required = ("left_hip", "right_hip")
    if any(full_pose.get(k) is None for k in required):
        mask[:] = 255
        return mask

    lh = np.array(full_pose["left_hip"], dtype=np.float32)
    rh = np.array(full_pose["right_hip"], dtype=np.float32)
    hip_w = max(24.0, float(np.linalg.norm(lh - rh)))
    hip_y = float((lh[1] + rh[1]) * 0.5)
    leg_bottom = float(h - 8)
    leg_region_all = np.zeros((h, w), dtype=np.uint8)
    if parsing:
        for k in ("left_leg", "right_leg", "pants", "skirt"):
            p = parsing.get(k)
            if p is not None:
                if p.shape[:2] != (h, w):
                    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)
                leg_region_all = cv2.bitwise_or(
                    leg_region_all, (p > 20).astype(np.uint8) * 255
                )
        ys, _ = np.where(leg_region_all > 20)
        if len(ys) > 100:
            leg_bottom = float(min(h - 4, max(ys.max(), hip_y + hip_w * 1.8)))
    leg_len = max(80.0, leg_bottom - hip_y)

    # v19.57: bóp ankle_r mọi style để cổ chân quần ôm sát mắt cá.
    # Trước: regular ankle_r=0.17·hip_w → cuff rộng gấp đôi mắt cá thật.
    if pants_style in {"skinny", "slim"}:
        knee_r = max(7, int(hip_w * 0.17)); ankle_r = max(5, int(hip_w * 0.095))
    elif pants_style in {"wide", "wide_leg", "loose"}:
        knee_r = max(10, int(hip_w * 0.25)); ankle_r = max(7, int(hip_w * 0.18))
    else:
        knee_r = max(8, int(hip_w * 0.20)); ankle_r = max(6, int(hip_w * 0.12))

    hip_cx = float((lh[0] + rh[0]) * 0.5)

    def _side_center_from_region(y: float, hip: np.ndarray) -> float | None:
        if int(cv2.countNonZero(leg_region_all)) <= 100:
            return None
        y0 = max(0, int(round(y)) - 7)
        y1 = min(h, int(round(y)) + 8)
        xs = np.where(leg_region_all[y0:y1, :] > 20)[1]
        if len(xs) < 6:
            return None
        if hip[0] < hip_cx:
            side_xs = xs[xs < hip_cx]
        else:
            side_xs = xs[xs >= hip_cx]
        if len(side_xs) < 4:
            side_xs = xs
        return float(np.median(side_xs))

    for side in ("left", "right"):
        hip = np.array(full_pose[f"{side}_hip"], dtype=np.float32)
        knee_fb = hip + np.array([0.0, leg_len * 0.48], dtype=np.float32)
        ankle_fb = hip + np.array([0.0, leg_len * 0.92], dtype=np.float32)
        knee_x = _side_center_from_region(knee_fb[1], hip)
        ankle_x = _side_center_from_region(ankle_fb[1], hip)
        if knee_x is not None:
            knee_fb[0] = knee_x
        if ankle_x is not None:
            ankle_fb[0] = ankle_x
        knee = np.array(full_pose.get(f"{side}_knee", knee_fb), dtype=np.float32)
        ankle = np.array(full_pose.get(f"{side}_ankle", ankle_fb), dtype=np.float32)
        if ankle[1] <= knee[1] + hip_w * 0.35:
            knee = knee_fb
            ankle = ankle_fb
        elif knee_x is not None or ankle_x is not None:
            if knee_x is not None:
                knee[0] = knee[0] * 0.55 + knee_x * 0.45
            if ankle_x is not None:
                ankle[0] = ankle[0] * 0.45 + ankle_x * 0.55
        steps = 16
        prev = knee
        for i in range(1, steps + 1):
            t = i / steps
            p = knee * (1.0 - t) + ankle * t
            r = int(round(knee_r * (1.0 - t) + ankle_r * t))
            cv2.line(
                mask, (int(prev[0]), int(prev[1])), (int(p[0]), int(p[1])),
                255, thickness=max(2 * r, 6), lineType=cv2.LINE_AA,
            )
            cv2.circle(mask, (int(p[0]), int(p[1])), r, 255, -1, lineType=cv2.LINE_AA)
            prev = p

    lk = np.array(full_pose["left_knee"], dtype=np.float32)
    rk = np.array(full_pose["right_knee"], dtype=np.float32)
    knee_y = int(min(lk[1], rk[1]))
    top_y = max(0, knee_y - int(hip_w * 0.35))
    mask[:top_y, :] = 255

    if parsing:
        leg_region = np.zeros((h, w), dtype=np.uint8)
        for k in ("left_leg", "right_leg", "pants", "skirt"):
            p = parsing.get(k)
            if p is not None:
                if p.shape[:2] != (h, w):
                    p = cv2.resize(p, (w, h), interpolation=cv2.INTER_NEAREST)
                leg_region = cv2.bitwise_or(leg_region, (p > 20).astype(np.uint8) * 255)
        if int(cv2.countNonZero(leg_region)) > 100:
            leg_region = cv2.dilate(leg_region, np.ones((9, 9), np.uint8), iterations=1)
            lower_only = np.zeros((h, w), dtype=np.uint8)
            lower_only[top_y:, :] = 255
            leg_region = cv2.bitwise_and(leg_region, lower_only)
            mask = cv2.bitwise_or(mask, leg_region)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return (mask > 20).astype(np.uint8) * 255


def _build_pants_diffusion_seed(
    init_tryon: np.ndarray,
    gen_mask_soft: np.ndarray,
    reference_cloth_rgb: np.ndarray | None,
    pants_type: str = "regular",
    warped_mask: np.ndarray | None = None,
    cleanup_mask: np.ndarray | None = None,
    cleanup_fill_rgb: np.ndarray | None = None,
) -> np.ndarray:
    return _pp_build_pants_diffusion_seed(
        init_tryon, gen_mask_soft, reference_cloth_rgb,
        safe_uint8=_safe_uint8, build_cloth_mask=build_cloth_mask,
        pants_type=pants_type, warped_mask=warped_mask,
        cleanup_mask=cleanup_mask, cleanup_fill_rgb=cleanup_fill_rgb,
    )


def _run_local_diffusion_refinement(
    init_tryon: np.ndarray,
    person_rgb: np.ndarray,
    warped_mask: np.ndarray,
    parsing: dict | None,
    pose_box,
    full_pose: dict | None,
    dress_pattern_reference: np.ndarray | None,
    style_prompt: str,
    gen_steps: int,
    gen_guidance: float,
    preserve_strength: float,
    refiner_mode: str,
    cloth_type: str,
    garment_category: str = "top",
    new_sleeve_type: str = "long",
    pants_type: str = "regular",
    reference_cloth_rgb: np.ndarray | None = None,
    gemini_negative_extra: str = "",
    top_subtype: str = "",
) -> tuple[np.ndarray, str, str, list[str]]:
    """v16.9: SOTA-inspired agnostic-mask diffusion.

    Key insight from GP-VTON/IDM-VTON/CatVTON research:
    - Build agnostic mask: erase entire torso garment + upper arms + neck
    - Keep face, hair, hands, lower body as context (NOT masked)
    - Let diffusion regenerate the full garment region with higher denoise
    - TPS output serves as init_image (spatial guide), not final output
    - Repaint composite: paste back original pixels outside mask after generation
    - Include 15-20px skin border in mask for natural garment-skin transition
    """
    pipeline_info = []
    warning_msg = ""

    h, w = init_tryon.shape[:2]
    binary_mask = (warped_mask > 20).astype(np.uint8) * 255
    shorts_wear_mask_for_seed = None
    shorts_seed_cleanup_mask = None

    if binary_mask.sum() < 500:
        return init_tryon, "", "Garment mask too small for diffusion", pipeline_info

    # ── AGNOSTIC MASK: erase garment region so diffusion can regenerate ──
    diffusion_base_mask = binary_mask
    if os.getenv("VTON_USE_HUMAN_MASK_PRIOR", "1").strip() == "1":
        human_prior_mask, human_prior_ok = _build_human_tryon_prior_mask(
            binary_mask,
            parsing,
            full_pose,
            garment_category,
            sleeve_type=new_sleeve_type,
            accessory_subtype=locals().get("accessory_subtype", ""),
            top_subtype=locals().get("top_subtype", ""),
        )
        if human_prior_ok:
            diffusion_base_mask = human_prior_mask
            pipeline_info.append("HumanMaskPrior:TripVVT:v18.0")
            _debug_save("09a_human_tryon_prior", human_prior_mask, is_mask=True)
        else:
            pipeline_info.append("HumanMaskPriorFallback:v18.0")

    gen_mask = diffusion_base_mask.copy()

    if garment_category == "pants":
        # For pants: erase existing pants/legs + warped pants region
        if parsing:
            old_pants = get_pants_mask(parsing)
            if old_pants is not None:
                gen_mask = cv2.bitwise_or(gen_mask, old_pants)
        # v19.25: stronger erase — union of pants/skirt/leg/belt parsing,
        # clipped to hip→knee band for shorts so we never paint below the knee.
        if parsing:
            old_lower = np.zeros((h, w), dtype=np.uint8)
            for _ok in ("pants", "skirt", "belt", "left_leg", "right_leg"):
                _op = parsing.get(_ok)
                if _op is not None:
                    old_lower = cv2.bitwise_or(old_lower, _op)
            if pants_type == "shorts" and full_pose is not None:
                old_lower = cv2.bitwise_and(
                    old_lower,
                    _build_shorts_edit_band((h, w), full_pose),
                )
            if int(cv2.countNonZero(old_lower)) > 0:
                gen_mask = cv2.bitwise_or(gen_mask, old_lower)
                pipeline_info.append("PantsOldJeansErase:v19.28")
        # PROTECT: face/hair/arms/torso — only legs change
        protect_keys = ("face", "hair", "hat", "sunglasses",
                        "upper_clothes", "dress", "left_arm", "right_arm")
        # v19.21: extra hard-clip against upper_clothes so the 21×21 dilate below
        # cannot push gen_mask up into the red top — that caused a black bleed
        # band onto the top edge in the hand-on-hip pose.
        if parsing:
            _ub_zone = np.zeros_like(gen_mask)
            for _k in ("upper_clothes", "dress", "left_arm", "right_arm",
                       "face", "hair"):
                _p = parsing.get(_k)
                if _p is not None:
                    _ub_zone = cv2.bitwise_or(_ub_zone, _p)
            if int(cv2.countNonZero(_ub_zone)) > 0:
                _ub_zone = cv2.dilate(_ub_zone, np.ones((9, 9), np.uint8), iterations=1)
                gen_mask = cv2.subtract(gen_mask, _ub_zone)
                pipeline_info.append("PantsUpperBodyHardClip:v19.21")

    elif garment_category == "dress":
        # v16.55: Generate dress like the TOP path: inpaint the FULL garment
        # footprint instead of only a narrow edge ring. The ring-only v16.53
        # avoided the duplicate layer, but it left the TPS warp almost intact,
        # so the result looked pasted on. Here diffusion owns the full warped
        # dress+sleeve mask and can add body-following folds/shading, while the
        # final DressSingleLayerClip still prevents pixels outside the footprint
        # from becoming a second dress.
        gen_mask = diffusion_base_mask.copy()
        gen_mask = cv2.morphologyEx(gen_mask, cv2.MORPH_CLOSE,
                                    np.ones((7, 7), np.uint8))
        gen_mask = cv2.dilate(gen_mask, np.ones((5, 5), np.uint8), iterations=1)

        # Keep neckline/collar cleanup close to the garment only. This mirrors
        # the top path's natural transition without adding arms/legs/skirt as a
        # broad separate dress silhouette.
        # v18.3: shrink bridge 17→6px to stop diffusion painting fake collars
        # onto bare neck/chest skin above scoop/cowl necklines. Only neck pixels
        # touching the garment seam are added; rest is left for protect to keep.
        if parsing:
            neck_diff = get_neck_mask(parsing)
            if neck_diff is not None:
                near_garment = cv2.dilate(binary_mask, np.ones((6, 6), np.uint8), iterations=1)
                gen_mask = cv2.bitwise_or(gen_mask, cv2.bitwise_and(neck_diff, near_garment))
        pipeline_info.append("DressFullGenMask:v18.3")

        # v20.8: union the OLD garment parsing into gen_mask so diffusion is
        # forced to repaint the original dress/skirt/upper_clothes footprint
        # even where it extends past the warped new-dress silhouette (e.g.
        # wide flared yellow dress behind narrower brown dress). Without this
        # the old garment leaks through behind the new one.
        if parsing:
            _old_dress_mask = np.zeros((h, w), dtype=np.uint8)
            for _k in ("dress", "skirt", "upper_clothes", "belt"):
                _p = parsing.get(_k)
                if _p is not None:
                    _old_dress_mask = cv2.bitwise_or(_old_dress_mask, _p)
            # Pull legs only where they overlap the old-garment shadow, so we
            # don't repaint full thighs/calves when the new dress is short.
            for _k in ("left_leg", "right_leg"):
                _p = parsing.get(_k)
                if _p is not None:
                    _leg_near = cv2.bitwise_and(
                        _p,
                        cv2.dilate(_old_dress_mask, np.ones((17, 17), np.uint8), iterations=1),
                    )
                    _old_dress_mask = cv2.bitwise_or(_old_dress_mask, _leg_near)
            if int(cv2.countNonZero(_old_dress_mask)) > 80:
                _old_dress_mask = cv2.dilate(
                    _old_dress_mask, np.ones((11, 11), np.uint8), iterations=1
                )
                _old_dress_mask = cv2.morphologyEx(
                    _old_dress_mask, cv2.MORPH_CLOSE,
                    np.ones((9, 9), np.uint8), iterations=1,
                )
                gen_mask = cv2.bitwise_or(gen_mask, _old_dress_mask)
                _debug_save("09d_dress_old_garment_erase_mask", _old_dress_mask, is_mask=True)
                pipeline_info.append("DressOldGarmentErase:v20.8")

        # Do not protect arms for dress: if sleeve warp produced sleeve mask,
        # arms inside binary_mask are part of the generated garment. Regions
        # outside binary_mask are restored after diffusion.
        protect_keys = ("face", "hair", "hat", "sunglasses",
                        "left_shoe", "right_shoe")

        # v18.4: protect bare neck skin (neck pixels OUTSIDE the warped garment).
        # Symptom: diffusion paints fake dark collars/scarves above scoop & cowl
        # necklines. Cause: SD inpaint imagines a collar to "complete" the dress
        # where TPS left bare skin. Fix: subtract neck-outside-garment from
        # gen_mask so diffusion only repaints inside binary_mask. Pixels of neck
        # that are inside the warped garment (turtleneck case) stay editable.
        # v18.7: also subtract chest/decolletage skin. SegFormer has no chest
        # label — get_skin_mask infers it by dilating face downward. When source
        # has high neckline but model originally wore a V/scoop top, the warped
        # garment top covers model's bare upper chest and diffusion paints a
        # dark "collar box" there. Subtracting skin_mask ∩ binary_mask (where
        # skin pixels coincide with the warped garment area) blocks that paint.
        if parsing:
            neck_skin = get_neck_mask(parsing)
            if neck_skin is not None:
                neck_outside = cv2.bitwise_and(
                    neck_skin, cv2.bitwise_not(binary_mask)
                )
                neck_outside = cv2.dilate(
                    neck_outside, np.ones((3, 3), np.uint8), iterations=1
                )
                gen_mask = cv2.subtract(gen_mask, neck_outside)
                pipeline_info.append("DressNeckSkinProtect:v18.4")
            # v18.11: build a CHEST-ONLY skin mask. Using get_skin_mask
            # included arm pixels — the shoulder/arm joint then got subtracted
            # from gen_mask, leaving an un-paintable notch at the shoulder
            # that bared the model's skin in the final composite. We rebuild
            # the protect mask from face dilated downward, then subtract arm
            # and clothing parsing labels so only neck+decolletage remain.
            face_parse = parsing.get("face")
            if face_parse is not None:
                # v18.21: shrink chest protect — was carving the new dress
                # neckline area too aggressively, leaving init-only patches
                # where old garment colour leaked through after diffusion.
                chest_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 25))
                chest_skin = cv2.dilate(face_parse, chest_kernel, iterations=1)
                for _k in ("left_arm", "right_arm", "left_leg", "right_leg",
                           "hair", "hat", "background", "bag", "scarf",
                           "sunglasses", "upper_clothes", "dress"):
                    if _k in parsing:
                        chest_skin = cv2.subtract(chest_skin, parsing[_k])
                ys_w, _ = np.where(binary_mask > 0)
                if len(ys_w) > 100:
                    g_top = int(ys_w.min())
                    g_bot = int(ys_w.max())
                    g_h = max(1, g_bot - g_top)
                    top_strip = np.zeros_like(binary_mask)
                    top_strip[g_top : g_top + int(g_h * 0.22), :] = 255
                    chest_in_top = cv2.bitwise_and(
                        cv2.bitwise_and(chest_skin, binary_mask), top_strip
                    )
                    chest_in_top = cv2.dilate(
                        chest_in_top, np.ones((3, 3), np.uint8), iterations=1
                    )
                    if int(chest_in_top.sum()) > 255 * 30:
                        gen_mask = cv2.subtract(gen_mask, chest_in_top)
                        pipeline_info.append("DressChestSkinProtect:v18.21")

            # v20.7: protect arm pixels OUTSIDE the dress silhouette so
            # diffusion can't paint dress fabric (or a blurry half-sleeve)
            # over a hand/forearm that lies outside the warped garment
            # mask. This is the same idea as the TOP pipeline already does
            # for face/arms via protect_keys. Arms inside binary_mask stay
            # editable so a long sleeve can still be generated.
            _arm_protect = np.zeros_like(binary_mask)
            for _ak in ("left_arm", "right_arm"):
                _av = parsing.get(_ak)
                if _av is not None:
                    _arm_protect = cv2.bitwise_or(_arm_protect, _av)
            if int(_arm_protect.sum()) > 255 * 50:
                _arm_outside = cv2.bitwise_and(
                    _arm_protect, cv2.bitwise_not(binary_mask)
                )
                _arm_outside = cv2.dilate(
                    _arm_outside, np.ones((3, 3), np.uint8), iterations=1
                )
                gen_mask = cv2.subtract(gen_mask, _arm_outside)
                pipeline_info.append("DressArmOutsideProtect:v20.7")

    else:
        # For tops: erase existing upper clothes + warped top region
        _is_hoodie_top = (top_subtype or "").lower() == "hoodie"
        if parsing:
            old_clothes = get_clothing_mask(parsing)
            if old_clothes is not None:
                if (
                    _is_hoodie_top
                    and os.getenv("VTON_HOODIE_CLIP_OLD_CLOTHES", "0").strip().lower()
                    in {"1", "true", "yes", "on"}
                ):
                    # Do not let the source/person hoodie silhouette expand the
                    # edit area after the category builder already made a tight
                    # hoodie prior. This was reintroducing wide shoulders,
                    # sleeves and waist through the old-clothes parsing mask.
                    _near_hoodie = cv2.dilate(
                        diffusion_base_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                        iterations=1,
                    )
                    old_clothes = cv2.bitwise_and(old_clothes, _near_hoodie)
                    pipeline_info.append("HoodieOldClothesClip:v22.2")
                gen_mask = cv2.bitwise_or(gen_mask, old_clothes)
            neck_diff = get_neck_mask(parsing)
            if neck_diff is not None:
                if _is_hoodie_top:
                    _near_neck = cv2.dilate(
                        diffusion_base_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                        iterations=1,
                    )
                    neck_diff = cv2.bitwise_and(neck_diff, _near_neck)
                gen_mask = cv2.bitwise_or(gen_mask, neck_diff)
        # PROTECT: face/hair/hat/sunglasses + lower body (pants untouched)
        protect_keys = ("face", "hair", "hat", "sunglasses",
                        "pants", "skirt", "left_leg", "right_leg",
                        "left_shoe", "right_shoe")

    # Dilate mask (dress already dilated above with larger kernel)
    # v19.24: pants need a gentler dilation; shorts sit right under the shirt
    # and 21x21 was pushing the mask into the upper body → diffusion repainted
    # the red shirt as a flat patch.
    if garment_category == "pants":
        _pk = 7 if pants_type == "shorts" else 11
        gen_mask = cv2.morphologyEx(gen_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        gen_mask = cv2.dilate(gen_mask, np.ones((_pk, _pk), np.uint8), iterations=1)
        pipeline_info.append(f"PantsDilate:{_pk}:v19.24")
        # v19.24: clip shorts mask to hip→mid-thigh band so diffusion can't
        # repaint anything below the knee or above the waistband.
        if pants_type == "shorts" and full_pose is not None:
            _allowed = _build_shorts_edit_band((h, w), full_pose)
            if parsing:
                _keep_old = np.zeros_like(gen_mask)
                for _kk in ("pants", "skirt", "belt"):
                    _kp = parsing.get(_kk)
                    if _kp is not None:
                        _keep_old = cv2.bitwise_or(_keep_old, _kp)
                _allowed = cv2.bitwise_or(_allowed, cv2.bitwise_and(_keep_old, _allowed))
            gen_mask = cv2.bitwise_and(gen_mask, _allowed)
            pipeline_info.append("ShortsRegionClip:v19.28")
        # v19.24: subtract upper body AGAIN after dilation (not only before),
        # because dilation re-introduces a few pixels onto the shirt.
        if parsing:
            _upper_protect = np.zeros((h, w), dtype=np.uint8)
            for _uk in ("upper_clothes", "dress", "left_arm", "right_arm", "face", "hair"):
                _up = parsing.get(_uk)
                if _up is not None:
                    _upper_protect = cv2.bitwise_or(_upper_protect, _up)
            _upper_protect = cv2.dilate(_upper_protect, np.ones((17, 17), np.uint8), iterations=1)
            gen_mask = cv2.subtract(gen_mask, _upper_protect)
            pipeline_info.append("PantsPostDilateUpperProtect:v19.24")
    elif garment_category != "dress":
        if (top_subtype or "").lower() == "hoodie":
            # Hoodie now follows the top/t-shirt diffusion contract: keep a
            # usable sleeve/torso edit band, but keep dilation modest so the
            # model cannot inflate the flat-lay hoodie into a boxy torso.
            gen_mask = cv2.morphologyEx(gen_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            gen_mask = cv2.dilate(gen_mask, np.ones((9, 9), np.uint8), iterations=1)
            pipeline_info.append("HoodieTopLikeGenMask:v23.1:dilate9")
        else:
            gen_mask = cv2.dilate(gen_mask, np.ones((21, 21), np.uint8), iterations=1)

    # v19.47 TopHairUnderlap (build EARLY) — before protect, so we can
    # carve hair_underlap OUT of the hair-protect region, and propagate
    # into warped_mask + binary_mask + gen_mask.
    _hair_underlap = np.zeros((h, w), dtype=np.uint8)
    # v19.51: chỉ áp dụng cho "top". Dress có pattern-injection riêng
    # (_preserve_dress_pattern_gpu + _apply_dress_primary_identity_guard)
    # và sample màu từ binary_mask — nếu mở rộng binary_mask sang vùng tóc/da
    # sẽ làm pattern extractor "tưởng tượng" sọc dọc → kết quả nhiễu loạn.
    _is_hoodie_top = garment_category == "top" and (top_subtype or "").lower() == "hoodie"
    if garment_category == "top":
        _hair_underlap = _build_top_hair_underlap_mask(
            (h, w), parsing, full_pose, base_torso_mask=binary_mask,
        )

    # Apply protection mask — v19.48: hair được protect+dilate RIÊNG,
    # rồi mới subtract _hair_underlap (sau dilate) → hair không regrow
    # đè vào vùng underlap đã mở ra.
    if parsing:
        protect_mask = np.zeros((h, w), dtype=np.uint8)
        hair_protect = np.zeros((h, w), dtype=np.uint8)
        for pkey in protect_keys:
            if pkey not in parsing:
                continue
            part = parsing[pkey]
            if pkey == "hair" and garment_category == "top":
                hair_protect = cv2.bitwise_or(hair_protect, part)
            else:
                protect_mask = cv2.bitwise_or(protect_mask, part)
        _protect_kernel = 13 if garment_category == "pants" else 5
        protect_mask = cv2.dilate(
            protect_mask, np.ones((_protect_kernel, _protect_kernel), np.uint8), iterations=1,
        )
        if garment_category == "top" and int(cv2.countNonZero(hair_protect)) > 0:
            hair_protect = cv2.dilate(
                hair_protect, np.ones((7, 7), np.uint8), iterations=1,
            )
            if int(cv2.countNonZero(_hair_underlap)) > 20:
                _underlap_soft = cv2.dilate(
                    _hair_underlap, np.ones((5, 5), np.uint8), iterations=1,
                )
                hair_protect = cv2.subtract(hair_protect, _underlap_soft)
            protect_mask = cv2.bitwise_or(protect_mask, hair_protect)
        gen_mask = cv2.subtract(gen_mask, protect_mask)

    # v19.48 TopHairUnderlap — propagate underlap vào warped/binary/gen mask
    # và seed init_tryon từ vùng warp thực sự (KHÔNG blur cả init_tryon, vì
    # quanh hair là skin/tóc — blur sẽ smear skin vào underlap).
    if garment_category == "top" and int(cv2.countNonZero(_hair_underlap)) > 20:
        gen_mask = cv2.bitwise_or(gen_mask, _hair_underlap)
        warped_mask = cv2.bitwise_or(warped_mask, _hair_underlap)
        # Snapshot binary_mask trước khi mở rộng để dùng làm garment-source
        _garment_src = (binary_mask > 20)
        binary_mask = cv2.bitwise_or(binary_mask, _hair_underlap)
        if int(_garment_src.sum()) > 200:
            _mean_rgb = init_tryon[_garment_src].mean(axis=0).astype(np.uint8)
            _garment_only = np.where(_garment_src[..., None], init_tryon, _mean_rgb)
            _seed_src = cv2.GaussianBlur(_garment_only, (11, 11), 2.5)
        else:
            _seed_src = cv2.GaussianBlur(init_tryon, (15, 15), 4.0)
        _alpha = cv2.GaussianBlur(
            (_hair_underlap > 20).astype(np.float32), (7, 7), 1.5,
        )[..., None]
        _alpha = np.clip(_alpha * 0.72, 0.0, 0.72)
        init_tryon = _safe_uint8(
            init_tryon.astype(np.float32) * (1.0 - _alpha)
            + _seed_src.astype(np.float32) * _alpha
        )
        _debug_save("09_top_hair_underlap_mask", _hair_underlap, is_mask=True)
        pipeline_info.append("TopHairUnderlap:v19.51")

    if garment_category == "top" and (top_subtype or "").lower() == "hoodie":
        # Final cap after old-clothes/neck/hair unions. Build it from pose +
        # parsing, not from the broad human prior, so hoodie diffusion keeps
        # top/t-shirt sleeve separation without growing into a square block.
        _hoodie_cap_base = (binary_mask > 20).astype(np.uint8) * 255
        if full_pose is not None:
            _hoodie_cap_base = _build_hoodie_pose_fit_mask(
                (h, w), full_pose, parsing, binary_mask,
            )
        if int(cv2.countNonZero(_hair_underlap)) > 20:
            _hoodie_cap_base = cv2.bitwise_or(_hoodie_cap_base, _hair_underlap)
        _hoodie_cap = cv2.dilate(
            _hoodie_cap_base,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        if os.getenv("VTON_HOODIE_FINAL_CAP", "1").strip().lower() not in {"0", "false", "no", "off"}:
            gen_mask = cv2.bitwise_and(gen_mask, _hoodie_cap)
            pipeline_info.append("HoodieHybridMaskCap:v23.1")
        else:
            pipeline_info.append("HoodieHybridMaskCapSkip:v23.1")
        _debug_save("09a1_hoodie_mask_cap", _hoodie_cap, is_mask=True)

    if garment_category == "pants" and pants_type != "shorts":
        pants_shape_mask = _build_pants_shape_mask((h, w), binary_mask, parsing, pants_type)
        if int(cv2.countNonZero(pants_shape_mask)) > 255:
            gen_mask = cv2.bitwise_or(gen_mask, pants_shape_mask)
            source_shape = cv2.dilate(
                ((binary_mask > 20).astype(np.uint8)) * 255,
                np.ones((9, 9), np.uint8), iterations=1,
            )
            gen_mask = cv2.bitwise_or(gen_mask, cv2.bitwise_and(source_shape, pants_shape_mask))

        # v19.34: Build a pose-driven leg mask (2 separate leg trapezoids from
        # hip to ankle) independent of the TPS warp footprint, which collapses
        # to a sliver when source garment is much wider than the person's legs.
        # This is what makes the mask look like 2 separate legs instead of a T.
        if full_pose is not None:
            try:
                lh = np.array(full_pose.get("left_hip", [0, 0]), dtype=np.float32)
                rh = np.array(full_pose.get("right_hip", [0, 0]), dtype=np.float32)
                lk = np.array(full_pose.get("left_knee", lh), dtype=np.float32)
                rk = np.array(full_pose.get("right_knee", rh), dtype=np.float32)
                la = np.array(full_pose.get("left_ankle", lk), dtype=np.float32)
                ra = np.array(full_pose.get("right_ankle", rk), dtype=np.float32)
                if pants_type == "cropped":
                    la = (lk + la) * 0.5 + (la - lk) * 0.3
                    ra = (rk + ra) * 0.5 + (ra - rk) * 0.3
                hip_w = float(np.linalg.norm(lh - rh))
                if hip_w < 12:
                    hip_w = w * 0.22
                # v19.40: slightly tighter legs (was 0.32/0.26/0.22). The
                # wider trapezoid made shape_mask cover the foot/sandal area.
                leg_hw_hip = max(7.0, hip_w * 0.28)
                leg_hw_knee = max(6.0, hip_w * 0.22)
                leg_hw_ankle = max(5.0, hip_w * 0.18)
                pose_leg = np.zeros((h, w), dtype=np.uint8)
                def _leg_poly(top, mid, bot, hw_top, hw_mid, hw_bot):
                    pts = np.array([
                        [top[0] - hw_top, top[1]],
                        [top[0] + hw_top, top[1]],
                        [mid[0] + hw_mid, mid[1]],
                        [bot[0] + hw_bot, bot[1]],
                        [bot[0] - hw_bot, bot[1]],
                        [mid[0] - hw_mid, mid[1]],
                    ], dtype=np.int32)
                    cv2.fillPoly(pose_leg, [pts], 255)
                _leg_poly(lh, lk, la, leg_hw_hip, leg_hw_knee, leg_hw_ankle)
                _leg_poly(rh, rk, ra, leg_hw_hip, leg_hw_knee, leg_hw_ankle)
                # Add a waistband rectangle to connect both legs at the top
                top_y = int(min(lh[1], rh[1]) - hip_w * 0.15)
                bot_y = int(max(lh[1], rh[1]) + hip_w * 0.05)
                left_x = int(min(lh[0], rh[0]) - leg_hw_hip)
                right_x = int(max(lh[0], rh[0]) + leg_hw_hip)
                cv2.rectangle(pose_leg, (left_x, top_y), (right_x, bot_y), 255, -1)
                # Smooth + dilate slightly so SD has room to render fabric
                pose_leg = cv2.dilate(pose_leg, np.ones((5, 5), np.uint8), iterations=1)
                gen_mask = cv2.bitwise_or(gen_mask, pose_leg)
                _debug_save("09d_pose_leg_mask", pose_leg, is_mask=True)
                pipeline_info.append("PoseLegMask:v19.34")
            except Exception as exc:
                pipeline_info.append(f"PoseLegMaskSkip:{type(exc).__name__}")

        # Re-apply upper-body hard clip so the pose leg mask cannot creep into
        # the top region near the waist.
        if parsing:
            _ub3 = np.zeros((h, w), dtype=np.uint8)
            for _k in ("upper_clothes", "dress", "left_arm", "right_arm",
                       "face", "hair"):
                _p = parsing.get(_k)
                if _p is not None:
                    _ub3 = cv2.bitwise_or(_ub3, _p)
            if int(cv2.countNonZero(_ub3)) > 0:
                _ub3 = cv2.dilate(_ub3, np.ones((7, 7), np.uint8), iterations=1)
                gen_mask = cv2.subtract(gen_mask, _ub3)

        # v19.35a: PARSING LEG GUARD — clip the wide pose trapezoid to the
        # actual leg silhouette from SegFormer (when reliable) so the mask
        # doesn't bleed outside the person. Used as a soft guard via OR with
        # a generous body envelope, not a strict AND (parser can fail on
        # dark-coloured tight pants).
        if parsing is not None and full_pose is not None:
            try:
                ll = parsing.get("left_leg")
                rl = parsing.get("right_leg")
                if ll is not None or rl is not None:
                    leg_seg = np.zeros((h, w), dtype=np.uint8)
                    if ll is not None:
                        leg_seg = cv2.bitwise_or(leg_seg, ll)
                    if rl is not None:
                        leg_seg = cv2.bitwise_or(leg_seg, rl)
                    # Include existing pants in the leg-region envelope so
                    # parser misses don't cut the mask.
                    for _k in ("pants", "skirt"):
                        _p = parsing.get(_k)
                        if _p is not None:
                            leg_seg = cv2.bitwise_or(leg_seg, _p)
                    if int(cv2.countNonZero(leg_seg)) > 200:
                        # Dilate so we keep a few px of slack outside the
                        # silhouette for fabric drape.
                        leg_env = cv2.dilate(
                            leg_seg, np.ones((15, 15), np.uint8), iterations=2,
                        )
                        # Only clip the LOWER half of gen_mask (below hip)
                        # so the waistband area stays attached to the body.
                        lh_y = float(min(full_pose.get("left_hip", [0, h])[1],
                                         full_pose.get("right_hip", [0, h])[1]))
                        upper_keep = np.zeros((h, w), dtype=np.uint8)
                        upper_keep[: int(lh_y) + 5, :] = 255
                        clip_zone = cv2.bitwise_or(leg_env, upper_keep)
                        gen_mask = cv2.bitwise_and(gen_mask, clip_zone)
                        pipeline_info.append("PantsParsingLegGuard:v19.35")
            except Exception as exc:
                pipeline_info.append(f"PantsParsingLegGuardSkip:{type(exc).__name__}")

        # v19.35b: CROTCH GAP CARVE — subtract a V-shaped wedge between the
        # two legs so diffusion sees two separate pant tubes instead of one
        # blob. Without this the legs render as a connected "skirt" shape.
        if full_pose is not None:
            try:
                lh = np.array(full_pose.get("left_hip", [0, 0]), dtype=np.float32)
                rh = np.array(full_pose.get("right_hip", [0, 0]), dtype=np.float32)
                lk = np.array(full_pose.get("left_knee", lh), dtype=np.float32)
                rk = np.array(full_pose.get("right_knee", rh), dtype=np.float32)
                hip_center = (lh + rh) * 0.5
                knee_center = (lk + rk) * 0.5
                hip_width = float(np.linalg.norm(lh - rh))
                if hip_width < 12:
                    hip_width = w * 0.22
                leg_height = float(abs(knee_center[1] - hip_center[1]))
                if leg_height < 12:
                    leg_height = h * 0.18

                # v19.43: shorten gap to upper thigh only — was 0.10–0.58
                # which still extended past mid-thigh as a long white tear.
                # Now 0.10–0.42 keeps crotch separation visible at the top
                # but legs touch / merge naturally below mid-thigh.
                top_y = int(hip_center[1] + leg_height * 0.10)
                bottom_y = int(hip_center[1] + leg_height * 0.42)
                top_half = max(4, int(hip_width * 0.10))
                bottom_half = max(2, int(hip_width * 0.04))
                cx_top = int(hip_center[0])
                cx_bottom = int(knee_center[0])

                crotch_gap = np.zeros((h, w), dtype=np.uint8)
                pts = np.array([
                    [cx_top - top_half, top_y],
                    [cx_top + top_half, top_y],
                    [cx_bottom + bottom_half, bottom_y],
                    [cx_bottom - bottom_half, bottom_y],
                ], dtype=np.int32)
                cv2.fillPoly(crotch_gap, [pts], 255)
                crotch_gap = cv2.GaussianBlur(crotch_gap, (9, 9), 2.5)
                crotch_gap = (crotch_gap > 30).astype(np.uint8) * 255
                gen_mask = cv2.bitwise_and(gen_mask, cv2.bitwise_not(crotch_gap))
                _debug_save("09e_crotch_gap", crotch_gap, is_mask=True)
                pipeline_info.append("PantsCrotchGapCarve:v19.40")
            except Exception as exc:
                pipeline_info.append(f"CrotchGapSkip:{type(exc).__name__}")

        # v19.40: Shoe/foot protect — don't repaint sandals or bare feet.
        # gen_mask was bleeding ~20px below the ankle, painting denim onto
        # the shoe area.
        # v19.44: strengthened with pose-derived heel/foot fallback + a
        # wider 15×15 dilation, and added ankle taper to bóp ống quần ở
        # cổ chân (skinny/straight jeans).
        try:
            shoe_protect = np.zeros((h, w), dtype=np.uint8)
            if parsing:
                for _k in ("left_shoe", "right_shoe"):
                    _p = parsing.get(_k)
                    if _p is not None:
                        shoe_protect = cv2.bitwise_or(shoe_protect, _p)
            # v19.44: pose-derived fallback even when parsing has shoes —
            # add ankle/heel/foot_index circles so soft mask edges below
            # the ankle line are also cut.
            if full_pose is not None:
                for _side in ("left", "right"):
                    for _kp in (f"{_side}_ankle", f"{_side}_heel", f"{_side}_foot_index"):
                        _pt = full_pose.get(_kp)
                        if _pt is not None:
                            cv2.circle(shoe_protect, (int(_pt[0]), int(_pt[1])), 14, 255, -1, lineType=cv2.LINE_AA)
            if int(cv2.countNonZero(shoe_protect)) > 80:
                shoe_protect = cv2.dilate(shoe_protect, np.ones((15, 15), np.uint8), iterations=1)
                gen_mask = cv2.subtract(gen_mask, shoe_protect)
                _debug_save("09g_foot_protect_mask", shoe_protect, is_mask=True)
                pipeline_info.append("PantsFootProtect:v19.44")
            elif full_pose is not None:
                la_y = full_pose.get("left_ankle", [0, 0])[1]
                ra_y = full_pose.get("right_ankle", [0, 0])[1]
                bot_limit = int(max(la_y, ra_y) + 10)
                if 0 < bot_limit < h:
                    gen_mask[bot_limit:, :] = 0
                    pipeline_info.append("PantsAnkleClip:v19.40")
        except Exception as exc:
            pipeline_info.append(f"PantsShoeProtectSkip:{type(exc).__name__}")

        # v19.44: Ankle taper — bóp chiều ngang ống quần ở cổ chân. AnkleClip
        # chỉ cắt dọc, không siết ngang nên ống vẫn phình rộng quanh mắt cá.
        # Taper dùng 2 capsule hip→knee (rộng) + knee→ankle (hẹp dần).
        try:
            if (
                pants_type != "shorts"
                and full_pose is not None
                and full_pose.get("left_hip") is not None
                and full_pose.get("right_hip") is not None
            ):
                _lh = np.array(full_pose["left_hip"], dtype=np.float32)
                _rh = np.array(full_pose["right_hip"], dtype=np.float32)
                _hip_w = max(20.0, float(np.linalg.norm(_lh - _rh)))
                _hip_y = float((_lh[1] + _rh[1]) * 0.5)
                _leg_bottom = float(h - 8)
                if parsing:
                    _old_lower = np.zeros((h, w), dtype=np.uint8)
                    for _k in ("left_leg", "right_leg", "pants", "skirt"):
                        _p = parsing.get(_k)
                        if _p is not None:
                            _old_lower = cv2.bitwise_or(_old_lower, _p)
                    _ys, _ = np.where(_old_lower > 20)
                    if len(_ys) > 100:
                        _leg_bottom = float(min(h - 4, max(_ys.max(), _hip_y + _hip_w * 1.8)))
                _leg_len = max(80.0, _leg_bottom - _hip_y)
                if pants_style in {"wide", "wide_leg", "loose"}:
                    _knee_r = max(10, int(_hip_w * 0.25))
                    _ankle_r = max(7, int(_hip_w * 0.18))
                elif pants_style in {"skinny", "slim"}:
                    _knee_r = max(7, int(_hip_w * 0.17))
                    _ankle_r = max(5, int(_hip_w * 0.095))
                else:
                    _knee_r = max(8, int(_hip_w * 0.20))
                    _ankle_r = max(6, int(_hip_w * 0.12))
                ankle_taper = np.zeros((h, w), dtype=np.uint8)
                for _side in ("left", "right"):
                    _hip = np.array(full_pose[f"{_side}_hip"], dtype=np.float32)
                    _knee_fb = _hip + np.array([0.0, _leg_len * 0.48], dtype=np.float32)
                    _ankle_fb = _hip + np.array([0.0, _leg_len * 0.92], dtype=np.float32)
                    _knee = np.array(full_pose.get(f"{_side}_knee", _knee_fb), dtype=np.float32)
                    _ankle = np.array(full_pose.get(f"{_side}_ankle", _ankle_fb), dtype=np.float32)
                    if _ankle[1] <= _knee[1] + _hip_w * 0.35:
                        _knee = _knee_fb
                        _ankle = _ankle_fb
                    cv2.line(
                        ankle_taper,
                        (int(_hip[0]), int(_hip[1])),
                        (int(_knee[0]), int(_knee[1])),
                        255, max(_knee_r * 2, 12), lineType=cv2.LINE_AA,
                    )
                    cv2.line(
                        ankle_taper,
                        (int(_knee[0]), int(_knee[1])),
                        (int(_ankle[0]), int(_ankle[1])),
                        255, max(_ankle_r * 2, 8), lineType=cv2.LINE_AA,
                    )
                    cv2.circle(ankle_taper, (int(_knee[0]), int(_knee[1])), _knee_r, 255, -1, lineType=cv2.LINE_AA)
                    cv2.circle(ankle_taper, (int(_ankle[0]), int(_ankle[1])), _ankle_r, 255, -1, lineType=cv2.LINE_AA)
                # Waistband bridge ở hông để không cắt vùng cạp
                _top_y = int(min(_lh[1], _rh[1]) - _hip_w * 0.30)
                _bot_y = int(max(_lh[1], _rh[1]) + _hip_w * 0.10)
                _left_x = int(min(_lh[0], _rh[0]) - _hip_w * 0.70)
                _right_x = int(max(_lh[0], _rh[0]) + _hip_w * 0.70)
                cv2.rectangle(ankle_taper, (_left_x, _top_y), (_right_x, _bot_y), 255, -1)
                # Union nhẹ với parsing leg để không quá mỏng
                if parsing:
                    _leg_region = np.zeros((h, w), dtype=np.uint8)
                    for _lk in ("left_leg", "right_leg", "pants", "skirt"):
                        _lp = parsing.get(_lk)
                        if _lp is not None:
                            if _lp.shape[:2] != (h, w):
                                _lp = cv2.resize(_lp, (w, h), interpolation=cv2.INTER_NEAREST)
                            _leg_region = cv2.bitwise_or(_leg_region, _lp)
                    if int(cv2.countNonZero(_leg_region)) > 100:
                        _leg_region = cv2.dilate(_leg_region, np.ones((9, 9), np.uint8), iterations=1)
                        ankle_taper = cv2.bitwise_or(ankle_taper, _leg_region)
                ankle_taper = cv2.morphologyEx(ankle_taper, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
                ankle_taper = (cv2.GaussianBlur(ankle_taper, (5, 5), 1.2) > 20).astype(np.uint8) * 255
                gen_mask = cv2.bitwise_and(gen_mask, ankle_taper)
                _debug_save("09f_pants_ankle_taper_mask", ankle_taper, is_mask=True)
                pipeline_info.append("PantsAnkleTaper:v19.50")
        except Exception as exc:
            pipeline_info.append(f"PantsAnkleTaperSkip:{type(exc).__name__}")

        # v19.46: lower-leg fit — bóp riêng vùng từ gối → mắt cá theo trục
        # knee/ankle. Vùng trên gối giữ nguyên để không ảnh hưởng hông/đùi.
        if pants_type != "shorts":
            try:
                lower_leg_fit = _build_pants_lower_leg_fit_mask(
                    (h, w), full_pose, parsing, pants_style=pants_style,
                )
                gen_mask = cv2.bitwise_and(gen_mask, lower_leg_fit)
                _debug_save("09h_pants_lower_leg_fit_mask", lower_leg_fit, is_mask=True)
                pipeline_info.append("PantsLowerLegFit:v19.50")
            except Exception as exc:
                pipeline_info.append(f"PantsLowerLegFitSkip:{type(exc).__name__}")

        _debug_save("09c_pants_shape_mask", gen_mask, is_mask=True)
        if "PantsShapeUnion:v19.32" not in pipeline_info:
            pipeline_info.append("PantsShapeUnion:v19.34")
            pipeline_info.append("PantsShapeUnion:v19.32")

    if garment_category == "pants" and pants_type == "shorts":
        shorts_seed_cleanup_mask = binary_mask.copy()
        _shorts_prompt_low = (style_prompt or "").lower()
        _is_denim_shorts_prompt = any(_kw in _shorts_prompt_low for _kw in (
            "denim", "jean", "jeans", "blue wash", "button", "zip",
            "zipper", "fly closure", "belt loop", "belt loops",
            "high-waist", "high waist", "high-waisted", "frayed hem",
        ))
        # v22.18: prefer parsing.pants (the actual worn shorts on the model)
        # as the dominant shape. SegFormer captures the real hip width, leg
        # opening curve and hem line, while MediaPipe's hip/knee keypoints
        # are often shifted inward and produce a too-narrow / wrong-form
        # envelope. We fall back to the pose-driven mask only when parsing
        # is unreliable.
        _parsing_shorts = None
        if parsing:
            _ps = np.zeros((h, w), dtype=np.uint8)
            for _k in ("pants", "skirt"):
                _p = parsing.get(_k)
                if _p is not None:
                    _ps = cv2.bitwise_or(_ps, _p)
            if int(cv2.countNonZero(_ps)) > 500:
                _ps_bin = ((_ps > 20).astype(np.uint8)) * 255
                _best_label = 0
                _best_score = -1.0
                try:
                    _hip_y = None
                    _ref_len = max(48.0, float(h) * 0.12)
                    if full_pose is not None:
                        _lh = full_pose.get("left_hip")
                        _rh = full_pose.get("right_hip")
                        _ls = full_pose.get("left_shoulder")
                        _rs = full_pose.get("right_shoulder")
                        if _lh is not None and _rh is not None:
                            _hip_y = float((_lh[1] + _rh[1]) * 0.5)
                            _hip_w = abs(float(_rh[0]) - float(_lh[0]))
                            _sw = abs(float(_rs[0]) - float(_ls[0])) if _ls is not None and _rs is not None else _hip_w * 2.0
                            _ref_len = max(48.0, _sw, _hip_w * 2.0)
                    _num, _labels, _stats, _cent = cv2.connectedComponentsWithStats(_ps_bin, connectivity=8)
                    for _i in range(1, _num):
                        _x, _y, _bw, _bh, _area = _stats[_i]
                        if _area < 500:
                            continue
                        _cy = float(_cent[_i][1])
                        if _hip_y is not None:
                            if _cy < _hip_y - _ref_len * 0.55 or _cy > _hip_y + _ref_len * 0.78:
                                continue
                            if _y > _hip_y + _ref_len * 0.55:
                                continue
                            _score = float(_area) - abs(_cy - _hip_y) * 20.0
                        else:
                            if _cy > h * 0.70:
                                continue
                            _score = float(_area)
                        if _score > _best_score:
                            _best_score = _score
                            _best_label = _i
                    if _best_label > 0:
                        _parsing_shorts = ((_labels == _best_label).astype(np.uint8)) * 255
                    else:
                        _parsing_shorts = _ps_bin
                except Exception:
                    _parsing_shorts = _ps_bin
                _parsing_shorts = cv2.morphologyEx(
                    _parsing_shorts, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1,
                )

        if _parsing_shorts is not None:
            # Use the real worn-shorts silhouette. Light dilation gives SD a
            # few pixels of fabric slack around the hip and hem.
            _shorts_base = cv2.dilate(
                _parsing_shorts, np.ones((7, 7), np.uint8), iterations=1,
            )
            # v22.20: derive STRAIGHT waist line + hem line from parsing.pants
            # so the warped reference (which carries hanging drawstrings and
            # a jagged waistband from the studio image) cannot pull the mask
            # into a wavy top edge or a dangling sash below the hem.
            # v22.21: relaxed hem slack — percentile 98 + 4px was clipping
            # the natural curved leg openings, producing notches/missing
            # pieces at the inseam and outer leg. Keep waist tight (it is
            # naturally horizontal) but give hem ~15px of fabric room.
            _ys_ps, _xs_ps = np.where(_parsing_shorts > 20)
            _waist_y = None
            _hem_y = None
            if len(_ys_ps) > 200:
                _cx = float(np.median(_xs_ps))
                _half = max(20.0, float(_xs_ps.max() - _xs_ps.min()) * 0.30)
                _central = (_xs_ps > _cx - _half) & (_xs_ps < _cx + _half)
                if int(_central.sum()) > 50:
                    _waist_y = int(np.percentile(_ys_ps[_central], 5))
                    _hem_y = int(np.percentile(_ys_ps, 99))
            # Union with the warped reference footprint so SD has anchor
            # pixels of the new garment color inside the mask.
            _warp_src = ((binary_mask > 20).astype(np.uint8)) * 255
            shorts_seed_cleanup_mask = _warp_src.copy()
            if int(cv2.countNonZero(_warp_src)) > 200:
                _warp_dil = cv2.dilate(_warp_src, np.ones((5, 5), np.uint8), iterations=1)
                # v22.21: tighter halo for shorts (15 instead of 21) so
                # drawstring tips of the warped reference cannot extend far
                # outside the worn-shorts silhouette, but still wide enough
                # for SD to draw natural leg openings/curves.
                _warp_zone = cv2.dilate(
                    _parsing_shorts, np.ones((15, 15), np.uint8), iterations=1,
                )
                _shorts_base = cv2.bitwise_or(
                    _shorts_base, cv2.bitwise_and(_warp_dil, _warp_zone),
                )
            # Close small holes the parser left around the waistband.
            _shorts_base = cv2.morphologyEx(
                _shorts_base, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1,
            )
            # v22.20: enforce STRAIGHT horizontal waistband + hem cap so the
            # final mask matches a real shorts silhouette instead of the
            # jagged parsing edge / warp drawstring blob.
            # v22.21: only tighten ABOVE the waist line — leave generous hem
            # room (+15px) so SD can paint natural curved leg openings.
            if _waist_y is not None and _hem_y is not None and _hem_y > _waist_y + 8:
                _waist_overlap = 18 if _is_denim_shorts_prompt else 6
                _waist_top_cut = max(0, _waist_y - _waist_overlap)
                _shorts_base[:_waist_top_cut, :] = 0
                _hem_bot_cut = min(h, _hem_y + 15)
                _shorts_base[_hem_bot_cut:, :] = 0
                # v22.21: fill central waist gaps. Parsing.pants for shorts
                # often has a notch at the very top center where the model's
                # belly button / waistband crease confuses the parser. Force
                # a thin solid waistband bar from waist_y down ~6px.
                _bar_half = max(_half, float(_xs_ps.max() - _xs_ps.min()) * (0.43 if _is_denim_shorts_prompt else 0.30))
                _bar_l = max(0, int(_cx - _bar_half))
                _bar_r = min(w, int(_cx + _bar_half))
                _bar_top = max(0, _waist_y - (15 if _is_denim_shorts_prompt else 0))
                _bar_b = min(h, _waist_y + (12 if _is_denim_shorts_prompt else 6))
                _shorts_base[_bar_top:_bar_b, _bar_l:_bar_r] = 255
            # v22.22: parsing shorts are one connected blob, so the bottom
            # center can become a small tab that SD reads as a crotch stripe or
            # fabric tail. Carve a modest V gap only in the lower center.
            def _carve_shorts_center_gap(_mask: np.ndarray) -> np.ndarray:
                _m = _mask.copy()
                _ys, _xs = np.where(_m > 20)
                if len(_xs) < 200:
                    return _m
                _x1, _x2 = int(_xs.min()), int(_xs.max())
                _y1, _y2 = int(_ys.min()), int(_ys.max())
                _width = max(1, _x2 - _x1)
                _height = max(1, _y2 - _y1)
                _cx = int(np.median(_xs))
                # v22.21: only carve the bottom ~25% (hem leg opening). The
                # old 54% start cut a tall V that produced a visible notch
                # up into the crotch/inner thigh area of the rendered shorts.
                _top = int(_y1 + _height * 0.78)
                _bot = min(h - 1, _y2 + max(2, int(_height * 0.04)))
                if _bot <= _top + 4:
                    return _m
                _top_half = max(2, int(_width * 0.015))
                _bot_half = max(6, int(_width * 0.055))
                _gap_poly = np.array([
                    [_cx - _top_half, _top],
                    [_cx + _top_half, _top],
                    [_cx + _bot_half, _bot],
                    [_cx - _bot_half, _bot],
                ], dtype=np.int32)
                cv2.fillPoly(_m, [_gap_poly], 0, lineType=cv2.LINE_AA)
                return (_m > 20).astype(np.uint8) * 255
            _shorts_base = _carve_shorts_center_gap(_shorts_base)
            gen_mask = _shorts_base
            # Upper-body protect (do NOT subtract "dress" — see v22.11 note).
            if parsing:
                _ub2 = np.zeros_like(gen_mask)
                _upper_clip_y = None
                for _k in ("upper_clothes", "left_arm", "right_arm", "face", "hair"):
                    _p = parsing.get(_k)
                    if _p is not None:
                        if _k == "upper_clothes" and _waist_y is not None:
                            _p = _p.copy()
                            _upper_clip_y = max(0, _waist_y - (18 if _is_denim_shorts_prompt else 6))
                            _p[_upper_clip_y:, :] = 0
                        _ub2 = cv2.bitwise_or(_ub2, _p)
                if int(cv2.countNonZero(_ub2)) > 0:
                    _ub2 = cv2.dilate(_ub2, np.ones((9, 9), np.uint8), iterations=1)
                    if _upper_clip_y is not None and _is_denim_shorts_prompt:
                        # Dilation of upper_clothes can creep back over the
                        # high-waist denim band; trim it again after dilation.
                        _ub2[_upper_clip_y:, :] = cv2.bitwise_and(
                            _ub2[_upper_clip_y:, :],
                            cv2.bitwise_not(cv2.dilate(
                                parsing.get("upper_clothes", np.zeros_like(_ub2))[_upper_clip_y:, :],
                                np.ones((3, 3), np.uint8),
                                iterations=1,
                            )),
                        )
                    gen_mask = cv2.subtract(gen_mask, _ub2)
            # Shoe protect using parsing + pose fallback (same logic as the
            # long-pants branch). Hem of worn shorts is well above shoes, so
            # this normally is a no-op but it guards against parser leak.
            try:
                _sp = np.zeros((h, w), dtype=np.uint8)
                for _k in ("left_shoe", "right_shoe"):
                    _p = parsing.get(_k) if parsing else None
                    if _p is not None:
                        _sp = cv2.bitwise_or(_sp, _p)
                if int(cv2.countNonZero(_sp)) > 80:
                    _sp = cv2.dilate(_sp, np.ones((9, 9), np.uint8), iterations=1)
                    gen_mask = cv2.subtract(gen_mask, _sp)
            except Exception:
                pass
            # Seed/anchor mask = the same parsing silhouette so reference
            # color is sampled exactly inside the worn shorts.
            shorts_wear_mask_for_seed = cv2.dilate(
                _parsing_shorts, np.ones((5, 5), np.uint8), iterations=1,
            )
            if _waist_y is not None and _hem_y is not None and _hem_y > _waist_y + 8:
                _seed_top = max(0, _waist_y - (16 if _is_denim_shorts_prompt else 4))
                shorts_wear_mask_for_seed[:_seed_top, :] = 0
                shorts_wear_mask_for_seed[min(h, _hem_y + 12):, :] = 0
                _seed_bar_l = locals().get("_bar_l", max(0, int(_cx - _half)))
                _seed_bar_r = locals().get("_bar_r", min(w, int(_cx + _half)))
                _seed_bar_top = max(0, _waist_y - (14 if _is_denim_shorts_prompt else 0))
                _seed_bar_b = min(h, _waist_y + (12 if _is_denim_shorts_prompt else 6))
                shorts_wear_mask_for_seed[_seed_bar_top:_seed_bar_b, _seed_bar_l:_seed_bar_r] = 255
            shorts_wear_mask_for_seed = _carve_shorts_center_gap(shorts_wear_mask_for_seed)
            _debug_save("09c_shorts_shape_mask", gen_mask, is_mask=True)
            _debug_save("09c0_shorts_parsing_base", _parsing_shorts, is_mask=True)
            _debug_save("09c1_shorts_wear_mask", shorts_wear_mask_for_seed, is_mask=True)
            pipeline_info.append(
                "ShortsParsingDriven:v22.24_denim_waist"
                if _is_denim_shorts_prompt else
                "ShortsParsingDriven:v22.22"
            )
        else:
            # Fallback: pose-driven envelope (legacy v22.12-v22.17 path).
            shorts_shape_mask = _build_shorts_shape_mask((h, w), binary_mask, parsing, full_pose)
            if int(cv2.countNonZero(shorts_shape_mask)) > 255:
                gen_mask = shorts_shape_mask
                if parsing:
                    _ub2 = np.zeros_like(gen_mask)
                    for _k in ("upper_clothes", "left_arm", "right_arm", "face", "hair"):
                        _p = parsing.get(_k)
                        if _p is not None:
                            _ub2 = cv2.bitwise_or(_ub2, _p)
                    if int(cv2.countNonZero(_ub2)) > 0:
                        _ub2 = cv2.dilate(_ub2, np.ones((11, 11), np.uint8), iterations=1)
                        gen_mask = cv2.subtract(gen_mask, _ub2)
                shorts_wear_mask_for_seed = _build_shorts_wear_mask(
                    (h, w), parsing, full_pose, gen_mask,
                )
                if int(cv2.countNonZero(shorts_wear_mask_for_seed)) > 200:
                    gen_mask = cv2.bitwise_or(gen_mask, shorts_wear_mask_for_seed)
                try:
                    _shorts_band = _pp_build_shorts_edit_band((h, w), full_pose)
                    if int(cv2.countNonZero(_shorts_band)) > 0:
                        gen_mask = cv2.bitwise_and(gen_mask, _shorts_band)
                        pipeline_info.append("ShortsBandReclip:v22.17")
                except Exception as exc:
                    pipeline_info.append(f"ShortsBandReclipSkip:{type(exc).__name__}")
                _debug_save("09c_shorts_shape_mask", gen_mask, is_mask=True)
                _debug_save("09c1_shorts_wear_mask", shorts_wear_mask_for_seed, is_mask=True)
                pipeline_info.append("ShortsPoseFallback:v22.18")

    if garment_category == "top" and (top_subtype or "").lower() == "hoodie":
        # v22.10: HoodieStructureAnchor + HoodieSleeveTorsoPrior were carving
        # holes in the gen_mask to "preserve" drawstrings/seams from the seed.
        # In practice this stamped hard vertical lines down the torso/arm
        # boundary that diffusion could not soften — visible as fake
        # sleeve-torso seams in the output. Disable: let diffusion paint
        # naturally inside the full mask. Reference garment + IP-adapter
        # carry the structure.
        pass

    # SOTA: Soft edge blur at mask boundary (CatVTON uses kernel=height/50)
    # v19.38: pants long/cropped — use a tighter feather (5px) to prevent
    # the soft-edge bleed that caused translucent halos. Other categories
    # keep the wider feather for fabric drape.
    if garment_category == "pants" and pants_type != "shorts":
        blur_k = 3
    elif garment_category == "pants" and pants_type == "shorts":
        # v22.19: tight feather for shorts so the hip waistband and white
        # trim do not bleed into a soft halo. Parsing-driven base already
        # has a precise edge.
        blur_k = 3
    elif garment_category == "top" and (top_subtype or "").lower() == "hoodie":
        blur_k = 3
    else:
        blur_k = max(7, (h // 50) | 1)  # ~10px at 512, always odd

    # Jacket subtype mask cleanup: prevent the two recurring failure modes
    # visible in current output — (a) fabric bleeding up onto the chin/jaw
    # because the 21px dilation pushes the mask over the lower face, and (b)
    # a vertical split at the bottom hem when the parsing left/right halves
    # are not perfectly continuous. Cap the top of the mask at the face
    # bottom + small margin, and horizontally close the lower hem band.
    if garment_category == "top" and (top_subtype or "").lower() == "jacket":
        try:
            if parsing and parsing.get("face") is not None:
                _face = parsing["face"]
                if _face.shape[:2] != (h, w):
                    _face = cv2.resize(_face, (w, h), interpolation=cv2.INTER_NEAREST)
                _face_ys = np.where(_face > 20)[0]
                if len(_face_ys):
                    _chin_y = int(_face_ys.max())
                    _chin_pad = max(4, int(h * 0.012))
                    _top_cut = max(0, _chin_y - _chin_pad)
                    _zero_band = np.zeros_like(gen_mask)
                    _zero_band[:_top_cut, :] = 255
                    # Erase mask above chin to keep jacket fabric off the jaw.
                    gen_mask = cv2.bitwise_and(gen_mask, cv2.bitwise_not(_zero_band))
                    pipeline_info.append("JacketChinCap:v1")
            # Only seal hem if there's an ACTUAL vertical split — i.e. a column
            # of zeros in the lower band that is bordered by mask on both sides.
            # An unconditional MORPH_CLOSE with a wide horizontal kernel smears
            # over real diffusion detail (zipper teeth, pocket seams).
            _ys_jk = np.where(gen_mask > 20)[0]
            if len(_ys_jk):
                _jk_top = int(_ys_jk.min())
                _jk_bot = int(_ys_jk.max())
                _jk_h = max(1, _jk_bot - _jk_top)
                _lo_y0 = int(_jk_top + _jk_h * 0.72)
                _lo_y1 = _jk_bot + 1
                _lower = gen_mask[_lo_y0:_lo_y1, :]
                if _lower.size:
                    _col_has = (_lower > 20).any(axis=0)
                    _nz_cols = np.where(_col_has)[0]
                    if len(_nz_cols) >= 4:
                        _cl, _cr = int(_nz_cols.min()), int(_nz_cols.max())
                        _interior = _col_has[_cl:_cr + 1]
                        _gap_cols = int((~_interior).sum())
                        if _gap_cols >= 3:
                            _lower_sealed = cv2.morphologyEx(
                                _lower, cv2.MORPH_CLOSE,
                                np.ones((3, 11), np.uint8), iterations=1,
                            )
                            gen_mask[_lo_y0:_lo_y1, :] = _lower_sealed
                            pipeline_info.append(f"JacketHemSeal:v2-gap{_gap_cols}")
        except Exception as _e:
            pipeline_info.append(f"JacketMaskCleanupSkip:{type(_e).__name__}")

    gen_mask_soft = cv2.GaussianBlur(gen_mask, (blur_k, blur_k), blur_k / 3.0)
    gen_mask_soft = np.clip(gen_mask_soft, 0, 255).astype(np.uint8)

    _debug_save("09_agnostic_mask", gen_mask_soft, is_mask=True)

    # Color stabilization before diffusion
    init_tryon = np.clip(init_tryon, 15, 240).astype(np.uint8)

    # v16.11f: Per-category diffusion config.
    # v16.12b: Dial back strength/guidance — 0.78/4.5 caused LCM to hallucinate
    # rainbow/tie-dye patterns, destroying the original leopard print. Keep
    # strength modest so the TPS/warped pattern is preserved, and keep the
    # skirt blend from pulling in a corrupted init.
    # v16.12b: Also save a CLEAN copy of init_tryon before any pre-fill mutation;
    # DressBlend uses this as the "ground truth" garment to preserve pattern.
    init_tryon_clean = init_tryon.copy()
    pattern_source_rgb = init_tryon_clean
    if garment_category == "dress" and dress_pattern_reference is not None:
        pattern_source_rgb = _merge_clean_dress_pattern_reference(
            dress_pattern_reference,
            init_tryon_clean,
            binary_mask,
        )
        _debug_save("09c_dress_pattern_source", pattern_source_rgb)
        pipeline_info.append("DressReferenceCondition:v18.2")
    elif garment_category == "dress":
        pattern_source_rgb = _merge_clean_dress_pattern_reference(
            None,
            init_tryon_clean,
            binary_mask,
        )
        _debug_save("09c_dress_pattern_source", pattern_source_rgb)
    # v19.52: detect plain (solid-color) dress fabric. High detail/chroma
    # strengths in _preserve_dress_pattern_gpu + _lock_dress_source_pattern_final
    # re-inject warp/weave high-freq artifacts as "pattern noise" (horizontal
    # stripes/scratches) when the source is uniform. Auto-soften strengths.
    # v19.53: relax thresholds (folds push L_std above 22 for taupe/brown);
    # log the decision so we can confirm detection at runtime.
    _dress_is_plain = False
    if garment_category == "dress" and pattern_source_rgb is not None:
        _src_lab = cv2.cvtColor(pattern_source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        _mask_bool = binary_mask > 127
        if int(_mask_bool.sum()) > 200:
            _ab = _src_lab[..., 1:3][_mask_bool]
            _L = _src_lab[..., 0][_mask_bool]
            _ab_std = float(_ab.std())
            _l_std = float(_L.std())
            # plain = low chroma variance (a/b uniform). L_std can be high from folds.
            _dress_is_plain = _ab_std < 8.0 and _l_std < 32.0
            print(f"[DRESS] plain-fabric detect: ab_std={_ab_std:.2f} l_std={_l_std:.2f} -> plain={_dress_is_plain}")
    diffusion_mode = (refiner_mode or "lcm").strip().lower()
    if diffusion_mode not in {"lcm", "hypersd", "dpm++", "euler", "base"}:
        diffusion_mode = "lcm"
    if (
        garment_category == "dress"
        and diffusion_mode in {"lcm", "hypersd"}
        and os.getenv("VTON_DRESS_FAST_REFINER", "0").strip() != "1"
    ):
        diffusion_mode = "dpm++"
        pipeline_info.append("DressQualityDPM:v17.2")

    if garment_category == "dress":
        # v17.16: Keep primary diffusion from redesigning patterned dresses
        # into grey base garments.  Drape still comes from SD, but the print
        # identity should remain close enough that post-guards do not need to
        # paint broad grey repair patches.
        _preserve = float(np.clip(preserve_strength, 0.25, 1.0))
        if diffusion_mode in {"dpm++", "euler", "base"}:
            _diff_strength = float(np.clip(0.94 - 0.10 * _preserve, 0.84, 0.92))
            _diff_guidance = float(np.clip(max(gen_guidance, 5.4), 5.2, 6.6))
        else:
            _diff_strength = float(np.clip(0.84 - 0.10 * _preserve, 0.74, 0.82))
            _diff_guidance = float(np.clip(max(gen_guidance, 4.6), 4.2, 5.8))
        pipeline_info.append("DressDiffusionCfg:v18.20")
        # Extract dominant dress colour from TPS torso region
        _dress_region = binary_mask > 127
        _dress_color = np.array([128, 128, 128], dtype=np.float32)
        if _dress_region.sum() > 200:
            _dr = float(pattern_source_rgb[:, :, 0][_dress_region].mean())
            _dg = float(pattern_source_rgb[:, :, 1][_dress_region].mean())
            _db = float(pattern_source_rgb[:, :, 2][_dress_region].mean())
            _dress_color = np.array([_dr, _dg, _db], dtype=np.float32)
            _dominant = "dark" if (_dr + _dg + _db) < 300 else "light"
            # v16.13: Inject concrete colour anchor + print-preservation hints to
            # stop LCM from hallucinating rainbow/tie-dye. We sample 3 anchor
            # colours (dark / mid / light quantile) from the garment region and
            # name them so the prompt acts as a colour clamp.
            try:
                _reg_pixels = pattern_source_rgb[_dress_region].reshape(-1, 3).astype(np.float32)
                _lum = _reg_pixels.mean(axis=1)
                _order = np.argsort(_lum)
                _n = len(_order)
                if _n >= 3:
                    _q_dark = _reg_pixels[_order[int(_n * 0.15)]]
                    _q_mid  = _reg_pixels[_order[int(_n * 0.50)]]
                    _q_light = _reg_pixels[_order[int(_n * 0.85)]]
                    def _name(c):
                        r, g, b = c
                        # Simple 11-bucket naming
                        if max(r, g, b) - min(r, g, b) < 18:
                            v = (r + g + b) / 3
                            if v < 55: return "black"
                            if r > b + 3 and v >= 105:
                                if v < 175: return "warm taupe beige"
                                if v < 225: return "warm ivory beige"
                                return "warm ivory white"
                            if v < 110: return "charcoal grey"
                            if v < 180: return "neutral grey"
                            if v < 225: return "light neutral grey"
                            return "white"
                        if r > g and r > b:
                            if g > 140 and b < 120: return "beige"
                            if r - g > 60 and r - b > 60: return "red"
                            return "brown"
                        if g > r and g > b: return "green"
                        if b > r and b > g:
                            return "navy" if b < 170 else "blue"
                        return "tan"
                    _anchor = f"{_name(_q_dark)} and {_name(_q_light)} {_name(_q_mid)}"
                    # Do not call the dress "leopard". That token makes SD
                    # redraw round spots.  Anchor the exact source dress and
                    # ask only for fold/lighting changes.
                    style_prompt = (
                        f"single fitted long sleeve mini dress on the person, "
                        f"preserve the exact original {_anchor} high contrast abstract print layout from the reference garment, "
                        f"rich dark print on light fabric, no grey cast, no silver wash, not faded, "
                        f"no scarf, no shawl, no cardigan, "
                        f"no coat, natural body-following fold shadows only, do not redraw or reinterpret the print, "
                        f"soft waist drape, subtle sleeve wrinkles, clean shoulders neckline, crisp hem, realistic fabric lighting"
                    ).strip(", ")
                    print(f"[DIFFUSION] Dress colour anchor: {_anchor}")
            except Exception:
                style_prompt = (
                    "single fitted long sleeve mini dress on the person, "
                    "preserve the exact original black and warm ivory beige high contrast abstract print layout from the reference garment, "
                    "rich dark print on light fabric, no grey cast, no silver wash, not faded, "
                    "no scarf, no shawl, no cardigan, "
                    "no coat, natural body-following fold shadows only, do not redraw or reinterpret the print, "
                    "soft waist drape, subtle sleeve wrinkles, clean shoulders neckline, crisp hem, realistic fabric lighting"
                )

        # v16.18b: SKIP tile pre-fill and color-stat nudge for dress.
        # With HardDressMask + DressFullErase, init_tryon already contains the
        # solid warped leopard pixels inside the tight gen_mask. Tile-fill was
        # sampling a horizontal strip and tiling it vertically, which produced
        # distorted fragments (read by diffusion as polka-dots / puffy sleeves).
        # Color-stat nudge was shifting pattern pixels toward the mean colour,
        # flattening the print. Both are now disabled for dress.
        pass
    else:
        _diff_strength = 0.65
        _diff_guidance = 4.0

    # Hoodie subtype: lower strength + higher guidance so SD refines the
    # warped reference (preserving kangaroo pocket, drawstring, fitted
    # silhouette) instead of re-inventing a puffy oversized shape.
    if garment_category == "top" and (top_subtype or "").lower() == "hoodie":
        hoodie_strength_cap = float(np.clip(
            float(os.getenv("VTON_HOODIE_STRENGTH_CAP", "0.62").strip() or 0.62),
            0.52,
            0.72,
        ))
        hoodie_guidance_cap = float(np.clip(
            float(os.getenv("VTON_HOODIE_GUIDANCE_CAP", "5.1").strip() or 5.1),
            4.9,
            5.6,
        ))
        _diff_strength = float(np.clip(min(preserve_strength, hoodie_strength_cap), 0.52, hoodie_strength_cap))
        _diff_guidance = float(np.clip(gen_guidance, 4.9, hoodie_guidance_cap))
        pipeline_info.append(f"HoodieDiffCfg:v22.8:str{_diff_strength:.2f}_cfg{_diff_guidance:.1f}")

    # v19.24: pants-specific override — lower strength + tighter prompt so SD
    # doesn't repaint the shorts as one flat black rectangle and doesn't
    # touch the existing shirt.
    if garment_category == "pants":
        # v19.42: with textured seed (warped jeans inside warp core, flat fill
        # only in extension area), strength can be moderate so diffusion
        # refines edges/drape without washing away denim texture/seams.
        # v22.16: after cleaning the CPU-warp T layer from the seed, shorts can
        # use enough denoise for SD to round/reshape the legs instead of just
        # preserving the deterministic mask silhouette.
        # v22.16: after cleaning the CPU-warp T layer from the seed, shorts can
        # use enough denoise for SD to round/reshape the legs instead of just
        # preserving the deterministic mask silhouette.
        # v22.13: lift shorts strength so SD redraws the warped drawstring blob
        # instead of preserving it; mirrors how the shirt path uses a clean
        # POSITIVE template + higher denoise to redraw clothing geometry.
        _diff_strength = 0.78 if pants_type == "shorts" else 0.68
        _diff_guidance = 5.2 if pants_type == "shorts" else 4.8
        if pants_type == "shorts":
            try:
                from src.prompts.category_prompts import (
                    PANTS_SHORTS_CONSTRAINT,
                    PANTS_SHORTS_DENIM_CONSTRAINT,
                    PANTS_SHORTS_POSITIVE,
                )
                _incoming_sp = (style_prompt or "").strip()
                _incoming_low = _incoming_sp.lower()
                _is_denim_shorts = any(_kw in _incoming_low for _kw in (
                    "denim", "jean", "jeans", "blue wash", "button", "zip",
                    "zipper", "fly closure", "belt loop", "belt loops",
                    "high-waist", "high waist", "high-waisted", "frayed hem",
                ))
                # v22.19: don't override the user/Gemini prompt when it
                # contains specific garment descriptors (colour, trim,
                # stripe, logo, pattern). Only fall back to PANTS_SHORTS_POSITIVE
                # when the prompt is truly empty / boilerplate.
                _has_detail = any(_kw in _incoming_low for _kw in (
                    "trim", "stripe", "stripes", "piping", "logo", "print",
                    "pattern", "band", "ribbon", "contrast", "two-tone",
                    "white", "red", "blue", "green", "yellow", "pink",
                    "purple", "orange", "grey", "gray", "beige", "navy",
                    "tan", "brown", "khaki", "olive",
                ))
                _is_generic_sp = (
                    not _incoming_low
                    or "realistic virtual try-on" in _incoming_low
                    or "matching the reference garment" == _incoming_low
                    or (
                        not _has_detail and (
                            "athletic shorts" in _incoming_low
                            or "a photo of a person wearing" in _incoming_low
                            or "a person wearing" in _incoming_low
                        )
                    )
                )
                if _is_generic_sp:
                    style_prompt = PANTS_SHORTS_POSITIVE
                    pipeline_info.append("ShortsPromptCfg:v22.22:override")
                else:
                    # Keep user/Gemini prompt verbatim + append our shorts
                    # constraint so trim/colour descriptors survive.
                    _shorts_constraint = (
                        PANTS_SHORTS_DENIM_CONSTRAINT
                        if _is_denim_shorts else
                        PANTS_SHORTS_CONSTRAINT
                    )
                    style_prompt = f"{_incoming_sp}, {_shorts_constraint}".strip(", ").strip()
                    pipeline_info.append(
                        "ShortsPromptCfg:v22.23:denim"
                        if _is_denim_shorts else
                        "ShortsPromptCfg:v22.22:keep"
                    )
            except Exception:
                pass
        elif pants_type == "cropped":
            _pants_noun = "cropped pants"
            _pants_shape = "cropped pants ending mid-calf, two straight leg openings"
        else:
            _pants_noun = "long pants"
            _pants_shape = (
                "full-length long pants covering the entire legs down to the ankles, "
                "two long straight leg openings, hem reaches the tops of the shoes, "
                "no old cuffs visible below the hem, any exposed ankle skin keeps natural skin tone"
            )
        if pants_type != "shorts":
            _incoming = (style_prompt or "").strip()
            _incoming_low = _incoming.lower()
            if pants_type != "cropped" and re.search(r"\bankle[- ]length\b|\bankle\s+length\b", _incoming_low):
                _incoming = re.sub(
                    r"\bankle[- ]length\b|\bankle\s+length\b",
                    "full-length",
                    _incoming,
                    flags=re.IGNORECASE,
                )
                _incoming_low = _incoming.lower()
                pipeline_info.append("PantsPromptLengthFix:v19.65:ankle_to_full")
            if re.search(r"\bshorts?\b", _incoming_low):
                _is_denim_long = any(_kw in _incoming_low for _kw in (
                    "denim", "jean", "jeans", "blue wash", "button", "zip",
                    "zipper", "fly closure", "belt loop", "belt loops",
                ))
                if pants_type == "cropped":
                    _incoming = (
                        "A model wearing cropped pants matching the reference garment, "
                        "two separate pant legs, finished hems, old lower garment fully covered"
                    )
                elif _is_denim_long:
                    _incoming = (
                        "A model wearing full-length light blue wash straight-leg jeans matching the reference garment, "
                        "high-waisted denim pants with button and zip closure, classic five-pocket styling, "
                        "continuous denim fabric from waistband to shoe tops, finished hems covering the old cuffs completely"
                    )
                else:
                    _incoming = (
                        "A model wearing full-length pants matching the reference garment, "
                        "two long straight pant legs, finished hems reaching the tops of the shoes, "
                        "old lower garment fully covered"
                    )
                pipeline_info.append("PantsPromptConflictFix:v19.64:shorts_to_long")
            _pants_tail = (
                f"replace the old lower garment with the reference {_pants_noun}, "
                f"{_pants_shape}, follow the reference fabric color and texture exactly, "
                "natural waistband at the hips, soft fabric folds, "
                "cover the previous bottom completely, no original pants visible under the hem, "
                "preserve original shirt, preserve torso and arms, "
                "do not redraw upper body, natural shadows at waist and thighs"
            )
            style_prompt = (
                f"{_incoming}, {_pants_tail}".strip(", ").strip()
                if _incoming else _pants_tail
            )
        pipeline_info.append(f"PantsPromptCfg:v19.29:{pants_type}")

    try:
        # v17.16: Dress DPM++ defaults to 640 to preserve print edges.  Users
        # can still override with VTON_DRESS_INFER=512/640/768, and high-res
        # generation retries at 512 on runtime failure.
        try:
            _env_infer_raw = os.environ.get("VTON_DRESS_INFER", "").strip()
            _env_infer = int(_env_infer_raw) if _env_infer_raw else 0
        except Exception:
            _env_infer = 0
        _infer_size = _env_infer if (garment_category == "dress" and _env_infer >= 512) else 0
        if (
            garment_category == "dress"
            and _infer_size == 0
            and diffusion_mode in {"dpm++", "euler", "base"}
            and os.getenv("VTON_DRESS_AUTO_INFER", "1").strip() != "0"
        ):
            _infer_size = 640
        if garment_category == "dress" and diffusion_mode in {"lcm", "hypersd"}:
            _infer_steps = int(np.clip(gen_steps, 10, 14))
        elif garment_category == "dress" and diffusion_mode in {"dpm++", "euler", "base"}:
            _infer_steps = int(np.clip(max(gen_steps, 32), 30, 38))
        else:
            _infer_steps = max(gen_steps, 20)
        _infer_label = _infer_size if _infer_size >= 512 else 512
        _gencfg_label = {"pants": "PantsGenCfg", "dress": "DressGenCfg", "top": "TopGenCfg"}.get(garment_category, "GenCfg")
        pipeline_info.append(f"{_gencfg_label}:v17.16:{diffusion_mode}:steps{_infer_steps}:cfg{_diff_guidance:.1f}:str{_diff_strength:.2f}:infer{_infer_label}")
        diffusion_init = init_tryon
        if garment_category == "pants":
            # v19.43→v19.45: split anchor into upper band (waist/hip/thigh,
            # wider dilation for waistband+pocket detail) and lower band
            # (calf/ankle, narrower so seed doesn't bleed onto shoes).
            texture_anchor_mask = binary_mask
            if pants_type == "shorts":
                if shorts_wear_mask_for_seed is None or int(cv2.countNonZero(shorts_wear_mask_for_seed)) < 200:
                    shorts_wear_mask_for_seed = _build_shorts_wear_mask(
                        (h, w), parsing, full_pose, gen_mask,
                    )
                if int(cv2.countNonZero(shorts_wear_mask_for_seed)) > 200:
                    texture_anchor_mask = shorts_wear_mask_for_seed
                    _debug_save("09b1_shorts_seed_wear_mask", texture_anchor_mask, is_mask=True)
                    pipeline_info.append("ShortsReferenceSeedMask:v22.22")
            else:
                ys, xs = np.where(binary_mask > 20)
                if len(ys) > 100:
                    y1, y2 = int(ys.min()), int(ys.max())
                    height = max(1, y2 - y1)
                    upper_band = np.zeros_like(binary_mask)
                    lower_band = np.zeros_like(binary_mask)
                    upper_band[y1:int(y1 + height * 0.52), :] = 255
                    lower_band[int(y1 + height * 0.52):y2 + 1, :] = 255
                    upper_core = cv2.bitwise_and(binary_mask, upper_band)
                    lower_core = cv2.bitwise_and(binary_mask, lower_band)
                    upper_core = cv2.dilate(
                        upper_core,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 23)),
                        iterations=1,
                    )
                    lower_core = cv2.dilate(
                        lower_core,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 15)),
                        iterations=1,
                    )
                    texture_anchor_mask = cv2.bitwise_or(upper_core, lower_core)
                else:
                    texture_anchor_mask = cv2.dilate(
                        binary_mask,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 21)),
                        iterations=1,
                    )
                if parsing is not None:
                    _ta_protect = np.zeros_like(texture_anchor_mask)
                    for _k in ("upper_clothes", "dress", "left_arm", "right_arm", "face", "hair"):
                        _p = parsing.get(_k)
                        if _p is not None:
                            _ta_protect = cv2.bitwise_or(_ta_protect, _p)
                    if int(cv2.countNonZero(_ta_protect)) > 0:
                        _ta_protect = cv2.dilate(_ta_protect, np.ones((9, 9), np.uint8), iterations=1)
                        texture_anchor_mask = cv2.subtract(texture_anchor_mask, _ta_protect)
                _debug_save("09b1_pants_texture_anchor_mask", texture_anchor_mask, is_mask=True)
            diffusion_init = _build_pants_diffusion_seed(
                init_tryon,
                gen_mask_soft,
                reference_cloth_rgb,
                pants_type=pants_type,
                warped_mask=texture_anchor_mask,
                cleanup_mask=shorts_seed_cleanup_mask if pants_type == "shorts" else None,
                cleanup_fill_rgb=person_rgb if pants_type == "shorts" else None,
            )
            _debug_save("09b_pants_diffusion_seed", diffusion_init)
            pipeline_info.append(
                "ShortsReferenceSeed:v22.22_clean_person_trim"
                if pants_type == "shorts" else
                "PantsReferenceSeed:v19.66_clean_anchor"
            )
        elif garment_category == "dress":
            diffusion_init = _build_dress_diffusion_seed(init_tryon, gen_mask_soft, binary_mask)
            _debug_save("09b_dress_diffusion_seed", diffusion_init)
        elif garment_category == "top" and (top_subtype or "").lower() == "hoodie":
            if os.getenv("VTON_HOODIE_SOFT_SEED", "0").strip().lower() in {"1", "true", "yes", "on"}:
                diffusion_init, hoodie_seed_anchor = _build_hoodie_diffusion_seed(
                    init_tryon,
                    gen_mask_soft,
                    binary_mask,
                )
                pipeline_info.append("HoodieSoftSeed:v22.7")
            else:
                diffusion_init = init_tryon.copy()
                hoodie_seed_anchor = _build_hoodie_structure_anchor_mask(diffusion_init, binary_mask)
                pipeline_info.append("HoodieTopLikeSeed:v23.0")
            _debug_save("09b_hoodie_diffusion_seed", diffusion_init)
            _debug_save("09b1_hoodie_seed_anchor", hoodie_seed_anchor, is_mask=True)
            # v22.10: disabled seed-side seam injection — was stamping hard
            # vertical lines into the diffusion seed at the sleeve/torso
            # boundary that diffusion then locked in as fake seams.
        dress_model_raw = None
        # Negative prompt rules moved to src/prompts/category_prompts.py.
        # Dress = full override; pants/top = base + category tail.
        _negative_prompt = build_category_negative(
            garment_category,
            base_negative=GenConfig().negative_prompt or "",
            subtype=top_subtype if garment_category == "top" else "",
        )
        if garment_category == "pants" and pants_type == "shorts":
            _style_low_for_neg = (style_prompt or "").lower()
            if any(_kw in _style_low_for_neg for _kw in ("denim", "jean", "jeans", "blue wash")):
                _negative_prompt = _negative_prompt.replace("blue denim shorts visible, ", "")
                _negative_prompt = _negative_prompt.replace("blue denim shorts visible", "")
                _negative_prompt = _negative_prompt.replace("old jeans visible, ", "")
                _negative_prompt = _negative_prompt.replace("old jeans visible", "")
                _negative_prompt = _negative_prompt.replace("covering stomach, ", "")
                _negative_prompt = _negative_prompt.replace("covering stomach", "")
                pipeline_info.append("ShortsDenimNegativeFix:v22.24")
            _shorts_extra_negative = (
                "T-shape silhouette, wide hip flare, flared shorts, skirt shape, "
                "fabric tail below crotch, shorts wider than hips, shorts width different from hip width, "
                "missing left front pocket, missing right front pocket, one pocket only, "
                "no pocket opening, pocket removed, blank front hip panel, "
                "shirt tail over shorts, scarf tail over shorts, white strip over shorts, "
                "upper garment hanging over shorts"
                if any(_kw in _style_low_for_neg for _kw in ("denim", "jean", "jeans", "blue wash")) else
                "horizontal fabric band across the waist, T-shape silhouette, "
                "wide hip flare, flared shorts, skirt shape, pleated waistband, "
                "fabric bar extending sideways past the hips, "
                "old waistband visible, second waistband, "
                "shorts wider than hips, shorts width different from hip width"
            )
            _negative_prompt = f"{_negative_prompt}, {_shorts_extra_negative}".strip(", ").strip()
        elif garment_category == "pants":
            _pants_extra_negative = (
                "shorts, denim shorts, short pants, cropped above ankle, capri pants, "
                "hem stops above ankle, ankle cuff exposed, old cuffs visible, "
                "original jeans cuff visible, old pants visible below hem, "
                "previous pants showing under new pants, black belt, dark belt, "
                "black waistband block, belt over jeans, blue ankle skin, "
                "denim-colored ankles, blue foot skin"
                if pants_type != "cropped" else
                "shorts, denim shorts, old cuffs visible, original pants visible below hem"
            )
            _negative_prompt = f"{_negative_prompt}, {_pants_extra_negative}".strip(", ").strip()
            pipeline_info.append("PantsNegativeHemLock:v19.64")
        if gemini_negative_extra:
            _negative_prompt = f"{_negative_prompt}, {gemini_negative_extra}".strip(", ").strip()

        # Hoodie subtype: if the incoming style_prompt is empty or the generic
        # placeholder ("realistic virtual try-on..."), replace it with a hoodie-
        # specific positive prompt so SD knows to draw the hood + drawstring +
        # long sleeves rather than collapsing the garment into a crewneck tee.
        if garment_category == "top" and (top_subtype or "").lower() == "hoodie":
            try:
                from src.prompts.category_prompts import TOP_HOODIE_CONSTRAINT, TOP_HOODIE_POSITIVE
                _incoming_top = (style_prompt or "").strip().lower()
                _is_generic = (
                    not _incoming_top
                    or "realistic virtual try-on" in _incoming_top
                    or "matching the reference garment" == _incoming_top
                )
                if _is_generic:
                    style_prompt = TOP_HOODIE_POSITIVE
                else:
                    _incoming_clean = re.sub(
                        r"\b(?:regular|relaxed|loose|oversized|baggy)\s+fit\b",
                        "fitted regular fit",
                        style_prompt,
                        flags=re.IGNORECASE,
                    )
                    style_prompt = f"{_incoming_clean}, {TOP_HOODIE_CONSTRAINT}".strip(", ").strip()
                pipeline_info.append("HoodiePromptCfg:v2")
            except Exception:
                pass

        # Jacket subtype: enforce a bomber-jacket prompt so SD draws the ribbed
        # hem, two side pockets and full-length zipper instead of collapsing
        # the jacket into a cropped tee/sweater silhouette (the recurring
        # "short hem, no pockets" failure mode).
        if garment_category == "top" and (top_subtype or "").lower() == "jacket":
            try:
                from src.prompts.category_prompts import TOP_JACKET_CONSTRAINT, TOP_JACKET_POSITIVE
                _incoming_jk = (style_prompt or "").strip().lower()
                _is_generic_jk = (
                    not _incoming_jk
                    or "realistic virtual try-on" in _incoming_jk
                    or "matching the reference garment" == _incoming_jk
                )
                if _is_generic_jk:
                    style_prompt = TOP_JACKET_POSITIVE
                else:
                    style_prompt = f"{style_prompt}, {TOP_JACKET_CONSTRAINT}".strip(", ").strip()
                pipeline_info.append("JacketPromptCfg:v1")
            except Exception:
                pass

        # Shirt subtype: force a button-up shirt prompt so SD keeps the point
        # collar + center button placket + buttoned cuffs and doesn't collapse
        # the result into a bomber jacket (stand collar, zipper, ribbed hem),
        # which is what happens with the generic top prompt when the warped
        # mask resembles a jacket silhouette.
        if garment_category == "top" and (top_subtype or "").lower() == "shirt":
            try:
                from src.prompts.category_prompts import TOP_SHIRT_CONSTRAINT, TOP_SHIRT_POSITIVE
                _incoming_sh = (style_prompt or "").strip().lower()
                _is_generic_sh = (
                    not _incoming_sh
                    or "realistic virtual try-on" in _incoming_sh
                    or "matching the reference garment" == _incoming_sh
                    or "bomber" in _incoming_sh
                    or "jacket" in _incoming_sh
                )
                if _is_generic_sh:
                    style_prompt = TOP_SHIRT_POSITIVE
                else:
                    style_prompt = f"{style_prompt}, {TOP_SHIRT_CONSTRAINT}".strip(", ").strip()
                pipeline_info.append("ShirtPromptCfg:v1")
            except Exception:
                pass

        # T-shirt subtype: append a strong constraint that locks the chest
        # graphic / typography so SD does not regenerate it into garbled text
        # or a new logo. Keeps the user-supplied prompt verbatim (Gemini usually
        # already describes the print accurately) and just appends the lock tail.
        if garment_category == "top" and (top_subtype or "").lower() == "tshirt":
            try:
                from src.prompts.category_prompts import TOP_TSHIRT_CONSTRAINT, TOP_TSHIRT_POSITIVE
                _incoming_ts = (style_prompt or "").strip()
                _incoming_ts_low = _incoming_ts.lower()
                _is_generic_ts = (
                    not _incoming_ts
                    or "realistic virtual try-on" in _incoming_ts_low
                    or "matching the reference garment" == _incoming_ts_low
                )
                if _is_generic_ts:
                    style_prompt = TOP_TSHIRT_POSITIVE
                else:
                    style_prompt = f"{_incoming_ts}, {TOP_TSHIRT_CONSTRAINT}".strip(", ").strip()
                pipeline_info.append("TshirtPromptCfg:v1")
            except Exception:
                pass

        def _generate_with_infer(infer_size: int) -> np.ndarray:
            return generate_tryon_image(
                init_tryon_rgb=diffusion_init,
                inpaint_mask_gray=gen_mask_soft,
                user_prompt=style_prompt,
                config=GenConfig(
                num_inference_steps=_infer_steps,
                guidance_scale=_diff_guidance,
                refiner_mode=diffusion_mode,
                cloth_type="dress" if garment_category == "dress" else cloth_type,
                use_cloth_lora=(garment_category != "dress"),
                strength=_diff_strength,
                infer_size=infer_size,
                reference_image_rgb=(
                    pattern_source_rgb if garment_category == "dress" else
                    reference_cloth_rgb if garment_category == "pants" else
                    reference_cloth_rgb if (garment_category == "top" and (top_subtype or "").lower() == "hoodie") else
                    reference_cloth_rgb if (garment_category == "top" and (top_subtype or "").lower() == "jacket") else
                    reference_cloth_rgb if (garment_category == "top" and (top_subtype or "").lower() == "shirt") else
                    reference_cloth_rgb if (garment_category == "top" and (top_subtype or "").lower() == "tshirt") else
                    None
                ),
                ip_adapter_scale=float(os.getenv(
                    "VTON_IP_ADAPTER_SCALE",
                    "0.70" if (garment_category == "pants" and pants_type == "shorts")
                    else ("0.65" if garment_category == "pants"
                    else ("0.36" if (garment_category == "top" and (top_subtype or "").lower() == "hoodie")
                          else ("0.48" if (garment_category == "top" and (top_subtype or "").lower() == "jacket")
                                else ("0.52" if (garment_category == "top" and (top_subtype or "").lower() == "shirt")
                                      else ("0.62" if (garment_category == "top" and (top_subtype or "").lower() == "tshirt")
                                            else "0.46"))))),
                )),
                negative_prompt=_negative_prompt,
                ),
            )

        try:
            generated = _generate_with_infer(_infer_size)
        except RuntimeError:
            if garment_category == "dress" and _infer_size >= 640:
                pipeline_info.append("DressInferFallback512:v17.16")
                generated = _generate_with_infer(0)
            else:
                raise

        # Sanitize + size-match
        generated = _sanitize_rgb_output(generated)
        generated = _fit_like(generated, init_tryon, is_mask=False)
        binary_mask = _fit_like(binary_mask, generated, is_mask=True)
        try:
            _paint_mask = (gen_mask_soft > 20)
            if int(_paint_mask.sum()) > 50:
                _delta = np.mean(
                    np.abs(
                        generated.astype(np.int16)[_paint_mask]
                        - diffusion_init.astype(np.int16)[_paint_mask]
                    ),
                    axis=1,
                )
                _changed_px = int((_delta > 8.0).sum())
                _changed_ratio = _changed_px / float(max(1, int(_paint_mask.sum())))
                pipeline_info.append(
                    f"DiffPaintDelta:v1:mad{float(_delta.mean()):.1f}:chg{_changed_ratio:.2f}"
                )
        except Exception:
            pass
        # Dress legacy fallback: use diffusion as the base only when it still
        # matches the intended dress geometry. If it loses sleeves, shortens the
        # hem, paints into face/hair, or drifts too wide at the waist, keep the
        # deterministic seed/warp and transfer only light fold detail.
        if garment_category == "dress":
            dress_model_raw = generated.copy()
            _debug_save("10a_dress_model_raw", generated)
            _dress_length_hint = _dress_prompt_length_hint(style_prompt)
            _shape_ok, _shape_reasons, _generated_dress_mask = _legacy_dress_diffusion_shape_guard(
                generated,
                pattern_source_rgb,
                binary_mask,
                full_pose,
                parsing,
                new_sleeve_type,
                _dress_length_hint,
            )
            _dress_color_mask = cv2.bitwise_and(binary_mask, _generated_dress_mask)
            if int(cv2.countNonZero(_dress_color_mask)) < 200:
                _dress_color_mask = binary_mask if _shape_ok else np.zeros_like(binary_mask)
            _debug_save("10a1_dress_generated_shape_mask", _generated_dress_mask, is_mask=True)
            _debug_save("10a2_dress_color_anchor_mask", _dress_color_mask, is_mask=True)

            # 1. Base = diffusion only if shape guard passes. Optional blend
            #    toward TPS composite via VTON_DRESS_TPS_BLEND.
            _tps_blend = float(np.clip(
                float(os.getenv("VTON_DRESS_TPS_BLEND", "0.0").strip() or 0.0),
                0.0, 0.6,
            ))
            if not _shape_ok:
                output = _blend_dress_luminance_detail(
                    pattern_source_rgb,
                    generated,
                    binary_mask,
                    strength=0.16,
                )
                pipeline_info.append(
                    "DressDiffusionShapeGuard:v21:" + ",".join(_shape_reasons[:5])
                )
            elif _tps_blend > 0.001:
                output = _safe_uint8(
                    generated.astype(np.float32) * (1.0 - _tps_blend)
                    + pattern_source_rgb.astype(np.float32) * _tps_blend
                )
            else:
                output = generated.copy()
            _debug_save("10b_v20_diffusion_base", output)

            # 2. Lock color only where diffusion still looks like the dress,
            #    not across the whole rectangular/over-wide binary mask.
            if int(cv2.countNonZero(_dress_color_mask)) > 200:
                output = _apply_color_consistency(
                    output,
                    pattern_source_rgb,
                    _dress_color_mask,
                    strength=0.35 if _shape_ok else 0.25,
                )
            else:
                pipeline_info.append("DressColorAnchorSkipped:v21")
            _debug_save("10c_v20_color_anchored", output)

            # 3. Feathered alpha composite outside dress mask: pull from
            #    init_tryon_clean (P1 cleaned base) so old yellow is gone.
            _alpha_v20 = cv2.GaussianBlur(binary_mask, (7, 7), 2.0).astype(np.float32) / 255.0
            _alpha_v20 = np.clip(_alpha_v20, 0.0, 1.0)[..., None]
            output = (
                init_tryon_clean.astype(np.float32) * (1.0 - _alpha_v20)
                + output.astype(np.float32) * _alpha_v20
            )
            output = _safe_uint8(output)
            _debug_save("10d_v20_composite", output)

            # 4. Arm/face restore — paste original skin where parsing says
            #    "left_arm/right_arm/face" AND the new dress mask does not
            #    cover that pixel. Without this, diffusion sometimes paints
            #    over the arm hanging by the side.
            if parsing:
                _id_mask = np.zeros(output.shape[:2], dtype=np.uint8)
                for _ik in ("left_arm", "right_arm", "face", "neck"):
                    _iv = parsing.get(_ik)
                    if _iv is not None:
                        _id_mask = cv2.bitwise_or(_id_mask, _iv)
                # Subtract dress mask so we don't paste arm over the new
                # sleeve when the dress is long-sleeved.
                _id_outside = cv2.subtract(_id_mask, binary_mask)
                if int(_id_outside.sum()) > 255 * 50:
                    output = _apply_foreground_layer(output, person_rgb, _id_outside)
                    _debug_save("10e_v20_arm_restore", output)
                    pipeline_info.append("DressArmRestore:v20.7")

            # 5. Edge cleanup — bleed/halo only, no pattern injection.
            _red_cleanup_mask = cv2.dilate(binary_mask, np.ones((11, 11), np.uint8), iterations=1)
            if parsing:
                _old_clothes_for_red = get_clothing_mask(parsing)
                if _old_clothes_for_red is not None:
                    _near_garment = cv2.dilate(binary_mask, np.ones((25, 25), np.uint8), iterations=1)
                    _red_cleanup_mask = cv2.bitwise_or(
                        _red_cleanup_mask,
                        cv2.bitwise_and(_old_clothes_for_red, _near_garment),
                    )
            output, _red_fixed = _inpaint_old_red_bleed(output, _red_cleanup_mask, parsing)
            if _red_fixed:
                pipeline_info.append("DressRedBleedInpaint:v16.67")
            output, _collar_fixed = _remove_old_collar_bleed(output, init_tryon_clean, binary_mask)
            if _collar_fixed:
                pipeline_info.append("DressCollarClean:v16.62")
            output, _light_collar_fixed = _inpaint_old_light_collar_bleed(output, binary_mask, parsing)
            if _light_collar_fixed:
                pipeline_info.append("DressLightCollarClean:v20.3")
            output = _suppress_dress_edge_halo(output, person_rgb, binary_mask)
            # v20.8: kill brown rectangular pattern-reference spill outside
            # the dress mask (most often visible below feet as a brown block).
            output, _rect_fixed = _remove_dress_rect_artifact(
                output, init_tryon_clean, binary_mask
            )
            if _rect_fixed:
                pipeline_info.append("DressRectArtifactClean:v20.8")
            pipeline_info.append(
                f"DressPipeline:v20.7:tps={_tps_blend:.2f}"
            )

        # Blackout guard
        mean_val = float(generated.mean())
        if mean_val < 10:
            warning_msg = f"Diffusion nearly black (mean={mean_val:.1f}), keeping CPU result"
            print(f"[DIFFUSION] SOFT-FAIL — {warning_msg}")
            return init_tryon, "", warning_msg, pipeline_info

        if _is_blackout_artifact(generated, init_tryon, binary_mask):
            warning_msg = "Diffusion unstable, keeping CPU result"
            print(f"[DIFFUSION] SOFT-FAIL — blackout artifact detected")
            return init_tryon, "", warning_msg, pipeline_info

        # Brightness guard — blend CPU garment back if too dark
        _garment_region = binary_mask > 127
        if _garment_region.sum() > 100:
            _input_brightness = float(init_tryon[_garment_region].mean())
            _output_brightness = float(generated[_garment_region].mean())
            if _input_brightness > 30 and _output_brightness < _input_brightness * 0.35:
                generated = _enforce_garment_identity(generated, init_tryon, binary_mask, 0.25)
                pipeline_info.append("DiffusionRecover")
                print(f"[DIFFUSION] RECOVER — brightness {_output_brightness:.0f} vs {_input_brightness:.0f}")

        # v16.9: NO duplicate repaint here.
        # gen_tryon.py already does repaint: keeps init_tryon (CPU output with new garment)
        # outside the mask, diffusion output inside the mask. Adding a second repaint
        # with person_rgb would overwrite the CPU garment with old shirt pixels.
        # The diffusion output (generated) already has correct compositing from gen_tryon.
        # v20.2: dress output is already finalized by the lean pipeline above.
        # For all other categories, assign diffusion output here.
        if garment_category != "dress":
            output = generated

        # Jacket: restore inner-shirt (e.g. white T at the open collar) which
        # gets repainted in the jacket color because parsing labels both layers
        # as "upper_clothes". Detect inner shirt as pixels inside upper_clothes
        # whose ORIGINAL color is far from the jacket reference dominant color,
        # restricted to the upper neckline strip where an inner layer can show.
        if garment_category == "top" and (top_subtype or "").lower() == "jacket":
            try:
                if (
                    parsing
                    and parsing.get("upper_clothes") is not None
                    and reference_cloth_rgb is not None
                ):
                    _h, _w = output.shape[:2]
                    _upper = parsing["upper_clothes"]
                    if _upper.shape[:2] != (_h, _w):
                        _upper = cv2.resize(_upper, (_w, _h), interpolation=cv2.INTER_NEAREST)
                    _upper_b = (_upper > 20)
                    _ref = reference_cloth_rgb
                    _ref_i = _ref.astype(np.int16)
                    _ref_valid = ~((_ref_i[..., 0] > 232) & (_ref_i[..., 1] > 232) & (_ref_i[..., 2] > 232))
                    if int(_ref_valid.sum()) < 200:
                        _ref_valid = np.ones(_ref.shape[:2], dtype=bool)
                    _jacket_mean = _ref[_ref_valid].astype(np.float32).mean(axis=0)
                    _orig_f = person_rgb.astype(np.float32)
                    _diff = np.linalg.norm(_orig_f - _jacket_mean[None, None, :], axis=2)
                    # Distinct from jacket color. Dark old jackets are also far
                    # from a camel reference, so this alone is not enough.
                    _far = _diff > 55.0
                    # Restrict to upper neckline band: top 36% of upper_clothes
                    _ys_u = np.where(_upper_b)[0]
                    _band = np.zeros((_h, _w), dtype=bool)
                    if len(_ys_u):
                        _y_top = int(_ys_u.min())
                        _y_bot = int(_ys_u.max())
                        _y_cut = int(_y_top + (_y_bot - _y_top) * 0.36)
                        _band[_y_top:_y_cut, :] = True
                    _center = np.zeros((_h, _w), dtype=bool)
                    try:
                        _ls = full_pose.get("left_shoulder") if full_pose else None
                        _rs = full_pose.get("right_shoulder") if full_pose else None
                        _lh = full_pose.get("left_hip") if full_pose else None
                        _rh = full_pose.get("right_hip") if full_pose else None
                        if _ls is not None and _rs is not None:
                            _cx = float((_ls[0] + _rs[0]) * 0.5)
                            _sw = max(32.0, abs(float(_rs[0]) - float(_ls[0])))
                        elif _lh is not None and _rh is not None:
                            _cx = float((_lh[0] + _rh[0]) * 0.5)
                            _sw = max(32.0, abs(float(_rh[0]) - float(_lh[0])) * 1.35)
                        else:
                            _ux = np.where(_upper_b)[1]
                            _cx = float(np.median(_ux)) if len(_ux) else _w * 0.5
                            _sw = max(32.0, float(_ux.max() - _ux.min()) * 0.45) if len(_ux) else _w * 0.22
                    except Exception:
                        _cx, _sw = _w * 0.5, _w * 0.22
                    _xx = np.arange(_w, dtype=np.float32)[None, :]
                    _center = np.abs(_xx - _cx) <= (_sw * 0.24)
                    # Exclude pixels close to skin tone (avoid restoring face/neck back
                    # into the jacket area — face/neck are handled elsewhere)
                    _r, _g, _b = _orig_f[..., 0], _orig_f[..., 1], _orig_f[..., 2]
                    _lum = _orig_f.mean(axis=2)
                    _chroma = _orig_f.max(axis=2) - _orig_f.min(axis=2)
                    _is_skin = (_r > _b + 6.0) & (_r > _g - 4.0) & (_lum > 70.0) & (_lum < 235.0) & (_chroma < 80.0)
                    # Shirt-like pixels: bright / neutral fabric. This keeps a
                    # white tee/shirt but rejects the old black leather jacket.
                    _shirt_like = (_lum > 125.0) & (_chroma < 86.0)
                    _inner = _upper_b & _band & _center & _far & _shirt_like & (~_is_skin)
                    _inner_u8 = _inner.astype(np.uint8) * 255
                    # Keep only the largest connected component to avoid noise
                    if int(_inner_u8.sum()) > 255 * 40:
                        _num, _lbl, _stats, _ = cv2.connectedComponentsWithStats(_inner_u8, connectivity=8)
                        if _num > 1:
                            _areas = _stats[1:, cv2.CC_STAT_AREA]
                            _keep = np.zeros_like(_inner_u8)
                            for _i, _a in enumerate(_areas, start=1):
                                if _a >= 80:
                                    _keep[_lbl == _i] = 255
                            _inner_u8 = _keep
                        # Erode slightly to stay inside the inner-shirt region, then soft blur
                        _inner_u8 = cv2.erode(_inner_u8, np.ones((3, 3), np.uint8), iterations=1)
                        if int(_inner_u8.sum()) > 255 * 30:
                            _alpha_in = cv2.GaussianBlur(_inner_u8, (0, 0), 1.6).astype(np.float32) / 255.0
                            _alpha_in = np.clip(_alpha_in, 0.0, 1.0)[..., None]
                            output = _safe_uint8(
                                output.astype(np.float32) * (1.0 - _alpha_in)
                                + person_rgb.astype(np.float32) * _alpha_in
                            )
                            _debug_save("10l_jacket_inner_shirt_mask", _inner_u8)
                            _debug_save("10m_jacket_inner_shirt_restore", output)
                            pipeline_info.append("JacketInnerShirtRestore:v2")
            except Exception as _e:
                pipeline_info.append(f"JacketInnerShirtRestoreSkip:{type(_e).__name__}")

        if garment_category == "top" and (top_subtype or "").lower() == "hoodie":
            output, hoodie_spill_mask = _remove_hoodie_edge_spill(
                output,
                person_rgb,
                binary_mask,
                parsing,
                full_pose,
                gen_mask_soft,
            )
            if int(cv2.countNonZero(hoodie_spill_mask)) > 15:
                _debug_save("10f_hoodie_edge_spill_mask", hoodie_spill_mask, is_mask=True)
                _debug_save("10g_hoodie_edge_spill_clean", output)
                pipeline_info.append("HoodieEdgeSpillClean:v23.3")
            output, hoodie_lower_tone_mask = _smooth_hoodie_lower_torso_tone(
                output,
                init_tryon_clean,
                binary_mask,
                full_pose,
            )
            if int(cv2.countNonZero(hoodie_lower_tone_mask)) > 200:
                _debug_save("10h_hoodie_lower_tone_mask", hoodie_lower_tone_mask, is_mask=True)
                _debug_save("10i_hoodie_lower_tone", output)
                pipeline_info.append("HoodieLowerTone:v22.5")
            output, hoodie_structure_mask = _restore_hoodie_structure_detail(
                output,
                init_tryon_clean,
                binary_mask,
            )
            if int(cv2.countNonZero(hoodie_structure_mask)) > 50:
                _debug_save("10j_hoodie_structure_mask", hoodie_structure_mask, is_mask=True)
                _debug_save("10k_hoodie_structure_restore", output)
                pipeline_info.append("HoodieStructureRestore:v23.3")
            output = _sharpen_hoodie_output(output, binary_mask)
            _debug_save("10l_hoodie_sharpen", output)
            pipeline_info.append("HoodieSharpen:v23.3")
            # v22.10: post-process sleeve/torso seam injection disabled — it
            # darkened a vertical strip along the sleeve/torso boundary that
            # read as a fake stitched seam in the final output.
        if garment_category == "pants" and pants_type == "shorts":
            output, shorts_guard_mask = _apply_shorts_shape_guard(
                output_rgb=output,
                person_rgb=person_rgb,
                init_tryon_rgb=diffusion_init,
                warped_mask=binary_mask,
                gen_mask_soft=gen_mask_soft,
                parsing=parsing,
                full_pose=full_pose,
                reference_cloth_rgb=reference_cloth_rgb,
                final_wear_mask=shorts_wear_mask_for_seed,
            )
            _debug_save("10f_shorts_shape_guard_mask", shorts_guard_mask, is_mask=True)
            _debug_save("10g_shorts_shape_guard", output)
            pipeline_info.append("ShortsShapeGuard:v22.22")
            _shorts_post_low = (style_prompt or "").lower()
            _shorts_is_denim_post = any(_kw in _shorts_post_low for _kw in (
                "denim", "jean", "jeans", "blue wash", "button", "zip",
                "zipper", "fly closure", "belt loop", "belt loops",
            ))
            if _shorts_is_denim_post:
                output = _pp_recover_pants_texture_detail(
                    output,
                    init_tryon_clean,
                    shorts_guard_mask,
                    safe_uint8=_safe_uint8,
                    detail_strength=0.40,
                    chroma_strength=0.12,
                    sharpen_strength=0.18,
                )
                _debug_save("10h0_shorts_denim_texture_recover", output)
                pipeline_info.append("ShortsDenimTextureRecover:v22.25")
            output, _shorts_outer_spill = _pp_cleanup_shorts_external_spill(
                output, person_rgb, shorts_guard_mask, safe_uint8=_safe_uint8,
            )
            if int(cv2.countNonZero(_shorts_outer_spill)) > 20:
                _debug_save("10h_shorts_outer_spill_mask", _shorts_outer_spill, is_mask=True)
                _debug_save("10i_shorts_outer_spill_clean", output)
                pipeline_info.append("ShortsOuterSpillClean:v22.22")
            if _shorts_is_denim_post:
                output, _shorts_old_hem_bleed = _pp_cleanup_shorts_old_hem_bleed(
                    output, shorts_guard_mask, safe_uint8=_safe_uint8,
                )
                if int(cv2.countNonZero(_shorts_old_hem_bleed)) > 8:
                    _debug_save("10j_shorts_old_hem_bleed_mask", _shorts_old_hem_bleed, is_mask=True)
                    _debug_save("10k_shorts_old_hem_bleed_clean", output)
                    pipeline_info.append("ShortsOldHemBleedClean:v22.26_crotch_gap")
                output, _shorts_upper_spill = _pp_cleanup_shorts_upper_cloth_spill(
                    output, shorts_guard_mask, safe_uint8=_safe_uint8,
                )
                if int(cv2.countNonZero(_shorts_upper_spill)) > 12:
                    _debug_save("10l_shorts_upper_cloth_spill_mask", _shorts_upper_spill, is_mask=True)
                    _debug_save("10m_shorts_upper_cloth_spill_clean", output)
                    pipeline_info.append("ShortsUpperClothSpillClean:v22.27")
                pipeline_info.append("ShortsCenterTrimSkipDenim:v22.24")
            else:
                output, _shorts_center_artifact = _pp_cleanup_shorts_center_trim_artifact(
                    output, shorts_guard_mask, safe_uint8=_safe_uint8,
                )
                if int(cv2.countNonZero(_shorts_center_artifact)) > 8:
                    _debug_save("10j_shorts_center_trim_mask", _shorts_center_artifact, is_mask=True)
                    _debug_save("10k_shorts_center_trim_clean", output)
                    pipeline_info.append("ShortsCenterTrimClean:v22.22")
        elif garment_category == "pants":
            if pants_type == "shorts":
                output, pants_guard_mask = _apply_pants_shape_guard(
                    output_rgb=output,
                    init_tryon_rgb=init_tryon_clean,
                    warped_mask=binary_mask,
                    gen_mask_soft=gen_mask_soft,
                    parsing=parsing,
                    pants_type=pants_type,
                )
                _debug_save("10f_pants_shape_guard_mask", pants_guard_mask, is_mask=True)
                _debug_save("10g_pants_shape_guard", output)
                pipeline_info.append("PantsShapeGuard:v19.31")
            else:
                # v19.37: For long/cropped pants, run a custom shape guard
                # using the pose-derived leg envelope (NOT the narrow TPS
                # warp). This kills the translucent blue wisps that leak
                # outside the leg silhouette due to feathered mask edges.
                try:
                    allowed = np.zeros((h, w), dtype=np.uint8)
                    # Parsing legs + existing pants envelope
                    if parsing is not None:
                        for _k in ("left_leg", "right_leg", "pants", "skirt"):
                            _p = parsing.get(_k)
                            if _p is not None:
                                allowed = cv2.bitwise_or(allowed, _p)
                    # Pose-trapezoid (slightly wider than parsing to allow
                    # fabric drape past the bare skin silhouette).
                    if full_pose is not None:
                        lh_g = np.array(full_pose.get("left_hip", [0, 0]), dtype=np.float32)
                        rh_g = np.array(full_pose.get("right_hip", [0, 0]), dtype=np.float32)
                        lk_g = np.array(full_pose.get("left_knee", lh_g), dtype=np.float32)
                        rk_g = np.array(full_pose.get("right_knee", rh_g), dtype=np.float32)
                        la_g = np.array(full_pose.get("left_ankle", lk_g), dtype=np.float32)
                        ra_g = np.array(full_pose.get("right_ankle", rk_g), dtype=np.float32)
                        hip_w_g = float(np.linalg.norm(lh_g - rh_g))
                        if hip_w_g < 12:
                            hip_w_g = w * 0.22
                        # v19.39: Tighten further (was 0.28/0.23/0.20).
                        # Wider envelope leaves a translucent halo just
                        # outside the leg silhouette where diffusion blue
                        # bleeds through soft mask edges.
                        # v19.57: bóp ankle envelope cho khớp với mắt cá.
                        hw_hip = max(6.0, hip_w_g * 0.22)
                        hw_knee = max(5.0, hip_w_g * 0.16)
                        hw_ankle = max(3.0, hip_w_g * 0.11)
                        def _leg_poly_g(top, mid, bot, htop, hmid, hbot):
                            pts = np.array([
                                [top[0] - htop, top[1]],
                                [top[0] + htop, top[1]],
                                [mid[0] + hmid, mid[1]],
                                [bot[0] + hbot, bot[1]],
                                [bot[0] - hbot, bot[1]],
                                [mid[0] - hmid, mid[1]],
                            ], dtype=np.int32)
                            cv2.fillPoly(allowed, [pts], 255)
                        _leg_poly_g(lh_g, lk_g, la_g, hw_hip, hw_knee, hw_ankle)
                        _leg_poly_g(rh_g, rk_g, ra_g, hw_hip, hw_knee, hw_ankle)
                        # Waistband bridge
                        top_yg = int(min(lh_g[1], rh_g[1]) - hip_w_g * 0.12)
                        bot_yg = int(max(lh_g[1], rh_g[1]) + hip_w_g * 0.05)
                        left_xg = int(min(lh_g[0], rh_g[0]) - hw_hip * 0.85)
                        right_xg = int(max(lh_g[0], rh_g[0]) + hw_hip * 0.85)
                        cv2.rectangle(allowed, (left_xg, top_yg), (right_xg, bot_yg), 255, -1)

                    if int(cv2.countNonZero(allowed)) > 200:
                        # v19.39: don't dilate `allowed` — we want a TIGHT
                        # envelope so spill includes every halo pixel.
                        allowed = cv2.morphologyEx(allowed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
                        allowed_full = allowed.copy()
                        # v19.46: bóp envelope ở lower-leg cùng helper với
                        # gen_mask, để spill từ gối → mắt cá vẫn bị bắt.
                        try:
                            _ll_fit = _build_pants_lower_leg_fit_mask(
                                (h, w), full_pose, parsing, pants_style=pants_style,
                            )
                            # v19.48: dilate _ll_fit trước khi AND để không
                            # siết envelope quá hẹp ở lower-leg. AND hẹp →
                            # spill catch ngay sát silhouette → restore từ
                            # init_tryon kéo nền tường trắng / warp rectangle
                            # vào, tạo vệt trắng. Dilate 13×13 + GaussianBlur
                            # giữ envelope vừa đủ ôm chân + đệm fabric drape.
                            _ll_fit_pad = cv2.dilate(
                                _ll_fit, np.ones((13, 13), np.uint8), iterations=1
                            )
                            _ll_fit_pad = (
                                cv2.GaussianBlur(_ll_fit_pad, (9, 9), 2.0) > 20
                            ).astype(np.uint8) * 255
                            # v19.58: lưu allowed_full TRƯỚC khi AND với
                            # capsule, để hard cut cuối dùng làm keep-region
                            # (tránh thay outside bằng person_rgb tại vùng
                            # parsing.pants gốc → vệt jeans đen lộ ra).
                            allowed = cv2.bitwise_and(allowed, _ll_fit_pad)
                            pipeline_info.append("PantsAllowedLowerLegFit:v19.50")
                        except Exception as exc:
                            pipeline_info.append(f"PantsAllowedLowerLegFitSkip:{type(exc).__name__}")
                        # Compute spill only where generated pixels are far
                        # outside the allowed envelope. Restoring the 1-6 px
                        # edge band pulls wall/old-pants pixels into the hem.
                        edge_slack = cv2.dilate(
                            allowed,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                            iterations=1,
                        )
                        spill = cv2.bitwise_and(gen_mask_soft, cv2.bitwise_not(edge_slack))
                        # v19.54: cắt spill dưới ankle để denim restore không
                        # kéo xuống dưới mắt cá / chân giày.
                        try:
                            if full_pose is not None and all(
                                full_pose.get(_k) is not None
                                for _k in ("left_ankle", "right_ankle")
                            ):
                                _la_y = float(full_pose["left_ankle"][1])
                                _ra_y = float(full_pose["right_ankle"][1])
                                _ankle_cut = int(min(h, max(_la_y, _ra_y) + 6))
                                if 0 < _ankle_cut < h:
                                    spill[_ankle_cut:, :] = 0
                        except Exception:
                            pass
                        if int(cv2.countNonZero(spill)) > 30:
                            # v19.54: dùng init_tryon_clean trực tiếp. Các thử
                            # nghiệm flat fill (v19.53) tạo trail denim rộng,
                            # blur (v19.52) làm pants mờ, person_rgb (v19.51)
                            # lộ nền tường. init_tryon_clean có warp ghost
                            # nhưng spill area mỏng nhất + ankle cut → ghost
                            # gần như không thấy.
                            restore_src = init_tryon_clean
                            # v19.39: HARD binary alpha (was 9x9 Gaussian, sigma 2.0).
                            # The soft restore was leaving a ~5px translucent
                            # halo where diffusion blue still showed through.
                            # Use a 3x3 minimal blur only for sub-pixel anti-alias.
                            alpha = cv2.GaussianBlur(
                                (spill > 4).astype(np.float32), (3, 3), 0.6,
                            )
                            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
                            cleaned = (
                                output.astype(np.float32) * (1.0 - alpha)
                                + restore_src.astype(np.float32) * alpha
                            )
                            output = _safe_uint8(cleaned)
                        _debug_save("10f_pants_shape_guard_mask", allowed, is_mask=True)
                        _debug_save("10g_pants_shape_guard", output)
                        pipeline_info.append("PantsPoseShapeGuard:v19.50")
                        # v19.44: remove small denim speckles outside silhouette
                        try:
                            output = _pp_cleanup_pants_speckles(
                                output, init_tryon_clean, allowed,
                                safe_uint8=_safe_uint8,
                            )
                            pipeline_info.append("PantsSpeckleCleanup:v19.44")
                        except Exception as exc:
                            pipeline_info.append(f"PantsSpeckleCleanupSkip:{type(exc).__name__}")
                        # v19.45: recover denim seams/pockets/waistband detail
                        # from init_tryon inside the pants core (LAB high-freq
                        # transfer + subtle sharpen). Leaves drape from
                        # diffusion intact, just sharpens texture.
                        try:
                            _texture_recover_mask = allowed
                            try:
                                _ref_tex = cv2.dilate(
                                    ((binary_mask > 20).astype(np.uint8)) * 255,
                                    np.ones((13, 13), np.uint8),
                                    iterations=1,
                                )
                                _texture_recover_mask = cv2.bitwise_and(allowed, _ref_tex)
                                if int(cv2.countNonZero(_texture_recover_mask)) < 300:
                                    _texture_recover_mask = allowed
                                else:
                                    _debug_save(
                                        "10h_pants_texture_recover_mask",
                                        _texture_recover_mask,
                                        is_mask=True,
                                    )
                            except Exception:
                                _texture_recover_mask = allowed
                            _recover_detail_strength = 0.72
                            _recover_chroma_strength = 0.32
                            _recover_sharpen_strength = 0.48
                            if pants_type != "shorts":
                                try:
                                    _ys_tr, _xs_tr = np.where(_texture_recover_mask > 20)
                                    if len(_ys_tr) > 200:
                                        _x1_tr, _x2_tr = int(_xs_tr.min()), int(_xs_tr.max())
                                        _y1_tr, _y2_tr = int(_ys_tr.min()), int(_ys_tr.max())
                                        _w_tr = max(1, _x2_tr - _x1_tr)
                                        _h_tr = max(1, _y2_tr - _y1_tr)
                                        _cx_tr = int(np.median(_xs_tr))
                                        _yy_tr, _xx_tr = np.indices((h, w))
                                        _unsafe_tex = (
                                            ((_yy_tr >= _y1_tr) & (_yy_tr <= int(_y1_tr + _h_tr * 0.16)))
                                            | (
                                                (_yy_tr >= int(_y1_tr + _h_tr * 0.32))
                                                & (_yy_tr <= int(_y1_tr + _h_tr * 0.56))
                                                & (np.abs(_xx_tr - _cx_tr) <= max(8, int(_w_tr * 0.07)))
                                            )
                                            | (_yy_tr >= int(_y1_tr + _h_tr * 0.86))
                                        )
                                        _texture_recover_mask = cv2.subtract(
                                            _texture_recover_mask,
                                            (_unsafe_tex.astype(np.uint8)) * 255,
                                        )
                                        _debug_save(
                                            "10h1_pants_texture_recover_safe_mask",
                                            _texture_recover_mask,
                                            is_mask=True,
                                        )
                                        pipeline_info.append("PantsTextureRecoverySafeZones:v19.66")
                                except Exception as exc:
                                    pipeline_info.append(f"PantsTextureRecoverySafeZonesSkip:{type(exc).__name__}")
                                _recover_detail_strength = 0.38
                                _recover_chroma_strength = 0.14
                                _recover_sharpen_strength = 0.20
                            output = _pp_recover_pants_texture_detail(
                                output, init_tryon_clean, _texture_recover_mask,
                                safe_uint8=_safe_uint8,
                                detail_strength=_recover_detail_strength,
                                chroma_strength=_recover_chroma_strength,
                                sharpen_strength=_recover_sharpen_strength,
                            )
                            pipeline_info.append("PantsTextureRecovery:v19.66_refmask_safe")
                        except Exception as exc:
                            pipeline_info.append(f"PantsTextureRecoverySkip:{type(exc).__name__}")
                        # v19.59 (restored from v19.61): hard composite cuối —
                        # silhouette tight từ đùi xuống dùng parsing.legs ∪
                        # parsing.pants ∪ lower_leg_capsule (không trapezoid).
                        try:
                            if full_pose is not None and all(
                                full_pose.get(_k) is not None
                                for _k in ("left_hip", "right_hip")
                            ):
                                _hip_w_55 = max(20.0, float(np.linalg.norm(
                                    np.array(full_pose["left_hip"], dtype=np.float32)
                                    - np.array(full_pose["right_hip"], dtype=np.float32)
                                )))
                                _hip_y_mid = float(
                                    (full_pose["left_hip"][1] + full_pose["right_hip"][1]) * 0.5
                                )
                                _hip_top = int(min(
                                    float(full_pose["left_hip"][1]),
                                    float(full_pose["right_hip"][1]),
                                ) - _hip_w_55 * 0.35)
                                _hip_top = max(0, _hip_top)
                                _below_hip_y = int(_hip_y_mid + _hip_w_55 * 0.10)
                                person_resized = person_rgb
                                if person_resized.shape[:2] != (h, w):
                                    person_resized = cv2.resize(
                                        person_resized, (w, h), interpolation=cv2.INTER_LINEAR
                                    )
                                lower_zone = np.zeros((h, w), dtype=np.uint8)
                                lower_zone[_hip_top:, :] = 255

                                # ZONE A (waistband/hip): keep = allowed_full.
                                _keep_hip = locals().get("allowed_full", allowed).copy()

                                # ZONE B (dưới hông): keep = parsing tight + capsule.
                                _keep_below = np.zeros((h, w), dtype=np.uint8)
                                if parsing is not None:
                                    for _k in ("pants", "skirt", "left_leg", "right_leg"):
                                        _p = parsing.get(_k)
                                        if _p is not None:
                                            if _p.shape[:2] != (h, w):
                                                _p = cv2.resize(_p, (w, h), interpolation=cv2.INTER_NEAREST)
                                            _keep_below = cv2.bitwise_or(
                                                _keep_below, (_p > 20).astype(np.uint8) * 255
                                            )
                                try:
                                    _capsule = _build_pants_lower_leg_fit_mask(
                                        (h, w), full_pose, parsing, pants_style=pants_style,
                                    )
                                    _capsule_pad = cv2.dilate(
                                        _capsule, np.ones((7, 7), np.uint8), iterations=1
                                    )
                                    _keep_below = cv2.bitwise_or(_keep_below, _capsule_pad)
                                except Exception:
                                    pass
                                try:
                                    _full_pants_keep = locals().get("allowed_full", allowed)
                                    _full_pants_keep = cv2.dilate(
                                        _full_pants_keep, np.ones((5, 5), np.uint8), iterations=1,
                                    )
                                    _keep_below = cv2.bitwise_or(_keep_below, _full_pants_keep)
                                    pipeline_info.append("PantsFinalHardCutHemKeep:v19.64")
                                except Exception as exc:
                                    pipeline_info.append(f"PantsFinalHardCutHemKeepSkip:{type(exc).__name__}")

                                _zone_a = np.zeros((h, w), dtype=np.uint8)
                                _zone_a[_hip_top:_below_hip_y, :] = 255
                                _zone_b = np.zeros((h, w), dtype=np.uint8)
                                _zone_b[_below_hip_y:, :] = 255
                                keep_region = cv2.bitwise_or(
                                    cv2.bitwise_and(_keep_hip, _zone_a),
                                    cv2.bitwise_and(_keep_below, _zone_b),
                                )
                                keep_region = cv2.dilate(
                                    keep_region, np.ones((3, 3), np.uint8), iterations=1
                                )
                                keep_soft = np.clip(
                                    cv2.GaussianBlur(
                                        keep_region.astype(np.float32) / 255.0,
                                        (5, 5), 1.0,
                                    ),
                                    0.0, 1.0,
                                )
                                _debug_save("10i_pants_final_keep_region", keep_region, is_mask=True)
                                replace_alpha = (
                                    (lower_zone.astype(np.float32) / 255.0)
                                    * (1.0 - keep_soft)
                                )[..., None]
                                output = _safe_uint8(
                                    output.astype(np.float32) * (1.0 - replace_alpha)
                                    + person_resized.astype(np.float32) * replace_alpha
                                )
                                pipeline_info.append("PantsFinalHardCut:v19.59")
                        except Exception as exc:
                            pipeline_info.append(f"PantsFinalHardCutSkip:{type(exc).__name__}")
                        try:
                            _pants_art_low = (style_prompt or "").lower()
                            if any(_kw in _pants_art_low for _kw in (
                                "denim", "jean", "jeans", "blue wash", "light blue wash",
                            )):
                                _artifact_base_mask = locals().get(
                                    "keep_region",
                                    locals().get("allowed_full", allowed),
                                )
                                output, _pants_denim_artifacts = _pp_cleanup_long_pants_denim_artifacts(
                                    output,
                                    _artifact_base_mask,
                                    safe_uint8=_safe_uint8,
                                )
                                if int(cv2.countNonZero(_pants_denim_artifacts)) > 20:
                                    _debug_save("10j_pants_denim_artifact_mask", _pants_denim_artifacts, is_mask=True)
                                    _debug_save("10k_pants_denim_artifact_clean", output)
                                    pipeline_info.append("PantsDenimArtifactClean:v19.67")
                        except Exception as exc:
                            pipeline_info.append(f"PantsDenimArtifactCleanSkip:{type(exc).__name__}")
                        try:
                            if pants_type != "shorts":
                                _ankle_base_mask = locals().get(
                                    "keep_region",
                                    locals().get("allowed_full", allowed),
                                )
                                output, _pants_ankle_skin = _pp_restore_long_pants_ankle_skin(
                                    output,
                                    person_rgb,
                                    _ankle_base_mask,
                                    parsing,
                                    full_pose,
                                    safe_uint8=_safe_uint8,
                                )
                                if int(cv2.countNonZero(_pants_ankle_skin)) > 20:
                                    _debug_save("10l_pants_ankle_skin_mask", _pants_ankle_skin, is_mask=True)
                                    _debug_save("10m_pants_ankle_skin_restore", output)
                                    pipeline_info.append("PantsAnkleSkinRestore:v19.68")
                        except Exception as exc:
                            pipeline_info.append(f"PantsAnkleSkinRestoreSkip:{type(exc).__name__}")
                    else:
                        pipeline_info.append("PantsPoseShapeGuard:skipped(empty)")
                except Exception as exc:
                    pipeline_info.append(f"PantsPoseShapeGuardSkip:{type(exc).__name__}")
        elif garment_category == "accessory":
            # ── Postprocess riêng cho phụ kiện (giày/boots/mũ/kính/thắt lưng/túi/khăn) ──
            try:
                from src.postprocess.accessory_postprocess import (
                    apply_accessory_postprocess as _pp_apply_accessory,
                )
                _acc_sub = locals().get("accessory_subtype", "") or ""
                # `allowed` = vùng cho phép vẽ phụ kiện (warped footprint + dilate)
                _acc_allowed = (binary_mask > 20).astype(np.uint8) * 255
                _acc_allowed = cv2.dilate(
                    _acc_allowed, np.ones((5, 5), np.uint8), iterations=1,
                )
                output, _acc_tag = _pp_apply_accessory(
                    output_rgb=output,
                    init_tryon_rgb=init_tryon_clean,
                    person_rgb=person_rgb,
                    allowed=_acc_allowed,
                    gen_mask_soft=gen_mask_soft,
                    parsing=parsing,
                    full_pose=full_pose,
                    subtype=_acc_sub,
                    fit_like=_fit_like,
                    safe_uint8=_safe_uint8,
                )
                _debug_save("10a_accessory_allowed", _acc_allowed, is_mask=True)
                _debug_save("10b_accessory_postprocess", output)
                pipeline_info.append(_acc_tag)
            except Exception as exc:
                pipeline_info.append(f"AccessoryPostprocessSkip:{type(exc).__name__}")
        print(f"[DIFFUSION] SUCCESS — SOTA agnostic mask, mean_brightness={float(output.mean()):.1f}")
        return output, f"Local Diffusion ({diffusion_mode})", warning_msg, pipeline_info
    except (RuntimeError, Exception) as exc:
        warning_msg = f"Local diffusion failed, keeping CPU pipeline result. ({exc})"
        print(f"[DIFFUSION] FAILED — {exc}")
        return init_tryon, "", warning_msg, pipeline_info


def _compact_pipeline_info(pipeline_info: list[str]) -> list[str]:
    """Return a stable, user-facing pipeline summary.

    Internal pipeline_info contains many historical debug tags (v16/v17).  The
    UI only needs major stages so regressions are still visible without making
    the pipeline look like dozens of separate products are stacked together.
    """
    compact: list[str] = []
    seen: set[str] = set()

    def add(label: str | None) -> None:
        if not label or label in seen:
            return
        compact.append(label)
        seen.add(label)

    for item in pipeline_info:
        if not item:
            continue
        if item in {
            "SoftErase", "BodyCopy:OFF", "Sleeves", "Layers", "DressGen",
            "Diffusion", "DrapeHint", "DressNeckUnprotect:v16.62",
            "DressDehaloCPU:v16.61",
        }:
            continue
        if item.startswith("Diffusion["):
            continue
        if item.startswith("Type:") or item.startswith("Trans:"):
            continue
        if item.startswith("Category:"):
            add(item)
        elif item in {"Parsing", "Pose", "Pose(basic)", "U2Net", "MaskFallback", "GarmentMaskEnsemble", "PreFit", "CloudVTON", "CloudFailed"}:
            add(item)
        elif item.startswith("MaskEdgeTrim"):
            add("MaskEdgeTrim")
        elif item.startswith(("Affine", "TPS", "Persp", "HipAlign", "ShoulderAlign", "RotAlign", "DriftFix")):
            add("Warp")
        elif item.startswith(("DressSleevePoseClip", "DressSkirtComplete", "DressBodyPoseFit", "DressShoulderSeal")):
            add("DressFit")
        elif item in {"ArmErase", "NeckErase", "SleeveErase"} or item.startswith(("DressEraseClip", "DressFullErase")):
            add("DressErase")
        elif item.startswith(("DressTextureFill", "DressPatternRef", "HardDressMask", "3LayerBlend", "EraseRestore")):
            add("DressComposite")
        elif item.startswith("HumanMaskPrior"):
            add("HumanMaskPrior")
        elif item.startswith("DressFullGenMask"):
            add("DiffusionMask")
        elif item.startswith(("DressReferenceCondition", "DressPatternLockGPU", "DressPatternTransfer", "DressSourcePatternLock", "DressPrimaryToneGuard", "DressPrimarySoftClip", "DressWhiteFill", "DressNaturalFolds", "DressCrispFinish")):
            add(item)
        elif item.startswith("DressDiffusionPrimary"):
            add(item)
        elif item.startswith(("DressFastLCM", "DressCpuTextureOptIn")):
            add("DressDiffusion")
        elif item.startswith("DressSingleLayerClip"):
            add("SingleLayerClip")
        elif item.startswith(("DressRedBleedInpaint", "DressCollarClean", "DressDehalo", "DressPostHairRedClean", "HairRestoreAfterClean")):
            add("Cleanup")
        elif item == "HairOverlay":
            add("HairOverlay")
        elif item in {"FacePreserve", "ColorMatch"}:
            add(item)
        else:
            add(item)

    return compact or pipeline_info


# ═══════════════════════════════════════════════════════════════════
#  Main try_on() — Cloud-Primary Architecture
# ═══════════════════════════════════════════════════════════════════

def try_on(
    person_img: np.ndarray,
    cloth_img: np.ndarray,
    fit_scale: float,
    alpha: float,
    y_offset: float,
    use_gen: bool,
    style_prompt: str,
    gen_steps: int,
    gen_guidance: float,
    preserve_strength: float,
    quality_preset: str,
    refiner_mode: str,
    cloth_type: str,
    use_catvton_cloud: bool,
    use_gemini_prompt: bool = True,
    prompt_mode: str = "auto",
):
    # `cloth_type` is the legacy form-field name; the UI now sends a category lock.
    category_lock = _normalize_category_lock(cloth_type)
    if person_img is None or cloth_img is None:
        raise gr.Error("Vui long tai ca anh nguoi mau va anh ao.")

    person_path = _save_temp_input(person_img, "person")
    cloth_path = _save_temp_input(cloth_img, "cloth")

    person_rgb = read_image_rgb(person_path)
    cloth_rgb = read_image_rgb(cloth_path)

    fit_scale, alpha, gen_steps, gen_guidance, preserve_strength, refiner_mode = _apply_quality_preset(
        quality_preset,
        fit_scale,
        alpha,
        gen_steps,
        gen_guidance,
        preserve_strength,
        refiner_mode,
    )

    pipeline_info = []
    backend_used = "CPU Pipeline"
    warning_msg = ""

    # ── Gemini Vision auto-prompt ────────────────────────────────────
    gemini_info: dict | None = None
    gemini_positive_prompt = ""
    gemini_negative_prompt = ""
    _mode = (prompt_mode or "auto").strip().lower()
    if _mode not in {"auto", "fallback", "manual"}:
        _mode = "auto"
    if _mode == "manual":
        pipeline_info.append("PromptMode:manual")
    elif _mode == "fallback":
        pipeline_info.append("PromptMode:fallback")
        if fallback_describe_garment is not None:
            try:
                gemini_info = fallback_describe_garment(category_lock)
                gemini_positive_prompt = (gemini_info or {}).get("positive_prompt", "").strip()
                gemini_negative_prompt = (gemini_info or {}).get("negative_prompt", "").strip()
                pipeline_info.append(f"CategoryPrompt:{(gemini_info or {}).get('category','')}")
            except Exception:
                pass
    elif use_gemini_prompt and analyze_garment_prompt_with_gemini is not None:
        try:
            gemini_info = analyze_garment_prompt_with_gemini(
                person_rgb=person_rgb,
                cloth_rgb=cloth_rgb,
                category_lock=category_lock,
                user_prompt=style_prompt or "",
            )
            gemini_positive_prompt = (gemini_info or {}).get("positive_prompt", "").strip()
            gemini_negative_prompt = (gemini_info or {}).get("negative_prompt", "").strip()
            pipeline_info.append("GeminiPrompt")
            _cat_hint = (gemini_info or {}).get("category", "")
            if _cat_hint:
                pipeline_info.append(f"GeminiCat:{_cat_hint}")
        except Exception as exc:
            pipeline_info.append(f"GeminiPromptSkipped:{type(exc).__name__}")
            if fallback_describe_garment is not None:
                try:
                    gemini_info = fallback_describe_garment(category_lock)
                    gemini_positive_prompt = (gemini_info or {}).get("positive_prompt", "").strip()
                    gemini_negative_prompt = (gemini_info or {}).get("negative_prompt", "").strip()
                    pipeline_info.append(f"GeminiFallback:{(gemini_info or {}).get('category','')}")
                except Exception:
                    pass

    final_style_prompt = (style_prompt or "").strip()
    if gemini_positive_prompt:
        final_style_prompt = (
            f"{gemini_positive_prompt}, {final_style_prompt}".strip(", ").strip()
            if final_style_prompt else gemini_positive_prompt
        )
    output = None
    parsing = None  # Set by CPU path; may be None if cloud succeeds
    warped_mask = None  # Set by CPU path; used by HairOverlay
    garment_category = None

    cloud_cloth_type = "upper"
    try:
        cloud_mask = build_cloth_mask(cloth_rgb)
        if category_lock == "auto":
            garment_category = detect_garment_category(cloud_mask)
            pipeline_info.append(f"Category:auto:{garment_category}")
        else:
            garment_category = _locked_garment_category(category_lock, cloud_mask)
            pipeline_info.append(f"CategoryLock:{category_lock}->{garment_category}")
        cloud_cloth_type = _cloud_type_from_category_lock(category_lock, garment_category)
    except Exception:
        garment_category = _locked_garment_category(category_lock, None)
        cloud_cloth_type = _cloud_type_from_category_lock(category_lock, garment_category)

    # ═══ PATH 0: DRESS PIPELINE v2 (gated) ═══
    # Standalone dress flow with pose-driven silhouette, hair-underlap aware
    # mask and Telea-inpainted seed. Skips legacy CPU geometric + local
    # refinement entirely when active. Set VTON_DRESS_PIPELINE_V2=0 to A/B
    # against the legacy path.
    _dress_v2_enabled = os.getenv("VTON_DRESS_PIPELINE_V2", "1").strip() != "0"
    if garment_category == "dress" and _dress_v2_enabled and use_gen:
        try:
            from src.pipelines.dress_pipeline import run_dress_pipeline_v2
            _v2_use_cloud = bool(use_catvton_cloud)
            v2_result = run_dress_pipeline_v2(
                person_rgb, cloth_rgb,
                style_prompt=final_style_prompt,
                use_cloud=_v2_use_cloud,
                gen_steps=int(gen_steps),
                gen_guidance=float(gen_guidance),
                preserve_strength=float(os.getenv("VTON_DRESS_STRENGTH", "0.85")),
            )
            output = v2_result.image
            pipeline_info.extend(v2_result.debug)
            pipeline_info.append("DressPipeline:v2")
            if any(item == "DressV2:diffusion=cloud" for item in v2_result.debug):
                backend_used = "DressV2-Cloud"
            elif any(item == "DressV2:diffusion=local" for item in v2_result.debug):
                backend_used = "DressV2-Local"
            else:
                backend_used = "DressV2-Seed"
        except Exception as exc:
            pipeline_info.append(f"DressV2Failed:{type(exc).__name__}:{exc}")

    # ═══ PATH A: CLOUD-PRIMARY (default when use_gen=True) ═══
    # Send raw person + cloth images directly to cloud API.
    # No CPU preprocessing needed — cloud handles everything.
    if output is None and use_gen and use_catvton_cloud:
        try:
            cloud_style_prompt = final_style_prompt.strip()
            if not cloud_style_prompt:
                if category_lock != "auto":
                    cloud_style_prompt = _category_prompt_from_lock(category_lock)
                elif cloud_cloth_type == "overall":
                    cloud_style_prompt = "a realistic full-body dress matching the garment reference"
                elif cloud_cloth_type == "lower":
                    cloud_style_prompt = "realistic lower-body clothing matching the garment reference"
                else:
                    cloud_style_prompt = "a realistic top garment matching the garment reference"
            cloud_result_path, cloud_backend = generate_with_cloud_router(
                person_image_path=person_path,
                cloth_image_path=cloth_path,
                style_prompt=cloud_style_prompt,
                steps=int(gen_steps),
                guidance=float(min(gen_guidance, 7.5)),
                seed=random.randint(0, 10000),
                cloth_type=cloud_cloth_type,
            )
            cloud_rgb = read_image_rgb(cloud_result_path)

            # Lightweight post-processing for identity preservation
            cloud_rgb, post_info = _postprocess_cloud_result(cloud_rgb, person_rgb, cloth_rgb)
            output = cloud_rgb
            backend_used = cloud_backend
            pipeline_info.append("CloudVTON")
            pipeline_info.extend(post_info)
        except (CloudVTONUnavailableError, Exception) as exc:
            warning_msg = f"Cloud VTON unavailable ({exc}), falling back to local pipeline."
            pipeline_info.append("CloudFailed")

    # ═══ PATH B: CPU GEOMETRIC FALLBACK ═══
    if output is None:
        cpu_output, warped_mask, parsing, pose_box, full_pose, cpu_info, tps_ok, garment_category, dress_pattern_reference, pants_type = _run_cpu_geometric_pipeline(
            person_rgb, cloth_rgb, fit_scale, alpha, y_offset,
            category_lock=category_lock,
        )
        output = cpu_output
        pipeline_info.extend(cpu_info)
        backend_used = "CPU Pipeline"

        # ═══ PATH C: LOCAL DIFFUSION REFINEMENT (on CPU output) ═══
        # v16.9: SOTA approach — diffusion GENERATES the garment, not just refines.
        # TPS output is init_image (spatial guide). Agnostic mask lets diffusion
        # regenerate the full garment region with body conformity, folds, and shading.
        if use_gen:
            # v16.10f: Add sleeve type to prompt so diffusion preserves garment shape
            _sleeve_hint = ""
            _diff_sleeve_type = "long"  # v18.17: default
            for info_item in cpu_info:
                if info_item.startswith("DressSleeveType:"):
                    # e.g. "DressSleeveType:sleeveless:v18.15" → "sleeveless"
                    _diff_sleeve_type = info_item.split(":")[1]
                if info_item.startswith("Type:"):
                    _stype = info_item.split(":")[1]
                    if garment_category != "dress":
                        _diff_sleeve_type = _stype
                    if garment_category == "dress":
                        if _stype == "short":
                            _sleeve_hint = "short sleeve dress, "
                        elif _stype == "long":
                            _sleeve_hint = "long sleeve dress, "
                    elif _stype == "short":
                        _sleeve_hint = "cap sleeve top, preserve cap-sleeve silhouette, "
                    elif _stype == "long":
                        if garment_category == "top" and _locked_top_subtype(category_lock) == "hoodie":
                            _sleeve_hint = "long hoodie sleeves following the arms, preserve fitted sleeve shape, "
                        else:
                            _sleeve_hint = "long sleeves shirt, preserve long sleeve shape, "
            if garment_category == "dress" and _diff_sleeve_type == "sleeveless":
                _sleeve_hint = "sleeveless dress, bare arms, "
            _diff_prompt = f"{_sleeve_hint}{final_style_prompt}".strip()

            diff_output, diff_backend, diff_warning, diff_info = _run_local_diffusion_refinement(
                init_tryon=cpu_output,
                person_rgb=person_rgb,
                warped_mask=warped_mask,
                parsing=parsing,
                pose_box=pose_box,
                full_pose=full_pose,
                dress_pattern_reference=dress_pattern_reference,
                style_prompt=_diff_prompt,
                gen_steps=gen_steps,
                gen_guidance=gen_guidance,
                preserve_strength=preserve_strength,
                refiner_mode=refiner_mode,
                cloth_type=category_lock if category_lock != "auto" else (garment_category or "auto"),
                garment_category=garment_category,
                new_sleeve_type=_diff_sleeve_type,
                pants_type=pants_type,
                reference_cloth_rgb=cloth_rgb,
                gemini_negative_extra=gemini_negative_prompt,
                top_subtype=(_locked_top_subtype(category_lock) if garment_category == "top" else ""),
            )
            output = diff_output
            pipeline_info.extend(diff_info)
            if diff_backend:
                backend_used = diff_backend
            if diff_warning:
                warning_msg = f"{warning_msg}\n{diff_warning}".strip()
            if garment_category == "pants":
                output = _restore_upper_body_for_pants(output, person_rgb, parsing)
                pipeline_info.append(
                    "PantsUpperBodyRestore:v22.23_waist_clip"
                    if pants_type == "shorts" else
                    "PantsUpperBodyRestore:v19.24"
                )

    # ═══ HAIR OVERLAY (v16.9c) — SOTA repaint: paste original hair on top ═══
    # Garment extends fully under hair (from Fix A/B/C/D).
    # Paste person_rgb's hair on top. Use erode (not dilate) to keep ONLY core hair
    # pixels and avoid old-shirt bleedthrough at the hair-garment boundary.
    if parsing and "hair" in parsing:
        output, _ = _paste_original_hair_layer(output, person_rgb, parsing)
        pipeline_info.append("HairOverlay")

        if (
            garment_category == "dress"
            and warped_mask is not None
            and os.getenv("VTON_POST_HAIR_RED_CLEAN", "0").strip() == "1"
        ):
            _post_red_mask = cv2.dilate(
                (warped_mask > 20).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
                iterations=1,
            )
            _old_upper = get_clothing_mask(parsing)
            if _old_upper is not None:
                _post_red_mask = cv2.bitwise_or(
                    _post_red_mask,
                    cv2.bitwise_and(
                        _old_upper,
                        cv2.dilate(_post_red_mask, np.ones((7, 7), np.uint8), iterations=1),
                    ),
                )
            output, _post_red_fixed = _inpaint_old_red_bleed(
                output,
                _post_red_mask,
                parsing,
                protect_hair=True,
                allow_large_upper=False,
            )
            if _post_red_fixed:
                pipeline_info.append("DressPostHairRedClean:v16.70")
                output, _hair_restored_after_clean = _paste_original_hair_layer(output, person_rgb, parsing)
                if _hair_restored_after_clean:
                    pipeline_info.append("HairRestoreAfterClean:v16.72")

    # ═══ SAVE & RETURN ═══
    output = _safe_uint8(output)
    output_file = storage.outputs_dir / f"tryon_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_image_rgb(output_file, output)

    if backend_used.startswith("Local Diffusion"):
        mode_label = "Local Diffusion"
    elif backend_used == "CPU Pipeline":
        mode_label = "CPU only"
    elif use_gen and use_catvton_cloud and "CloudFailed" not in pipeline_info:
        mode_label = "Cloud AI"
    else:
        mode_label = "Local Diffusion" if use_gen else "CPU only"
    public_pipeline_info = _compact_pipeline_info(pipeline_info)
    info = (
        f"Saved: {output_file}\n"
        f"Storage base: {storage.base_dir}\n"
        f"Mode: {mode_label}\n"
        f"Backend: {backend_used}\n"
        f"Preset: {quality_preset}\n"
        f"Pipeline: {' -> '.join(public_pipeline_info)}\n"
    )

    if warning_msg:
        info = f"{info}\n{warning_msg}"

    return output, info


# ═══════════════════════════════════════════════════════════════════
#  Gradio UI (kept for backward compatibility / standalone use)
# ═══════════════════════════════════════════════════════════════════

with gr.Blocks(title="Virtual Try-On") as demo:
    gr.Markdown("# Virtual Try-On (Cloud-Primary + CPU Fallback)")
    gr.Markdown(
        "Architecture: Cloud VTON (CatVTON / IDM-VTON / Fal.ai) as primary engine. "
        "Falls back to CPU geometric pipeline + local SD-inpaint when cloud is unavailable."
    )

    with gr.Row():
        person_input = gr.Image(type="numpy", label="Anh nguoi mau")
        cloth_input = gr.Image(type="numpy", label="Anh ao/quan")

    with gr.Row():
        fit_scale = gr.Slider(0.8, 1.5, value=1.12, step=0.01, label="Do rong trang phuc")
        alpha = gr.Slider(0.4, 1.0, value=0.65, step=0.01, label="Do hoa tron")
        y_offset = gr.Slider(-0.15, 0.2, value=-0.01, step=0.01, label="Dich doc")

    with gr.Row():
        use_gen = gr.Checkbox(value=True, label="Enable AI refinement (cloud + local)")
        style_prompt = gr.Textbox(
            value="",
            placeholder="Leave empty to preserve original garment",
            label="Garment description (prompt)",
        )

    with gr.Row():
        gen_steps = gr.Slider(4, 30, value=24, step=1, label="Refine steps")
        gen_guidance = gr.Slider(0.5, 8.0, value=5.2, step=0.1, label="Refine guidance")
        preserve_strength = gr.Slider(0.25, 1.0, value=0.82, step=0.01, label="Preserve garment texture")

    quality_preset = gr.Radio(
        choices=["fast", "balanced", "hq"],
        value="hq",
        label="Quality preset",
    )

    with gr.Row():
        refiner_mode = gr.Dropdown(
            choices=["lcm", "hypersd", "dpm++", "euler", "base"],
            value="dpm++",
            label="Refiner mode",
        )
        cloth_type = gr.Dropdown(
            choices=CATEGORY_LOCK_CHOICES,
            value="auto",
            label="Category Lock",
            info="Auto = tự nhận. Chọn category để khóa pipeline (top/pants/dress/skirt/accessory).",
        )

    use_catvton_cloud = gr.Checkbox(value=True, label="Enable Cloud VTON (recommended)")
    use_gemini_prompt = gr.Checkbox(
        value=True,
        label="Gemini auto prompt",
        info="Gemini Vision phân tích ảnh trang phục để tự sinh positive/negative prompt.",
    )

    run_btn = gr.Button("Try On", variant="primary")

    with gr.Row():
        output_img = gr.Image(type="numpy", label="Result")
        output_info = gr.Textbox(label="Info", lines=4)

    run_btn.click(
        fn=try_on,
        inputs=[
            person_input,
            cloth_input,
            fit_scale,
            alpha,
            y_offset,
            use_gen,
            style_prompt,
            gen_steps,
            gen_guidance,
            preserve_strength,
            quality_preset,
            refiner_mode,
            cloth_type,
            use_catvton_cloud,
            use_gemini_prompt,
        ],
        outputs=[output_img, output_info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
