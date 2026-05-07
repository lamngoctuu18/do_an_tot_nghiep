from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import random

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
from src.tps_warp import tps_warp_cloth, warp_sleeves_to_arms, classify_garment_type, simple_affine_warp_cloth, detect_garment_category, detect_pants_landmarks, detect_pants_type
from src.gen_tryon import GenConfig, generate_tryon_image
from src.cloud_vton_router import CloudVTONUnavailableError, generate_with_cloud_router
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


def _build_dress_diffusion_seed(
    init_tryon_rgb: np.ndarray,
    gen_mask: np.ndarray,
    garment_mask: np.ndarray,
) -> np.ndarray:
    """Soften the CPU dress guide before inpainting.

    Passing the sharp warped dress directly to SD-inpaint makes LCM reconstruct
    that CPU composite almost exactly.  Keep the coarse color/print placement,
    but remove high-frequency pasted pixels so diffusion has room to redraw
    fabric folds and sleeve structure.
    """
    gen_mask = _fit_like(gen_mask, init_tryon_rgb, is_mask=True)
    garment_mask = _fit_like(garment_mask, init_tryon_rgb, is_mask=True)
    active = gen_mask > 20
    if int(active.sum()) < 500:
        return init_tryon_rgb

    soft_alpha = cv2.GaussianBlur(active.astype(np.float32), (11, 11), 2.6)
    soft_alpha = np.clip(soft_alpha * 0.84, 0.0, 0.84)[..., None]

    init_lab = cv2.cvtColor(init_tryon_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    guide_lab = init_lab.copy()
    light_low = cv2.GaussianBlur(init_lab[:, :, 0], (0, 0), 4.2)
    chroma_a = cv2.GaussianBlur(init_lab[:, :, 1], (0, 0), 1.6)
    chroma_b = cv2.GaussianBlur(init_lab[:, :, 2], (0, 0), 1.6)
    guide_lab[:, :, 0] = init_lab[:, :, 0] * 0.40 + light_low * 0.60
    guide_lab[:, :, 1] = init_lab[:, :, 1] * 0.72 + chroma_a * 0.28
    guide_lab[:, :, 2] = init_lab[:, :, 2] * 0.72 + chroma_b * 0.28
    guide = cv2.cvtColor(_safe_uint8(guide_lab), cv2.COLOR_LAB2RGB).astype(np.float32)

    garment_active = (garment_mask > 20) & active
    if int(garment_active.sum()) > 200:
        median_rgb = np.median(init_tryon_rgb[garment_active], axis=0).astype(np.float32)
        guide[active] = guide[active] * 0.92 + median_rgb * 0.08

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

        lab = cv2.cvtColor(_safe_uint8(guide), cv2.COLOR_RGB2LAB).astype(np.float32)
        fold_mask = cv2.GaussianBlur(garment_active.astype(np.float32), (17, 17), 5.0)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + fold_delta * fold_mask, 0, 255)
        guide = cv2.cvtColor(_safe_uint8(lab), cv2.COLOR_LAB2RGB).astype(np.float32)

    seed = (
        init_tryon_rgb.astype(np.float32) * (1.0 - soft_alpha)
        + guide.astype(np.float32) * soft_alpha
    )
    return _safe_uint8(seed)


def _restore_dress_print_detail(
    generated_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    garment_mask: np.ndarray,
    detail_strength: float = 0.55,
) -> np.ndarray:
    """Put source print detail back without flattening diffusion lighting."""
    reference_rgb = _fit_like(reference_rgb, generated_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, generated_rgb, is_mask=True)
    mask = garment_mask > 20
    if int(mask.sum()) < 500:
        return generated_rgb

    mask_core = cv2.erode(mask.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
    mask_f = cv2.GaussianBlur((mask_core > 20).astype(np.float32), (9, 9), 2.2)
    mask_f = np.clip(mask_f * float(np.clip(detail_strength, 0.0, 1.0)), 0.0, 1.0)

    gen_lab = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    ref_l = ref_lab[:, :, 0]
    gen_l = gen_lab[:, :, 0]
    ref_detail = ref_l - cv2.GaussianBlur(ref_l, (0, 0), 2.4)
    ref_detail = np.clip(ref_detail, -38.0, 38.0)
    gen_lab[:, :, 0] = np.clip(gen_l + ref_detail * mask_f, 0, 255)

    chroma_alpha = np.clip(mask_f * 0.72, 0.0, 0.72)
    gen_lab[:, :, 1] = gen_lab[:, :, 1] * (1.0 - chroma_alpha) + ref_lab[:, :, 1] * chroma_alpha
    gen_lab[:, :, 2] = gen_lab[:, :, 2] * (1.0 - chroma_alpha) + ref_lab[:, :, 2] * chroma_alpha
    return cv2.cvtColor(_safe_uint8(gen_lab), cv2.COLOR_LAB2RGB)


def _restore_dress_crisp_source_texture(
    refined_rgb: np.ndarray,
    source_rgb: np.ndarray,
    garment_mask: np.ndarray,
    detail_strength: float = 0.88,
    chroma_strength: float = 0.90,
) -> np.ndarray:
    """Restore sharp source print while keeping refined broad lighting."""
    source_rgb = _fit_like(source_rgb, refined_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, refined_rgb, is_mask=True)
    mask = garment_mask > 20
    if int(mask.sum()) < 500:
        return refined_rgb

    core = cv2.erode(mask.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
    mask_f = cv2.GaussianBlur((core > 20).astype(np.float32), (5, 5), 1.1)
    mask_f = np.clip(mask_f * float(np.clip(detail_strength, 0.0, 1.0)), 0.0, 1.0)

    refined_lab = cv2.cvtColor(refined_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    refined_l = refined_lab[:, :, 0]
    source_l = source_lab[:, :, 0]
    refined_low = cv2.GaussianBlur(refined_l, (0, 0), 7.0)
    source_low = cv2.GaussianBlur(source_l, (0, 0), 7.0)
    source_detail = np.clip(source_l - source_low, -58.0, 58.0)
    crisp_l = np.clip(refined_low + source_detail, 0, 255)

    refined_lab[:, :, 0] = refined_l * (1.0 - mask_f) + crisp_l * mask_f
    chroma_alpha = np.clip(mask_f * float(np.clip(chroma_strength, 0.0, 1.0)), 0.0, 1.0)
    refined_lab[:, :, 1] = refined_lab[:, :, 1] * (1.0 - chroma_alpha) + source_lab[:, :, 1] * chroma_alpha
    refined_lab[:, :, 2] = refined_lab[:, :, 2] * (1.0 - chroma_alpha) + source_lab[:, :, 2] * chroma_alpha
    return cv2.cvtColor(_safe_uint8(refined_lab), cv2.COLOR_LAB2RGB)


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
    upper_r = max(5, int(sw * 0.13))
    lower_r = max(4, int(sw * 0.105))

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
            k = (max(5, int(sw * 0.11)) | 1)
            arm_u8 = cv2.dilate(
                arm_u8,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
                iterations=1,
            )
            env = cv2.bitwise_or(env, arm_u8)

    top_limit = max(0, int(sh_p[1] - sw * 0.18))
    env[:top_limit, :] = 0
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
    allowed = np.clip(allowed * 1.12, 0.0, 1.0)
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
            int(np.clip(max(gen_steps, 25), 12, 30)),
            float(np.clip(gen_guidance, 1.0, 3.0)),
            float(np.clip(preserve_strength, 0.80, 1.0)),
            mode,
        )

    # balanced
    return (
        float(np.clip(fit_scale, 0.95, 1.15)),
        float(np.clip(alpha, 0.88, 1.0)),
        int(np.clip(gen_steps, 6, 20)),
        float(np.clip(gen_guidance, 0.8, 2.5)),
        float(np.clip(preserve_strength, 0.75, 0.95)),
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
        for key in ("face", "hair", "hat", "sunglasses"):
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


def _apply_dress_gpu_refine_layer(
    source_rgb: np.ndarray,
    diffusion_rgb: np.ndarray,
    garment_mask: np.ndarray,
    strength: float = 0.58,
) -> np.ndarray:
    """Use GPU diffusion for dress folds/lighting without redrawing the print.

    This is the dress equivalent of the top GPU refine path: diffusion owns the
    cloth lighting and drape cues, while source RGB keeps the exact print so SD
    does not hallucinate a second/generated pattern layer.
    """
    diffusion_rgb = _fit_like(diffusion_rgb, source_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, source_rgb, is_mask=True)
    mask = garment_mask > 20
    if int(mask.sum()) < 500:
        return source_rgb

    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (19, 19), 6.0)
    mask_f = np.clip(mask_f * float(np.clip(strength, 0.0, 1.0)) * 0.82, 0.0, 0.82)

    src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dif_lab = cv2.cvtColor(diffusion_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_l = src_lab[:, :, 0]
    dif_l = cv2.GaussianBlur(dif_lab[:, :, 0], (0, 0), 1.8)

    src_low = cv2.GaussianBlur(src_l, (0, 0), 16.0)
    dif_low = cv2.GaussianBlur(dif_l, (0, 0), 16.0)
    broad_delta = dif_low - src_low
    broad_delta = broad_delta - float(np.median(broad_delta[mask]))
    broad_delta = np.clip(broad_delta, -28.0, 28.0)

    src_mid = cv2.GaussianBlur(src_l, (0, 0), 4.8) - src_low
    dif_mid = cv2.GaussianBlur(dif_l, (0, 0), 4.8) - dif_low
    fold_delta = cv2.GaussianBlur(dif_mid - src_mid, (0, 0), 3.2)
    fold_delta = np.clip(fold_delta, -24.0, 24.0)

    delta = np.clip(
        broad_delta * 0.72 + fold_delta * 0.34,
        -30.0,
        30.0,
    )
    src_lab[:, :, 0] = np.clip(src_l + delta * mask_f, 0, 255)
    return cv2.cvtColor(_safe_uint8(src_lab), cv2.COLOR_LAB2RGB)


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
    min_hem_width = max(54, int(ref_width * 0.92))
    min_mid_width = max(min_hem_width, int(ref_width * 1.02))

    body_center_x = float(np.median(xs))
    if full_pose is not None and "left_hip" in full_pose and "right_hip" in full_pose:
        body_center_x = float((full_pose["left_hip"][0] + full_pose["right_hip"][0]) * 0.5)

    result = mask.copy()
    start_y = y1 + int(dress_h * 0.52)
    for row in range(start_y, y2 + 1):
        progress = (row - start_y) / max(1, y2 - start_y)
        target_w = int((1.0 - progress) * min_mid_width + progress * min_hem_width)
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
) -> tuple[np.ndarray, bool]:
    """Keep the dress footprint close to the model shoulder/waist/hip size."""
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
    width_curve = np.array([
        [0.00, sw * 0.62],
        [0.18, sw * 0.56],
        [0.42, max(sw * 0.48, hip_w * 0.72)],
        [0.66, max(sw * 0.50, hip_w * 0.78)],
        [1.00, max(sw * 0.54, hip_w * 0.86)],
    ], dtype=np.float32)

    for y in range(top_y, min(h, bot_y + 1)):
        f = (y - top_y) / max(1.0, float(bot_y - top_y))
        cx_f = min(1.0, max(0.0, (y - shoulder_y) / max(1.0, hip_y - shoulder_y)))
        cx = shoulder_cx * (1.0 - cx_f) + hip_cx * cx_f
        half_w = float(np.interp(f, width_curve[:, 0], width_curve[:, 1]))
        x_l = max(0, int(round(cx - half_w)))
        x_r = min(w - 1, int(round(cx + half_w)))
        env[y, x_l:x_r + 1] = 255

    env = cv2.GaussianBlur(env, (9, 9), 2.0)
    env = (env > 24).astype(np.uint8) * 255
    clipped = cv2.bitwise_and(mask, env)

    old_area = int(cv2.countNonZero(mask))
    new_area = int(cv2.countNonZero(clipped))
    if new_area < max(500, int(old_area * 0.58)):
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


def _garment_texture_valid_mask(image_rgb: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """Select real garment texture pixels, excluding flat neutral warp fill."""
    base = base_mask > 20
    if int(base.sum()) < 50:
        return base

    img_i = image_rgb.astype(np.int16)
    rgb_sum = img_i.sum(axis=2)
    chroma = img_i.max(axis=2) - img_i.min(axis=2)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    local_sq = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    base_pixels = image_rgb[base]
    median_rgb = np.median(base_pixels, axis=0).astype(np.float32) if len(base_pixels) else np.array([128, 128, 128], dtype=np.float32)
    color_dist = np.mean(np.abs(image_rgb.astype(np.float32) - median_rgb[None, None, :]), axis=2)

    neutral_fill = (rgb_sum > 330) & (rgb_sum < 505) & (chroma < 28) & (local_std < 9.0)
    flat_fill = (rgb_sum > 250) & (chroma < 34) & (local_std < 4.5)
    too_dark = rgb_sum < 45
    pattern_detail = (color_dist > 26.0) | (chroma > 32) | (rgb_sum < 330) | (rgb_sum > 520)
    valid = base & pattern_detail & ~neutral_fill & ~flat_fill & ~too_dark

    if int(valid.sum()) < max(30, int(base.sum() * 0.08)):
        valid = base & ~neutral_fill & ~flat_fill & ~too_dark
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

    valid = _garment_texture_valid_mask(warped_cloth, existing.astype(np.uint8) * 255)
    row_filled, row_fill_mask = _repeat_row_texture_into_mask(warped_cloth, support, valid)

    return _propagate_texture_into_mask(
        row_filled,
        support,
        valid | row_fill_mask,
        max_iter=120,
    )


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
) -> tuple[np.ndarray, np.ndarray, dict | None, object, dict | None, list[str], bool]:
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
    try:
        cloth_mask = segment_cloth_u2net(cloth_rgb)
        pipeline_info.append("U2Net")
    except Exception:
        cloth_mask = build_cloth_mask(cloth_rgb)
        pipeline_info.append("MaskFallback")

    # Optional SegFormer merge for better mask quality
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
    except Exception:
        pass

    # v16.7f: Erode cloth mask by 3px BEFORE any further processing.
    # U2Net mask edges contain semi-transparent pixels that blend garment with
    # white background. Stronger erode removes this fringe at source.
    cloth_mask = cv2.erode(cloth_mask, np.ones((5, 5), np.uint8), iterations=2)
    # Re-threshold to binary after erode (remove any partial values)
    cloth_mask = ((cloth_mask > 127).astype(np.uint8)) * 255

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
    garment_category = detect_garment_category(cloth_mask)
    pipeline_info.append(f"Category:{garment_category}")
    print(f"[GARMENT] Category detected: {garment_category}")

    if garment_category == "pants" and full_pose is not None:
        pants_type = detect_pants_type(cloth_mask)
        pants_landmarks = detect_pants_landmarks(cloth_mask)
        leg_meas = compute_leg_measurements(full_pose)
        print(f"[GARMENT] Pants type: {pants_type}")
        print(f"[GARMENT] Pants landmarks: { {k: f'({v[0]:.0f},{v[1]:.0f})' for k, v in pants_landmarks.items()} }")
        print(f"[GARMENT] Leg measurements: hip_w={leg_meas['hip_width']:.0f}px leg_len={leg_meas['leg_length']:.0f}px")
        pipeline_info.append(f"PantsType:{pants_type}")
    elif garment_category == "dress":
        print(f"[GARMENT] Dress detected — full-body coverage mode")

    # ── Step 3b: Classify garment type (short/long sleeve, loose/tight) ──
    garment_info = classify_garment_type(cloth_mask)
    new_sleeve_type = garment_info["sleeve_type"]  # "short", "long", "sleeveless"
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

                # Scale pants to hip width (waistband ~1.1x hip width)
                scale_ratio = float(np.clip((hip_w * 1.1) / cloth_w, 0.4, 2.5))
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
    elif full_pose is not None and garment_category != "dress":
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
            scale_x = (target_width * 1.20) / cloth_w_actual
            if garment_category == "dress":
                # v16.11d: Dress scale — shoulder width + full body height (shoulder to ankle)
                la_y = full_pose.get("left_ankle", full_pose.get("left_knee", [0, h_out * 0.92]))[1]
                ra_y = full_pose.get("right_ankle", full_pose.get("right_knee", [0, h_out * 0.92]))[1]
                body_h = max(abs(la_y - ls[1]), abs(ra_y - rs[1]), target_width * 2.0)
                cloth_h_actual = max(1, cloth_y2 - cloth_y1)
                scale_y = (body_h * 0.95) / cloth_h_actual
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
                scale_y = (torso_h * 1.10) / cloth_h_actual
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
    if full_pose is not None and garment_category not in ("pants", "dress"):
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
        # v16.11c: Pants use affine warping (simple hip-based alignment)
        try:
            warped_cloth, warped_mask = simple_affine_warp_cloth(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=(h_out, w_out),
            )
            pipeline_info.append("Affine_pants")
        except Exception as e:
            pipeline_info.append(f"Affine_pants_fail({e})")
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
    if full_pose is not None:
        _ls = full_pose["left_shoulder"]
        _rs = full_pose["right_shoulder"]
        _lh = full_pose.get("left_hip", _ls)
        _rh = full_pose.get("right_hip", _rs)
        _sw = abs(_rs[0] - _ls[0])
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
        pose_fit_body, _pose_fit_applied = _fit_dress_body_mask_to_pose(
            dress_body_support_mask,
            full_pose,
        )
        if _pose_fit_applied:
            dress_body_support_mask = pose_fit_body
            pipeline_info.append("DressBodyPoseFit:v16.72")

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

        # Build per-pixel fill: dress colour only inside the real warp footprint
        # (tiny 1px safety dilation), background everywhere else. This removes
        # the brown/grey under-dress silhouette at the source.
        _wm_inside = cv2.dilate(
            (garment_support_mask > 20).astype(np.uint8) * 255,
            np.ones((9, 9), np.uint8),
            iterations=1,
        )
        _inside_f = (_wm_inside > 0).astype(np.float32)[..., None]
        _fill_rgb = (
            _inside_f * _dress_rgb[None, None, :]
            + (1.0 - _inside_f) * _bg_rgb[None, None, :]
        )
        person_cleaned = _safe_uint8(
            person_cleaned.astype(np.float32) * (1.0 - _hard_erase)
            + _fill_rgb * _hard_erase
        )
        pipeline_info.append("DressFullErase:v16.54")
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
        pipeline_info.append("DressTextureFill:v16.61")

    # ── Step 6b: Body Curve Map + DrapeHint ──
    # v16.9: DISABLED when diffusion is primary (strength 0.75).
    # These CPU-side shading hacks fight with diffusion's own fold/shadow generation.
    # Diffusion at 0.75 strength regenerates garment texture including natural folds,
    # body curvature shading, and drape. CPU curve/drape only adds noise to the
    # init_image that diffusion then has to correct.
    # Only kept as light hint — reduced from previous strengths.
    if full_pose is not None and garment_category != "dress":
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
        if garment_category == "dress":
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
    return init_tryon, warped_mask_full, parsing, pose_box, full_pose, pipeline_info, tps_ok, garment_category


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


def _run_local_diffusion_refinement(
    init_tryon: np.ndarray,
    person_rgb: np.ndarray,
    warped_mask: np.ndarray,
    parsing: dict | None,
    pose_box,
    style_prompt: str,
    gen_steps: int,
    gen_guidance: float,
    preserve_strength: float,
    refiner_mode: str,
    cloth_type: str,
    garment_category: str = "top",
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

    if binary_mask.sum() < 500:
        return init_tryon, "", "Garment mask too small for diffusion", pipeline_info

    # ── AGNOSTIC MASK: erase garment region so diffusion can regenerate ──
    gen_mask = binary_mask.copy()

    if garment_category == "pants":
        # For pants: erase existing pants/legs + warped pants region
        if parsing:
            old_pants = get_pants_mask(parsing)
            if old_pants is not None:
                gen_mask = cv2.bitwise_or(gen_mask, old_pants)
        # PROTECT: face/hair/arms/torso — only legs change
        protect_keys = ("face", "hair", "hat", "sunglasses",
                        "upper_clothes", "dress", "left_arm", "right_arm")

    elif garment_category == "dress":
        # v16.55: Generate dress like the TOP path: inpaint the FULL garment
        # footprint instead of only a narrow edge ring. The ring-only v16.53
        # avoided the duplicate layer, but it left the TPS warp almost intact,
        # so the result looked pasted on. Here diffusion owns the full warped
        # dress+sleeve mask and can add body-following folds/shading, while the
        # final DressSingleLayerClip still prevents pixels outside the footprint
        # from becoming a second dress.
        gen_mask = binary_mask.copy()
        gen_mask = cv2.morphologyEx(gen_mask, cv2.MORPH_CLOSE,
                                    np.ones((7, 7), np.uint8))
        gen_mask = cv2.dilate(gen_mask, np.ones((5, 5), np.uint8), iterations=1)

        # Keep neckline/collar cleanup close to the garment only. This mirrors
        # the top path's natural transition without adding arms/legs/skirt as a
        # broad separate dress silhouette.
        if parsing:
            neck_diff = get_neck_mask(parsing)
            if neck_diff is not None:
                near_garment = cv2.dilate(binary_mask, np.ones((17, 17), np.uint8), iterations=1)
                gen_mask = cv2.bitwise_or(gen_mask, cv2.bitwise_and(neck_diff, near_garment))
        pipeline_info.append("DressFullGenMask:v16.55")

        # Do not protect arms for dress: if sleeve warp produced sleeve mask,
        # arms inside binary_mask are part of the generated garment. Regions
        # outside binary_mask are restored after diffusion.
        protect_keys = ("face", "hair", "hat", "sunglasses",
                        "left_shoe", "right_shoe")

    else:
        # For tops: erase existing upper clothes + warped top region
        if parsing:
            old_clothes = get_clothing_mask(parsing)
            if old_clothes is not None:
                gen_mask = cv2.bitwise_or(gen_mask, old_clothes)
            neck_diff = get_neck_mask(parsing)
            if neck_diff is not None:
                gen_mask = cv2.bitwise_or(gen_mask, neck_diff)
        # PROTECT: face/hair/hat/sunglasses + lower body (pants untouched)
        protect_keys = ("face", "hair", "hat", "sunglasses",
                        "pants", "skirt", "left_leg", "right_leg",
                        "left_shoe", "right_shoe")

    # Dilate mask (dress already dilated above with larger kernel)
    if garment_category != "dress":
        gen_mask = cv2.dilate(gen_mask, np.ones((21, 21), np.uint8), iterations=1)

    # Apply protection mask
    if parsing:
        protect_mask = np.zeros((h, w), dtype=np.uint8)
        for pkey in protect_keys:
            if pkey in parsing:
                protect_mask = cv2.bitwise_or(protect_mask, parsing[pkey])
        protect_mask = cv2.dilate(protect_mask, np.ones((5, 5), np.uint8), iterations=1)
        gen_mask = cv2.subtract(gen_mask, protect_mask)

    # SOTA: Soft edge blur at mask boundary (CatVTON uses kernel=height/50)
    blur_k = max(7, (h // 50) | 1)  # ~10px at 512, always odd
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
        # v17.3: Keep dress diffusion real, but stop asking SD to redesign the
        # garment.  DPM++ fp32 now generates correctly; high strength/guidance
        # made it invent a new dress print instead of adding folds to the seed.
        _preserve = float(np.clip(preserve_strength, 0.25, 1.0))
        if diffusion_mode in {"dpm++", "euler", "base"}:
            _diff_strength = float(np.clip(0.76 - 0.10 * _preserve, 0.62, 0.74))
            _diff_guidance = 3.2
        else:
            _diff_strength = float(np.clip(0.72 - 0.08 * _preserve, 0.60, 0.70))
            _diff_guidance = 3.6
        # Extract dominant dress colour from TPS torso region
        _dress_region = binary_mask > 127
        _dress_color = np.array([128, 128, 128], dtype=np.float32)
        if _dress_region.sum() > 200:
            _dr = float(init_tryon[:, :, 0][_dress_region].mean())
            _dg = float(init_tryon[:, :, 1][_dress_region].mean())
            _db = float(init_tryon[:, :, 2][_dress_region].mean())
            _dress_color = np.array([_dr, _dg, _db], dtype=np.float32)
            _dominant = "dark" if (_dr + _dg + _db) < 300 else "light"
            # v16.13: Inject concrete colour anchor + print-preservation hints to
            # stop LCM from hallucinating rainbow/tie-dye. We sample 3 anchor
            # colours (dark / mid / light quantile) from the garment region and
            # name them so the prompt acts as a colour clamp.
            try:
                _reg_pixels = init_tryon[_dress_region].reshape(-1, 3).astype(np.float32)
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
                            if v < 110: return "dark grey"
                            if v < 180: return "grey"
                            if v < 225: return "light grey"
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
                        f"same exact source dress on the person, preserve the original {_anchor} abstract print and color layout, "
                        f"do not redesign the dress, natural body-following fabric folds, soft waist drape, "
                        f"subtle sleeve wrinkles, clean shoulders neckline, crisp hem"
                    ).strip(", ")
                    print(f"[DIFFUSION] Dress colour anchor: {_anchor}")
            except Exception:
                style_prompt = (
                    "same exact source dress on the person, preserve the original black beige abstract print and color layout, "
                    "do not redesign the dress, natural body-following fabric folds, soft waist drape, "
                    "subtle sleeve wrinkles, clean shoulders neckline, crisp hem"
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

    try:
        # v16.13: For dress, request higher inference resolution (640) on GPU so
        # complex prints (animal print, paisley, etc.) survive the VAE round-trip.
        # generate_tryon_image falls back to 512 on CPU automatically.
        # v16.13b: Default OFF (=512) to avoid OOM on 4GB GPUs. Opt-in with
        # env VTON_DRESS_INFER=640 (or 768) when VRAM allows.
        try:
            _env_infer = int(os.environ.get("VTON_DRESS_INFER", "0"))
        except Exception:
            _env_infer = 0
        _infer_size = _env_infer if (garment_category == "dress" and _env_infer >= 512) else 0
        if garment_category == "dress" and diffusion_mode in {"lcm", "hypersd"}:
            _infer_steps = int(np.clip(gen_steps, 10, 14))
        elif garment_category == "dress" and diffusion_mode in {"dpm++", "euler", "base"}:
            _infer_steps = int(np.clip(max(gen_steps, 24), 22, 30))
        else:
            _infer_steps = max(gen_steps, 20)
        diffusion_init = init_tryon
        if garment_category == "dress":
            diffusion_init = _build_dress_diffusion_seed(init_tryon, gen_mask_soft, binary_mask)
            _debug_save("09b_dress_diffusion_seed", diffusion_init)
        generated = generate_tryon_image(
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
                infer_size=_infer_size,
                negative_prompt=(
                    "logo, text, graphic, animal face, different dress, changed print, "
                    "leopard spots, spotted print, polka dots, dalmatian print, round spots, "
                    "new pattern, geometric panels, vertical panels, diagonal panels, triangular panels, "
                    "metallic panels, color block panels, glossy satin, armor, eyes, redesigned dress, "
                    "plain fabric, inner dress, second dress, double layer, inner panel, "
                    "shoulder pads, puffy shoulders, bulky shoulders, cape shoulders, "
                    "duplicate sleeves, extra sleeves, pasted sleeves, arm overlay, bare arm under sleeve, old red sleeve, "
                    "cinched waist, tight waist, belt, sash, old red shirt, "
                    "shorts visible, transparent skirt, missing skirt, open skirt, "
                    "old clothing visible, cpu warp, geometric warp, flat pasted cloth, sticker effect, no folds, "
                    "vertical streaks, scratched texture, torn fabric, grey patch, blurry, low quality, bad anatomy"
                ) if garment_category == "dress" else GenConfig().negative_prompt,
            ),
        )

        # Sanitize + size-match
        generated = _sanitize_rgb_output(generated)
        generated = _fit_like(generated, init_tryon, is_mask=False)
        binary_mask = _fit_like(binary_mask, generated, is_mask=True)
        if garment_category == "dress":
            _debug_save("10a_dress_model_raw", generated)
            generated = _apply_color_consistency(generated, init_tryon_clean, binary_mask, strength=0.16)
            generated = _restore_dress_print_detail(generated, init_tryon_clean, binary_mask, detail_strength=0.62)
            _debug_save("10b_dress_detail_restored", generated)

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
                # v16.9: Lighter recovery (0.50) — keep more diffusion output
                generated = _enforce_garment_identity(generated, init_tryon, binary_mask, 0.50)
                pipeline_info.append("DiffusionRecover")
                print(f"[DIFFUSION] RECOVER — brightness {_output_brightness:.0f} vs {_input_brightness:.0f}")

        # v16.9: NO duplicate repaint here.
        # gen_tryon.py already does repaint: keeps init_tryon (CPU output with new garment)
        # outside the mask, diffusion output inside the mask. Adding a second repaint
        # with person_rgb would overwrite the CPU garment with old shirt pixels.
        # The diffusion output (generated) already has correct compositing from gen_tryon.
        output = generated
        if garment_category == "dress":
            _debug_save("10_dress_diffusion_raw", generated)
            if os.getenv("VTON_DRESS_DIFFUSION_PRIMARY", "0").strip() == "1":
                output = generated.copy()
                pipeline_info.append("DressDiffusionPrimaryOptIn:v17.4")
            else:
                output = _apply_dress_gpu_refine_layer(
                    init_tryon_clean,
                    generated,
                    binary_mask,
                    strength=float(np.clip(preserve_strength, 0.25, 1.0)),
                )
                output = _restore_dress_crisp_source_texture(
                    output,
                    init_tryon_clean,
                    binary_mask,
                    detail_strength=0.92,
                    chroma_strength=0.94,
                )
                _debug_save("10c_dress_gpu_light_refine", output)
                pipeline_info.append("DressGpuLightRefine:v17.5")
                pipeline_info.append("DressCrispTexture:v17.5")

        # v16.17: DressShapeLock REMOVED. The previous post-blend of 45% TPS
        # onto diffusion was re-introducing red/pink bleed because
        # init_tryon_clean carried residual colour contamination from
        # person_cleaned (old red shirt pixels on arms). With v16.17's full
        # body erase + neutral grey fill, diffusion already receives a clean
        # init — output is the final answer.
        if garment_category == "dress":
            # v16.53 DressSingleLayerClip: Hard-clip diffusion output OUTSIDE
            # the warp footprint back to the CPU init. Even with a tightened
            # gen_mask, LCM can leak dress pixels into adjacent regions via
            # the soft-mask repaint. Restoring init_tryon outside binary_mask
            # guarantees only ONE visible dress layer (the warp) and removes
            # any second/ghost dress silhouette behind it.
            try:
                _wm_u8 = (binary_mask > 20).astype(np.uint8) * 255
                _wm_clip = cv2.dilate(_wm_u8, np.ones((3, 3), np.uint8), iterations=1)
                _wm_alpha = cv2.GaussianBlur(
                    (_wm_clip > 20).astype(np.float32), (5, 5), 1.2
                )[..., None]
                _wm_alpha = np.clip(_wm_alpha, 0.0, 1.0)
                output = _safe_uint8(
                    init_tryon_clean.astype(np.float32) * (1.0 - _wm_alpha)
                    + output.astype(np.float32) * _wm_alpha
                )
                # Restore identity regions (face/hair/arms) from person_rgb so
                # any diffusion residue on skin is removed.
                if parsing:
                    _restore = np.zeros(_wm_u8.shape, dtype=np.uint8)
                    for _rk in ("face", "hair", "hat", "sunglasses",
                                "left_arm", "right_arm"):
                        _rv = parsing.get(_rk)
                        if _rv is not None:
                            _restore = cv2.bitwise_or(_restore, _rv)
                    _restore = cv2.subtract(
                        _restore,
                        cv2.dilate(_wm_u8, np.ones((5, 5), np.uint8), iterations=1),
                    )
                    _rf = cv2.GaussianBlur(
                        (_restore > 20).astype(np.float32), (7, 7), 2.0
                    )[..., None]
                    output = _safe_uint8(
                        output.astype(np.float32) * (1.0 - _rf)
                        + person_rgb.astype(np.float32) * _rf
                    )
                pipeline_info.append("DressSingleLayerClip:v16.53")
            except Exception as _clip_exc:
                print(f"[DIFFUSION] DressSingleLayerClip skipped: {_clip_exc}")
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
            output = _suppress_dress_edge_halo(output, person_rgb, binary_mask)
            pipeline_info.append("DressDehalo:v16.59")
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
        elif item in {"Parsing", "Pose", "Pose(basic)", "U2Net", "MaskFallback", "PreFit", "CloudVTON", "CloudFailed"}:
            add(item)
        elif item.startswith(("Affine", "TPS", "Persp", "HipAlign", "ShoulderAlign", "RotAlign", "DriftFix")):
            add("Warp")
        elif item.startswith(("DressSleevePoseClip", "DressSkirtComplete", "DressBodyPoseFit", "DressShoulderSeal")):
            add("DressFit")
        elif item in {"ArmErase", "NeckErase", "SleeveErase"} or item.startswith(("DressEraseClip", "DressFullErase")):
            add("DressErase")
        elif item.startswith(("DressTextureFill", "HardDressMask", "3LayerBlend", "EraseRestore")):
            add("DressComposite")
        elif item.startswith("DressFullGenMask"):
            add("DiffusionMask")
        elif item.startswith(("DressFastLCM", "DressDiffusionPrimary", "DressCpuTextureOptIn")):
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
):
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
    output = None
    parsing = None  # Set by CPU path; may be None if cloud succeeds
    warped_mask = None  # Set by CPU path; used by HairOverlay
    garment_category = None

    cloud_cloth_type = "upper"
    try:
        cloud_mask = build_cloth_mask(cloth_rgb)
        garment_category = detect_garment_category(cloud_mask)
        if garment_category == "dress":
            cloud_cloth_type = "overall"
        elif garment_category == "pants":
            cloud_cloth_type = "lower"
    except Exception:
        garment_category = None
        cloud_cloth_type = "upper"

    # ═══ PATH A: CLOUD-PRIMARY (default when use_gen=True) ═══
    # Send raw person + cloth images directly to cloud API.
    # No CPU preprocessing needed — cloud handles everything.
    if use_gen and use_catvton_cloud:
        try:
            cloud_style_prompt = (style_prompt or "").strip()
            if not cloud_style_prompt:
                if cloud_cloth_type == "overall":
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
        cpu_output, warped_mask, parsing, pose_box, full_pose, cpu_info, tps_ok, garment_category = _run_cpu_geometric_pipeline(
            person_rgb, cloth_rgb, fit_scale, alpha, y_offset,
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
            for info_item in cpu_info:
                if info_item.startswith("Type:"):
                    _stype = info_item.split(":")[1]
                    if garment_category == "dress":
                        if _stype == "short":
                            _sleeve_hint = "short sleeve dress, "
                        elif _stype == "long":
                            _sleeve_hint = "long sleeve dress, "
                    elif _stype == "short":
                        _sleeve_hint = "cap sleeve top, preserve cap-sleeve silhouette, "
                    elif _stype == "long":
                        _sleeve_hint = "long sleeves shirt, preserve long sleeve shape, "
            _diff_prompt = f"{_sleeve_hint}{style_prompt}".strip()

            diff_output, diff_backend, diff_warning, diff_info = _run_local_diffusion_refinement(
                init_tryon=cpu_output,
                person_rgb=person_rgb,
                warped_mask=warped_mask,
                parsing=parsing,
                pose_box=pose_box,
                style_prompt=_diff_prompt,
                gen_steps=gen_steps,
                gen_guidance=gen_guidance,
                preserve_strength=preserve_strength,
                refiner_mode=refiner_mode,
                cloth_type=cloth_type,
                garment_category=garment_category,
            )
            output = diff_output
            pipeline_info.extend(diff_info)
            if diff_backend:
                backend_used = diff_backend
            if diff_warning:
                warning_msg = f"{warning_msg}\n{diff_warning}".strip()

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

    mode_label = (
        "Cloud+Refine"
        if use_gen and use_catvton_cloud
        else ("Local Diffusion" if use_gen else "CPU only")
    )
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
        gen_steps = gr.Slider(4, 30, value=20, step=1, label="Refine steps")
        gen_guidance = gr.Slider(0.5, 8.0, value=1.5, step=0.1, label="Refine guidance")
        preserve_strength = gr.Slider(0.25, 1.0, value=0.60, step=0.01, label="Preserve garment texture")

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
            choices=["auto", "tshirt", "hoodie", "jacket", "dress",
                     "pants", "jeans", "shorts", "generic"],
            value="auto",
            label="Cloth type",
        )

    use_catvton_cloud = gr.Checkbox(value=True, label="Enable Cloud VTON (recommended)")

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
        ],
        outputs=[output_img, output_info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
