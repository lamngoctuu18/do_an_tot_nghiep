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
from src.tps_warp import tps_warp_cloth, warp_sleeves_to_arms, classify_garment_type, simple_affine_warp_cloth, body_fit_warp_dress, detect_garment_category, detect_pants_landmarks, detect_pants_type
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


def _apply_texture_preserving_dress_folds(
    garment_rgb: np.ndarray,
    garment_mask: np.ndarray,
    full_pose: dict | None,
) -> np.ndarray:
    """Add subtle fold lighting without changing the source print."""
    mask = garment_mask > 30
    if int(mask.sum()) < 500:
        return garment_rgb

    h, w = garment_mask.shape[:2]
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    if full_pose is not None and "left_shoulder" in full_pose and "right_shoulder" in full_pose:
        ls = np.array(full_pose["left_shoulder"], dtype=np.float32)
        rs = np.array(full_pose["right_shoulder"], dtype=np.float32)
        center_x = float((ls[0] + rs[0]) * 0.5)
    else:
        center_x = float((x1 + x2) * 0.5)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx - center_x) / max(1.0, width * 0.5)
    yn = (yy - y1) / max(1.0, height)

    side_shadow = -0.105 * np.exp(-((np.abs(xn) - 0.62) ** 2) / 0.035)
    center_highlight = 0.065 * np.exp(-(xn ** 2) / 0.18) * (0.35 + 0.65 * yn)
    fine_folds = 0.044 * np.sin((xn * 3.8 + yn * 1.4) * np.pi) * (0.25 + 0.75 * yn)
    waist_shadow = -0.040 * np.exp(-((yn - 0.42) ** 2) / 0.018) * np.exp(-(xn ** 2) / 0.72)
    hem_shadow = -0.060 * np.clip((yn - 0.72) / 0.28, 0.0, 1.0)

    fold = side_shadow + center_highlight + fine_folds + waist_shadow + hem_shadow
    fold = cv2.GaussianBlur(fold.astype(np.float32), (0, 0), 5.0)
    fold_mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 5.0)
    fold = fold * np.clip(fold_mask, 0.0, 1.0)

    out = garment_rgb.astype(np.float32) * (1.0 + fold[..., None])
    return _safe_uint8(out)


def _blend_luminance_from_diffusion(
    base_rgb: np.ndarray,
    diffusion_rgb: np.ndarray,
    garment_mask: np.ndarray,
    strength: float = 0.45,
    broad_strength: float = 0.45,
) -> np.ndarray:
    """Use diffusion for fold lighting only, preserving base color/print."""
    diffusion_rgb = _fit_like(diffusion_rgb, base_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, base_rgb, is_mask=True)

    mask_f = (garment_mask > 30).astype(np.float32)
    if int(mask_f.sum()) < 500:
        return base_rgb
    mask_f = cv2.GaussianBlur(mask_f, (15, 15), 5.0)
    mask_f = np.clip(mask_f * np.clip(strength, 0.0, 1.0), 0.0, 1.0)

    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff_lab = cv2.cvtColor(diffusion_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_l = base_lab[:, :, 0]
    diff_l = diff_lab[:, :, 0]

    # Keep medium-frequency fold lighting and a controlled amount of broad
    # drape shading. Removing only global exposure drift avoids recoloring the
    # original garment while still letting GPU add body-following folds.
    base_blur = cv2.GaussianBlur(base_l, (0, 0), 11.0)
    diff_blur = cv2.GaussianBlur(diff_l, (0, 0), 11.0)
    medium_delta = (diff_l - diff_blur) - (base_l - base_blur)

    broad_delta = diff_blur - base_blur
    active = mask_f > 0.05
    if int(active.sum()) > 200:
        broad_delta = broad_delta - float(np.median(broad_delta[active]))

    delta = np.clip(
        medium_delta + broad_delta * np.clip(broad_strength, 0.0, 1.0),
        -42.0,
        42.0,
    )

    base_lab[:, :, 0] = np.clip(base_l + delta * mask_f, 0, 255)
    return cv2.cvtColor(_safe_uint8(base_lab), cv2.COLOR_LAB2RGB)


def _apply_body_shade_from_person(
    garment_rgb: np.ndarray,
    person_rgb: np.ndarray,
    garment_mask: np.ndarray,
    strength: float = 0.55,
) -> np.ndarray:
    """v16.46: Transfer real body shading (chest/waist/side shadows) from the
    original person photo onto the warped garment.

    The original person already wears clothing whose luminance follows the body
    curvature (highlights on chest/abdomen, shadows on the sides where the body
    rounds away from light, darker waist crease, hem shadow, etc.). We extract
    that low-frequency luminance pattern from `person_rgb` inside the garment
    region, normalise it around its mean, and modulate the garment luminance
    accordingly. Pattern/colour are preserved (we only scale L in LAB)."""
    person_rgb = _fit_like(person_rgb, garment_rgb, is_mask=False)
    garment_mask = _fit_like(garment_mask, garment_rgb, is_mask=True)

    mask_bool = garment_mask > 30
    if int(mask_bool.sum()) < 800:
        return garment_rgb

    person_lab = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    person_l = person_lab[:, :, 0]

    # Low-frequency body shading only — large blur removes fabric texture / print.
    body_shade = cv2.GaussianBlur(person_l, (0, 0), 22.0)
    inside_vals = body_shade[mask_bool]
    if inside_vals.size < 400:
        return garment_rgb
    mean_val = float(np.median(inside_vals))
    shade_delta = body_shade - mean_val
    # Limit magnitude so pattern luminance is not crushed.
    shade_delta = np.clip(shade_delta, -28.0, 22.0)

    mask_f = cv2.GaussianBlur(mask_bool.astype(np.float32), (21, 21), 7.0)
    mask_f = np.clip(mask_f * float(np.clip(strength, 0.0, 1.5)), 0.0, 1.0)

    garment_lab = cv2.cvtColor(garment_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    garment_lab[:, :, 0] = np.clip(garment_lab[:, :, 0] + shade_delta * mask_f, 0, 255)
    return cv2.cvtColor(_safe_uint8(garment_lab), cv2.COLOR_LAB2RGB)


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
            # v16.47: Body-fit dress warp — anchors neck/shoulder/waist/hip on
            # the person while preserving the original dress's per-row flare
            # below the hip. This fixes the "pasted-on" feel: the dress now
            # follows the actual torso shape (snug at waist, full at hips,
            # skirt extends to ankles with the source dress's silhouette).
            warped_cloth, warped_mask = body_fit_warp_dress(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=(h_out, w_out),
            )
            pipeline_info.append("BodyFit_dress")
        except Exception as e:
            pipeline_info.append(f"BodyFit_dress_fail({e})")
            try:
                warped_cloth, warped_mask = simple_affine_warp_cloth(
                    cloth_rgb=scaled_cloth,
                    cloth_mask=scaled_mask,
                    pose=full_pose,
                    output_shape=(h_out, w_out),
                    garment_category="dress",
                )
                pipeline_info.append("Affine_dress")
            except Exception as e2:
                pipeline_info.append(f"Affine_dress_fail({e2})")
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

        # v16.28: For DRESS, clip warped_mask to the PERSON SILHOUETTE so the
        # paste never extends beyond the actual body (which causes a visible
        # outline of the warped torso sitting on top of the diffusion result,
        # i.e. the 'two-layer' look). Diffusion still owns the boundary ring.
        # v16.36: Use ERODED silhouette (5px) instead of dilated, so the warp
        # never bleeds outside body — the phantom dress on the left was the
        # warp leaking past the silhouette boundary.
        if garment_category == "dress" and parsing:
            _sil = np.zeros(warped_mask.shape, dtype=np.uint8)
            for _sk in (
                "face", "hair", "hat", "sunglasses",
                "upper_clothes", "dress", "coat", "scarf",
                "left_arm", "right_arm", "neck",
                "pants", "skirt", "left_leg", "right_leg",
                "left_shoe", "right_shoe",
            ):
                _sv = parsing.get(_sk)
                if _sv is not None:
                    _sil = cv2.bitwise_or(_sil, _sv)
            if int(_sil.sum()) > 255 * 500:
                _sil = cv2.morphologyEx(_sil, cv2.MORPH_CLOSE,
                                        np.ones((11, 11), np.uint8))
                # v16.36: ERODE 3px instead of dilate 5px so warp stays
                # strictly inside body silhouette, no overflow.
                _sil = cv2.erode(_sil, np.ones((3, 3), np.uint8), iterations=1)
                warped_mask = cv2.bitwise_and(warped_mask, _sil)
                pipeline_info.append("WarpClipSil")

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

    # ── Step 4c: Sleeve warp (LONG sleeves) ──
    # v16.41: Allow dress long sleeves again. Affine dress warp preserves the
    # torso print, but it often misses the user's arms; because dress erase
    # removes full arms, skipping sleeve warp leaves gray fill columns.
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
                    sleeve_data[side] = (s_rgb, s_mask_f)
                    _debug_save(f"04c_sleeve_{side}_rgb", s_rgb)
                    _debug_save(f"04c_sleeve_{side}_mask", (s_mask_f * 255).astype(np.uint8), is_mask=True)
                pipeline_info.append("Sleeves")
        except Exception:
            pass  # Sleeve warp failed, keep torso-only result

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
        if face_m is not None:
            neck_skin_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 10))
            neck_skin = cv2.dilate(face_m, neck_skin_k, iterations=1)
            face_ys = np.where(face_m > 0)[0]
            if len(face_ys) > 5:
                neck_skin[:int(face_ys.max() * 0.97), :] = 0
            preserve_mask = cv2.bitwise_or(preserve_mask, neck_skin)
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
                # v16.26: For dress, ERASE FULL ARMS (not just sleeve overlap).
                # The dress sleeve from affine warp will cover them; if some
                # arm pixels remain, the composite shows old skin / old shirt
                # poking out next to the dress -> looks like "layered" output.
                # Skin will be repainted by the dress mean colour fill below,
                # then refined by diffusion at the boundary ring.
                _arm_any = get_arm_mask(parsing)
                if _arm_any is not None and int(_arm_any.sum()) > 255 * 50:
                    _arm_dil = cv2.dilate(_arm_any, np.ones((9, 9), np.uint8), iterations=1)
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

    # Also erase where the new garment will go (union with warped mask)
    wm_bin = (warped_mask > 20).astype(np.uint8) * 255
    erase_mask = cv2.bitwise_or(erase_mask, wm_bin)

    # Subtract preserve areas — face/hair/neck/arms must NOT be erased
    erase_mask = cv2.subtract(erase_mask, preserve_mask)
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
        # v16.25: Split the fill by WHETHER the pixel lies inside the warped
        # dress or not:
        #   - Inside warped_mask (dilated a bit) -> dress mean colour, so any
        #     feathering at the dress alpha edge blends invisibly.
        #   - Outside warped_mask (arms / neck / exposed skin) -> SKIN colour
        #     sampled from the face, so those areas look like bare skin.
        # This prevents the "body-shaped brown silhouette" artifact when the
        # warp is narrower than the erase region.
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
                    _face_pixels = person_rgb[_fb].astype(np.float32)
                    # v16.32: Filter out RED-DOMINANT pixels (red shirt collar
                    # / lipstick / red trim leaking into the face mask). These
                    # bias _skin_rgb toward pink, then DressFullErase paints
                    # arms/cuffs/neck pink, then diffusion locks in a fake
                    # red trim look. Keep only pixels where R is not >> G+B.
                    _r = _face_pixels[:, 0]
                    _g = _face_pixels[:, 1]
                    _b = _face_pixels[:, 2]
                    _not_red = (_r < (_g + _b) * 0.75 + 25) & (_r < 230)
                    _filtered = _face_pixels[_not_red]
                    if len(_filtered) > 50:
                        _skin_rgb = _filtered.mean(axis=0)
                    else:
                        _skin_rgb = _face_pixels.mean(axis=0)
                    break
        if _skin_rgb is None:
            _skin_rgb = _dress_rgb.copy()
        # v16.28: Fill dress_mean ONLY inside the actual warp footprint
        # (dilated a few px so the alpha-edge feather blends invisibly).
        # Outside the warp -> SKIN colour. The previous v16.27 wide column
        # filled the entire body bbox with dress_mean, which produced a flat
        # pinkish-grey blob at the sides that looked like a SECOND layer
        # under the real warped torso ('chong lop' look). With skin outside,
        # diffusion has clean source pixels to either extend the dress sleeve
        # or leave bare arm skin.
        _wm_inside = cv2.dilate((warped_mask > 20).astype(np.uint8) * 255,
                                np.ones((5, 5), np.uint8), iterations=1)
        _inside_f = (_wm_inside > 0).astype(np.float32)[..., None]
        _fill_rgb = (
            _inside_f * _dress_rgb[None, None, :]
            + (1.0 - _inside_f) * _skin_rgb[None, None, :]
        )
        person_cleaned = _safe_uint8(
            person_cleaned.astype(np.float32) * (1.0 - _hard_erase)
            + _fill_rgb * _hard_erase
        )
        pipeline_info.append("DressFullErase")
    pipeline_info.append("SoftErase")

    # ── Step 5b: BodyCopy REMOVED (v16) ──
    # v13-v15 BodyCopy was filling exposed areas (where old garment existed but
    # new garment doesn't cover) with body_blurred pixels. This was the #1 cause
    # of artifacts: re-introducing old shirt colors, dark patches, and destroying
    # garment texture. SoftErase in Step 5 already provides a clean body background.
    # BodyCopy is no longer needed — the garment covers the torso, and any small
    # gaps are handled by the feathered blend in Step 7.
    pipeline_info.append("BodyCopy:OFF")

    # ── Step 6: Color match cloth to person lighting ──
    # Match warped cloth brightness to person BEFORE blending.
    # This prevents the garment looking "pasted on" with different lighting.
    warped_cloth_matched = _match_cloth_brightness(warped_cloth_prefilled, person_cleaned, warped_mask)
    if garment_category == "dress":
        warped_cloth_matched = _apply_texture_preserving_dress_folds(
            warped_cloth_matched,
            warped_mask,
            full_pose,
        )
        pipeline_info.append("DressFoldShade")
        # v16.46: BodyShade — borrow real body shading (chest highlight, side
        # shadows, waist crease) from the original person photo so the dress
        # follows the actual 3D body shape instead of looking like a flat
        # cutout pasted on. Pattern/colour are unchanged (luminance only).
        warped_cloth_matched = _apply_body_shade_from_person(
            warped_cloth_matched,
            person_rgb,
            warped_mask,
            strength=0.65,
        )
        pipeline_info.append("BodyShade")

    # ── Step 6b: Body Curve Map + DrapeHint ──
    # v16.9: DISABLED when diffusion is primary (strength 0.75).
    # These CPU-side shading hacks fight with diffusion's own fold/shadow generation.
    # Diffusion at 0.75 strength regenerates garment texture including natural folds,
    # body curvature shading, and drape. CPU curve/drape only adds noise to the
    # init_image that diffusion then has to correct.
    # Only kept as light hint — reduced from previous strengths.
    if full_pose is not None:
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
    warped_mask_full = warped_mask.copy()

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
        if face_m is not None:
            face_ys = np.where(face_m > 0)[0]
            if len(face_ys) > 5:
                neck_skin_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 10))
                neck_skin = cv2.dilate(face_m, neck_skin_k, iterations=1)
                neck_skin[:int(face_ys.max() * 0.97), :] = 0
                face_only_mask = cv2.bitwise_or(face_only_mask, neck_skin)
        face_only_mask = cv2.dilate(face_only_mask, np.ones((3, 3), np.uint8), iterations=1)

    if face_only_mask.sum() > 0:
        warped_mask = cv2.subtract(warped_mask, face_only_mask)

    _debug_save("06a_warped_mask_after_protect", warped_mask, is_mask=True)

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
        # v16.29: Slightly softer warp edge (sigma 0.6 -> 1.4) so the warp/print
        # transition into diffusion is gradient, not a hard cut. The wider
        # diffusion ring below will fully cover this feather, eliminating any
        # visible seam at neck / waist / sleeve boundary.
        # v16.46: Increase feather (sigma 1.4 -> 2.6, erode 1 -> 2) so the
        # garment edge fades gently into skin/background instead of looking
        # like a sharp cutout pasted on top of the person.
        torso_soft = _soft_mask(warped_mask, blur_sigma=2.6, erode_px=2)
        pipeline_info.append("HardDressMask")
    else:
        torso_soft = _soft_mask(warped_mask, blur_sigma=1.2, erode_px=2)
    _debug_save("06_torso_mask", (torso_soft * 255).astype(np.uint8), is_mask=True)
    torso_alpha = torso_soft[..., None]

    init_tryon = (
        person_cleaned.astype(np.float32) * (1.0 - torso_alpha)
        + warped_cloth_matched.astype(np.float32) * torso_alpha
    )
    init_tryon = _safe_uint8(init_tryon)

    # Layer 2+3: SLEEVES (v16: tighter seam ring)
    # v16: 3x3 dilate (was 5x5) and (5,5) blur (was 7,7) — narrower seam band
    torso_u8 = (torso_soft * 255).clip(0, 255).astype(np.uint8)
    torso_dilated = cv2.dilate(torso_u8, np.ones((3, 3), np.uint8), iterations=1)
    seam_ring = cv2.subtract(torso_dilated, torso_u8)  # thin border ring
    seam_f = cv2.GaussianBlur(
        seam_ring.astype(np.float32) / 255.0, (5, 5), 1.0,
    )

    for side in ("left", "right"):
        if side not in sleeve_data:
            continue
        s_rgb, s_mask_f = sleeve_data[side]

        # v15: Trap gray-border artifacts in sleeve RGB.
        # The sleeve warp fills non-sleeve areas with (128,128,128) gray.
        # If the mask bleeds slightly, these gray pixels blend in → dark patches.
        # Replace near-gray pixels inside the mask with nearest cloth colors.
        s_bright = s_rgb.astype(np.float32).sum(axis=2)
        gray_leak = (s_mask_f > 0.1) & (np.abs(s_bright - 384.0) < 30.0)  # 128*3=384
        if gray_leak.sum() > 10:
            s_blurred = cv2.GaussianBlur(s_rgb, (11, 11), 3.0)
            s_rgb = s_rgb.copy()
            s_rgb[gray_leak] = s_blurred[gray_leak]

        # Remove torso overlap from sleeve mask (sleeve only outside torso)
        s_exclusive = np.clip(s_mask_f - torso_soft, 0.0, 1.0)

        # In the seam ring: cross-fade sleeve at 60% opacity for smooth join
        s_in_seam = np.minimum(s_mask_f, seam_f) * 0.6
        s_final = np.clip(s_exclusive + s_in_seam, 0.0, 1.0)

        # Soft edge: gaussian blur the final sleeve alpha for anti-aliasing
        s_final_u8 = (s_final * 255).clip(0, 255).astype(np.uint8)
        s_final = cv2.GaussianBlur(
            s_final_u8.astype(np.float32) / 255.0, (5, 5), 1.0,
        )

        s_alpha = s_final[..., None]
        init_tryon = (
            init_tryon.astype(np.float32) * (1.0 - s_alpha)
            + s_rgb.astype(np.float32) * s_alpha
        )
        init_tryon = _safe_uint8(init_tryon)
        _debug_save(f"06_after_sleeve_{side}", init_tryon)

    pipeline_info.append("3LayerBlend")

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
    wm_garment = (warped_mask > 20).astype(np.uint8) * 255
    for side in sleeve_data:
        s_rgb, s_mask_f = sleeve_data[side]
        sleeve_uint8 = (s_mask_f * 255).clip(0, 255).astype(np.uint8)
        wm_garment = cv2.bitwise_or(wm_garment, sleeve_uint8)
    warped_mask = wm_garment

    # ── Step 8: Layer foreground on top ──
    # Two-pass overlay:
    #   Pass 1: Arms + face (with sleeve_protect subtraction for arm-sleeve overlap)
    #   Pass 2: Hair overlay (ALWAYS on top — garment renders UNDER hair)
    # This split is critical: if sleeve_protect subtracts from hair mask,
    # hair disappears where it overlaps garment → unnatural.
    if parsing:
        # Pass 1: Arms + face (NOT hair)
        # v16.41: For dress, paste arms back only where the garment/sleeve mask
        # does NOT cover them. This removes gray erase-fill columns while still
        # keeping real dress sleeves on top where they exist.
        if garment_category == "dress":
            _fg_keys = ("left_arm", "right_arm", "face", "hat", "sunglasses")
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
        hair_mask = parsing.get("hair")
        if hair_mask is not None and hair_mask.sum() > 0:
            # Erode to keep only confident hair pixels (no uncertain boundary pixels)
            hair_core = cv2.erode(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
            hair_alpha = cv2.GaussianBlur(
                hair_core.astype(np.float32) / 255.0, (7, 7), 2.0
            )[..., None]

            init_tryon = _safe_uint8(
                init_tryon.astype(np.float32) * (1.0 - hair_alpha)
                + person_rgb.astype(np.float32) * hair_alpha
            )

        pipeline_info.append("Layers")

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
        # v16.50: IDM/CatVTON-style dress agnostic mask. For dress/overall
        # try-on, modern pipelines mask a broad changeable region (old upper
        # clothes, skirt/pants/legs, arms/neck) and protect only identity parts
        # such as face/hair/shoes/hands. Our previous seam-only mask preserved
        # print, but left DressFullErase's skin/grey fill visible outside the
        # warped dress -> the output looked layered. Keep the dress core locked
        # by subtracting dress_core, but let diffusion refill the surrounding
        # changeable region as a single visible garment layer.
        # v16.52: Shrink dress_outer dilation from 19 -> 5 px so diffusion does
        # NOT paint a wider dress silhouette beyond the actual warp footprint.
        # Previously the 19-px outward ring produced a second outer dress shape
        # surrounding the locked CPU core ("two layers" artifact). Keep the
        # core lock tight (erode 9) so most of the warp interior is preserved
        # but the seam ring is thin and entirely on top of the warp.
        dress_bin = (warped_mask > 20).astype(np.uint8) * 255
        dress_core = cv2.erode(dress_bin, np.ones((9, 9), np.uint8), iterations=1)
        dress_outer = cv2.dilate(dress_bin, np.ones((5, 5), np.uint8), iterations=1)
        gen_mask = cv2.subtract(dress_outer, dress_core)
        if parsing:
            old_clothes = get_clothing_mask(parsing)
            if old_clothes is not None:
                old_clothes = cv2.subtract(old_clothes, dress_core)
                gen_mask = cv2.bitwise_or(gen_mask, old_clothes)
            neck_diff = get_neck_mask(parsing)
            if neck_diff is not None:
                gen_mask = cv2.bitwise_or(gen_mask, neck_diff)
            for _dk in (
                "dress", "coat", "scarf", "pants", "skirt",
                "left_leg", "right_leg", "left_arm", "right_arm",
            ):
                _dm = parsing.get(_dk)
                if _dm is not None:
                    _dm = cv2.subtract(_dm, dress_core)
                    gen_mask = cv2.bitwise_or(gen_mask, _dm)
        # PROTECT: face/hair/hat/sunglasses/shoes. Do not protect pants/legs
        # for dress: DressCode/IDM treat them as changeable for dresses.
        # Dress core is not protected here because it has already been removed
        # from gen_mask.
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

    if garment_category == "dress":
        # v16.52: tighten dilation from 13 -> 3 so any old-clothes/arm regions
        # added to gen_mask only get a thin seam ring, not a wide repaint band
        # that would extend beyond the warp and create an outer ghost dress.
        gen_mask = cv2.dilate(gen_mask, np.ones((3, 3), np.uint8), iterations=1)
        # Re-subtract the dress core so the central pattern stays from CPU warp.
        gen_mask = cv2.subtract(gen_mask, dress_core)
        # v16.52: HARD CLIP gen_mask to the dress_outer (warp + 5px). This
        # guarantees diffusion never paints dress fabric outside the warp
        # footprint, eliminating the second-layer halo. Old shirt residue
        # outside the warp is handled by DressFullErase's skin fill instead.
        gen_mask = cv2.bitwise_and(gen_mask, dress_outer)
    else:
        gen_mask = cv2.dilate(gen_mask, np.ones((21, 21), np.uint8), iterations=1)

    # Apply protection mask
    if parsing:
        protect_mask = np.zeros((h, w), dtype=np.uint8)
        for pkey in protect_keys:
            if pkey in parsing:
                protect_mask = cv2.bitwise_or(protect_mask, parsing[pkey])
        if garment_category == "dress":
            # v16.50: IDM/CatVTON protect hands while masking arms for sleeves.
            # Our parser does not expose separate hand labels, so approximate
            # them as the lower/distal end of each arm component. This keeps
            # fingers/skin from being repainted while still allowing sleeve
            # fabric to cover the upper/mid arm.
            _hand_protect = np.zeros((h, w), dtype=np.uint8)
            for _ak in ("left_arm", "right_arm"):
                _am = parsing.get(_ak)
                if _am is None or int(_am.sum()) < 255 * 30:
                    continue
                _ays, _axs = np.where(_am > 0)
                if len(_ays) < 30:
                    continue
                _cut_y = int(np.percentile(_ays, 78))
                _distal = np.zeros((h, w), dtype=np.uint8)
                _distal[(_am > 0) & (np.indices((h, w))[0] >= _cut_y)] = 255
                _distal = cv2.dilate(_distal, np.ones((5, 5), np.uint8), iterations=1)
                _hand_protect = cv2.bitwise_or(_hand_protect, _distal)
            if int(_hand_protect.sum()) > 0:
                protect_mask = cv2.bitwise_or(protect_mask, _hand_protect)
                pipeline_info.append("HandProtect")
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
    if garment_category == "dress":
        # v16.45: stronger seam/agnostic pass so diffusion can extend the
        # dress fabric into the grey erase ring around the warp (old shirt
        # silhouette). The dress core is still locked by DressLightMerge so
        # the leopard print is preserved; this pass only refills the gap.
        _diff_strength = 0.55
        _diff_guidance = 3.5
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
                    # v16.39: DON'T override style_prompt at all. The user-
                    # provided prompt (or empty) is best. Adding 'fitted
                    # dress' / colour hints made LCM bias toward generic
                    # dress samples and destroyed the leopard print. Just
                    # log the anchor for debug; keep prompt as user gave it.
                    print(f"[DIFFUSION] Dress colour anchor: {_anchor}")
            except Exception:
                pass

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
        generated = generate_tryon_image(
            init_tryon_rgb=init_tryon,
            inpaint_mask_gray=gen_mask_soft,
            user_prompt=style_prompt,
            config=GenConfig(
                num_inference_steps=max(gen_steps, 20),
                guidance_scale=_diff_guidance,
                refiner_mode="lcm",
                cloth_type=cloth_type,
                # v16.40b: Dress LoRA overpowers complex prints and pulls the
                # result toward generic dress samples. Keep LoRA for tops /
                # pants, but let dress follow the init warp.
                use_cloth_lora=(garment_category != "dress"),
                strength=_diff_strength,
                infer_size=_infer_size,
            ),
        )

        # Sanitize + size-match
        generated = _sanitize_rgb_output(generated)
        generated = _fit_like(generated, init_tryon, is_mask=False)
        binary_mask = _fit_like(binary_mask, generated, is_mask=True)

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

        if garment_category == "dress":
            # v16.43: DressLightMerge fixes the CPU-warp vs diffusion double
            # composite loop. gen_tryon.py already composites diffusion over
            # init_tryon; then DressCoreLock previously pasted CPU warp back at
            # 0.88, erasing most visible GPU changes. For dress, keep CPU warp
            # as the single colour/print source, transfer only luminance from
            # diffusion over the garment core. For the agnostic gap OUTSIDE the
            # warped dress, use diffusion RGB strongly; otherwise the skin/grey
            # DressFullErase underlayer remains visible as a second layer.
            _core_lock = cv2.erode(binary_mask, np.ones((17, 17), np.uint8), iterations=1)
            _edge_mask = cv2.subtract(gen_mask_soft, _core_lock)
            _edge_alpha = cv2.GaussianBlur((_edge_mask > 20).astype(np.float32), (9, 9), 3.0)[..., None]
            _edge_alpha = np.clip(_edge_alpha * 0.35, 0.0, 0.35)

            # v16.51: Disable broad RGB gap fill. It removed grey fill, but it
            # also synthesized a second translucent dress layer around the real
            # warped dress. Keep RGB diffusion only on the narrow seam band;
            # final single-layer clipping below removes any outer generated
            # residue after DrapeLightPass.
            _merge_alpha = _edge_alpha

            _light_base = _blend_luminance_from_diffusion(
                init_tryon_clean,
                generated,
                binary_mask,
                strength=0.72,
                broad_strength=0.32,
            )
            generated = _safe_uint8(
                _light_base.astype(np.float32) * (1.0 - _merge_alpha)
                + generated.astype(np.float32) * _merge_alpha
            )
            pipeline_info.append("DressLightMerge")

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

        # v16.42: GPU DrapeLightPass. Previous FoldPass directly accepted the
        # diffusion RGB and could repaint leopard into zebra. This pass still
        # asks GPU for natural folds, but only transfers medium-frequency
        # luminance (shadows/highlights) back onto the locked CPU/diffusion
        # result. Colour + print remain from the original warped dress.
        if garment_category == "dress":
            try:
                _fold_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)
                if parsing:
                    _ps = np.zeros(_fold_mask.shape, dtype=np.uint8)
                    for _sk in (
                        "upper_clothes", "dress", "coat", "scarf",
                        "left_arm", "right_arm", "neck",
                        "pants", "skirt", "left_leg", "right_leg",
                    ):
                        _sv = parsing.get(_sk)
                        if _sv is not None:
                            _ps = cv2.bitwise_or(_ps, _sv)
                    if int(_ps.sum()) > 255 * 500:
                        _ps = cv2.morphologyEx(_ps, cv2.MORPH_CLOSE,
                                               np.ones((15, 15), np.uint8))
                        _ps = cv2.erode(_ps, np.ones((3, 3), np.uint8), iterations=1)
                        _fold_mask = cv2.bitwise_and(_fold_mask, _ps)
                _protect = np.zeros(_fold_mask.shape, dtype=np.uint8)
                if parsing:
                    for _pk in ("face", "hair", "hat", "sunglasses",
                                "left_shoe", "right_shoe"):
                        _pv = parsing.get(_pk)
                        if _pv is not None:
                            _protect = cv2.bitwise_or(_protect, _pv)
                    _protect = cv2.dilate(_protect, np.ones((5, 5), np.uint8), iterations=1)
                _fold_mask = cv2.subtract(_fold_mask, _protect)
                _fold_mask_soft = cv2.GaussianBlur(_fold_mask,
                                                   (blur_k, blur_k),
                                                   blur_k / 3.0)
                _fold_mask_soft = np.clip(_fold_mask_soft, 0, 255).astype(np.uint8)
                _debug_save("09b_drape_light_mask", _fold_mask_soft, is_mask=True)

                if int((_fold_mask_soft > 30).sum()) > 500:
                    _fold_prompt = (
                        "same dress on body, realistic vertical fabric folds, "
                        "soft waist drape, hip shadows, natural cloth wrinkles, "
                        "clean neckline, no text, no logo"
                    )
                    fold_out = generate_tryon_image(
                        init_tryon_rgb=output,
                        inpaint_mask_gray=_fold_mask_soft,
                        user_prompt=_fold_prompt,
                        config=GenConfig(
                            num_inference_steps=14,
                            guidance_scale=3.0,
                            refiner_mode="lcm",
                            cloth_type="dress",
                            use_cloth_lora=False,
                            strength=0.50,
                            infer_size=_infer_size,
                        ),
                    )
                    fold_out = _sanitize_rgb_output(fold_out)
                    fold_out = _fit_like(fold_out, output, is_mask=False)
                    if float(fold_out.mean()) > 20 and not _is_blackout_artifact(fold_out, output, binary_mask):
                        output = _blend_luminance_from_diffusion(
                            output,
                            fold_out,
                            _fold_mask_soft,
                            strength=0.78,
                            broad_strength=0.42,
                        )
                        pipeline_info.append("DrapeLightPass")
                        print(f"[DIFFUSION] DRAPE LIGHT PASS done, mean={float(output.mean()):.1f}")
            except Exception as _fold_exc:
                print(f"[DIFFUSION] DrapeLightPass skipped: {_fold_exc}")

        if garment_category == "dress":
            # v16.52: Single visible dress layer. Aggressively clip every
            # diffusion-generated pixel outside the actual warp footprint back
            # to the clean CPU composite. Reduced dilation 7 -> 2 px so the
            # outer dress halo (caused by diffusion painting beyond the warp)
            # is entirely removed.
            _visible = cv2.dilate(binary_mask, np.ones((3, 3), np.uint8), iterations=1)
            _visible = cv2.GaussianBlur((_visible > 20).astype(np.float32), (5, 5), 1.2)[..., None]
            _visible = np.clip(_visible, 0.0, 1.0)
            output = _safe_uint8(
                init_tryon_clean.astype(np.float32) * (1.0 - _visible)
                + output.astype(np.float32) * _visible
            )
            # Restore identity/skin regions outside the single dress layer so
            # hands/arms/face do not keep diffusion residue.
            if parsing:
                _restore = np.zeros(binary_mask.shape, dtype=np.uint8)
                for _rk in ("face", "hair", "hat", "sunglasses", "left_arm", "right_arm"):
                    _rv = parsing.get(_rk)
                    if _rv is not None:
                        _restore = cv2.bitwise_or(_restore, _rv)
                _restore = cv2.subtract(_restore, cv2.dilate(binary_mask, np.ones((5, 5), np.uint8), iterations=1))
                _restore_f = cv2.GaussianBlur((_restore > 20).astype(np.float32), (7, 7), 2.0)[..., None]
                output = _safe_uint8(
                    output.astype(np.float32) * (1.0 - _restore_f)
                    + person_rgb.astype(np.float32) * _restore_f
                )
            pipeline_info.append("DressSingleLayerClip:v16.52")

        # v16.17: DressShapeLock REMOVED. The previous post-blend of 45% TPS
        # onto diffusion was re-introducing red/pink bleed because
        # init_tryon_clean carried residual colour contamination from
        # person_cleaned (old red shirt pixels on arms). With v16.17's full
        # body erase + neutral grey fill, diffusion already receives a clean
        # init — output is the final answer.
        if garment_category == "dress":
            pipeline_info.append("DressGen")

        # v16.23: Expose GPU/CPU status so user can verify from the UI.
        try:
            import torch as _torch_dbg
            _is_cuda = _torch_dbg.cuda.is_available()
            pipeline_info.append(f"Diffusion[{'GPU' if _is_cuda else 'CPU'}]")
        except Exception:
            pipeline_info.append("Diffusion")
        print(f"[DIFFUSION] SUCCESS — SOTA agnostic mask, mean_brightness={float(output.mean()):.1f}")
        return output, "Local Diffusion (lcm)", warning_msg, pipeline_info
    except (RuntimeError, Exception) as exc:
        warning_msg = f"Local diffusion failed, keeping CPU pipeline result. ({exc})"
        print(f"[DIFFUSION] FAILED — {exc}")
        return init_tryon, "", warning_msg, pipeline_info


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

    # ═══ PATH A: CLOUD-PRIMARY (default when use_gen=True) ═══
    # Send raw person + cloth images directly to cloud API.
    # No CPU preprocessing needed — cloud handles everything.
    if use_gen:
        try:
            cloud_result_path, cloud_backend = generate_with_cloud_router(
                person_image_path=person_path,
                cloth_image_path=cloth_path,
                style_prompt=style_prompt,
                steps=int(gen_steps),
                guidance=float(min(gen_guidance, 7.5)),
                seed=random.randint(0, 10000),
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
                    if _stype == "short":
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
        _hair_raw = parsing["hair"]
        # ERODE 2px to remove uncertain edge pixels where old shirt might leak
        _hair_core = cv2.erode(_hair_raw, np.ones((3, 3), np.uint8), iterations=1)
        # Soft edge: blur the eroded mask for natural transition
        _hair_alpha = cv2.GaussianBlur(
            _hair_core.astype(np.float32) / 255.0, (7, 7), 2.0
        )[..., None]

        # Direct paste: person_rgb hair onto output (which has garment under hair)
        output = _safe_uint8(
            output.astype(np.float32) * (1.0 - _hair_alpha)
            + person_rgb.astype(np.float32) * _hair_alpha
        )
        pipeline_info.append("HairOverlay")

    # ═══ SAVE & RETURN ═══
    output = _safe_uint8(output)
    output_file = storage.outputs_dir / f"tryon_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_image_rgb(output_file, output)

    info = (
        f"Saved: {output_file}\n"
        f"Storage base: {storage.base_dir}\n"
        f"Mode: {'Cloud+Refine' if use_gen else 'CPU only'}\n"
        f"Backend: {backend_used}\n"
        f"Preset: {quality_preset}\n"
        f"Pipeline: {' -> '.join(pipeline_info)}\n"
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
