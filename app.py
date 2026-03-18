from __future__ import annotations

import datetime as dt
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
    detect_full_pose,
    detect_upper_body_box,
    erase_clothing_region,
    full_pose_to_box,
    poisson_blend,
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
    get_skin_mask,
)
from src.tps_warp import tps_warp_cloth
from src.gen_tryon import GenConfig, generate_tryon_image
from src.cloud_vton_router import CloudVTONUnavailableError, generate_with_cloud_router
from src.storage import resolve_storage_config


load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)


storage = resolve_storage_config()


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

    dilate_kernel = np.ones((25, 25), np.uint8)
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

    return cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_LAB2RGB)


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


def _apply_foreground_layer(
    base_rgb: np.ndarray,
    person_rgb: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Composite person's arms/face/hair on top of the try-on result."""
    fg = cv2.GaussianBlur(fg_mask, (7, 7), 0)
    fg_f = (fg.astype(np.float32) / 255.0)[..., None]
    result = base_rgb.astype(np.float32) * (1.0 - fg_f) + person_rgb.astype(np.float32) * fg_f
    return np.clip(result, 0, 255).astype(np.uint8)


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
            6,
            1.0,
            0.85,
            "lcm",
        )
    if p == "hq":
        mode = "dpm++" if refiner_mode == "base" else refiner_mode
        return (
            float(np.clip(fit_scale, 1.0, 1.20)),
            float(np.clip(alpha, 0.55, 0.75)),
            18,
            1.3,
            0.80,
            mode,
        )

    # balanced
    return (
        float(np.clip(fit_scale, 0.95, 1.15)),
        float(np.clip(alpha, 0.88, 1.0)),
        10,
        1.3,
        0.80,
        "lcm" if refiner_mode == "base" else refiner_mode,
    )


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
        raise gr.Error("Vui lòng tải cả ảnh người mẫu và ảnh áo.")

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

    # ═══════════════════════════════════════════════════════════════
    #  Professional CPU-only Pipeline (no GPU required)
    # ═══════════════════════════════════════════════════════════════
    pipeline_info = []

    # ── Step 1: Human Parsing (SegFormer) ──────────────────────────
    parsing = parse_human(person_rgb)
    if parsing:
        pipeline_info.append("Parsing")

    # ── Step 2: Full Pose Estimation ───────────────────────────────
    full_pose = None
    try:
        full_pose = detect_full_pose(person_rgb)
        # Smooth landmarks to reduce TPS distortion from pose noise
        full_pose = smooth_pose_landmarks(full_pose, person_rgb.shape[:2])
        pose_box = full_pose_to_box(full_pose)
        pipeline_info.append("Pose+Smooth")
    except Exception:
        pose_box = detect_upper_body_box(person_rgb)
        pipeline_info.append("Pose(basic)")

    # ── Step 3: Cloth Segmentation + Body Measurement + Pre-fit ────
    # Hybrid segmentation: U2Net + SegFormer parsing on cloth image.
    # This helps recover missing belly/collar areas when one model under-segments.
    try:
        mask_u2net = segment_cloth_u2net(cloth_rgb)
        pipeline_info.append("U2Net")
    except Exception:
        mask_u2net = build_cloth_mask(cloth_rgb)
        pipeline_info.append("MaskFallback")

    mask_segformer = None
    try:
        cloth_parsing = parse_human(cloth_rgb)
        if cloth_parsing:
            mask_segformer = get_clothing_mask(cloth_parsing)
    except Exception:
        mask_segformer = None

    if mask_segformer is not None and int(mask_segformer.sum()) > 255 * 100:
        cloth_mask = cv2.bitwise_or(mask_u2net, mask_segformer)
        cloth_mask = cv2.morphologyEx(
            cloth_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2,
        )
        # Slight expand so torso/collar isn't cut too tight.
        cloth_mask = cv2.dilate(cloth_mask, np.ones((9, 9), np.uint8), iterations=1)
        pipeline_info.append("SegFormerMerge")
    else:
        cloth_mask = mask_u2net
    scaled_cloth, scaled_mask = cloth_rgb, cloth_mask
    if full_pose is not None:
        measurements = compute_body_measurements(full_pose)
        scaled_cloth, scaled_mask = prefit_scale_cloth(
            cloth_rgb, cloth_mask, measurements,
        )
        pipeline_info.append("ShapePreFit")

    # ── Step 4: TPS Warp (body-conforming) ─────────────────────────
    if full_pose is not None:
        try:
            warped_cloth, warped_mask = tps_warp_cloth(
                cloth_rgb=scaled_cloth,
                cloth_mask=scaled_mask,
                pose=full_pose,
                output_shape=person_rgb.shape[:2],
                fit_scale=fit_scale,
                y_offset_ratio=y_offset,
            )
            pipeline_info.append("Affine+TPS")
        except Exception:
            warped_cloth, warped_mask = warp_cloth_to_torso(
                person_rgb=person_rgb, cloth_rgb=cloth_rgb,
                cloth_mask=cloth_mask, box=pose_box,
                fit_scale=fit_scale, y_offset_ratio=y_offset,
            )
            pipeline_info.append("Persp(fallback)")
    else:
        warped_cloth, warped_mask = warp_cloth_to_torso(
            person_rgb=person_rgb, cloth_rgb=cloth_rgb,
            cloth_mask=cloth_mask, box=pose_box,
            fit_scale=fit_scale, y_offset_ratio=y_offset,
        )
        pipeline_info.append("Perspective")

    # ── Step 5: Cloth edge smoothing ─────────────────────────────
    # Soften warped cloth edges so they blend more naturally with skin.
    warped_cloth = cv2.GaussianBlur(warped_cloth, (3, 3), 0)
    # ── Step 6: Full torso erase (skeleton-based) ───────────────
    # Use body skeleton to define erase area — NOT the old garment
    # parsing.  This prevents the old garment shape (sports bra,
    # crop top, etc.) from leaking through.
    skin_mask = get_skin_mask(parsing) if parsing else None

    if full_pose is not None:
        # Skeleton-based erase covers shoulders-to-hips + sleeve corridors
        skeleton_erase = build_skeleton_erase_mask(full_pose, person_rgb.shape[:2])
        # Union with warped_mask: ensure we erase everywhere the new garment goes
        erase_mask = cv2.bitwise_or(skeleton_erase, (warped_mask > 20).astype(np.uint8) * 255)
        # Also add neck/collar region
        if parsing:
            neck_mask = get_neck_mask(parsing)
            if neck_mask is not None:
                erase_mask = cv2.bitwise_or(erase_mask, neck_mask)
        person_cleaned = erase_clothing_region(
            person_rgb, erase_mask, parsing_skin_mask=skin_mask,
        )
        pipeline_info.append("SkeletonErase")
    elif parsing:
        old_clothes = get_clothing_mask(parsing)
        if old_clothes is not None:
            erase_mask = cv2.dilate(
                old_clothes, np.ones((25, 25), np.uint8), iterations=1,
            )
            neck_mask = get_neck_mask(parsing)
            if neck_mask is not None:
                erase_mask = cv2.bitwise_or(erase_mask, neck_mask)
            person_cleaned = erase_clothing_region(
                person_rgb, erase_mask, parsing_skin_mask=skin_mask,
            )
            pipeline_info.append("ParseErase+Neck")
        else:
            person_cleaned = erase_clothing_region(
                person_rgb, warped_mask, parsing_skin_mask=skin_mask,
            )
            pipeline_info.append("MaskErase")
    else:
        person_cleaned = erase_clothing_region(person_rgb, warped_mask)
        pipeline_info.append("MaskErase")

    # ── Step 6b: Reconstruct exposed skin in erased region ────────
    # Fill any remaining flat patches with surrounding skin texture
    # by blending the inpainted person with original skin pixels.
    if parsing and skin_mask is not None:
        skin_binary = skin_mask > 0
        if skin_binary.sum() > 200:
            # Keep original skin pixels (arms, face, legs, neck) on top
            safe_skin = skin_mask.copy()
            # Do not restore skin over garment area, otherwise torso holes appear.
            garment_block = cv2.dilate((warped_mask > 20).astype(np.uint8) * 255, np.ones((11, 11), np.uint8), 1)
            safe_skin = cv2.subtract(safe_skin, garment_block)
            skin_alpha = (safe_skin.astype(np.float32) / 255.0)[..., None]
            person_cleaned = (
                person_cleaned.astype(np.float32) * (1.0 - skin_alpha)
                + person_rgb.astype(np.float32) * skin_alpha
            ).clip(0, 255).astype(np.uint8)
            pipeline_info.append("SkinRestore")

    # ── Step 7: Layer compositing (VITON-HD style) ─────────────────
    # Order: background → body skin → warped cloth → foreground
    #
    # person_cleaned = original body with old clothes removed, skin intact
    # warped_cloth   = new garment warped to body shape
    # warped_mask    = where the new garment goes
    #
    # Mask refinement: hard threshold → edge feathering.
    # This removes anti-aliased white halo from warped mask edges.
    wm_binary = (warped_mask > 128).astype(np.uint8) * 255   # hard cut
    # Erode 1px to pull mask inward away from white fringe
    wm_erode = cv2.erode(wm_binary, np.ones((3, 3), np.uint8), iterations=1)
    # Detect edge band via Canny and feather it
    wm_edge = cv2.Canny(wm_erode, 50, 150)
    wm_edge = cv2.dilate(wm_edge, np.ones((3, 3), np.uint8), iterations=1)
    wm_feather = cv2.GaussianBlur(wm_erode, (5, 5), 0)
    cloth_alpha = (wm_feather.astype(np.float32) / 255.0)[..., None]

    init_tryon = (
        person_cleaned.astype(np.float32) * (1.0 - cloth_alpha)
        + warped_cloth.astype(np.float32) * cloth_alpha
    ).clip(0, 255).astype(np.uint8)
    pipeline_info.append("EdgeFeather")

    # Optional Poisson blend for seamless colour matching at edges
    try:
        init_tryon = poisson_blend(init_tryon, warped_cloth, warped_mask)
        pipeline_info.append("Poisson")
    except Exception:
        pipeline_info.append("AlphaBlend")

    # ── Step 8: Layer arms / face / hair on top ────────────────────
    if parsing:
        keep_mask = get_foreground_keep_mask(parsing)
        if keep_mask is not None and keep_mask.shape[:2] == person_rgb.shape[:2]:
            # Prevent arm layer from erasing generated sleeves.
            arm_mask = get_arm_mask(parsing)
            sleeve_protect = _build_sleeve_protect_mask(warped_mask, arm_mask)
            if sleeve_protect is not None:
                keep_mask = cv2.subtract(keep_mask, sleeve_protect)
            init_tryon = _apply_foreground_layer(init_tryon, person_rgb, keep_mask)
            pipeline_info.append("Layers")
    output = init_tryon
    backend_used = "CPU Pipeline"
    warning_msg = ""
    should_run_local_diffusion = True

    # ── Optional refinement stage (cloud / local) ─
    if use_gen and use_catvton_cloud and backend_used in {"CPU Pipeline", "Local Masked Diffusion"}:
        try:
            cloud_result_path, cloud_backend = generate_with_cloud_router(
                person_image_path=person_path,
                cloth_image_path=cloth_path,
                style_prompt=style_prompt,
                steps=int(gen_steps),
                guidance=float(min(gen_guidance, 7.5)),
                seed=random.randint(0, 10000),
            )
            if backend_used in {"CPU Pipeline", "Local Masked Diffusion"}:
                output = read_image_rgb(cloud_result_path)
                backend_used = cloud_backend
        except (CloudVTONUnavailableError, Exception) as exc:
            if backend_used in {"CPU Pipeline", "Local Masked Diffusion"}:
                warning_msg = (
                    f"{warning_msg}\nℹ️ Cloud VTON không khả dụng, tự fallback sang local masked diffusion. ({exc})"
                ).strip()
                backend_used = "Local Masked Diffusion"

    if use_gen and should_run_local_diffusion and backend_used in {"CPU Pipeline", "Local Masked Diffusion"}:
        binary_mask = (warped_mask > 20).astype(np.uint8) * 255
        core_mask, gen_mask = _build_masks_for_garment_preservation(
            binary_mask=binary_mask,
            image_shape=person_rgb.shape[:2],
            pose_box=pose_box,
        )

        # Edge-only diffusion mask (IDM-VTON approach):
        # Expand edge mask so diffusion can naturally redraw sleeves/collar seams.
        edge = cv2.Canny(binary_mask, 50, 150)
        edge_mask = cv2.dilate(edge, np.ones((12, 12), np.uint8), iterations=1)
        # Also include neck/collar region for neckline refinement.
        if parsing:
            neck_diff = get_neck_mask(parsing)
            if neck_diff is not None:
                edge_mask = cv2.bitwise_or(edge_mask, neck_diff)
        gen_mask_narrow = cv2.GaussianBlur(edge_mask, (9, 9), 0)

        try:
            generated = generate_tryon_image(
                init_tryon_rgb=init_tryon,
                inpaint_mask_gray=gen_mask_narrow,
                user_prompt=style_prompt,
                config=GenConfig(
                    num_inference_steps=gen_steps,
                    guidance_scale=gen_guidance,
                    refiner_mode=refiner_mode,
                    cloth_type=cloth_type,
                    use_cloth_lora=True,
                ),
            )
            restored = _restore_core_garment(
                generated_rgb=generated,
                init_tryon_rgb=init_tryon,
                core_mask=core_mask,
                preserve_strength=preserve_strength,
            )
            output = _apply_color_consistency(
                generated_rgb=restored,
                reference_rgb=init_tryon,
                garment_mask=binary_mask,
                strength=0.6,
            )
            # Texture lock: diffusion only touches edges/folds, core texture from warp.
            # final = warped_cloth * preserve + diffusion * (1 - preserve)
            output = _enforce_garment_identity(
                generated_rgb=output,
                init_tryon_rgb=init_tryon,
                garment_mask=binary_mask,
                strength=max(0.55, float(preserve_strength)),
            )
            backend_used = f"Local Diffusion ({refiner_mode})"
            pipeline_info.append("MaskedDiffusion")
        except RuntimeError as exc:
            warning_msg = f"{warning_msg}\nℹ️ Local diffusion không chạy được, giữ kết quả CPU pipeline. ({exc})".strip()

    output_file = storage.outputs_dir / f"tryon_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_image_rgb(output_file, output)

    info = (
        f"✅ Đã lưu kết quả: {output_file}\n"
        f"Storage base: {storage.base_dir}\n"
        f"Mode: {'Refine enabled' if use_gen else 'CPU only'}\n"
        f"Backend: {backend_used}\n"
        f"Preset: {quality_preset}\n"
        f"Pipeline: {' → '.join(pipeline_info)}\n"
        "Tip: máy yếu nên dùng refiner_mode=lcm, steps=6-12, guidance=1.0-3.0."
    )

    if warning_msg:
        info = f"{info}\n{warning_msg}"

    return output, info


with gr.Blocks(title="Virtual Try-On") as demo:
    gr.Markdown("# 👕 Virtual Try-On (CPU Pipeline)")
    gr.Markdown(
        "Pipeline: Parsing → Pose Smoothing → U2Net Cloth Seg → Body Measurement → "
        "Shape-Constrained PreFit → Affine Align + TPS Warp → "
        "Skeleton Erase → Skin Restore → Edge Feather → Poisson Blend → "
        "Layer Compositing → Edge-Only Masked Diffusion (optional)."
    )

    with gr.Row():
        person_input = gr.Image(type="numpy", label="Ảnh người mẫu")
        cloth_input = gr.Image(type="numpy", label="Ảnh áo/quần (khuyến nghị áo nền sáng)")

    with gr.Row():
        fit_scale = gr.Slider(0.8, 1.5, value=1.12, step=0.01, label="Độ rộng trang phục")
        alpha = gr.Slider(0.4, 1.0, value=0.65, step=0.01, label="Độ hòa trộn")
        y_offset = gr.Slider(-0.15, 0.2, value=-0.01, step=0.01, label="Dịch dọc")

    with gr.Row():
        use_gen = gr.Checkbox(value=True, label="Bật masked diffusion refine (khuyến nghị)")
        style_prompt = gr.Textbox(
            value="",
            placeholder="(Để trống để giữ nguyên áo gốc — chỉ nhập nếu muốn thay đổi phong cách)",
            label="Mô tả trang phục (prompt)",
        )

    with gr.Row():
        gen_steps = gr.Slider(4, 30, value=18, step=1, label="Refine steps")
        gen_guidance = gr.Slider(0.5, 8.0, value=1.3, step=0.1, label="Refine guidance")
        preserve_strength = gr.Slider(0.25, 1.0, value=0.80, step=0.01, label="Giữ texture áo gốc")

    quality_preset = gr.Radio(
        choices=["fast", "balanced", "hq"],
        value="hq",
        label="Preset chất lượng",
    )

    with gr.Row():
        refiner_mode = gr.Dropdown(
            choices=["lcm", "hypersd", "dpm++", "euler", "base"],
            value="dpm++",
            label="Refiner mode (VRAM thấp: lcm, chi tiết cao: dpm++)",
        )
        cloth_type = gr.Dropdown(
            choices=["auto", "tshirt", "hoodie", "jacket", "dress", "generic"],
            value="auto",
            label="Cloth type cho LoRA routing",
        )

    use_catvton_cloud = gr.Checkbox(value=False, label="Ưu tiên Cloud VTON trước local refine (optional)")

    run_btn = gr.Button("Thử đồ", variant="primary")

    with gr.Row():
        output_img = gr.Image(type="numpy", label="Kết quả")
        output_info = gr.Textbox(label="Thông tin", lines=4)

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
