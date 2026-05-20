"""Dress Pipeline v2 — standalone orchestrator.

A single entry point `run_dress_pipeline_v2(person_rgb, cloth_rgb, ...)`
that owns the whole dress flow: pose + parsing -> dress analysis ->
pose-driven target silhouette -> agnostic mask (with hair underlap) ->
inpainted seed (no flat-colour fill) -> diffusion (local or cloud) ->
occluder restore (hair_front / hands / shoes) -> postprocess cleanup.

This module never imports from `app.py` to keep responsibility one-way and
avoid circular imports. Small helpers it needs (soft mask, foreground layer)
are duplicated locally — they are tiny.
"""
from __future__ import annotations

import datetime as dt
import os
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.garment_silhouettes import detect_dress_type
from src.geometry.dress_geometry import build_target_silhouette
from src.human_parsing import parse_human
from src.image_ops import build_cloth_mask, detect_dress_pose, detect_full_pose, segment_cloth_ensemble
from src.masks.dress_mask_builder import DressMasks, build_dress_masks
from src.occlusion.dress_occlusion import restore_occluders
from src.postprocess.dress_postprocess_v2 import (
    apply_color_anchor,
    clean_hem,
    remove_old_dress_ghost,
    remove_rect_artifact,
    telea_inpaint_seed,
)
from src.storage import resolve_storage_config


@dataclass
class DressPipelineResult:
    image: np.ndarray
    masks: DressMasks | None
    analysis: dict
    debug: list[str] = field(default_factory=list)


def _debug_enabled() -> bool:
    return os.getenv("VTON_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _debug_dir() -> Path | None:
    if not _debug_enabled():
        return None
    try:
        if os.getenv("VTON_DEBUG_DIR", "").strip():
            base = Path(os.getenv("VTON_DEBUG_DIR", "")).expanduser().resolve()
        else:
            base = resolve_storage_config().base_dir / "debug"
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        return None


def _save_debug(name: str, img: np.ndarray, is_mask: bool = False, run_id: str | None = None) -> None:
    d = _debug_dir()
    if d is None:
        return
    prefix = run_id or dt.datetime.now().strftime("%H%M%S")
    p = d / f"{prefix}_{name}.png"
    if is_mask:
        cv2.imwrite(str(p), img)
    else:
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _save_debug_text(name: str, text: str, run_id: str | None = None) -> None:
    d = _debug_dir()
    if d is None:
        return
    prefix = run_id or dt.datetime.now().strftime("%H%M%S")
    try:
        (d / f"{prefix}_{name}.txt").write_text(text, encoding="utf-8")
    except OSError:
        return


def _pose_xy(pose: dict | None, key: str) -> tuple[float, float] | None:
    if not pose:
        return None
    v = pose.get(key)
    if v is None:
        return None
    return float(v[0]), float(v[1])


def _row_width(mask: np.ndarray, row: int) -> float:
    row = max(0, min(mask.shape[0] - 1, int(row)))
    nz = np.where(mask[row] > 0)[0]
    return float(nz.max() - nz.min()) if len(nz) >= 4 else 0.0


def _row_segments(mask: np.ndarray, row: int, min_width: int = 4) -> list[tuple[int, int]]:
    row = max(0, min(mask.shape[0] - 1, int(row)))
    nz = np.where(mask[row] > 0)[0]
    if len(nz) < min_width:
        return []
    segs: list[tuple[int, int]] = []
    start = prev = int(nz[0])
    for x in nz[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
        else:
            if prev - start + 1 >= min_width:
                segs.append((start, prev))
            start = prev = x
    if prev - start + 1 >= min_width:
        segs.append((start, prev))
    return segs


def _long_sleeve_pose_mask(pose: dict | None, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.uint8)
    ls = _pose_xy(pose, "left_shoulder")
    rs = _pose_xy(pose, "right_shoulder")
    if ls is None or rs is None:
        return out
    shoulder_w = max(24.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    upper_r = max(8, int(shoulder_w * 0.16))
    lower_r = max(7, int(shoulder_w * 0.13))
    for side in ("left", "right"):
        sh = _pose_xy(pose, f"{side}_shoulder")
        el = _pose_xy(pose, f"{side}_elbow")
        wr = _pose_xy(pose, f"{side}_wrist")
        if sh is None or el is None:
            continue
        if wr is None:
            wr = el
        sh_i = (int(round(sh[0])), int(round(sh[1])))
        el_i = (int(round(el[0])), int(round(el[1])))
        wr_i = (int(round(wr[0])), int(round(wr[1])))
        cv2.line(out, sh_i, el_i, 255, thickness=upper_r * 2, lineType=cv2.LINE_AA)
        cv2.line(out, el_i, wr_i, 255, thickness=lower_r * 2, lineType=cv2.LINE_AA)
        cv2.circle(out, sh_i, upper_r, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, el_i, upper_r, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, wr_i, lower_r, 255, thickness=-1, lineType=cv2.LINE_AA)
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)


def _reference_lab_stats(ref_rgb: np.ndarray, ref_mask: np.ndarray | None = None) -> tuple[np.ndarray, float, float, float] | None:
    if ref_rgb is None or ref_rgb.size == 0:
        return None
    ref_lab = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    if ref_mask is not None and ref_mask.shape[:2] == ref_rgb.shape[:2]:
        valid = ref_mask > 20
    else:
        # Cloth photos usually sit on a light background. Keep non-background
        # pixels when no reliable mask is available.
        rgb = ref_rgb.astype(np.int16)
        valid = ~((rgb[..., 0] > 232) & (rgb[..., 1] > 232) & (rgb[..., 2] > 232))
    if int(valid.sum()) < 50:
        valid = np.ones(ref_rgb.shape[:2], dtype=bool)
    lab = ref_lab[valid]
    ab = lab[:, 1:3]
    center = np.median(ab, axis=0)
    spread = float(np.median(np.linalg.norm(ab - center[None, :], axis=1)))
    l_center = float(np.median(lab[:, 0]))
    l_spread = float(np.median(np.abs(lab[:, 0] - l_center)))
    return center.astype(np.float32), spread, l_center, l_spread


def _cloth_like_mask(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    target_mask: np.ndarray,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    stats = _reference_lab_stats(reference_rgb, reference_mask)
    target = target_mask > 20
    if stats is None or int(target.sum()) < 50:
        return (target.astype(np.uint8) * 255)
    center, spread, l_center, l_spread = stats
    cand_lab = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    dist = np.linalg.norm(cand_lab[..., 1:3] - center[None, None, :], axis=2)
    ab_threshold = float(np.clip(18.0 + spread * 2.2, 20.0, 46.0))
    l_threshold = float(np.clip(24.0 + l_spread * 2.4, 30.0, 78.0))
    l_ok = np.abs(cand_lab[..., 0] - l_center) <= l_threshold
    cloth_like = ((dist <= ab_threshold) & l_ok & target).astype(np.uint8) * 255
    cloth_like = cv2.morphologyEx(cloth_like, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    cloth_like = cv2.morphologyEx(cloth_like, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    return cloth_like


def _diffusion_geometry_guard(
    candidate_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    target_mask: np.ndarray,
    pose: dict | None,
    parsing: dict | None,
    *,
    sleeve_type: str,
    length: str,
    reference_mask: np.ndarray | None = None,
) -> tuple[bool, list[str], np.ndarray]:
    """Validate that diffusion drew the expected dress footprint.

    The guard intentionally prefers the deterministic seed when uncertain.
    """
    h, w = candidate_rgb.shape[:2]
    target = (target_mask > 20).astype(np.uint8) * 255
    if target.shape[:2] != (h, w):
        target = cv2.resize(target, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(target > 0)
    if len(xs) < 300:
        return True, [], target

    dress_like = _cloth_like_mask(candidate_rgb, reference_rgb, target, reference_mask)
    core = cv2.erode(target, np.ones((5, 5), np.uint8), iterations=1)
    core_count = int(cv2.countNonZero(core))
    coverage = float(cv2.countNonZero(cv2.bitwise_and(dress_like, core))) / max(1.0, float(core_count))

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
    if cy2 < ty1 + int(th * 0.72):
        reasons.append("hem_too_short")

    split_rows = 0
    split_gap_max = 0
    for frac in (0.52, 0.62, 0.72, 0.82, 0.90):
        row = int(round(ty1 + th * frac))
        target_segs = _row_segments(target, row, min_width=8)
        gen_segs = _row_segments(dress_like, row, min_width=8)
        if len(target_segs) != 1 or len(gen_segs) < 2:
            continue
        target_w = target_segs[0][1] - target_segs[0][0] + 1
        gaps = [gen_segs[i + 1][0] - gen_segs[i][1] - 1 for i in range(len(gen_segs) - 1)]
        if not gaps:
            continue
        gap = max(gaps)
        if gap >= max(8, int(target_w * 0.06)):
            split_rows += 1
            split_gap_max = max(split_gap_max, gap)
    if split_rows >= 2:
        reasons.append(f"skirt_split={split_rows}x{split_gap_max}")

    ls = _pose_xy(pose, "left_shoulder")
    rs = _pose_xy(pose, "right_shoulder")
    lh = _pose_xy(pose, "left_hip")
    rh = _pose_xy(pose, "right_hip")
    lk = _pose_xy(pose, "left_knee")
    rk = _pose_xy(pose, "right_knee")
    la = _pose_xy(pose, "left_ankle")
    ra = _pose_xy(pose, "right_ankle")
    if ls and rs and lh and rh:
        shoulder_w = max(24.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
        waist_y = int(round((min(ls[1], rs[1]) * 0.35) + (max(lh[1], rh[1]) * 0.65)))
        expected_waist = _row_width(target, waist_y)
        actual_waist = _row_width(dress_like, waist_y)
        if expected_waist > 12 and actual_waist > expected_waist * 1.28:
            reasons.append(f"waist_too_wide={actual_waist:.0f}/{expected_waist:.0f}")
        if expected_waist > 12 and actual_waist < expected_waist * 0.42:
            reasons.append(f"waist_missing={actual_waist:.0f}/{expected_waist:.0f}")

        if sleeve_type == "long":
            sleeve_mask = _long_sleeve_pose_mask(pose, (h, w))
            sleeve_mask = cv2.bitwise_and(sleeve_mask, target)
            sleeve_count = int(cv2.countNonZero(sleeve_mask))
            if sleeve_count > 80:
                sleeve_cov = float(cv2.countNonZero(cv2.bitwise_and(dress_like, sleeve_mask))) / float(sleeve_count)
                if sleeve_cov < 0.20:
                    reasons.append(f"sleeve_missing={sleeve_cov:.2f}")

        if lk and rk and la and ra and length in {"midi", "maxi"}:
            knee_y = float((lk[1] + rk[1]) * 0.5)
            ankle_y = float((la[1] + ra[1]) * 0.5)
            midi_min = knee_y + (ankle_y - knee_y) * 0.22
            if cy2 < midi_min:
                reasons.append("hem_above_midi_band")
            if cy2 > ankle_y + shoulder_w * 0.18:
                reasons.append("hem_hits_shoes")

    if parsing:
        protect = np.zeros((h, w), dtype=np.uint8)
        for key in ("face", "hair"):
            region = parsing.get(key)
            if region is not None:
                if region.shape[:2] != (h, w):
                    region = cv2.resize(region, (w, h), interpolation=cv2.INTER_NEAREST)
                protect = cv2.bitwise_or(protect, (region > 20).astype(np.uint8) * 255)
        if int(cv2.countNonZero(protect)) > 0:
            overlap = cv2.countNonZero(cv2.bitwise_and(dress_like, protect))
            if overlap > max(20, int(cv2.countNonZero(protect) * 0.035)):
                reasons.append("dress_over_face_or_hair")

    return len(reasons) == 0, reasons, dress_like


def _blend_luminance_detail(
    base_rgb: np.ndarray,
    detail_rgb: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.18,
) -> np.ndarray:
    if int(cv2.countNonZero(mask)) < 50:
        return base_rgb
    base_lab = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    detail_lab = cv2.cvtColor(detail_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    alpha = cv2.GaussianBlur((mask > 20).astype(np.float32), (9, 9), 2.0)
    alpha = np.clip(alpha * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0)
    base_lab[..., 0] = base_lab[..., 0] * (1.0 - alpha) + detail_lab[..., 0] * alpha
    return cv2.cvtColor(np.clip(base_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _soft_dress_alpha(mask: np.ndarray, sigma: float = 3.6, radius: int = 11) -> np.ndarray:
    """Soft alpha for seed/fallback paths so target-mask edges do not look cut out."""
    if mask is None or mask.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    m = (mask > 20).astype(np.uint8) * 255
    if int(cv2.countNonZero(m)) < 20:
        return m.astype(np.float32) / 255.0
    radius = max(3, int(radius) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), float(sigma))
    return np.clip(alpha, 0.0, 1.0)


def _build_seed(
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    masks: DressMasks,
) -> np.ndarray:
    """Compose the diffusion init image.

    1. Telea-inpaint old-clothes pixels on the person so no flat fill.
    2. Stamp a centered, scaled crop of the cloth into the target footprint.
    3. Outside the target footprint, keep the inpainted person.
    """
    erase = cv2.bitwise_or(masks.old_clothes_mask, masks.target_mask)
    erase = cv2.subtract(erase, masks.hair_front_mask)
    erase = cv2.subtract(erase, masks.shoe_protect_mask)
    seed = telea_inpaint_seed(person_rgb, erase)

    ys, xs = np.where(masks.target_mask > 0)
    if len(xs) < 100:
        return seed
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    tw, th = max(1, x2 - x1 + 1), max(1, y2 - y1 + 1)

    cys, cxs = np.where(cloth_mask > 0)
    if len(cxs) < 50:
        return seed
    cy1, cy2 = int(cys.min()), int(cys.max())
    cx1, cx2 = int(cxs.min()), int(cxs.max())
    cloth_crop = cloth_rgb[cy1:cy2 + 1, cx1:cx2 + 1]
    cloth_msk = cloth_mask[cy1:cy2 + 1, cx1:cx2 + 1]
    cloth_resized = cv2.resize(cloth_crop, (tw, th), interpolation=cv2.INTER_CUBIC)
    # Mild unsharp only — strong unsharp on the cloth resize caused vertical
    # curtain-stripe artefacts when the source dress had vertical folds.
    _blur = cv2.GaussianBlur(cloth_resized, (0, 0), 2.0)
    cloth_resized = cv2.addWeighted(cloth_resized, 1.08, _blur, -0.08, 0)
    cloth_resized = np.clip(cloth_resized, 0, 255).astype(np.uint8)
    msk_resized = cv2.resize(cloth_msk, (tw, th), interpolation=cv2.INTER_NEAREST)

    valid = cloth_msk > 0
    if int(valid.sum()) > 20:
        median_rgb = np.median(cloth_crop[valid], axis=0).astype(np.float32)
    else:
        median_rgb = np.median(cloth_rgb.reshape(-1, 3), axis=0).astype(np.float32)

    # Pose-aware guide: target_mask owns geometry. Do not let the flat-lay
    # product mask stamp straight sleeves or collar shape into the seed.
    cloth_texture = cloth_resized.astype(np.float32)
    cloth_texture[msk_resized <= 0] = median_rgb
    cloth_texture = cv2.GaussianBlur(cloth_texture, (3, 3), 0.6)
    cloth_texture = cloth_texture * 0.88 + median_rgb[None, None, :] * 0.12
    cloth_texture = np.clip(cloth_texture, 0, 255).astype(np.uint8)
    target_crop = masks.target_mask[y1:y2 + 1, x1:x2 + 1]

    region = seed[y1:y2 + 1, x1:x2 + 1].copy()
    msk_f = _soft_dress_alpha(target_crop, sigma=3.4, radius=11)[..., None]
    region = region.astype(np.float32) * (1.0 - msk_f) + cloth_texture.astype(np.float32) * msk_f

    seed[y1:y2 + 1, x1:x2 + 1] = np.clip(region, 0, 255).astype(np.uint8)

    inside_target = (masks.target_mask > 0).astype(np.uint8) * 255
    placed_cloth_mask = np.zeros_like(inside_target)
    placed_cloth_mask[y1:y2 + 1, x1:x2 + 1] = target_crop
    holes = cv2.subtract(inside_target, cv2.bitwise_and(inside_target, placed_cloth_mask))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if int(holes.sum()) > 255 * 50:
        seed = cv2.inpaint(seed, holes, 4, cv2.INPAINT_TELEA)

    return seed


def _run_local_diffusion(
    seed_rgb: np.ndarray,
    mask_gray: np.ndarray,
    prompt: str,
    cloth_rgb: np.ndarray,
    steps: int,
    guidance: float,
    strength: float,
    refiner_mode: str = "dpm++",
) -> np.ndarray:
    from src.gen_tryon import GenConfig, generate_tryon_image
    _use_ip = os.getenv("VTON_USE_IP_ADAPTER", "0").strip().lower() in {"1", "true", "yes", "on"}
    cfg = GenConfig(
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        strength=float(strength),
        refiner_mode=str(refiner_mode or "dpm++"),
        cloth_type="dress",
        infer_size=512,
        reference_image_rgb=cloth_rgb if _use_ip else None,
        ip_adapter_scale=float(os.getenv("VTON_IP_ADAPTER_SCALE", "0.0")),
        negative_prompt=(
            "front slit, split skirt, open front, wrap opening, coat, blazer, jacket, lapels, buttons, "
            "scarf, shawl, cowl neck, turtleneck, collar covering chin, straight flat-lay sleeves, "
            "sleeves hanging vertically instead of following arms, "
            "curtain-like vertical stripes, flat pleated curtain, stiff straight tube dress, "
            "missing sleeves, bare forearms, old yellow dress visible, "
            "old dress visible, duplicated dress, brown block, square artifact, "
            "pasted cloth, flat texture, wrong hem, deformed legs, extra skirt layer, "
            "blurry pattern, broken waist, distorted arms, shoes painted over, "
            "deformed body, bad anatomy, blurry, low quality, multiple garments"
        ),
    )
    return generate_tryon_image(seed_rgb, mask_gray, prompt, cfg)


def _run_cloud_diffusion(
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    style_prompt: str,
    steps: int,
    guidance: float,
) -> np.ndarray:
    from src.cloud_vton_router import generate_with_cloud_router
    with tempfile.TemporaryDirectory() as td:
        p_path = Path(td) / "person.png"
        c_path = Path(td) / "cloth.png"
        cv2.imwrite(str(p_path), cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(c_path), cv2.cvtColor(cloth_rgb, cv2.COLOR_RGB2BGR))
        out_path, _backend = generate_with_cloud_router(
            person_image_path=p_path,
            cloth_image_path=c_path,
            style_prompt=style_prompt or "a realistic full-body dress matching the garment reference",
            steps=int(steps),
            guidance=float(min(guidance, 7.5)),
            seed=random.randint(0, 10000),
            cloth_type="overall",
        )
        img = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("cloud router returned no image")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_dress_pipeline_v2(
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    *,
    style_prompt: str = "",
    use_cloud: bool = False,
    gen_steps: int = 24,
    gen_guidance: float = 3.8,
    preserve_strength: float = 0.72,
    pose: dict | None = None,
    parsing: dict | None = None,
    cloth_mask: np.ndarray | None = None,
) -> DressPipelineResult:
    info: list[str] = ["DressV2:start"]
    run_id = dt.datetime.now().strftime("%H%M%S")
    h, w = person_rgb.shape[:2]
    _save_debug("dressv2_00_person_input", person_rgb, run_id=run_id)
    _save_debug("dressv2_00_cloth_input", cloth_rgb, run_id=run_id)

    if pose is None:
        try:
            pose = detect_dress_pose(person_rgb)
            info.append("DressV2:pose=mediapipe_heavy")
        except Exception as _exc:
            info.append(f"DressV2:pose_heavy_fail={type(_exc).__name__}; fallback=light")
            pose = detect_full_pose(person_rgb)
    if parsing is None:
        parsing = parse_human(person_rgb)
    if cloth_mask is None:
        try:
            cloth_mask = segment_cloth_ensemble(cloth_rgb)
        except Exception:
            cloth_mask = build_cloth_mask(cloth_rgb)

    analysis = detect_dress_type(cloth_mask, cloth_rgb)

    # Gemini auto-prompt is the most reliable source of garment semantics.
    # detect_dress_type misreads flat-lay long-sleeve A-line dresses as
    # sheath/sleeveless. If the user-supplied prompt (typically generated by
    # Gemini Vision) names a silhouette/length/sleeve, trust it.
    _prompt_locked: set[str] = set()
    try:
        _p = (style_prompt or "").lower()
        if _p:
            _hits = []
            # Silhouette
            if "a-line" in _p or "a line" in _p:
                analysis["silhouette"] = "a_line"; _hits.append("a_line"); _prompt_locked.add("silhouette")
            elif "fit and flare" in _p or "fit-and-flare" in _p or "fit & flare" in _p:
                analysis["silhouette"] = "fit_and_flare"; _hits.append("fit_and_flare"); _prompt_locked.add("silhouette")
            elif "ball gown" in _p or "ballgown" in _p:
                analysis["silhouette"] = "ball_gown"; _hits.append("ball_gown"); _prompt_locked.add("silhouette")
            elif "empire" in _p:
                analysis["silhouette"] = "empire"; _hits.append("empire"); _prompt_locked.add("silhouette")
            elif "mermaid" in _p or "trumpet" in _p:
                analysis["silhouette"] = "mermaid"; _hits.append("mermaid"); _prompt_locked.add("silhouette")
            elif "sheath" in _p:
                analysis["silhouette"] = "sheath"; _hits.append("sheath"); _prompt_locked.add("silhouette")
            elif "shift" in _p:
                analysis["silhouette"] = "shift"; _hits.append("shift"); _prompt_locked.add("silhouette")
            # Length
            for _k, _v in (("maxi", "maxi"), ("midi", "midi"), ("mini", "mini"),
                           ("knee-length", "knee"), ("knee length", "knee"),
                           ("thigh", "thigh")):
                if _k in _p:
                    analysis["length"] = _v; _hits.append(f"len={_v}"); _prompt_locked.add("length")
                    break
            # Sleeve
            if "long sleeve" in _p or "long-sleeve" in _p or "long sleeved" in _p or "long-sleeved" in _p:
                analysis["sleeve"] = "long"; _hits.append("long_sleeve"); _prompt_locked.add("sleeve")
            elif "short sleeve" in _p or "short-sleeve" in _p:
                analysis["sleeve"] = "short"; _hits.append("short_sleeve"); _prompt_locked.add("sleeve")
            elif "sleeveless" in _p or "tank" in _p:
                analysis["sleeve"] = "sleeveless"; _hits.append("sleeveless"); _prompt_locked.add("sleeve")
            if _hits:
                info.append("DressV2:prompt_override=" + ",".join(_hits))
    except Exception as _exc:
        info.append(f"DressV2:prompt_parse_skip={type(_exc).__name__}")

    # Override mis-detected analysis from cloth bbox aspect + width curve.
    # detect_dress_type is unreliable for flat-lay long-sleeve A-line dresses
    # (frequently reads them as sheath/sleeveless/midi). Prompt-locked fields
    # are NOT overwritten — the Gemini description wins.
    try:
        cys, cxs = np.where(cloth_mask > 20)
        if len(cxs) > 100:
            cmw = int(cxs.max() - cxs.min() + 1)
            cmh = int(cys.max() - cys.min() + 1)
            aspect = float(cmh) / max(1.0, float(cmw))
            # Width at three vertical bands
            top_q = cys.min() + int(cmh * 0.2)
            mid_q = cys.min() + int(cmh * 0.5)
            bot_q = cys.min() + int(cmh * 0.85)
            def _band_w(y0: int) -> int:
                y_lo = max(0, y0 - 4)
                y_hi = min(cloth_mask.shape[0] - 1, y0 + 4)
                xs_band = np.where(cloth_mask[y_lo:y_hi + 1] > 20)[1]
                return int(xs_band.max() - xs_band.min() + 1) if len(xs_band) > 4 else 0
            top_w = _band_w(top_q)
            mid_w = _band_w(mid_q)
            bot_w = _band_w(bot_q)
            if "sleeve" not in _prompt_locked and top_w > mid_w * 1.25:
                analysis["sleeve"] = "long"
            if "silhouette" not in _prompt_locked:
                if bot_w > mid_w * 1.10:
                    analysis["silhouette"] = "a_line"
                elif bot_w > mid_w * 1.25:
                    analysis["silhouette"] = "fit_and_flare"
            if "length" not in _prompt_locked:
                if aspect > 1.6:
                    analysis["length"] = "maxi"
                elif aspect > 1.35:
                    analysis["length"] = "midi"
            info.append(
                f"DressV2:cloth_override aspect={aspect:.2f} top/mid/bot={top_w}/{mid_w}/{bot_w}"
            )
    except Exception as _exc:
        info.append(f"DressV2:override_skip={type(_exc).__name__}")

    info.append(
        f"DressV2:analysis silhouette={analysis.get('silhouette')} "
        f"length={analysis.get('length')} sleeve={analysis.get('sleeve')}"
    )
    _save_debug("dressv2_00_cloth_mask", cloth_mask, is_mask=True, run_id=run_id)

    target = build_target_silhouette(pose, analysis, (h, w))
    _save_debug("dressv2_01_target_silhouette", target, is_mask=True, run_id=run_id)

    # Pass "auto" so the mask builder always unions the sleeve pose envelope.
    # Source-cloth sleeve detection often misfires on long-sleeve flat-lay
    # photos (reads as "sleeveless") which previously starved the diffusion
    # mask of any arm coverage. "auto" -> include sleeves + hand-only protect.
    _sleeve_for_mask = str(analysis.get("sleeve") or "auto").lower()
    if _sleeve_for_mask == "sleeveless":
        _sleeve_for_mask = "auto"
    masks = build_dress_masks(target, parsing, pose, sleeve_type=_sleeve_for_mask)
    _save_debug("dressv2_02_target_mask", masks.target_mask, is_mask=True, run_id=run_id)
    _save_debug("dressv2_02_agnostic_mask", masks.agnostic_mask, is_mask=True, run_id=run_id)
    _save_debug("dressv2_03_hair_front", masks.hair_front_mask, is_mask=True, run_id=run_id)
    _save_debug("dressv2_03_hair_underlap", masks.hair_underlap_mask, is_mask=True, run_id=run_id)

    seed = _build_seed(person_rgb, cloth_rgb, cloth_mask, masks)
    _save_debug("dressv2_04_seed", seed, run_id=run_id)
    target_alpha_dbg = (_soft_dress_alpha(masks.target_mask, sigma=3.6, radius=11) * 255.0).astype(np.uint8)
    _save_debug("dressv2_04a_target_alpha", target_alpha_dbg, is_mask=True, run_id=run_id)

    diffusion_mask = cv2.dilate(masks.agnostic_mask, np.ones((3, 3), np.uint8), iterations=1)
    _save_debug("dressv2_04b_diffusion_mask", diffusion_mask, is_mask=True, run_id=run_id)

    _sleeve = str(analysis.get("sleeve") or "auto").lower()
    _length = str(analysis.get("length") or "midi").lower()
    _sleeve_clause = "long sleeves fully covering both forearms, " if _sleeve == "long" else (
        "short sleeves, " if _sleeve == "short" else ""
    )
    base_prompt = (style_prompt or "").strip() or (
        f"a full body photo of a woman wearing a taupe {_sleeve_clause}A-line {_length} dress, "
        "crew neck, natural soft jersey knit fabric, subtle vertical folds, "
        "realistic drape around waist and hips, clean neckline, single dress layer, "
        "dress covers old clothing completely, realistic shadow"
    )
    strict_head = (
        "closed-front one-piece taupe brown A-line midi dress, "
        "long sleeves follow the model shoulders elbows and wrists, "
        "crew neck at base of neck below chin, continuous skirt panel"
    )
    prompt = f"{strict_head}, {base_prompt}"

    def _validate_diffused(img: np.ndarray) -> bool:
        """Return False if the output looks like junk (mostly white/black/uniform)."""
        if img is None or img.size == 0:
            return False
        if img.shape[:2] != (h, w):
            return True  # will be resized; trust caller
        # Sample inside the dress target — diffusion should have painted SOMETHING
        # close to dress colour there, not pure white/background.
        ys, xs = np.where(masks.target_mask > 20)
        if len(xs) < 50:
            return True
        sample = img[ys, xs].astype(np.float32)
        mean = sample.mean(axis=0)
        # Cloud router sometimes returns the cloth crop on white BG; that means
        # >90% of target-area pixels are near-white. Reject in that case.
        white_frac = float(((sample > 225).all(axis=1)).mean())
        if white_frac > 0.40:
            return False
        if mean.mean() < 8 or mean.mean() > 232:
            return False
        # Reject if variance is near-zero (uniform flat fill, often pure white)
        if float(sample.std()) < 6.0:
            return False
        return True

    diffused = None
    if use_cloud:
        try:
            cand = _run_cloud_diffusion(seed, cloth_rgb, prompt, gen_steps, gen_guidance)
            if cand.shape[:2] != (h, w):
                cand = cv2.resize(cand, (w, h), interpolation=cv2.INTER_LINEAR)
            if _validate_diffused(cand):
                diffused = cand
                info.append("DressV2:diffusion=cloud")
            else:
                info.append("DressV2:cloud_returned_junk")
                _save_debug("dressv2_05_cloud_rejected", cand, run_id=run_id)
        except Exception as exc:
            info.append(f"DressV2:cloud_fail={type(exc).__name__}:{exc}")
            _save_debug_text("dressv2_05_cloud_exception", repr(exc), run_id=run_id)

    if diffused is None:
        try:
            cand = _run_local_diffusion(
                seed, diffusion_mask, prompt, cloth_rgb,
                gen_steps, gen_guidance, preserve_strength,
            )
            if _validate_diffused(cand):
                diffused = cand
                info.append("DressV2:diffusion=local")
            else:
                info.append("DressV2:local_returned_junk")
                _save_debug("dressv2_05_local_rejected", cand, run_id=run_id)
                diffused = seed
        except Exception as exc:
            info.append(f"DressV2:local_fail={type(exc).__name__}:{exc}")
            _save_debug_text("dressv2_05_diffusion_exception", repr(exc), run_id=run_id)
            _save_debug("dressv2_05_diffusion_failed_fallback_seed", seed, run_id=run_id)
            diffused = seed

    _save_debug("dressv2_05_diffusion", diffused, run_id=run_id)

    shape_ok, shape_reasons, generated_dress_mask = _diffusion_geometry_guard(
        diffused,
        cloth_rgb,
        masks.target_mask,
        pose,
        parsing,
        sleeve_type=str(analysis.get("sleeve") or "auto").lower(),
        length=str(analysis.get("length") or "midi").lower(),
        reference_mask=cloth_mask,
    )
    color_anchor_mask = masks.target_mask
    # Only reject diffusion for HARD failures (NaN/all-white/no-coverage). Soft
    # reasons (hem position, sleeve coverage, waist width) come from possibly
    # wrong silhouette/sleeve detection — trusting the seed-blend over a real
    # diffusion output makes the final image look pasted. Mirror how the top
    # pipeline does it: trust diffusion inside target_mask, just color-anchor.
    _fatal = {"no_dress_like_pixels"}
    fatal_fail = (not shape_ok) and any(r in _fatal for r in shape_reasons) or (
        not shape_ok and any(r.startswith("coverage=") for r in shape_reasons)
        and any(float(r.split("=")[1]) < 0.15 for r in shape_reasons if r.startswith("coverage="))
    ) or (
        not shape_ok and any(r.startswith("skirt_split=") for r in shape_reasons)
    )
    if shape_ok:
        color_anchor_mask = cv2.bitwise_and(masks.target_mask, generated_dress_mask)
        if int(cv2.countNonZero(color_anchor_mask)) < 255:
            color_anchor_mask = masks.target_mask
        info.append("DressV2:geometry_guard=ok")
    elif fatal_fail:
        info.append("DressV2:geometry_guard_FATAL=" + ",".join(shape_reasons[:5]))
        diffused = _blend_luminance_detail(seed, diffused, masks.target_mask, strength=0.08)
        _save_debug("dressv2_05b_guarded_seed_detail", diffused, run_id=run_id)
    else:
        # Soft warnings only — keep the diffusion result, just log.
        info.append("DressV2:geometry_guard_soft=" + ",".join(shape_reasons[:5]))

    alpha = _soft_dress_alpha(masks.target_mask, sigma=3.6, radius=11)[..., None]
    composite = seed.astype(np.float32) * (1.0 - alpha) + diffused.astype(np.float32) * alpha
    out = np.clip(composite, 0, 255).astype(np.uint8)

    out = apply_color_anchor(out, cloth_rgb, color_anchor_mask, strength=0.25, ref_mask=cloth_mask)
    out = restore_occluders(out, person_rgb, masks)
    _save_debug("dressv2_06_color_occluder_restored", out, run_id=run_id)

    out, ghost_fixed = remove_old_dress_ghost(out, person_rgb, masks.old_clothes_mask, masks.target_mask)
    if ghost_fixed:
        info.append("DressV2:ghost_removed")
        _save_debug("dressv2_06b_ghost_removed", out, run_id=run_id)
    out = clean_hem(out, person_rgb, masks.target_mask, masks.shoe_protect_mask)
    _save_debug("dressv2_06c_hem_cleaned", out, run_id=run_id)
    out, rect_fixed = remove_rect_artifact(out, seed, masks.target_mask)
    if rect_fixed:
        info.append("DressV2:rect_artifact_cleaned")
        _save_debug("dressv2_06d_rect_cleaned", out, run_id=run_id)

    _save_debug("dressv2_07_final", out, run_id=run_id)
    info.append("DressV2:done")
    _save_debug_text(
        "dressv2_99_info",
        "\n".join(info + [f"prompt={prompt}", f"use_cloud={use_cloud}", f"steps={gen_steps}", f"guidance={gen_guidance}"]),
        run_id=run_id,
    )

    return DressPipelineResult(image=out, masks=masks, analysis=analysis, debug=info)
