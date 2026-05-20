from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseBox:
    left_shoulder: tuple[int, int]
    right_shoulder: tuple[int, int]
    left_hip: tuple[int, int]
    right_hip: tuple[int, int]


POSE_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def _ensure_model_cache_env() -> Path:
    """Force model downloads/caches to VTO_BASE_DIR, defaulting to drive E."""
    base_dir = Path(os.getenv("VTO_BASE_DIR", "E:/virtual_try_on_data")).expanduser().resolve()
    hf_home = base_dir / "huggingface"
    hf_hub = hf_home / "hub"
    hf_assets = hf_home / "assets"
    torch_home = base_dir / "torch"
    xdg_cache_home = base_dir / "cache"
    u2net_home = base_dir / "u2net"
    rembg_home = base_dir / "rembg"
    mediapipe_home = base_dir / "cache" / "mediapipe"

    for directory in (hf_home, hf_hub, hf_assets, torch_home, xdg_cache_home, u2net_home, rembg_home, mediapipe_home):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_hub)
    os.environ["HF_HUB_CACHE"] = str(hf_hub)
    os.environ["HF_ASSETS_CACHE"] = str(hf_assets)
    os.environ["DIFFUSERS_CACHE"] = str(hf_hub)
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)
    os.environ["U2NET_HOME"] = str(u2net_home)
    os.environ["REMBG_HOME"] = str(rembg_home)
    os.environ["MEDIAPIPE_HOME"] = str(mediapipe_home)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return base_dir


_ensure_model_cache_env()


def read_image_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def detect_upper_body_box(person_rgb: np.ndarray) -> PoseBox:
    if hasattr(mp, "solutions"):
        return _detect_pose_with_solutions(person_rgb)
    return _detect_pose_with_tasks(person_rgb)


def detect_full_pose(person_rgb: np.ndarray) -> dict[str, tuple[int, int]]:
    """Detect extended body landmarks (shoulders, elbows, wrists, hips)."""
    if hasattr(mp, "solutions"):
        return _detect_full_pose_solutions(person_rgb)
    return _detect_full_pose_tasks(person_rgb)


def full_pose_to_box(pose: dict[str, tuple[int, int]]) -> PoseBox:
    return PoseBox(
        left_shoulder=pose["left_shoulder"],
        right_shoulder=pose["right_shoulder"],
        left_hip=pose["left_hip"],
        right_hip=pose["right_hip"],
    )


def _detect_pose_with_solutions(person_rgb: np.ndarray) -> PoseBox:
    height, width = person_rgb.shape[:2]

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(person_rgb)

    if not result.pose_landmarks:
        raise ValueError("Không tìm thấy pose. Hãy dùng ảnh rõ toàn thân nửa trên.")

    lm = result.pose_landmarks.landmark

    def _xy(landmark_id: int) -> tuple[int, int]:
        point = lm[landmark_id]
        return int(point.x * width), int(point.y * height)

    ls = _xy(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
    rs = _xy(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)
    lh = _xy(mp.solutions.pose.PoseLandmark.LEFT_HIP)
    rh = _xy(mp.solutions.pose.PoseLandmark.RIGHT_HIP)

    return PoseBox(ls, rs, lh, rh)


def _detect_full_pose_solutions(person_rgb: np.ndarray) -> dict[str, tuple[int, int]]:
    height, width = person_rgb.shape[:2]
    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.5,
    ) as pose:
        result = pose.process(person_rgb)
    if not result.pose_landmarks:
        raise ValueError("Không tìm thấy pose. Hãy dùng ảnh rõ toàn thân nửa trên.")
    lm = result.pose_landmarks.landmark
    PL = mp.solutions.pose.PoseLandmark

    def _xy(lid) -> tuple[int, int]:
        p = lm[lid]
        return int(p.x * width), int(p.y * height)

    return {
        "nose": _xy(PL.NOSE),
        "left_eye_inner": _xy(PL.LEFT_EYE_INNER),
        "left_eye": _xy(PL.LEFT_EYE),
        "left_eye_outer": _xy(PL.LEFT_EYE_OUTER),
        "right_eye_inner": _xy(PL.RIGHT_EYE_INNER),
        "right_eye": _xy(PL.RIGHT_EYE),
        "right_eye_outer": _xy(PL.RIGHT_EYE_OUTER),
        "left_ear": _xy(PL.LEFT_EAR),
        "right_ear": _xy(PL.RIGHT_EAR),
        "mouth_left": _xy(PL.MOUTH_LEFT),
        "mouth_right": _xy(PL.MOUTH_RIGHT),
        "left_shoulder": _xy(PL.LEFT_SHOULDER),
        "right_shoulder": _xy(PL.RIGHT_SHOULDER),
        "left_elbow": _xy(PL.LEFT_ELBOW),
        "right_elbow": _xy(PL.RIGHT_ELBOW),
        "left_wrist": _xy(PL.LEFT_WRIST),
        "right_wrist": _xy(PL.RIGHT_WRIST),
        "left_pinky": _xy(PL.LEFT_PINKY),
        "right_pinky": _xy(PL.RIGHT_PINKY),
        "left_index": _xy(PL.LEFT_INDEX),
        "right_index": _xy(PL.RIGHT_INDEX),
        "left_thumb": _xy(PL.LEFT_THUMB),
        "right_thumb": _xy(PL.RIGHT_THUMB),
        "left_hip": _xy(PL.LEFT_HIP),
        "right_hip": _xy(PL.RIGHT_HIP),
        "left_knee": _xy(PL.LEFT_KNEE),
        "right_knee": _xy(PL.RIGHT_KNEE),
        "left_ankle": _xy(PL.LEFT_ANKLE),
        "right_ankle": _xy(PL.RIGHT_ANKLE),
        "left_heel": _xy(PL.LEFT_HEEL),
        "right_heel": _xy(PL.RIGHT_HEEL),
        "left_foot_index": _xy(PL.LEFT_FOOT_INDEX),
        "right_foot_index": _xy(PL.RIGHT_FOOT_INDEX),
    }


_DRESS_POSE_KEYS = (
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def detect_dress_pose(
    person_rgb: np.ndarray,
    *,
    visibility_threshold: float = 0.35,
) -> dict[str, tuple[int, int] | float]:
    """High-accuracy pose for the dress pipeline.

    Uses MediaPipe `model_complexity=2` (heavy) and returns only keypoints with
    visibility above `visibility_threshold` so the dress mask builder can rely
    on what it sees and mirror what's missing. Visibility scores are also
    stored under `{key}_v` so callers can inspect them.

    Returns a dict that:
    - sets each landmark name to `(x, y)` when reliable, otherwise `None`
    - sets `f"{name}_v"` to the float visibility score (0..1)
    """
    if not hasattr(mp, "solutions"):
        # Tasks API fallback returns no visibility — fall back gracefully.
        return _detect_full_pose_tasks(person_rgb)  # type: ignore[return-value]

    height, width = person_rgb.shape[:2]
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as pose:
        result = pose.process(person_rgb)
    if not result.pose_landmarks:
        raise ValueError("Không tìm thấy pose. Hãy dùng ảnh rõ toàn thân.")
    lm = result.pose_landmarks.landmark
    PL = mp.solutions.pose.PoseLandmark

    def _pt(lid) -> tuple[tuple[int, int], float]:
        p = lm[lid]
        return (int(p.x * width), int(p.y * height)), float(getattr(p, "visibility", 1.0))

    name_to_id = {
        "nose": PL.NOSE,
        "left_ear": PL.LEFT_EAR,
        "right_ear": PL.RIGHT_EAR,
        "left_shoulder": PL.LEFT_SHOULDER,
        "right_shoulder": PL.RIGHT_SHOULDER,
        "left_elbow": PL.LEFT_ELBOW,
        "right_elbow": PL.RIGHT_ELBOW,
        "left_wrist": PL.LEFT_WRIST,
        "right_wrist": PL.RIGHT_WRIST,
        "left_hip": PL.LEFT_HIP,
        "right_hip": PL.RIGHT_HIP,
        "left_knee": PL.LEFT_KNEE,
        "right_knee": PL.RIGHT_KNEE,
        "left_ankle": PL.LEFT_ANKLE,
        "right_ankle": PL.RIGHT_ANKLE,
        "left_heel": PL.LEFT_HEEL,
        "right_heel": PL.RIGHT_HEEL,
        "left_foot_index": PL.LEFT_FOOT_INDEX,
        "right_foot_index": PL.RIGHT_FOOT_INDEX,
    }

    out: dict[str, tuple[int, int] | float | None] = {}
    for name, lid in name_to_id.items():
        xy, vis = _pt(lid)
        out[f"{name}_v"] = vis
        out[name] = xy if vis >= visibility_threshold else None
    return out  # type: ignore[return-value]


def _detect_pose_with_tasks(person_rgb: np.ndarray) -> PoseBox:
    height, width = person_rgb.shape[:2]

    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _ensure_pose_task_model()

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_rgb)
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raise ValueError("Không tìm thấy pose. Hãy dùng ảnh rõ toàn thân nửa trên.")

    lm = result.pose_landmarks[0]

    def _xy(index: int) -> tuple[int, int]:
        point = lm[index]
        return int(point.x * width), int(point.y * height)

    ls = _xy(11)
    rs = _xy(12)
    lh = _xy(23)
    rh = _xy(24)

    return PoseBox(ls, rs, lh, rh)


def _detect_full_pose_tasks(person_rgb: np.ndarray) -> dict[str, tuple[int, int]]:
    height, width = person_rgb.shape[:2]
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _ensure_pose_task_model()
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_rgb)
        result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        raise ValueError("Không tìm thấy pose. Hãy dùng ảnh rõ toàn thân nửa trên.")
    lm = result.pose_landmarks[0]

    def _xy(idx: int) -> tuple[int, int]:
        p = lm[idx]
        return int(p.x * width), int(p.y * height)

    return {
        "nose": _xy(0),
        "left_eye_inner": _xy(1), "left_eye": _xy(2), "left_eye_outer": _xy(3),
        "right_eye_inner": _xy(4), "right_eye": _xy(5), "right_eye_outer": _xy(6),
        "left_ear": _xy(7), "right_ear": _xy(8),
        "mouth_left": _xy(9), "mouth_right": _xy(10),
        "left_shoulder": _xy(11), "right_shoulder": _xy(12),
        "left_elbow": _xy(13), "right_elbow": _xy(14),
        "left_wrist": _xy(15), "right_wrist": _xy(16),
        "left_pinky": _xy(17), "right_pinky": _xy(18),
        "left_index": _xy(19), "right_index": _xy(20),
        "left_thumb": _xy(21), "right_thumb": _xy(22),
        "left_hip": _xy(23), "right_hip": _xy(24),
        "left_knee": _xy(25), "right_knee": _xy(26),
        "left_ankle": _xy(27), "right_ankle": _xy(28),
        "left_heel": _xy(29), "right_heel": _xy(30),
        "left_foot_index": _xy(31), "right_foot_index": _xy(32),
    }


def _ensure_pose_task_model() -> Path:
    base_dir = _ensure_model_cache_env()
    model_dir = Path(os.getenv("MEDIAPIPE_HOME", str(base_dir / "cache" / "mediapipe")))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "pose_landmarker_lite.task"
    if not model_path.exists():
        urlretrieve(POSE_TASK_MODEL_URL, str(model_path))

    return model_path


def build_cloth_mask(cloth_rgb: np.ndarray, threshold: int = 245) -> np.ndarray:
    gray = cv2.cvtColor(cloth_rgb, cv2.COLOR_RGB2GRAY)
    init_mask = (gray < threshold).astype(np.uint8) * 255

    total = cloth_rgb.shape[0] * cloth_rgb.shape[1]
    fg = int(cv2.countNonZero(init_mask))

    # If threshold-based mask is unreliable, try GrabCut
    if fg < total * 0.05 or fg > total * 0.95:
        gc_mask = _grabcut_cloth_mask(cloth_rgb)
        if gc_mask is not None:
            init_mask = gc_mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(init_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dilate mask outward so cloth covers full silhouette.
    # Use moderate dilation to extend coverage for sleeve edges without
    # pulling in excessive background.
    dilate_k = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, dilate_k, iterations=1)
    # Fill internal holes via contour-fill
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    return mask


def _grabcut_cloth_mask(cloth_rgb: np.ndarray) -> np.ndarray | None:
    """GrabCut fallback for non-white backgrounds."""
    try:
        h, w = cloth_rgb.shape[:2]
        margin = max(5, int(min(h, w) * 0.05))
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)
        gc_mask = np.zeros((h, w), np.uint8)
        bg = np.zeros((1, 65), np.float64)
        fg = np.zeros((1, 65), np.float64)
        bgr = cv2.cvtColor(cloth_rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(bgr, gc_mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
        return np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0,
        ).astype(np.uint8)
    except Exception:
        return None


def warp_cloth_to_torso(
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    box: PoseBox,
    fit_scale: float,
    y_offset_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = person_rgb.shape[:2]

    shoulder_width = np.linalg.norm(np.array(box.left_shoulder) - np.array(box.right_shoulder))
    hip_width = np.linalg.norm(np.array(box.left_hip) - np.array(box.right_hip))

    # Garment must be WIDER than skeleton — clothes hang loose
    # Add ~15% overhang beyond body edges for natural look
    scale = float(np.clip(fit_scale, 0.90, 1.25))
    top_half_w = int(shoulder_width * scale * 0.62)   # was /2 (0.50) → too tight
    bot_half_w = int(hip_width * scale * 0.62)

    torso_height = int(np.mean([
        abs(box.left_hip[1] - box.left_shoulder[1]),
        abs(box.right_hip[1] - box.right_shoulder[1]),
    ]) * (1.15 + y_offset_ratio))

    if top_half_w < 10 or bot_half_w < 10 or torso_height < 20:
        raise ValueError("Pose không hợp lệ để thử đồ.")

    top_cx = int((box.left_shoulder[0] + box.right_shoulder[0]) / 2)
    bot_cx = int((box.left_hip[0] + box.right_hip[0]) / 2)
    top_y = int(min(box.left_shoulder[1], box.right_shoulder[1]) + y_offset_ratio * 30)
    bottom_y = min(top_y + torso_height, height - 1)

    # Trapezoid destination: follows body taper from shoulders to hips
    dst = np.float32([
        [max(0, top_cx - top_half_w), max(0, top_y)],
        [min(width - 1, top_cx + top_half_w), max(0, top_y)],
        [min(width - 1, bot_cx + bot_half_w), bottom_y],
        [max(0, bot_cx - bot_half_w), bottom_y],
    ])

    h_c, w_c = cloth_rgb.shape[:2]
    src = np.float32([
        [0, 0],
        [w_c - 1, 0],
        [w_c - 1, h_c - 1],
        [0, h_c - 1],
    ])

    transform = cv2.getPerspectiveTransform(src, dst)
    # Use BORDER_REPLICATE for cloth (no black edges), BORDER_CONSTANT for mask
    warped_cloth = cv2.warpPerspective(
        cloth_rgb, transform, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpPerspective(
        cloth_mask, transform, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    return warped_cloth, warped_mask


def erase_clothing_region(
    person_rgb: np.ndarray,
    erase_mask: np.ndarray,
    parsing_skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove original clothing — direct inpaint strategy.

    KEY FIX v3: Previous two-zone strategy (core fill + edge inpaint)
    created a visible flat beige patch at torso center.

    New approach: Use Telea inpainting on the ENTIRE erase area directly.
    Since we've already tightened the erase_mask to only cover actual
    clothing pixels (not skin), the inpaint area is small enough for
    Telea to produce natural results without smearing.

    For very large erase areas (>15% of image), fall back to a
    shrink-then-inpaint approach to avoid Telea artifacts.
    """
    h, w = person_rgb.shape[:2]
    binary = (erase_mask > 30).astype(np.uint8) * 255

    erase_pixels = int(binary.sum()) // 255
    total_pixels = h * w

    if erase_pixels < 100:
        return person_rgb.copy()

    result_bgr = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR)

    if erase_pixels < total_pixels * 0.15:
        # Small area: direct Telea inpaint (natural, uses surrounding pixels)
        result_bgr = cv2.inpaint(result_bgr, binary, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    else:
        # Large area: shrink inpaint in layers (outside-in) to avoid smear
        # Layer 1: inpaint outer ring (edge band)
        k = np.ones((11, 11), np.uint8)
        inner = cv2.erode(binary, k, iterations=2)
        outer_ring = cv2.subtract(binary, inner)
        result_bgr = cv2.inpaint(result_bgr, outer_ring, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        # Layer 2: inpaint remaining inner core (now surrounded by inpainted pixels)
        if inner.sum() > 255 * 50:
            result_bgr = cv2.inpaint(result_bgr, inner, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def poisson_blend(
    person_rgb: np.ndarray,
    warped_cloth: np.ndarray,
    warped_mask: np.ndarray,
) -> np.ndarray:
    """Poisson seamless clone — merges garment colours/gradients naturally."""
    full_mask = (warped_mask > 20).astype(np.uint8) * 255
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    # Clone only a conservative core region; keep outer boundary for alpha blend.
    core_mask = cv2.erode(
        full_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )

    # Compute centre of the core mask
    ys, xs = np.where(core_mask > 0)
    if len(xs) < 100:
        return blend_tryon(person_rgb, warped_cloth, warped_mask, alpha=0.92)
    cx, cy = int(xs.mean()), int(ys.mean())

    # seamlessClone requires BGR
    try:
        core_clone = cv2.seamlessClone(
            cv2.cvtColor(warped_cloth, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR),
            core_mask,
            (cx, cy),
            cv2.NORMAL_CLONE,
        )

        core_clone_rgb = cv2.cvtColor(core_clone, cv2.COLOR_BGR2RGB)

        # Feathered merge from Poisson-core to original around edges.
        core_alpha = cv2.GaussianBlur(core_mask.astype(np.float32), (21, 21), 0) / 255.0
        core_alpha = core_alpha[..., None]
        blended = (
            core_clone_rgb.astype(np.float32) * core_alpha
            + person_rgb.astype(np.float32) * (1.0 - core_alpha)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)
    except cv2.error:
        # Fallback to alpha blend when mask is near image border
        return blend_tryon(person_rgb, warped_cloth, warped_mask, alpha=0.92)


def refine_with_optical_flow(
    warped_cloth: np.ndarray,
    warped_mask: np.ndarray,
    person_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Use dense Optical Flow to nudge warped cloth towards body edges."""
    gray_person = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2GRAY)
    gray_cloth = cv2.cvtColor(warped_cloth, cv2.COLOR_RGB2GRAY)

    # Farneback dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray_cloth, gray_person,
        None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )

    h, w = person_rgb.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    # Only apply flow where warped mask exists, with gentle strength
    strength = 0.20
    mask_f = (warped_mask > 30).astype(np.float32)
    map_x = x_coords + flow[..., 0] * mask_f * strength
    map_y = y_coords + flow[..., 1] * mask_f * strength

    refined_cloth = cv2.remap(
        warped_cloth, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    refined_mask = cv2.remap(
        warped_mask, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return refined_cloth, refined_mask


def warpflow_hybrid_refine(
    tps_cloth: np.ndarray,
    tps_mask: np.ndarray,
    person_rgb: np.ndarray,
    flow_strength: float = 0.22,
    edge_blend: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """WarpFlow hybrid: TPS base + optical-flow edge refinement.

    Strategy:
    1) Keep TPS result as geometric backbone.
    2) Compute dense flow and remap cloth/mask.
    3) Blend flow result mostly on garment boundary to avoid body-shape collapse.
    """
    flow_cloth, flow_mask = refine_with_optical_flow(tps_cloth, tps_mask, person_rgb)

    binary = (tps_mask > 30).astype(np.uint8) * 255
    edge_band = cv2.subtract(
        cv2.dilate(binary, np.ones((7, 7), np.uint8), iterations=1),
        cv2.erode(binary, np.ones((7, 7), np.uint8), iterations=1),
    )
    edge_band = cv2.GaussianBlur(edge_band, (9, 9), 0)
    edge_alpha = (edge_band.astype(np.float32) / 255.0) * float(np.clip(edge_blend, 0.0, 1.0))
    edge_alpha = edge_alpha[..., None]

    hybrid_cloth = (
        tps_cloth.astype(np.float32) * (1.0 - edge_alpha)
        + flow_cloth.astype(np.float32) * edge_alpha
    ).clip(0, 255).astype(np.uint8)

    # Keep mask conservative: union then close to avoid holes.
    hybrid_mask = cv2.bitwise_or(tps_mask, flow_mask)
    hybrid_mask = cv2.morphologyEx(hybrid_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    return hybrid_cloth, hybrid_mask


def blend_tryon(person_rgb: np.ndarray, warped_cloth: np.ndarray, warped_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Two-zone blend: hard core (full occlusion) + soft feathered edges."""
    # Binary mask from warped result
    binary = (warped_mask > 30).astype(np.uint8) * 255

    # Core mask: eroded — fully opaque, completely hides original clothing
    erode_k = np.ones((7, 7), np.uint8)
    core = cv2.erode(binary, erode_k, iterations=2)

    # Edge mask: border region between core and outer boundary, blurred for smooth transition
    dilate_k = np.ones((5, 5), np.uint8)
    outer = cv2.dilate(binary, dilate_k, iterations=1)
    edge = cv2.subtract(outer, core)
    edge = cv2.GaussianBlur(edge, (13, 13), 0)

    # Core = full opacity (1.0), edge = alpha-blended
    core_f = core.astype(np.float32) / 255.0
    edge_f = edge.astype(np.float32) / 255.0 * alpha
    final_mask = np.clip(core_f + edge_f, 0.0, 1.0)[..., None]

    output = person_rgb.astype(np.float32) * (1.0 - final_mask) + warped_cloth.astype(np.float32) * final_mask
    return np.clip(output, 0, 255).astype(np.uint8)


def compute_body_measurements(
    pose: dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Compute body dimensions from pose landmarks."""
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    lh = np.array(pose["left_hip"], dtype=np.float64)
    rh = np.array(pose["right_hip"], dtype=np.float64)

    shoulder_w = float(np.linalg.norm(ls - rs))
    hip_w = float(np.linalg.norm(lh - rh))
    torso_h = float(np.mean([abs(lh[1] - ls[1]), abs(rh[1] - rs[1])]))

    return {
        "shoulder_width": shoulder_w,
        "hip_width": hip_w,
        "torso_height": torso_h,
        "shoulder_hip_ratio": shoulder_w / max(hip_w, 1.0),
        "torso_aspect": torso_h / max(shoulder_w, 1.0),
    }


def compute_leg_measurements(
    pose: dict[str, tuple[int, int]],
) -> dict[str, float]:
    """Compute leg dimensions from pose landmarks.

    v16.11c: Added for pants support.

    Returns:
      - 'hip_width': distance between left/right hips
      - 'leg_length': average length from hip to ankle
      - 'knee_width': average distance between knees (if available)
    """
    lh = np.array(pose.get("left_hip", (0, 0)), dtype=np.float64)
    rh = np.array(pose.get("right_hip", (0, 0)), dtype=np.float64)
    lk = np.array(pose.get("left_knee", (0, 0)), dtype=np.float64)
    rk = np.array(pose.get("right_knee", (0, 0)), dtype=np.float64)
    la = np.array(pose.get("left_ankle", (0, 0)), dtype=np.float64)
    ra = np.array(pose.get("right_ankle", (0, 0)), dtype=np.float64)

    hip_w = float(np.linalg.norm(lh - rh))

    # Leg length: hip to ankle
    left_leg_len = float(np.linalg.norm(lh - la)) if la[1] > 0 else float(np.linalg.norm(lh - lk))
    right_leg_len = float(np.linalg.norm(rh - ra)) if ra[1] > 0 else float(np.linalg.norm(rh - rk))
    leg_len = np.mean([left_leg_len, right_leg_len]) if left_leg_len > 0 and right_leg_len > 0 else max(left_leg_len, right_leg_len)

    # Knee width (optional)
    knee_w = float(np.linalg.norm(lk - rk)) if lk[1] > 0 and rk[1] > 0 else hip_w * 0.95

    return {
        "hip_width": hip_w,
        "leg_length": leg_len,
        "knee_width": knee_w,
    }


def build_skeleton_erase_mask(
    pose: dict[str, tuple[int, int]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Build a full-torso erase mask from skeleton keypoints.

    Covers the entire area from above shoulders to below hips,
    wide enough to include sleeves/armholes.  This ensures the
    old garment is completely removed regardless of its shape
    (sports bra, tank top, crop top, etc.).
    """
    h, w = image_shape
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    lh = np.array(pose["left_hip"], dtype=np.float64)
    rh = np.array(pose["right_hip"], dtype=np.float64)

    sw = float(np.linalg.norm(ls - rs))
    hw = float(np.linalg.norm(lh - rh))
    torso_h = float(np.mean([abs(lh[1] - ls[1]), abs(rh[1] - rs[1])]))

    # Conservative coverage: small extension above shoulders (just for collar),
    # moderate extension below hips.
    # Previous bug: 0.15 * torso_h above shoulders was too much → white V at neckline
    top_y = int(min(ls[1], rs[1]) - 0.05 * torso_h)   # was 0.15 → too aggressive
    bot_y = int(max(lh[1], rh[1]) + 0.08 * torso_h)
    cx = int((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
    half_w = int(max(sw, hw) * 0.68)  # was 0.72 → slightly tighter to avoid over-erase

    top_y = max(0, top_y)
    bot_y = min(h - 1, bot_y)
    lx = max(0, cx - half_w)
    rx = min(w - 1, cx + half_w)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (lx, top_y), (rx, bot_y), 255, thickness=-1)

    # Add shoulder circles for sleeve coverage (smaller to avoid over-erase)
    sleeve_r = max(8, int(0.20 * sw))   # was 0.30 → too big, erased too much skin
    cv2.circle(mask, (int(ls[0]), int(ls[1])), sleeve_r, 255, thickness=-1)
    cv2.circle(mask, (int(rs[0]), int(rs[1])), sleeve_r, 255, thickness=-1)

    # If elbows are available, draw arm corridors for long sleeves
    for elbow_key, shoulder_pt in [("left_elbow", ls), ("right_elbow", rs)]:
        if elbow_key in pose:
            ep = np.array(pose[elbow_key], dtype=np.float64)
            arm_r = max(8, int(0.12 * sw))
            cv2.line(mask, tuple(shoulder_pt.astype(int)), tuple(ep.astype(int)),
                     255, thickness=arm_r * 2)

    # Extend corridor from elbow to wrist to cover full long-sleeve area.
    # Radius intentionally narrower (0.08 vs 0.12) since the sleeve tapers
    # toward the cuff compared to the upper arm.
    for wrist_key, elbow_key in [("left_wrist", "left_elbow"), ("right_wrist", "right_elbow")]:
        if wrist_key in pose and elbow_key in pose:
            ep = np.array(pose[elbow_key], dtype=np.float64)
            wp = np.array(pose[wrist_key], dtype=np.float64)
            arm_r = max(6, int(0.08 * sw))   # narrower than upper-arm corridor
            cv2.line(mask, tuple(ep.astype(int)), tuple(wp.astype(int)),
                     255, thickness=arm_r * 2)

    # Smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    return mask


def prefit_scale_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    measurements: dict[str, float],
    preserve_ratio: float = 0.88,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-scale cloth to match person's body — garment-preserve approach.

    Two-axis scaling:
    1. WIDTH: scale garment TORSO width to person shoulder * loose_factor
       Uses CORE torso width (not including sleeves) for accurate ratio.
    2. HEIGHT: ensure garment covers full torso height (collar→hip+margin)

    Garment type classification (tight/normal/loose) adjusts the width factor
    so loose garments aren't squeezed onto slim bodies.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 50:
        return cloth_rgb, cloth_mask

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    cloth_h = max(1, y2 - y1)
    cloth_w = max(1, x2 - x1)
    if cloth_h < 10:
        return cloth_rgb, cloth_mask

    h_mask = cloth_mask.shape[0]

    def _row_width(rel_y: float) -> float:
        row = max(0, min(h_mask - 1, y1 + int(cloth_h * rel_y)))
        nz = np.where(cloth_mask[row] > 0)[0]
        return float(nz.max() - nz.min()) if len(nz) > 4 else 0.0

    # ── Measure CORE torso width (excluding sleeves) ──
    # Use minimum width in the 40%-65% band — this is the true torso
    # body. At shoulder level (12%), sleeves inflate the width → wrong ratio.
    core_widths = []
    for frac in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        w = _row_width(frac)
        if w > 0:
            core_widths.append(w)

    # Also get shoulder-level width for comparison
    w_shoulder = _row_width(0.12)

    if not core_widths or w_shoulder < 10:
        return cloth_rgb, cloth_mask

    cloth_torso_w = min(core_widths)  # true torso width, no sleeves

    # If shoulder is much wider than core (sleeves present), use CORE for ratio.
    # Otherwise use shoulder width (sleeveless/tank top).
    if w_shoulder > cloth_torso_w * 1.15:
        # Sleeves present — use core torso for accurate width matching
        cloth_ref_w = cloth_torso_w
    else:
        # No significant sleeves — use shoulder width directly
        cloth_ref_w = w_shoulder

    if cloth_ref_w < 10:
        return cloth_rgb, cloth_mask

    # ── Measure cloth width at mid-torso (~55%) ──
    cloth_mid_w = _row_width(0.55) if _row_width(0.55) > 0 else cloth_ref_w

    # ── Classify garment type ──
    aspect_ratio = cloth_w / max(1, cloth_h)
    shoulder_to_mid = w_shoulder / max(1, cloth_mid_w)

    if aspect_ratio > 1.2 or shoulder_to_mid > 1.25:
        loose_factor = 1.25  # Loose/oversized garment
    elif aspect_ratio > 0.85:
        loose_factor = 1.15  # Normal fit
    else:
        loose_factor = 1.08  # Slim/fitted garment

    person_sw = measurements["shoulder_width"]
    person_torso_h = measurements["torso_height"]

    # ── WIDTH scale: person_shoulder * loose_factor / cloth_torso_width ──
    # Using core torso width prevents sleeves from deflating the scale ratio.
    # Add 5% oversize to prevent garment looking too small on body.
    sx = (person_sw * loose_factor * 1.05) / cloth_ref_w

    # ── HEIGHT scale: ensure garment covers full torso + 10% margin ──
    target_h = person_torso_h * 1.10
    sy = target_h / cloth_h

    # PRIMARY: use width-based scale (preserves garment proportions).
    # Only apply separate height scale if garment would be too SHORT with uniform sx.
    sx_final = sx
    sy_final = sx  # uniform scale by default (preserves aspect ratio)

    # If uniform scale leaves garment too short for torso, stretch height slightly
    if sy > sx * 1.05:
        # Garment is shorter than torso at this width scale — add height
        # But cap the stretch to avoid significant distortion (max 15% non-uniform)
        sy_final = min(sy, sx * 1.15)

    # Clamp: don't shrink below 85% or grow beyond 2.5x
    sx_final = float(np.clip(sx_final, 0.85, 2.5))
    sy_final = float(np.clip(sy_final, 0.85, 2.5))

    h, w = cloth_rgb.shape[:2]
    new_w, new_h = max(10, int(w * sx_final)), max(10, int(h * sy_final))

    scaled_rgb = cv2.resize(cloth_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(cloth_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return scaled_rgb, scaled_mask


# ── U2Net cloth segmentation ───────────────────────────────────────────

def segment_cloth_u2net(cloth_rgb: np.ndarray) -> np.ndarray:
    """Use rembg (U2Net) for precise cloth foreground segmentation.

    Falls back to threshold+GrabCut if rembg is not installed.
    """
    _ensure_model_cache_env()
    try:
        from rembg import remove
        from PIL import Image
        import io

        pil_img = Image.fromarray(cloth_rgb)
        # rembg returns RGBA with accurate alpha channel
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        result = remove(buf.getvalue())
        rgba = Image.open(io.BytesIO(result)).convert("RGBA")
        alpha = np.array(rgba)[:, :, 3]
        # Clean binary mask
        mask = (alpha > 128).astype(np.uint8) * 255
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        # Fill internal holes: find contours and fill so sleeve interior
        # is not lost due to U2Net under-segmenting thin arm areas.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask
    except BaseException:
        # Catch SystemExit from rembg when onnxruntime is missing
        return build_cloth_mask(cloth_rgb)


# ── Pose landmark smoothing ────────────────────────────────────────────

def _clean_garment_mask(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Normalize garment foreground mask without over-expanding the silhouette."""
    h, w = image_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    mask = np.nan_to_num(mask.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    mask = (np.clip(mask, 0, 255) > 127).astype(np.uint8) * 255

    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

    inv = cv2.bitwise_not(mask)
    flood = inv.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    holes = flood > 0
    if holes.any():
        num, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
        max_hole_area = max(80, int(h * w * 0.010))
        for idx in range(1, num):
            if int(stats[idx, cv2.CC_STAT_AREA]) <= max_hole_area:
                mask[labels == idx] = 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas):
            largest = int(areas.max())
            keep = np.zeros_like(mask)
            min_area = max(64, int(largest * 0.015))
            for idx in range(1, num):
                if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
                    keep[labels == idx] = 255
            mask = keep

    return ((mask > 127).astype(np.uint8)) * 255


@lru_cache(maxsize=1)
def _get_rmbg2_model():
    """Load BRIA RMBG-2.0 once per process."""
    _ensure_model_cache_env()
    import torch
    from transformers import AutoModelForImageSegmentation

    requested = os.getenv("VTON_RMBG2_DEVICE", "").strip().lower()
    if requested in {"cpu", "cuda"}:
        device = requested if requested == "cpu" or torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = os.getenv("VTON_RMBG2_MODEL", "briaai/RMBG-2.0").strip() or "briaai/RMBG-2.0"
    try:
        cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
        model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            cache_dir=cache_dir,
        )
    except TypeError:
        model = AutoModelForImageSegmentation.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE"),
        )
    model.to(device)
    model.eval()
    return model, device


def segment_cloth_rmbg2(cloth_rgb: np.ndarray) -> np.ndarray:
    """BRIA RMBG-2.0 foreground mask for garment product images."""
    import torch
    from PIL import Image

    model, device = _get_rmbg2_model()

    image = Image.fromarray(cloth_rgb).convert("RGB")
    orig_w, orig_h = image.size
    infer_size = int(os.getenv("VTON_RMBG2_SIZE", "1024"))
    infer_size = max(512, min(infer_size, 1536))
    resized = image.resize((infer_size, infer_size), Image.BILINEAR)

    arr = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean[None, None, :]) / std[None, None, :]
    input_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        preds = model(input_tensor)[-1].sigmoid().detach().cpu().float()

    pred = preds[0].squeeze().numpy()
    pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    threshold = float(os.getenv("VTON_RMBG2_THRESH", "0.50"))
    mask = (pred > threshold).astype(np.uint8) * 255
    return _clean_garment_mask(mask, cloth_rgb.shape[:2])


def _mask_bbox(mask: np.ndarray, pad_ratio: float = 0.04) -> np.ndarray | None:
    ys, xs = np.where(mask > 20)
    if len(xs) < 50:
        return None
    h, w = mask.shape[:2]
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(max(x2 - x1 + 1, y2 - y1 + 1) * pad_ratio)
    return np.array([
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(w - 1, x2 + pad),
        min(h - 1, y2 + pad),
    ], dtype=np.float32)


@lru_cache(maxsize=1)
def _get_sam2_predictor():
    """Load SAM2 predictor if package/checkpoint are available."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = os.getenv("VTON_SAM2_CHECKPOINT", "").strip()
    model_cfg = os.getenv("VTON_SAM2_CONFIG", "sam2_hiera_l.yaml").strip()
    if not checkpoint:
        raise RuntimeError("VTON_SAM2_CHECKPOINT is not set")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(model_cfg, checkpoint, device=device)
    return SAM2ImagePredictor(model)


def segment_cloth_sam2_refine(cloth_rgb: np.ndarray, rough_mask: np.ndarray) -> np.ndarray:
    """Optional SAM2 box-refinement for an existing garment mask."""
    predictor = _get_sam2_predictor()
    box = _mask_bbox(rough_mask)
    if box is None:
        return rough_mask
    predictor.set_image(cloth_rgb)
    masks, scores, _logits = predictor.predict(
        box=box,
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        return rough_mask
    best_idx = int(np.argmax(scores)) if scores is not None and len(scores) else 0
    sam_mask = masks[best_idx].astype(np.uint8) * 255
    return _clean_garment_mask(sam_mask, cloth_rgb.shape[:2])


def segment_cloth_ensemble(cloth_rgb: np.ndarray) -> np.ndarray:
    """Garment mask ensemble: U2Net + BRIA RMBG-2.0 + optional SAM2 refine."""
    masks: list[np.ndarray] = []
    weights: list[float] = []

    try:
        mask_u2net = segment_cloth_u2net(cloth_rgb)
        if mask_u2net is not None and int(mask_u2net.sum()) > 255 * 100:
            masks.append(mask_u2net.astype(np.float32))
            weights.append(0.45)
    except BaseException as exc:
        print(f"[garment_mask] U2Net unavailable: {exc}")

    use_rmbg2 = os.getenv("VTON_USE_RMBG2", "1").strip().lower() not in {"0", "false", "no"}
    if use_rmbg2:
        try:
            mask_rmbg = segment_cloth_rmbg2(cloth_rgb)
            if mask_rmbg is not None and int(mask_rmbg.sum()) > 255 * 100:
                masks.append(mask_rmbg.astype(np.float32))
                weights.append(0.55)
        except BaseException as exc:
            print(f"[garment_mask] RMBG-2.0 unavailable: {exc}")

    if not masks:
        return build_cloth_mask(cloth_rgb)

    weights_np = np.array(weights, dtype=np.float32)
    weights_np = weights_np / max(1e-6, float(weights_np.sum()))
    merged = np.zeros(cloth_rgb.shape[:2], dtype=np.float32)
    for mask, weight in zip(masks, weights_np):
        if mask.shape[:2] != merged.shape:
            mask = cv2.resize(mask, (merged.shape[1], merged.shape[0]), interpolation=cv2.INTER_LINEAR)
        merged += mask * float(weight)

    mask = _clean_garment_mask((merged > 127).astype(np.uint8) * 255, cloth_rgb.shape[:2])

    use_sam2 = os.getenv("VTON_USE_SAM2_GARMENT_MASK", "0").strip().lower() in {"1", "true", "yes"}
    if use_sam2:
        try:
            mask_sam2 = segment_cloth_sam2_refine(cloth_rgb, mask)
            if mask_sam2 is not None and int(mask_sam2.sum()) > 255 * 100:
                merged = 0.80 * mask.astype(np.float32) + 0.20 * mask_sam2.astype(np.float32)
                mask = _clean_garment_mask((merged > 127).astype(np.uint8) * 255, cloth_rgb.shape[:2])
        except BaseException as exc:
            print(f"[garment_mask] SAM2 refine unavailable: {exc}")

    return mask


def smooth_pose_landmarks(
    pose: dict[str, tuple[int, int]],
    image_shape: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    """Smooth pose landmarks to reduce TPS warping artefacts from noise.

    Applies bilateral symmetry correction and midpoint stabilisation
    so that slight pose asymmetry doesn't cause garment distortion.
    """
    h, w = image_shape
    result = dict(pose)

    # Symmetry pairs: average left/right Y-coordinates to reduce tilt
    sym_pairs = [
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_elbow", "right_elbow"),
    ]
    for lk, rk in sym_pairs:
        if lk in result and rk in result:
            lx, ly = result[lk]
            rx, ry = result[rk]
            avg_y = int((ly + ry) / 2)
            result[lk] = (lx, avg_y)
            result[rk] = (rx, avg_y)

    # Clamp to image bounds
    for key in result:
        x, y = result[key]
        result[key] = (max(0, min(w - 1, x)), max(0, min(h - 1, y)))

    return result


# ── Lighting / color transfer ──────────────────────────────────────────

def match_cloth_lighting(
    warped_cloth: np.ndarray,
    person_rgb: np.ndarray,
    warped_mask: np.ndarray,
) -> np.ndarray:
    """Transfer person's lighting/color stats to cloth (Reinhard color transfer).

    Matches mean and std of L channel in LAB space so the cloth looks like
    it was photographed under the same lighting as the person.
    """
    mask_bool = warped_mask > 30
    if mask_bool.sum() < 500:
        return warped_cloth

    # Get the person's torso region (behind where cloth will go)
    cloth_lab = cv2.cvtColor(warped_cloth, cv2.COLOR_RGB2LAB).astype(np.float32)
    person_lab = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    result = cloth_lab.copy()
    eps = 1e-6

    # Only transfer L (lightness) channel to preserve cloth color/saturation
    src_vals = cloth_lab[:, :, 0][mask_bool]
    tgt_vals = person_lab[:, :, 0][mask_bool]

    src_mean, src_std = float(src_vals.mean()), float(src_vals.std() + eps)
    tgt_mean, tgt_std = float(tgt_vals.mean()), float(tgt_vals.std() + eps)

    # Gentle transfer (strength 0.4) to avoid washing out cloth
    strength = 0.4
    adjusted = (src_vals - src_mean) * (tgt_std / src_std) + tgt_mean
    blended = src_vals * (1.0 - strength) + adjusted * strength
    result[:, :, 0][mask_bool] = np.clip(blended, 0, 255)

    return cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2RGB)
