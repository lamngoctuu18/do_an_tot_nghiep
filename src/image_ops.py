from __future__ import annotations

from dataclasses import dataclass
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
        "left_shoulder": _xy(PL.LEFT_SHOULDER),
        "right_shoulder": _xy(PL.RIGHT_SHOULDER),
        "left_elbow": _xy(PL.LEFT_ELBOW),
        "right_elbow": _xy(PL.RIGHT_ELBOW),
        "left_wrist": _xy(PL.LEFT_WRIST),
        "right_wrist": _xy(PL.RIGHT_WRIST),
        "left_hip": _xy(PL.LEFT_HIP),
        "right_hip": _xy(PL.RIGHT_HIP),
    }


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
        "left_shoulder": _xy(11), "right_shoulder": _xy(12),
        "left_elbow": _xy(13), "right_elbow": _xy(14),
        "left_wrist": _xy(15), "right_wrist": _xy(16),
        "left_hip": _xy(23), "right_hip": _xy(24),
    }


def _ensure_pose_task_model() -> Path:
    base_dir = Path(os.getenv("VTO_BASE_DIR", "E:/virtual_try_on_data"))
    model_dir = base_dir / "cache" / "mediapipe"
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

    # Dilate mask outward so cloth covers full silhouette (prevents waist pinch).
    # Keep dilation moderate (5×5) to avoid pulling in white background pixels.
    dilate_k = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, dilate_k, iterations=1)
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

    # Body-conforming: separate widths for shoulders and hips
    top_half_w = int(shoulder_width * fit_scale / 2)
    bot_half_w = int(hip_width * fit_scale / 2)

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
    # Pre-multiply cloth by mask to prevent white background bleeding
    mask_f = (cloth_mask.astype(np.float32) / 255.0)[..., None]
    cloth_premul = (cloth_rgb.astype(np.float32) * mask_f).clip(0, 255).astype(np.uint8)
    warped_cloth = cv2.warpPerspective(cloth_premul, transform, (width, height), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(cloth_mask, transform, (width, height), flags=cv2.INTER_LINEAR)

    return warped_cloth, warped_mask


def erase_clothing_region(
    person_rgb: np.ndarray,
    erase_mask: np.ndarray,
    parsing_skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove original clothing by inpainting the masked area.

    Only the garment pixels (upper_clothes from parsing) should be in
    erase_mask.  Skin / neck / torso must NOT be included.

    If a skin mask is provided, sample real skin texture to seed the
    inpainter with a plausible base rather than flat colour.
    """
    h, w = person_rgb.shape[:2]

    # Tight binary mask – small dilation to catch garment edge remnants
    binary = (erase_mask > 30).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Pre-fill the erase region with plausible skin.
    prefilled = person_rgb.copy()
    if parsing_skin_mask is not None:
        skin_area = parsing_skin_mask > 0
        if skin_area.sum() > 200:
            mean_skin = prefilled[skin_area].mean(axis=0).astype(np.uint8)
            erase_area = binary > 0
            prefilled[erase_area] = mean_skin

    # Navier-Stokes inpainting with tighter radius for realistic skin
    inpainted = cv2.inpaint(
        cv2.cvtColor(prefilled, cv2.COLOR_RGB2BGR),
        binary, inpaintRadius=6, flags=cv2.INPAINT_NS,
    )
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)


def poisson_blend(
    person_rgb: np.ndarray,
    warped_cloth: np.ndarray,
    warped_mask: np.ndarray,
) -> np.ndarray:
    """Poisson seamless clone — merges garment colours/gradients naturally."""
    binary = (warped_mask > 30).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Compute centre of the mask
    ys, xs = np.where(binary > 0)
    if len(xs) < 100:
        return person_rgb  # nothing to blend
    cx, cy = int(xs.mean()), int(ys.mean())

    # seamlessClone requires BGR
    try:
        result = cv2.seamlessClone(
            cv2.cvtColor(warped_cloth, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(person_rgb, cv2.COLOR_RGB2BGR),
            binary,
            (cx, cy),
            cv2.MIXED_CLONE,
        )
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    except cv2.error:
        # Fallback to alpha blend when mask is near image border
        return blend_tryon(person_rgb, warped_cloth, warped_mask, alpha=0.95)


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

    # Generous coverage: extend above shoulders and below hips
    top_y = int(min(ls[1], rs[1]) - 0.15 * torso_h)
    bot_y = int(max(lh[1], rh[1]) + 0.10 * torso_h)
    cx = int((ls[0] + rs[0] + lh[0] + rh[0]) / 4)
    half_w = int(max(sw, hw) * 0.72)  # wide enough for sleeves

    top_y = max(0, top_y)
    bot_y = min(h - 1, bot_y)
    lx = max(0, cx - half_w)
    rx = min(w - 1, cx + half_w)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (lx, top_y), (rx, bot_y), 255, thickness=-1)

    # Add shoulder circles for sleeve coverage
    sleeve_r = max(12, int(0.30 * sw))
    cv2.circle(mask, (int(ls[0]), int(ls[1])), sleeve_r, 255, thickness=-1)
    cv2.circle(mask, (int(rs[0]), int(rs[1])), sleeve_r, 255, thickness=-1)

    # If elbows are available, draw arm corridors for long sleeves
    for elbow_key, shoulder_pt in [("left_elbow", ls), ("right_elbow", rs)]:
        if elbow_key in pose:
            ep = np.array(pose[elbow_key], dtype=np.float64)
            arm_r = max(8, int(0.12 * sw))
            cv2.line(mask, tuple(shoulder_pt.astype(int)), tuple(ep.astype(int)),
                     255, thickness=arm_r * 2)

    # Smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1)
    return mask


def prefit_scale_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    measurements: dict[str, float],
    preserve_ratio: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-scale cloth with Garment Shape Constraint (CP-VTON / HR-VITON).

    Scales by max(shoulder, hip) width so garment is never narrower than
    the body.  ``preserve_ratio`` ensures the garment keeps at least 90%
    of its original width — prevents TPS from over-squeezing.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 50:
        return cloth_rgb, cloth_mask

    cloth_w = float(xs.max() - xs.min())
    cloth_h = float(ys.max() - ys.min())
    if cloth_w < 10 or cloth_h < 10:
        return cloth_rgb, cloth_mask

    person_sw = measurements["shoulder_width"]
    person_hw = measurements["hip_width"]
    body_width = max(person_sw, person_hw)

    # Garment Shape Constraint: target width is at least
    #   max(body_width * 1.12, cloth_width * preserve_ratio)
    # This prevents the garment from being squeezed smaller than its
    # natural shape while still fitting the body.
    target_w = max(body_width * 1.12, cloth_w * preserve_ratio)
    target_h = measurements["torso_height"] * 1.25

    sx = target_w / cloth_w
    sy = target_h / cloth_h
    sx = np.clip(sx, 0.6, 2.5)
    sy = np.clip(sy, 0.6, 2.5)

    h, w = cloth_rgb.shape[:2]
    new_w, new_h = max(10, int(w * sx)), max(10, int(h * sy))

    scaled_rgb = cv2.resize(cloth_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(cloth_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Dilate mask slightly to prevent tight-cropping that causes
    # TPS to squeeze chest/waist.
    dilate_k = np.ones((11, 11), np.uint8)
    scaled_mask = cv2.dilate(scaled_mask, dilate_k, iterations=1)

    return scaled_rgb, scaled_mask


# ── U2Net cloth segmentation ───────────────────────────────────────────

def segment_cloth_u2net(cloth_rgb: np.ndarray) -> np.ndarray:
    """Use rembg (U2Net) for precise cloth foreground segmentation.

    Falls back to threshold+GrabCut if rembg is not installed.
    """
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
        return mask
    except BaseException:
        # Catch SystemExit from rembg when onnxruntime is missing
        return build_cloth_mask(cloth_rgb)


# ── Pose landmark smoothing ────────────────────────────────────────────

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
