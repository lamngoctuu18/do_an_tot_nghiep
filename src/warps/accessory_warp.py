"""Accessory warp dispatcher.

Per-subtype warp strategy (rigidity-driven). All warps return:
    (warped_rgb, warped_mask)
both H×W matching the person image, uint8.

Subtype strategies:
  - shoes/boots : per-foot affine using {ankle, heel, foot_index}
  - hat         : affine using {head_top, left_ear, right_ear}
  - sunglasses  : affine using {left_eye_outer, right_eye_outer, nose_bridge}
  - belt        : piecewise affine across waistline {l_hip, hip_center, r_hip}
  - bag         : paste with shoulder/hip routing (no warp)
  - scarf       : ring-around-neck affine (TPS-ready point set)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _empty(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = shape
    return np.zeros((h, w, 3), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)


def _bbox_of_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 20)
    if len(xs) < 10:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _paste_affine(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = out_shape
    M = cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))
    warped_rgb = cv2.warpAffine(cloth_rgb, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_mask = cv2.warpAffine(cloth_mask, M, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_rgb, warped_mask


def _split_shoes_by_components(cloth_mask: np.ndarray):
    """Tách 2 chiếc giày bằng connected components.

    Trả về list [(mask_blob, centroid_x), ...] sắp xếp theo centroid_x tăng dần.
    Nếu chỉ có 1 blob (ảnh chỉ chứa 1 chiếc giày), trả về list 1 phần tử —
    caller sẽ mirror để dùng cho chân còn lại.
    """
    bin_mask = (cloth_mask > 20).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    blobs = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 200:
            continue
        m = (labels == i).astype(np.uint8) * 255
        blobs.append((m, float(centroids[i, 0]), area))
    if not blobs:
        return []
    blobs.sort(key=lambda b: -b[2])
    blobs = blobs[:2]
    blobs.sort(key=lambda b: b[1])
    return [(m, cx) for (m, cx, _) in blobs]


def _shoe_principal_axis(cloth_mask_side: np.ndarray):
    """Tìm trục chính của giày qua minAreaRect.

    Trả về (center, (long_len, short_len), angle_deg) với angle là góc của
    cạnh DÀI so với trục X (đầu mũi giày → gót).
    """
    ys, xs = np.where(cloth_mask_side > 20)
    if len(xs) < 20:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w_, h_), ang = rect
    if w_ >= h_:
        long_len, short_len = w_, h_
        long_angle = ang
    else:
        long_len, short_len = h_, w_
        long_angle = ang + 90.0
    return (cx, cy), (long_len, short_len), long_angle


def _warp_one_foot(
    cloth_rgb: np.ndarray,
    cloth_mask_side: np.ndarray,
    ankle: Tuple[float, float],
    heel: Optional[Tuple[float, float]],
    toe: Optional[Tuple[float, float]],
    out_shape: Tuple[int, int],
    *,
    extend_up: float = 0.0,
    mirror_src: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    # Mirror nguồn TRƯỚC khi phân tích trục — để mọi tính toán đều nhất quán.
    if mirror_src:
        cloth_rgb = cv2.flip(cloth_rgb, 1)
        cloth_mask_side = cv2.flip(cloth_mask_side, 1)

    axis = _shoe_principal_axis(cloth_mask_side)
    if axis is None:
        return _empty(out_shape)
    (scx, scy), (long_len, short_len), long_angle_deg = axis

    # Convention: u (trục dài) hướng +x (mũi giày bên phải trong ảnh sản phẩm),
    # v (trục ngắn) hướng -y (đỉnh giày phía trên ảnh).
    rad = np.deg2rad(long_angle_deg)
    ux, uy = np.cos(rad), np.sin(rad)
    if ux < 0:
        ux, uy = -ux, -uy
    vx, vy = -uy, ux
    if vy > 0:
        vx, vy = -vx, -vy

    half_long = long_len * 0.5
    half_short = short_len * 0.5
    toe_src = (scx + ux * half_long, scy + uy * half_long)
    heel_src = (scx - ux * half_long, scy - uy * half_long)
    top_src = (scx + vx * half_short, scy + vy * half_short)

    if heel is None and toe is None:
        return _empty(out_shape)
    if heel is None:
        heel = (ankle[0] - 0.5 * long_len, ankle[1] + 0.15 * long_len)
    if toe is None:
        toe = (ankle[0] + 0.7 * long_len, ankle[1] + 0.15 * long_len)

    foot_len = float(np.hypot(toe[0] - heel[0], toe[1] - heel[1])) or long_len
    foot_h = max(foot_len * (0.55 + extend_up * 1.4), 18.0)

    # Vector vuông góc với (heel→toe), hướng LÊN trên ảnh (y giảm).
    fdx = (toe[0] - heel[0]) / max(foot_len, 1e-3)
    fdy = (toe[1] - heel[1]) / max(foot_len, 1e-3)
    nfx, nfy = -fdy, fdx
    if nfy > 0:
        nfx, nfy = -nfx, -nfy

    mid_x = (heel[0] + toe[0]) * 0.5
    mid_y = (heel[1] + toe[1]) * 0.5
    top_dst = (
        mid_x + nfx * foot_h * (0.5 + extend_up),
        mid_y + nfy * foot_h * (0.5 + extend_up),
    )

    src_tri = np.array([heel_src, toe_src, top_src], dtype=np.float32)
    dst_tri = np.array([list(heel), list(toe), list(top_dst)], dtype=np.float32)

    return _paste_affine(cloth_rgb, cloth_mask_side, src_tri, dst_tri, out_shape)


def warp_shoes(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    anchors: Dict[str, Optional[Tuple[float, float]]],
    out_shape: Tuple[int, int],
    *,
    boot_height: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = out_shape
    blobs = _split_shoes_by_components(cloth_mask)
    out_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    out_mask = np.zeros((h, w), dtype=np.uint8)
    if not blobs:
        return out_rgb, out_mask

    # Gán blob → side theo centroid_x của chân (ankle x trên ảnh người).
    sides = []
    for s in ("left", "right"):
        a = anchors.get(f"{s}_ankle")
        if a is not None:
            sides.append((s, a[0]))
    if not sides:
        return out_rgb, out_mask
    sides.sort(key=lambda x: x[1])

    if len(blobs) == 1:
        # Một chiếc giày → dùng cho cả 2 chân (mirror cho bên còn lại).
        blob_mask, _ = blobs[0]
        assignments = [(sides[0][0], blob_mask, False)]
        if len(sides) > 1:
            assignments.append((sides[1][0], blob_mask, True))
    else:
        assignments = []
        # blobs sorted by centroid_x ascending; sides also by ankle x ascending.
        for (side, _), (m, _) in zip(sides, blobs):
            assignments.append((side, m, False))

    for side, m_side, mirror in assignments:
        ankle = anchors.get(f"{side}_ankle")
        heel = anchors.get(f"{side}_heel")
        toe = anchors.get(f"{side}_foot_index")
        if ankle is None:
            continue
        wr, wm = _warp_one_foot(
            cloth_rgb, m_side, ankle, heel, toe, out_shape,
            extend_up=boot_height, mirror_src=mirror,
        )
        out_mask = np.maximum(out_mask, wm)
        alpha = (wm > 20)[..., None]
        out_rgb = np.where(alpha, wr, out_rgb)

    return out_rgb, out_mask


def warp_hat(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    anchors: Dict[str, Optional[Tuple[float, float]]],
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    bb = _bbox_of_mask(cloth_mask)
    if bb is None:
        return _empty(out_shape)
    le = anchors.get("left_ear"); re = anchors.get("right_ear")
    top = anchors.get("head_top"); ctr = anchors.get("head_center")
    if le is None or re is None or top is None or ctr is None:
        return _empty(out_shape)

    x0, y0, x1, y1 = bb
    ear_gap = float(np.hypot(le[0] - re[0], le[1] - re[1]))
    head_w = max(40.0, ear_gap * 1.25)
    head_h = max(40.0, abs(ctr[1] - top[1]) * 2.05)

    src_tri = np.array([[x0, y0], [x1, y0], [(x0 + x1) * 0.5, y1]], dtype=np.float32)
    dst_tri = np.array([
        [ctr[0] - head_w * 0.5, top[1]],
        [ctr[0] + head_w * 0.5, top[1]],
        [ctr[0], top[1] + head_h * 1.05],
    ], dtype=np.float32)
    return _paste_affine(cloth_rgb, cloth_mask, src_tri, dst_tri, out_shape)


def warp_sunglasses(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    anchors: Dict[str, Optional[Tuple[float, float]]],
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    bb = _bbox_of_mask(cloth_mask)
    if bb is None:
        return _empty(out_shape)
    le = anchors.get("left_eye_outer") or anchors.get("left_eye")
    re = anchors.get("right_eye_outer") or anchors.get("right_eye")
    nb = anchors.get("nose_bridge") or anchors.get("nose")
    if le is None or re is None or nb is None:
        return _empty(out_shape)

    x0, y0, x1, y1 = bb
    eye_gap = float(np.hypot(le[0] - re[0], le[1] - re[1]))
    frame_w = max(40.0, eye_gap * 1.55)
    frame_h = max(20.0, frame_w * 0.36)
    cx = (le[0] + re[0]) * 0.5
    cy = (le[1] + re[1]) * 0.5

    src_tri = np.array([[x0, y0], [x1, y0], [(x0 + x1) * 0.5, y1]], dtype=np.float32)
    dst_tri = np.array([
        [cx - frame_w * 0.5, cy - frame_h * 0.5],
        [cx + frame_w * 0.5, cy - frame_h * 0.5],
        [cx, cy + frame_h * 0.55],
    ], dtype=np.float32)
    return _paste_affine(cloth_rgb, cloth_mask, src_tri, dst_tri, out_shape)


def warp_belt(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    anchors: Dict[str, Optional[Tuple[float, float]]],
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    bb = _bbox_of_mask(cloth_mask)
    if bb is None:
        return _empty(out_shape)
    lh = anchors.get("left_hip"); rh = anchors.get("right_hip")
    if lh is None or rh is None:
        return _empty(out_shape)

    x0, y0, x1, y1 = bb
    hip_w = max(60.0, float(np.hypot(lh[0] - rh[0], lh[1] - rh[1])))
    belt_w = hip_w * 1.18
    belt_h = max(10.0, belt_w * 0.075)
    cx = (lh[0] + rh[0]) * 0.5
    cy = (lh[1] + rh[1]) * 0.5 - belt_h * 0.20  # belt sits slightly above hip line

    src_tri = np.array([[x0, y0], [x1, y0], [(x0 + x1) * 0.5, y1]], dtype=np.float32)
    dst_tri = np.array([
        [cx - belt_w * 0.5, cy - belt_h * 0.5],
        [cx + belt_w * 0.5, cy - belt_h * 0.5],
        [cx, cy + belt_h * 0.55],
    ], dtype=np.float32)
    return _paste_affine(cloth_rgb, cloth_mask, src_tri, dst_tri, out_shape)


def warp_bag(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    anchors: Dict[str, Optional[Tuple[float, float]]],
    out_shape: Tuple[int, int],
    *,
    bag_style: str = "shoulder_bag",
) -> Tuple[np.ndarray, np.ndarray]:
    bb = _bbox_of_mask(cloth_mask)
    if bb is None:
        return _empty(out_shape)
    shoulder = anchors.get("carry_shoulder")
    opp_hip = anchors.get("opposite_hip")
    if shoulder is None:
        return _empty(out_shape)

    x0, y0, x1, y1 = bb
    src_w = max(2, x1 - x0)
    src_h = max(2, y1 - y0)
    aspect = src_w / float(src_h)

    if bag_style == "crossbody" and opp_hip is not None:
        # bag sits at hip on the opposite side
        cx = opp_hip[0]
        cy = (shoulder[1] + opp_hip[1]) * 0.55
    else:
        # shoulder bag hangs ~40% from shoulder to hip
        ref = opp_hip if opp_hip is not None else (shoulder[0], shoulder[1] + 220.0)
        cx = (shoulder[0] + ref[0]) * 0.5
        cy = shoulder[1] + abs(ref[1] - shoulder[1]) * 0.45

    target_h = max(80.0, abs((opp_hip[1] if opp_hip else shoulder[1] + 220.0) - shoulder[1]) * 0.50)
    target_w = target_h * aspect

    src_tri = np.array([[x0, y0], [x1, y0], [(x0 + x1) * 0.5, y1]], dtype=np.float32)
    dst_tri = np.array([
        [cx - target_w * 0.5, cy - target_h * 0.5],
        [cx + target_w * 0.5, cy - target_h * 0.5],
        [cx, cy + target_h * 0.55],
    ], dtype=np.float32)
    return _paste_affine(cloth_rgb, cloth_mask, src_tri, dst_tri, out_shape)


def warp_scarf(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    anchors: Dict[str, Optional[Tuple[float, float]]],
    out_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    bb = _bbox_of_mask(cloth_mask)
    if bb is None:
        return _empty(out_shape)
    ls = anchors.get("left_shoulder"); rs = anchors.get("right_shoulder")
    neck = anchors.get("neck_center")
    if ls is None or rs is None or neck is None:
        return _empty(out_shape)

    x0, y0, x1, y1 = bb
    shoulder_w = max(60.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    scarf_w = shoulder_w * 1.45
    scarf_h = max(70.0, scarf_w * 0.55)
    cx = neck[0]
    cy = neck[1] + scarf_h * 0.10

    src_tri = np.array([[x0, y0], [x1, y0], [(x0 + x1) * 0.5, y1]], dtype=np.float32)
    dst_tri = np.array([
        [cx - scarf_w * 0.5, cy - scarf_h * 0.40],
        [cx + scarf_w * 0.5, cy - scarf_h * 0.40],
        [cx, cy + scarf_h * 0.60],
    ], dtype=np.float32)
    return _paste_affine(cloth_rgb, cloth_mask, src_tri, dst_tri, out_shape)


def warp_accessory(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    subtype: str,
    out_shape: Tuple[int, int],
    *,
    pose_anchors: Optional[Dict] = None,
    face_anchors: Optional[Dict] = None,
    bag_style: str = "shoulder_bag",
) -> Tuple[np.ndarray, np.ndarray]:
    """Subtype dispatcher."""
    sub = (subtype or "").lower()
    anchors = dict(pose_anchors or {})
    if face_anchors:
        anchors.update(face_anchors)

    if sub == "shoes":
        return warp_shoes(cloth_rgb, cloth_mask, anchors, out_shape, boot_height=0.0)
    if sub == "boots":
        return warp_shoes(cloth_rgb, cloth_mask, anchors, out_shape, boot_height=0.85)
    if sub == "hat":
        return warp_hat(cloth_rgb, cloth_mask, anchors, out_shape)
    if sub == "sunglasses":
        return warp_sunglasses(cloth_rgb, cloth_mask, anchors, out_shape)
    if sub == "belt":
        return warp_belt(cloth_rgb, cloth_mask, anchors, out_shape)
    if sub == "bag":
        return warp_bag(cloth_rgb, cloth_mask, anchors, out_shape, bag_style=bag_style)
    if sub == "scarf":
        return warp_scarf(cloth_rgb, cloth_mask, anchors, out_shape)
    return _empty(out_shape)


__all__ = [
    "warp_accessory",
    "warp_shoes",
    "warp_hat",
    "warp_sunglasses",
    "warp_belt",
    "warp_bag",
    "warp_scarf",
]
