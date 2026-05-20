"""Thin-Plate-Spline (TPS) cloth warping — **Graduated Body-Follow** approach.

Upper garment (shoulders) preserves garment silhouette (65% garment, 35% body).
Lower garment (mid/hem) follows body curvature (40% garment, 60% body).

This graduated blend gives natural draping: the shoulder seam stays near the
garment's original position while the waist and hem conform to body shape.

Key design: 24 landmarks (collar, shoulder_top L/R,
shoulder L/R, chest_center, chest L/R, side_torso L/R,
waist L/R, mid-torso L/R, hem L/R, armpit L/R,
under_bust L/R, sleeve_tip L/R, sleeve_outer L/R).
chest_center provides HORIZONTAL CENTERING constraint (prevents logo drift).
waist points fill the gap between side (35%) and mid (55%) for natural taper.
sleeve_tip/sleeve_outer anchor the sleeve hem for short-sleeve garments.
Destination half-widths are computed as a weighted blend of GARMENT
silhouette widths (primary) and body skeleton widths (secondary).
"""
from __future__ import annotations

import cv2
import numpy as np


# ── Garment type classification ─────────────────────────────────────────

def detect_garment_category(cloth_mask: np.ndarray) -> str:
    """Classify garment into category: 'top', 'pants', or 'dress'.

    v19.3:
    - Stronger pants-first routing.
    - Fix shorts bị nhận thành top.
    - Fix wide-leg jeans / trousers bị nhận thành dress.
    - Ưu tiên tín hiệu quần trước khi xét dress.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 100:
        return "top"

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    aspect_ratio = h / max(1, w)

    def _row_pixels(frac: float) -> np.ndarray:
        y = max(0, min(cloth_mask.shape[0] - 1, y1 + int(h * frac)))
        return cloth_mask[y, x1:x2 + 1]

    def _row_width(frac: float) -> float:
        row = _row_pixels(frac)
        nz = np.where(row > 0)[0]
        return float(nz.max() - nz.min()) if len(nz) > 4 else 0.0

    def _count_segments(frac: float) -> tuple[int, list[int]]:
        row = _row_pixels(frac)
        in_seg = False
        n_seg = 0
        gap_widths: list[int] = []
        cur_gap = 0
        seen_fg = False
        for px in row:
            if px > 0:
                seen_fg = True
                if not in_seg:
                    in_seg = True
                    n_seg += 1
                    if cur_gap > 0:
                        gap_widths.append(cur_gap)
                        cur_gap = 0
            else:
                if in_seg:
                    in_seg = False
                if seen_fg:
                    cur_gap += 1
        return n_seg, gap_widths

    def _center_gap_ratio(frac_a: float, frac_b: float, band_ratio: float = 0.12) -> float:
        """Fraction filled inside a vertical band at the center of the mask.
        Pants have crotch gap → low ratio. Dresses are solid → high ratio."""
        ya = max(0, min(cloth_mask.shape[0] - 1, y1 + int(h * frac_a)))
        yb = max(0, min(cloth_mask.shape[0] - 1, y1 + int(h * frac_b)))
        if yb <= ya:
            yb = min(cloth_mask.shape[0] - 1, ya + 1)
        cx = int((x1 + x2) * 0.5)
        half = max(3, int(w * band_ratio * 0.5))
        xa = max(0, cx - half)
        xb = min(cloth_mask.shape[1] - 1, cx + half)
        band = cloth_mask[ya:yb + 1, xa:xb + 1]
        if band.size == 0:
            return 1.0
        return float((band > 0).mean())

    w_top   = _row_width(0.08)
    w_waist = _row_width(0.22)
    w_hip   = _row_width(0.38)
    w_mid   = _row_width(0.55)
    w_low   = _row_width(0.72)
    w_hem   = _row_width(0.90)

    bbox_w = max(1.0, float(w))
    hem_coverage = w_hem / bbox_w
    low_coverage = w_low / bbox_w
    mid_coverage = w_mid / bbox_w

    # 1) Pants-first detection: long pants / wide-leg pants
    if aspect_ratio > 1.05:
        two_leg_votes = 0
        meaningful_gap_votes = 0
        for frac in (0.52, 0.62, 0.72, 0.82, 0.92):
            n_seg, gaps = _count_segments(frac)
            if n_seg >= 2:
                two_leg_votes += 1
                # v19.22: lowered gap threshold so a shallow crotch notch on
                # shorts hanging from a hanger (which inflates aspect_ratio)
                # still counts as a leg-split signal.
                if gaps and max(gaps) >= max(3, int(w * 0.022)):
                    meaningful_gap_votes += 1
        center_gap_lower = _center_gap_ratio(0.52, 0.92, band_ratio=0.13)
        if two_leg_votes >= 1 and meaningful_gap_votes >= 1:
            return "pants"
        if two_leg_votes >= 2:
            return "pants"
        if center_gap_lower < 0.42 and low_coverage >= 0.45:
            return "pants"
        if hem_coverage > 0.48 and center_gap_lower < 0.55 and aspect_ratio > 1.20:
            return "pants"

    # 2) Shorts detection
    if 0.35 <= aspect_ratio <= 1.35:
        two_leg_votes = 0
        meaningful_gap_votes = 0
        # v19.22: also scan very close to the hem where the V-notch is most
        # visible on compact boxer-style shorts; lower the min-gap threshold
        # so a small crotch notch still counts.
        for frac in (0.55, 0.68, 0.80, 0.88, 0.94, 0.97):
            n_seg, gaps = _count_segments(frac)
            if n_seg >= 2:
                two_leg_votes += 1
                if gaps and max(gaps) >= max(3, int(w * 0.022)):
                    meaningful_gap_votes += 1
        center_gap_short = _center_gap_ratio(0.48, 0.96, band_ratio=0.16)
        if two_leg_votes >= 1 and meaningful_gap_votes >= 1:
            return "pants"
        if center_gap_short < 0.45 and w_hem > max(8.0, w_waist * 0.55):
            return "pants"
        if aspect_ratio < 0.85 and center_gap_short < 0.60 and hem_coverage > 0.42:
            return "pants"
        # v19.22b: compact-shorts fallback — rectangular cloth in shorts aspect
        # range with stable mid/low/hem widths is almost always shorts even if
        # the leg gap is too small for the segment scanner to detect.
        widths = [w_waist, w_hip, w_mid, w_low, w_hem]
        widths = [x for x in widths if x > 0]
        if len(widths) >= 4:
            wmax = max(widths)
            wmin = min(widths)
            if (
                wmin / wmax >= 0.72
                and hem_coverage >= 0.50
                and mid_coverage >= 0.62
                and aspect_ratio <= 1.30
            ):
                return "pants"

    # 3) Dress detection: only if no pants signal
    # v19.41: tightened to stop tees being mis-classified as dresses.
    # A t-shirt has sleeves at chest level → w_waist (≈22%) ≥ w_hem because
    # the bbox width is set by the sleeves, not the body. A true dress is
    # both longer (AR ≥ 1.45) and has a hem at least as wide as the chest
    # (w_hem / w_waist ≥ 0.92), reflecting a skirt that doesn't narrow.
    if aspect_ratio > 1.45:
        center_fill_lower = _center_gap_ratio(0.55, 0.95, band_ratio=0.16)
        hem_vs_waist = w_hem / max(1.0, w_waist)
        if (
            hem_coverage > 0.35
            and low_coverage > 0.35
            and center_fill_lower > 0.58
            and hem_vs_waist >= 0.92
        ):
            return "dress"
        nonzero_widths = [x for x in (w_waist, w_hip, w_low, w_hem) if x > 0]
        if nonzero_widths:
            width_variation = max(w_waist, w_hip, w_low, w_hem) / max(1.0, min(nonzero_widths))
            if (
                aspect_ratio > 1.60
                and width_variation < 1.45
                and center_fill_lower > 0.64
                and hem_coverage > 0.28
                and hem_vs_waist >= 0.95
            ):
                return "dress"
    # v19.41: mini-dress catch — short AR (1.20–1.45) but the hem flares
    # clearly past the chest (w_hem ≥ w_waist * 1.15) is a flared dress, not
    # a top. Tees don't flare; their hem is ≤ chest row width.
    if 1.20 <= aspect_ratio <= 1.45:
        if w_waist > 0 and w_hem / w_waist >= 1.15 and hem_coverage >= 0.55:
            center_fill_lower = _center_gap_ratio(0.55, 0.95, band_ratio=0.16)
            if center_fill_lower > 0.62:
                return "dress"

    # 4) v19.22: last-chance shorts catch — anything roughly square or wider
    # than tall with strong hem/mid coverage is almost certainly shorts.
    # v19.41: skip when the waist row is much wider than the hem — that's a
    # short-sleeve top whose bbox is set by the sleeves, not a pair of shorts.
    if 0.55 <= aspect_ratio <= 1.25 and hem_coverage >= 0.45 and mid_coverage >= 0.55:
        hem_vs_waist = w_hem / max(1.0, w_waist)
        if hem_vs_waist >= 0.85:
            return "pants"

    # 5) Default
    return "top"


def _sample_width_at_y(cloth_mask: np.ndarray, y: int) -> float:
    """Get the width of the mask at a specific y coordinate."""
    if y < 0 or y >= cloth_mask.shape[0]:
        return 0.0
    row = cloth_mask[y, :]
    xs = np.where(row > 0)[0]
    if len(xs) < 2:
        return 0.0
    return float(xs.max() - xs.min())


def _has_pants_crotch_split(cloth_mask: np.ndarray, y1: int, h: int, w: int,
                            fracs=(0.55, 0.62, 0.70, 0.78, 0.85, 0.92, 0.97)) -> bool:
    """Scan many rows in the lower half for a clear gap between 2 segments.
    Wide-leg jeans often have legs touching at the top — but lower rows still split."""
    min_gap = max(5, int(w * 0.04))
    H = cloth_mask.shape[0]
    for frac in fracs:
        row_y = max(0, min(H - 1, y1 + int(h * frac)))
        row = cloth_mask[row_y, :]
        in_seg = False
        segs: list[tuple[int, int]] = []
        start = 0
        for i, px in enumerate(row):
            if px > 0 and not in_seg:
                in_seg = True
                start = i
            elif px == 0 and in_seg:
                in_seg = False
                segs.append((start, i - 1))
        if in_seg:
            segs.append((start, len(row) - 1))
        if len(segs) >= 2:
            gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
            if max(gaps) >= min_gap:
                return True
    return False


def detect_pants_type(cloth_mask: np.ndarray) -> str:
    """Classify pants type: 'shorts', 'cropped', 'regular', 'long'.

    v19.3: use garment OWN bbox aspect ratio (h/w) — independent of frame
    padding. Product photos with lots of whitespace no longer mislead this.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 100:
        return "regular"

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    aspect = h / w  # garment-own h/w

    if aspect < 1.15:
        return "shorts"
    elif aspect < 1.75:
        return "cropped"
    elif aspect < 2.40:
        return "regular"
    else:
        return "long"


def detect_pants_style(cloth_mask: np.ndarray) -> str:
    """Classify pants silhouette style.

    v19.3: gate by aspect — for shorts return 'regular' (no wide_leg/skinny).
    Returns one of: 'skinny', 'wide_leg', 'straight', 'regular'.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 100:
        return "regular"

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    aspect = h / w
    # Shorts/cropped: style heuristics on hem vs hip are noisy — default safe.
    if aspect < 1.40:
        return "regular"

    def row_width(frac: float) -> float:
        y = max(0, min(cloth_mask.shape[0] - 1, y1 + int(h * frac)))
        nz = np.where(cloth_mask[y] > 0)[0]
        return float(nz.max() - nz.min()) if len(nz) > 4 else 0.0

    hip_w = row_width(0.25)
    knee_w = row_width(0.65)
    hem_w = row_width(0.90)

    if hip_w <= 0:
        return "regular"

    if hem_w < hip_w * 0.45:
        return "skinny"
    if hem_w > hip_w * 0.95:
        return "wide_leg"
    if abs(hem_w - knee_w) < hip_w * 0.12:
        return "straight"
    return "regular"


# ── Piecewise pants warp (v19.3) ─────────────────────────────────────────

def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _split_pants_source_mask(cloth_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split source pants into hip block, left leg, right leg.
    Works for shorts, jeans, trousers, wide-leg pants. Bbox-center fallback."""
    bbox = _mask_bbox(cloth_mask)
    if bbox is None:
        z = np.zeros_like(cloth_mask, dtype=np.uint8)
        return z, z, z

    x1, y1, x2, y2 = bbox
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    cx = int((x1 + x2) * 0.5)

    mask_bin = (cloth_mask > 20).astype(np.uint8) * 255

    hip_mask = np.zeros_like(mask_bin)
    hip_y2 = min(y2, y1 + int(h * 0.42))
    hip_mask[y1:hip_y2 + 1, x1:x2 + 1] = mask_bin[y1:hip_y2 + 1, x1:x2 + 1]

    lower_y1 = y1 + int(h * 0.42)
    lower_y2 = y2

    best_score = None
    best_x = cx
    search_half = max(4, int(w * 0.18))
    for xx in range(max(x1, cx - search_half), min(x2, cx + search_half) + 1):
        col = mask_bin[lower_y1:lower_y2 + 1, xx]
        score = int((col > 0).sum())
        score_tuple = (score, abs(xx - cx))
        if best_score is None or score_tuple < best_score:
            best_score = score_tuple
            best_x = xx
    split_x = int(best_x)

    left_mask = np.zeros_like(mask_bin)
    right_mask = np.zeros_like(mask_bin)
    left_mask[lower_y1:y2 + 1, x1:split_x + 1] = mask_bin[lower_y1:y2 + 1, x1:split_x + 1]
    right_mask[lower_y1:y2 + 1, split_x:x2 + 1] = mask_bin[lower_y1:y2 + 1, split_x:x2 + 1]

    overlap_y1 = y1 + int(h * 0.30)
    overlap_y2 = y1 + int(h * 0.52)
    overlap_w = max(3, int(w * 0.05))
    left_mask[overlap_y1:overlap_y2 + 1, x1:min(x2, split_x + overlap_w) + 1] = \
        mask_bin[overlap_y1:overlap_y2 + 1, x1:min(x2, split_x + overlap_w) + 1]
    right_mask[overlap_y1:overlap_y2 + 1, max(x1, split_x - overlap_w):x2 + 1] = \
        mask_bin[overlap_y1:overlap_y2 + 1, max(x1, split_x - overlap_w):x2 + 1]

    k5 = np.ones((5, 5), np.uint8)
    hip_mask = cv2.morphologyEx(hip_mask, cv2.MORPH_CLOSE, k5, 1)
    left_mask = cv2.morphologyEx(left_mask, cv2.MORPH_CLOSE, k5, 1)
    right_mask = cv2.morphologyEx(right_mask, cv2.MORPH_CLOSE, k5, 1)
    return hip_mask, left_mask, right_mask


def _find_shorts_source_hem_points(
    mask_bin: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    src_cx: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Find real left/right leg-opening anchors for shorts."""
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    min_seg_w = max(6, int(w * 0.08))
    best: tuple[float, int, tuple[int, int], tuple[int, int]] | None = None

    for yy in range(y1 + int(h * 0.66), y2 + 1):
        row = mask_bin[yy] > 20
        segs: list[tuple[int, int]] = []
        start = None
        for xx in range(x1, x2 + 1):
            if row[xx] and start is None:
                start = xx
            elif not row[xx] and start is not None:
                segs.append((start, xx - 1))
                start = None
        if start is not None:
            segs.append((start, x2))

        left_candidates = [seg for seg in segs if seg[1] < src_cx + w * 0.08]
        right_candidates = [seg for seg in segs if seg[0] > src_cx - w * 0.08]
        if not left_candidates or not right_candidates:
            continue

        left_seg = max(left_candidates, key=lambda seg: seg[1] - seg[0])
        right_seg = max(right_candidates, key=lambda seg: seg[1] - seg[0])
        left_w = left_seg[1] - left_seg[0] + 1
        right_w = right_seg[1] - right_seg[0] + 1
        if left_w < min_seg_w or right_w < min_seg_w:
            continue

        # Prefer wide, real leg openings instead of the very bottom crotch sliver.
        score = float(left_w + right_w) + (yy - y1) * 0.08
        if best is None or score > best[0]:
            best = (score, yy, left_seg, right_seg)

    if best is None:
        return None

    _, yy, left_seg, right_seg = best
    left = np.array([(left_seg[0] + left_seg[1]) * 0.5, yy], dtype=np.float32)
    right = np.array([(right_seg[0] + right_seg[1]) * 0.5, yy], dtype=np.float32)
    return left, right


def _affine_piece(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    out_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    h_out, w_out = out_shape
    if src_pts.shape[0] < 3 or dst_pts.shape[0] < 3:
        return (
            np.zeros((h_out, w_out, 3), dtype=np.uint8),
            np.zeros((h_out, w_out), dtype=np.uint8),
        )
    M = cv2.getAffineTransform(src_pts[:3].astype(np.float32), dst_pts[:3].astype(np.float32))
    rgb = cv2.warpAffine(
        cloth_rgb, M, (w_out, h_out),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    m = cv2.warpAffine(
        cloth_mask, M, (w_out, h_out),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    m = (m > 20).astype(np.uint8) * 255
    return rgb, m


def _composite_piece(
    base_rgb: np.ndarray,
    base_mask: np.ndarray,
    piece_rgb: np.ndarray,
    piece_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if cv2.countNonZero(piece_mask) < 20:
        return base_rgb, base_mask
    alpha = cv2.GaussianBlur((piece_mask > 20).astype(np.float32), (7, 7), 1.8)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    out_rgb = (
        base_rgb.astype(np.float32) * (1.0 - alpha)
        + piece_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)
    out_mask = cv2.bitwise_or(base_mask, piece_mask)
    return out_rgb, out_mask


def _pose_get(pose: dict, key: str, fallback: np.ndarray) -> np.ndarray:
    val = pose.get(key) if pose else None
    if val is None:
        return fallback.astype(np.float32)
    return np.array(val, dtype=np.float32)


def _bbox_warp_pants_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict,
    output_shape: tuple[int, int],
    pants_type: str,
    pants_style: str,
    fit_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable pants warp for cropped people with missing knees/ankles."""
    h_out, w_out = output_shape
    canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    out_mask = np.zeros((h_out, w_out), dtype=np.uint8)
    bbox = _mask_bbox(cloth_mask)
    if bbox is None or not pose or "left_hip" not in pose or "right_hip" not in pose:
        return canvas, out_mask

    x1, y1, x2, y2 = bbox
    lh = np.array(pose["left_hip"], dtype=np.float32)
    rh = np.array(pose["right_hip"], dtype=np.float32)
    hip_c = (lh + rh) * 0.5
    pose_hip_w = float(np.linalg.norm(lh - rh))
    ls = pose.get("left_shoulder")
    rs = pose.get("right_shoulder")
    if ls is not None and rs is not None:
        sw_ref = float(np.linalg.norm(np.array(ls, dtype=np.float32) - np.array(rs, dtype=np.float32)))
    else:
        sw_ref = pose_hip_w
    hip_w = max(24.0, pose_hip_w, sw_ref * 0.72)

    top = int(max(0, hip_c[1] - hip_w * 0.44))
    la = pose.get("left_ankle")
    ra = pose.get("right_ankle")
    lk = pose.get("left_knee")
    rk = pose.get("right_knee")
    if la is not None and ra is not None:
        ankle_y = float((la[1] + ra[1]) * 0.5)
        if pants_type == "cropped":
            bottom = int(hip_c[1] + (ankle_y - hip_c[1]) * 0.78)
        elif pants_type == "regular":
            bottom = int(hip_c[1] + (ankle_y - hip_c[1]) * 0.92)
        else:
            bottom = int(ankle_y + hip_w * 0.08)
    elif lk is not None and rk is not None:
        knee_y = float((lk[1] + rk[1]) * 0.5)
        bottom = int(hip_c[1] + (knee_y - hip_c[1]) * (1.75 if pants_type != "cropped" else 1.25))
    elif pants_type == "cropped":
        bottom = int(hip_c[1] + hip_w * 1.18)
    elif pants_type == "regular":
        bottom = int(hip_c[1] + hip_w * 1.42)
    else:
        bottom = int(hip_c[1] + hip_w * 1.62)
    bottom = min(h_out - 2, max(top + 48, bottom))

    width_mul = 1.52
    if pants_style == "wide_leg":
        width_mul = 1.74
    elif pants_style == "straight":
        width_mul = 1.60
    elif pants_style == "skinny":
        width_mul = 1.34
    target_w = int(max(42, hip_w * width_mul * fit_scale))
    target_h = int(max(48, bottom - top))
    left = int(round(hip_c[0] - target_w * 0.5))
    right = left + target_w
    if left < 0:
        right -= left
        left = 0
    if right > w_out:
        left -= right - w_out
        right = w_out
    left = max(0, left)
    target_w = max(1, right - left)

    prepared = _prepare_cloth_for_warp(cloth_rgb, cloth_mask)
    crop_rgb = prepared[y1:y2 + 1, x1:x2 + 1]
    crop_mask = ((cloth_mask[y1:y2 + 1, x1:x2 + 1] > 20).astype(np.uint8)) * 255
    if crop_rgb.size == 0 or cv2.countNonZero(crop_mask) < 100:
        return canvas, out_mask

    resized_rgb = cv2.resize(crop_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = (resized_mask > 20).astype(np.uint8) * 255

    y_end = min(h_out, top + target_h)
    x_end = min(w_out, left + target_w)
    paste_rgb = resized_rgb[: y_end - top, : x_end - left]
    paste_mask = resized_mask[: y_end - top, : x_end - left]
    canvas[top:y_end, left:x_end] = paste_rgb
    out_mask[top:y_end, left:x_end] = paste_mask
    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), 1)
    return canvas, out_mask


def piecewise_warp_pants_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict,
    output_shape: tuple[int, int],
    pants_type: str = "long",
    pants_style: str = "regular",
    fit_scale: float = 1.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp pants by hip block + left/right leg pieces (v19.3).
    Estimates knees/ankles from hips if pose lacks them."""
    h_out, w_out = output_shape
    canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    out_mask = np.zeros((h_out, w_out), dtype=np.uint8)

    bbox = _mask_bbox(cloth_mask)
    if bbox is None:
        return canvas, out_mask
    x1, y1, x2, y2 = bbox
    src_h = max(1, y2 - y1)
    src_w = max(1, x2 - x1)
    src_cx = (x1 + x2) * 0.5

    if not pose or "left_hip" not in pose or "right_hip" not in pose:
        return canvas, out_mask

    if pants_type != "shorts":
        bbox_rgb, bbox_mask = _bbox_warp_pants_cloth(
            cloth_rgb,
            cloth_mask,
            pose,
            output_shape,
            pants_type,
            pants_style,
            fit_scale,
        )
        if cv2.countNonZero(bbox_mask) >= 300:
            return bbox_rgb, bbox_mask

    lh = np.array(pose["left_hip"], dtype=np.float32)
    rh = np.array(pose["right_hip"], dtype=np.float32)
    hip_c = (lh + rh) * 0.5
    # v19.25: MediaPipe hip x-distance is often <50px on frontal poses,
    # which collapses the whole pants warp into a thin black band. Fall
    # back to shoulder width * 0.58 when hip width is unrealistically small.
    pose_hip_w = float(np.linalg.norm(lh - rh))
    _ls_pt = pose.get("left_shoulder")
    _rs_pt = pose.get("right_shoulder")
    if _ls_pt is not None and _rs_pt is not None:
        _sw_ref = float(np.linalg.norm(np.array(_ls_pt, dtype=np.float32) - np.array(_rs_pt, dtype=np.float32)))
    else:
        _sw_ref = pose_hip_w
    if pants_type == "shorts":
        hip_w = max(24.0, pose_hip_w, _sw_ref * 0.84)
    else:
        hip_w = max(24.0, pose_hip_w, _sw_ref * 0.72)

    body_leg_len = max(80.0, h_out - hip_c[1] - 8.0)
    lk_fb = lh + np.array([-hip_w * 0.08, body_leg_len * 0.45], dtype=np.float32)
    rk_fb = rh + np.array([ hip_w * 0.08, body_leg_len * 0.45], dtype=np.float32)
    la_fb = lh + np.array([-hip_w * 0.10, body_leg_len * 0.92], dtype=np.float32)
    ra_fb = rh + np.array([ hip_w * 0.10, body_leg_len * 0.92], dtype=np.float32)

    lk = _pose_get(pose, "left_knee", lk_fb)
    rk = _pose_get(pose, "right_knee", rk_fb)
    la = _pose_get(pose, "left_ankle", la_fb)
    ra = _pose_get(pose, "right_ankle", ra_fb)

    if pants_type == "shorts":
        # v19.4: shorts should reach mid-thigh, not stay near crotch.
        left_end = lh * 0.38 + lk * 0.62
        right_end = rh * 0.38 + rk * 0.62
        # v19.27: on cropped photos, pose knees can collapse to hip level →
        # shorts shrink to a sliver near the waistband. Enforce a minimum
        # vertical extent of hip_w · 1.1 below the hip line.
        _min_drop = hip_w * 0.98
        if left_end[1] - lh[1] < _min_drop:
            left_end = np.array([left_end[0], lh[1] + _min_drop], dtype=np.float32)
        if right_end[1] - rh[1] < _min_drop:
            right_end = np.array([right_end[0], rh[1] + _min_drop], dtype=np.float32)
    elif pants_type == "cropped":
        left_end = lk * 0.30 + la * 0.70
        right_end = rk * 0.30 + ra * 0.70
    elif pants_type == "regular":
        left_end = lk * 0.12 + la * 0.88
        right_end = rk * 0.12 + ra * 0.88
    else:
        left_end = la
        right_end = ra

    if pants_style == "skinny":
        thigh_scale, hem_scale = 0.24, 0.13
    elif pants_style == "wide_leg":
        thigh_scale, hem_scale = 0.34, 0.30
    elif pants_style == "straight":
        thigh_scale, hem_scale = 0.30, 0.23
    else:
        thigh_scale, hem_scale = 0.29, 0.20
    if pants_type == "shorts":
        # v19.4: sport/casual shorts need wider leg openings.
        thigh_scale = max(thigh_scale * 1.18, 0.34)
        hem_scale = max(hem_scale, 0.34)

    thigh_half = hip_w * thigh_scale * fit_scale
    hem_half = hip_w * hem_scale * fit_scale

    waist_y = hip_c[1] - hip_w * 0.34
    crotch_y = hip_c[1] + hip_w * 0.38
    waist_left = np.array([hip_c[0] - hip_w * 0.68 * fit_scale, waist_y], dtype=np.float32)
    waist_right = np.array([hip_c[0] + hip_w * 0.68 * fit_scale, waist_y], dtype=np.float32)
    crotch = np.array([hip_c[0], crotch_y], dtype=np.float32)

    hip_mask, left_mask, right_mask = _split_pants_source_mask(cloth_mask)
    prepared_rgb = _prepare_cloth_for_warp(cloth_rgb, cloth_mask)

    src_waist_l = np.array([x1, y1 + src_h * 0.06], dtype=np.float32)
    src_waist_r = np.array([x2, y1 + src_h * 0.06], dtype=np.float32)
    src_crotch  = np.array([src_cx, y1 + src_h * 0.46], dtype=np.float32)
    src_left_hem  = np.array([x1 + src_w * 0.25, y2], dtype=np.float32)
    src_right_hem = np.array([x2 - src_w * 0.25, y2], dtype=np.float32)
    if pants_type in {"shorts", "cropped", "regular", "long"}:
        hem_points = _find_shorts_source_hem_points(
            (cloth_mask > 20).astype(np.uint8) * 255,
            x1,
            y1,
            x2,
            y2,
            src_cx,
        )
        if hem_points is not None:
            src_left_hem, src_right_hem = hem_points

    # Hip block
    hip_rgb, hip_warp_mask = _affine_piece(
        prepared_rgb, hip_mask,
        np.float32([src_waist_l, src_waist_r, src_crotch]),
        np.float32([waist_left, waist_right, crotch]),
        output_shape,
    )
    canvas, out_mask = _composite_piece(canvas, out_mask, hip_rgb, hip_warp_mask)

    # Left leg
    src_left_pts = np.float32([
        [x1, y1 + src_h * 0.36],
        [src_cx, y1 + src_h * 0.46],
        src_left_hem,
    ])
    dst_left_outer = lh + np.array([-thigh_half, hip_w * 0.18], dtype=np.float32)
    dst_left_hem   = np.array([left_end[0] - hem_half, left_end[1]], dtype=np.float32)
    left_rgb, left_warp_mask = _affine_piece(
        prepared_rgb, left_mask, src_left_pts,
        np.float32([dst_left_outer, crotch, dst_left_hem]),
        output_shape,
    )
    canvas, out_mask = _composite_piece(canvas, out_mask, left_rgb, left_warp_mask)

    # Right leg
    src_right_pts = np.float32([
        [x2, y1 + src_h * 0.36],
        [src_cx, y1 + src_h * 0.46],
        src_right_hem,
    ])
    dst_right_outer = rh + np.array([thigh_half, hip_w * 0.18], dtype=np.float32)
    dst_right_hem   = np.array([right_end[0] + hem_half, right_end[1]], dtype=np.float32)
    right_rgb, right_warp_mask = _affine_piece(
        prepared_rgb, right_mask, src_right_pts,
        np.float32([dst_right_outer, crotch, dst_right_hem]),
        output_shape,
    )
    canvas, out_mask = _composite_piece(canvas, out_mask, right_rgb, right_warp_mask)

    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 1)
    out_mask = cv2.dilate(out_mask, np.ones((3, 3), np.uint8), 1)
    if cv2.countNonZero(out_mask) < 80:
        return canvas, out_mask

    top_guard = max(0, int(waist_y - hip_w * 0.18))
    out_mask[:top_guard, :] = 0
    canvas[:top_guard, :] = 0
    return canvas, out_mask


def classify_garment_type(cloth_mask: np.ndarray) -> dict[str, object]:
    """Classify garment type from its mask silhouette.

    Returns dict with:
      - sleeve_type: "short", "long", or "sleeveless"
      - is_loose: bool
      - sleeve_ratio: float (sleeve width / torso width at shoulder level)

    This drives downstream logic: skip sleeve warp for short sleeves,
    erase old sleeves when switching garment types, etc.

    Classification strategy:
      1. Find CORE torso width = minimum width in the vertical middle band (40%-70%)
         This avoids measuring at a sleeve row and thinking the torso IS that wide.
      2. Compare widths at multiple rows against core width.
         If any upper row (10%-40%) is >1.15x core → sleeves exist.
      3. For long vs short: check if the garment extends wide on LEFT or RIGHT
         edges past 55% height. Long sleeves maintain width far down.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        return {"sleeve_type": "short", "is_loose": False, "sleeve_ratio": 0.0}

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    ch = max(1, y2 - y1)
    cw = max(1, x2 - x1)
    h_mask = cloth_mask.shape[0]

    def _row_width(rel_y: float) -> float:
        row = max(0, min(h_mask - 1, y1 + int(ch * rel_y)))
        nz = np.where(cloth_mask[row] > 0)[0]
        return float(nz.max() - nz.min()) if len(nz) > 4 else 0.0

    def _row_lr(rel_y: float) -> tuple[float, float] | None:
        row = max(0, min(h_mask - 1, y1 + int(ch * rel_y)))
        nz = np.where(cloth_mask[row] > 0)[0]
        if len(nz) > 4:
            return float(nz.min()), float(nz.max())
        return None

    # ── Step 1: Find CORE torso width ──
    # The narrowest width in the mid-band (40%-70%) represents the actual
    # torso body, excluding any sleeve influence. For a long-sleeve garment,
    # the upper rows are wide because sleeves extend the silhouette, but the
    # mid-torso area should still be narrower (just the body).
    core_widths = []
    for frac in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        w = _row_width(frac)
        if w > 0:
            core_widths.append(w)

    if not core_widths:
        return {"sleeve_type": "short", "is_loose": False, "sleeve_ratio": 0.0}

    # Core = minimum width in mid-band (the true torso, no sleeves)
    w_core = min(core_widths)
    w_mid_avg = sum(core_widths) / len(core_widths)

    # ── Step 2: Check upper rows for sleeve presence ──
    # If ANY upper row (10%-40%) is significantly wider than core → has sleeves
    upper_widths = []
    for frac in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        w = _row_width(frac)
        if w > 0:
            upper_widths.append(w)

    max_upper = max(upper_widths) if upper_widths else 0.0
    sleeve_ratio = max_upper / max(1.0, w_core)

    has_sleeves = sleeve_ratio > 1.15

    # ── Step 3: Long vs Short ──
    # For long sleeves, the garment remains wide on the sides far below
    # the shoulder area (past 50-60% height). Check if LEFT or RIGHT edges
    # extend significantly beyond the core torso at lower rows.
    sleeve_type = "sleeveless"

    if has_sleeves:
        # Find how far down the wide portion extends
        # Check rows from top down: find where width narrows to ~core level
        last_wide_frac = 0.10
        for frac in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
            w = _row_width(frac)
            if w > w_core * 1.12:
                last_wide_frac = frac

        if last_wide_frac >= 0.50:
            sleeve_type = "long"
        elif last_wide_frac >= 0.20:
            sleeve_type = "short"
        else:
            # Wide only at very top — could be puffy shoulders, not real sleeves
            sleeve_type = "sleeveless"
    else:
        # Also check: for long-sleeve garments where width is UNIFORM top to bottom
        # (e.g., oversized hoodie), the ratio test above may barely miss.
        # Fallback: if the garment is tall (aspect < 0.9) and reasonably wide
        # at 60% height on sides, it's likely long-sleeve.
        lr_60 = _row_lr(0.60)
        lr_20 = _row_lr(0.20)
        if lr_60 is not None and lr_20 is not None:
            w_60 = lr_60[1] - lr_60[0]
            w_20 = lr_20[1] - lr_20[0]
            # If width at 60% is still >90% of width at 20%, garment extends
            # uniformly down → likely long-sleeve (not narrowing at waist)
            if w_20 > 0 and w_60 / w_20 > 0.90 and cw / max(1, ch) > 0.7:
                sleeve_type = "long"

    # Loose detection
    w_shoulder = _row_width(0.12)
    aspect = cw / max(1, ch)
    is_loose = aspect > 1.1 or (w_shoulder / max(1.0, w_mid_avg) > 1.20)

    return {
        "sleeve_type": sleeve_type,
        "is_loose": is_loose,
        "sleeve_ratio": sleeve_ratio,
    }


def _prepare_cloth_for_warp(cloth_rgb: np.ndarray, cloth_mask: np.ndarray) -> np.ndarray:
    """Fill background around garment so interpolation doesn't pull pure white/black edges.

    We keep garment colors, then inpaint the outside area from nearby garment pixels.
    This reduces halo artifacts after affine/TPS remap.
    Uses wider inpaint band (61px) to cover more edge area.
    """
    if cloth_rgb.shape[:2] != cloth_mask.shape[:2]:
        return cloth_rgb

    fg = (cloth_mask > 0).astype(np.uint8) * 255
    if int(fg.sum()) < 255 * 100:
        return cloth_rgb

    inpaint_area = cv2.bitwise_not(fg)
    # Wide band around garment to fill background with cloth-like colors
    band = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)), iterations=1)
    inpaint_area = cv2.bitwise_and(inpaint_area, band)
    if int(inpaint_area.sum()) < 255 * 50:
        return cloth_rgb

    # Pre-fill outer area with average edge color (better starting point for inpaint)
    edge_band = cv2.subtract(
        cv2.dilate(fg, np.ones((7, 7), np.uint8), iterations=1),
        cv2.erode(fg, np.ones((7, 7), np.uint8), iterations=1),
    )
    edge_pixels = cloth_rgb[edge_band > 0]
    if len(edge_pixels) > 50:
        avg_color = edge_pixels.mean(axis=0).astype(np.uint8)
        result = cloth_rgb.copy()
        result[inpaint_area > 0] = avg_color
    else:
        result = cloth_rgb.copy()

    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, inpaint_area, inpaintRadius=12, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def _landmarks_are_sane(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_shape: tuple[int, int],
    garment_category: str = "top",
) -> bool:
    """Basic sanity checks to avoid unstable TPS solves (24-point version v16.8).

    Points order: collar, sh_top_l, sh_top_r, shoulder_l, shoulder_r,
                  chest_center, chest_l, chest_r, side_l, side_r,
                  waist_l, waist_r, mid_l, mid_r, hem_l, hem_r,
                  armpit_l, armpit_r, under_bust_l, under_bust_r
    """
    if src_pts.shape != dst_pts.shape or src_pts.shape[0] < 16:
        return False
    if not np.isfinite(src_pts).all() or not np.isfinite(dst_pts).all():
        return False

    h_out, w_out = output_shape

    # Must be within a reasonable extended canvas.
    # v16.12: Dresses can legitimately place hem slightly past canvas when the
    # person is cropped above ankles; allow up to 1.5*h for dresses.
    y_upper = 1.50 * h_out if garment_category == "dress" else 1.25 * h_out
    if (dst_pts[:, 0] < -0.25 * w_out).any() or (dst_pts[:, 0] > 1.25 * w_out).any():
        return False
    if (dst_pts[:, 1] < -0.25 * h_out).any() or (dst_pts[:, 1] > y_upper).any():
        return False

    # Expected left-right ordering for all pairs
    # 3,4=shoulder  6,7=chest  8,9=side  10,11=waist  12,13=mid  14,15=hem  16,17=armpit  18,19=under_bust  20,21=sleeve_tip  22,23=sleeve_outer
    pair_ids = [(3, 4), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23)]
    for li, ri in pair_ids:
        if li < len(dst_pts) and ri < len(dst_pts):
            if dst_pts[li, 0] >= dst_pts[ri, 0]:
                return False

    shoulder_w = float(dst_pts[4, 0] - dst_pts[3, 0])
    mid_w = float(dst_pts[13, 0] - dst_pts[12, 0]) if len(dst_pts) > 13 else shoulder_w
    hem_w = float(dst_pts[15, 0] - dst_pts[14, 0]) if len(dst_pts) > 15 else mid_w

    if min(shoulder_w, mid_w, hem_w) < 14.0:
        return False

    # Keep torso profile smooth
    if not (0.35 * shoulder_w <= mid_w <= 1.70 * shoulder_w):
        return False
    if not (0.30 * shoulder_w <= hem_w <= 2.50 * shoulder_w):  # wider for dress skirt flare
        return False

    # Top-to-bottom progression
    collar_y = float(dst_pts[0, 1])
    sh_y = float((dst_pts[3, 1] + dst_pts[4, 1]) * 0.5)
    mid_y = float((dst_pts[12, 1] + dst_pts[13, 1]) * 0.5) if len(dst_pts) > 13 else sh_y
    hem_y = float((dst_pts[14, 1] + dst_pts[15, 1]) * 0.5) if len(dst_pts) > 15 else mid_y

    if collar_y > sh_y + 28.0:
        return False
    if not (sh_y <= mid_y <= hem_y + 8.0):
        return False

    return True


# ── TPS math ────────────────────────────────────────────────────────────

def _solve_tps(
    dst_pts: np.ndarray,
    src_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve TPS coefficients for inverse mapping dst → src.

    **v10: NORMALIZED TPS** — points are centered/scaled before solving to
    prevent numerical instability when pixel coordinates are large (200-800px).
    Without normalization, r²log(r²) kernel produces huge values → ill-conditioned
    linear system → local collapse (black blob).

    dst_pts : Nx2  –  control points in output (person) space
    src_pts : Nx2  –  corresponding points in source (cloth) space
    Returns : (coeffs, dst_norm, dst_mean, dst_scale) for use in _apply_tps
    """
    n = len(dst_pts)

    # ── NORMALIZE dst to zero-mean, unit-scale ──
    dst_mean = dst_pts.mean(axis=0)
    dst_scale = dst_pts.std() + 1e-6
    dst_n = (dst_pts - dst_mean) / dst_scale

    # NORMALIZE src with same strategy
    src_mean = src_pts.mean(axis=0)
    src_scale = src_pts.std() + 1e-6
    src_n = (src_pts - src_mean) / src_scale

    diff = dst_n[:, None, :] - dst_n[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K = np.zeros_like(r2)
    pos = r2 > 0
    K[pos] = r2[pos] * np.log(r2[pos] + 1e-20)

    P = np.hstack([np.ones((n, 1)), dst_n])

    # Regularisation (5e-1) — prevents extreme deformation
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K + np.eye(n) * 5e-1
    L[:n, n:] = P
    L[n:, :n] = P.T

    rhs = np.zeros((n + 3, 2), dtype=np.float64)
    rhs[:n] = src_n
    return (np.linalg.solve(L, rhs), dst_n, dst_mean, dst_scale,
            src_mean, src_scale)


def _apply_tps(
    coeffs: np.ndarray,
    dst_norm: np.ndarray,
    dst_mean: np.ndarray,
    dst_scale: float,
    src_mean: np.ndarray,
    src_scale: float,
    query: np.ndarray,
) -> np.ndarray:
    """Map *query* points (in output space) back to source space.

    v10: query is normalized with dst stats, result is denormalized with src stats.
    """
    n = len(dst_norm)
    # Normalize query with same dst stats used during solve
    q_n = (query - dst_mean) / dst_scale

    diff = q_n[:, None, :] - dst_norm[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K = np.zeros_like(r2)
    pos = r2 > 0
    K[pos] = r2[pos] * np.log(r2[pos] + 1e-20)

    P = np.hstack([np.ones((len(q_n), 1)), q_n])
    result_n = K @ coeffs[:n] + P @ coeffs[n:]

    # Denormalize back to pixel space using src stats
    return result_n * src_scale + src_mean


# ── Cloth landmark detection ────────────────────────────────────────────

def detect_cloth_landmarks(cloth_mask: np.ndarray) -> dict[str, tuple[float, float]]:
    """Detect 24 stable landmarks on the garment from its mask silhouette.

    24 points: collar, shoulder_top_l/r, shoulder_l/r, chest_center,
               chest_l/r, side_l/r, waist_l/r, mid_l/r, hem_l/r,
               armpit_l/r (v11), under_bust_l/r (v11),
               sleeve_tip_l/r (v16.8), sleeve_outer_l/r (v16.8).
    sleeve_tip: at sleeve hem Y, inset from edge — anchors sleeve end shape.
    sleeve_outer: at sleeve hem Y, near edge — controls sleeve width at tip.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        h, w = cloth_mask.shape[:2]
        return {
            "collar":              (w * 0.50, h * 0.05),
            "shoulder_top_left":   (w * 0.35, h * 0.08),
            "shoulder_top_right":  (w * 0.65, h * 0.08),
            "shoulder_left":       (w * 0.20, h * 0.12),
            "shoulder_right":      (w * 0.80, h * 0.12),
            "chest_center":        (w * 0.50, h * 0.22),
            "chest_left":          (w * 0.20, h * 0.22),
            "chest_right":         (w * 0.80, h * 0.22),
            "side_left":           (w * 0.18, h * 0.35),
            "side_right":          (w * 0.82, h * 0.35),
            "waist_left":          (w * 0.20, h * 0.42),
            "waist_right":         (w * 0.80, h * 0.42),
            "mid_left":            (w * 0.22, h * 0.55),
            "mid_right":           (w * 0.78, h * 0.55),
            "hem_left":            (w * 0.22, h * 0.92),
            "hem_right":           (w * 0.78, h * 0.92),
            "armpit_left":         (w * 0.22, h * 0.18),
            "armpit_right":        (w * 0.78, h * 0.18),
            "under_bust_left":     (w * 0.22, h * 0.28),
            "under_bust_right":    (w * 0.78, h * 0.28),
            "sleeve_tip_left":     (w * 0.15, h * 0.15),
            "sleeve_tip_right":    (w * 0.85, h * 0.15),
            "sleeve_outer_left":   (w * 0.10, h * 0.15),
            "sleeve_outer_right":  (w * 0.90, h * 0.15),
        }

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    ch = max(1, y2 - y1)
    cw = max(1, x2 - x1)

    def _row_lr(rel_y: float) -> tuple[float, float]:
        """Left-most and right-most cloth pixel at relative height."""
        row_y = max(0, min(cloth_mask.shape[0] - 1, y1 + int(ch * rel_y)))
        nz = np.where(cloth_mask[row_y] > 0)[0]
        if len(nz) > 4:
            return float(nz.min()), float(nz.max())
        return float(x1), float(x2)

    # Collar: center of the top 8% band
    top_band_rows = np.where(
        (cloth_mask > 0) &
        (np.indices(cloth_mask.shape)[0] <= y1 + max(6, int(ch * 0.08)))
    )
    collar_x = float(np.mean(top_band_rows[1])) if len(top_band_rows[1]) > 10 else (x1 + x2) / 2.0
    collar_y = float(y1 + max(3, int(ch * 0.04)))

    # Shoulder: Use 12% height row, but INSET from edges
    shl_raw, shr_raw = _row_lr(0.12)
    mdl, mdr = _row_lr(0.55)
    hml, hmr = _row_lr(0.88)

    shoulder_row_w = shr_raw - shl_raw
    mid_row_w = mdr - mdl

    # If shoulders are much wider than mid-torso, sleeves are extending the silhouette.
    if shoulder_row_w > mid_row_w * 1.3:
        center_x = (shl_raw + shr_raw) / 2.0
        seam_half_w = mid_row_w * 0.55
        shl = center_x - seam_half_w
        shr = center_x + seam_half_w
    else:
        shl, shr = shl_raw, shr_raw

    sh_y = float(y1 + int(ch * 0.12))

    # Shoulder top: halfway between collar and shoulder, interpolated x
    # v16.7g: 0.45 (was 0.55) — closer to collar center for tighter neckline arc.
    # Combined with narrower destination hw_st, this preserves collar roundness.
    st_y = float(y1 + int(ch * 0.07))
    stl_x = collar_x + (shl - collar_x) * 0.45
    str_x = collar_x + (shr - collar_x) * 0.45

    # Side torso: at 35% height — CRITICAL for horizontal spread (anti-collinear)
    sdl, sdr = _row_lr(0.35)
    sd_y = float(y1 + int(ch * 0.35))

    # Chest: at 22% height — between shoulder and side, gives upper torso curve
    chl, chr = _row_lr(0.22)
    ch_y = float(y1 + int(ch * 0.22))

    # Chest center: garment center at chest height (centering constraint)
    chest_center_x = (chl + chr) / 2.0

    # Waist: at 42% height — fills gap between side (35%) and mid (55%)
    wstl, wstr = _row_lr(0.42)
    wst_y = float(y1 + int(ch * 0.42))

    # Armpit: at 18% height — INSET from shoulder edges (between shoulder and chest)
    # Critical junction where torso meets sleeve
    apl, apr = _row_lr(0.18)
    ap_y = float(y1 + int(ch * 0.18))
    # Inset armpit inward from raw edges (closer to torso center than shoulder)
    ap_inset = (shr - shl) * 0.08  # 8% of shoulder width inward
    apl_x = apl + ap_inset
    apr_x = apr - ap_inset

    # Under-bust: at 28% height — between chest (22%) and side (35%)
    # Prevents chest compression by constraining the mid-chest shape
    ubl, ubr = _row_lr(0.28)
    ub_y = float(y1 + int(ch * 0.28))

    # Sleeve tip: at ~15% height — short-sleeve hem area.
    # These points anchor the lower edge of the sleeve to prevent TPS from
    # freely interpolating in the sleeve zone (which causes flat/stiff sleeves).
    # Use the actual garment edges at 15% height.
    stl_15, str_15 = _row_lr(0.15)
    st_15_y = float(y1 + int(ch * 0.15))
    # Inset sleeve_tip slightly from outer edge (mid-sleeve hem position)
    st_15_mid_l = stl_15 + (collar_x - stl_15) * 0.25
    st_15_mid_r = str_15 + (collar_x - str_15) * 0.25

    return {
        "collar":              (collar_x,  collar_y),
        "shoulder_top_left":   (stl_x,     st_y),
        "shoulder_top_right":  (str_x,     st_y),
        "shoulder_left":       (shl,       sh_y),
        "shoulder_right":      (shr,       sh_y),
        "chest_center":        (chest_center_x, ch_y),
        "chest_left":          (chl,       ch_y),
        "chest_right":         (chr,       ch_y),
        "side_left":           (sdl,       sd_y),
        "side_right":          (sdr,       sd_y),
        "waist_left":          (wstl,      wst_y),
        "waist_right":         (wstr,      wst_y),
        "mid_left":            (mdl, float(y1 + int(ch * 0.55))),
        "mid_right":           (mdr, float(y1 + int(ch * 0.55))),
        "hem_left":            (hml, float(y1 + int(ch * 0.88))),
        "hem_right":           (hmr, float(y1 + int(ch * 0.88))),
        "armpit_left":         (apl_x,     ap_y),
        "armpit_right":        (apr_x,     ap_y),
        "under_bust_left":     (ubl,       ub_y),
        "under_bust_right":    (ubr,       ub_y),
        "sleeve_tip_left":     (st_15_mid_l, st_15_y),
        "sleeve_tip_right":    (st_15_mid_r, st_15_y),
        "sleeve_outer_left":   (stl_15,    st_15_y),
        "sleeve_outer_right":  (str_15,    st_15_y),
    }


# ── Garment silhouette measurement ─────────────────────────────────────

def _measure_cloth_silhouette(
    cloth_mask: np.ndarray,
    landmarks: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Measure the garment's own silhouette half-widths at key rows.

    These are used as the PRIMARY shape reference (70-80%) for destination
    points — the body skeleton only provides light adjustment (20-30%).
    Returns half-widths (from center) at shoulder, mid, and hem rows.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        return {}

    h, w = cloth_mask.shape[:2]

    # Center x of the garment (from collar landmark)
    cx = landmarks["collar"][0]

    def _halfwidth_at_y(y: float) -> float:
        """Half-width of the garment at a given y row (from center)."""
        row_y = max(0, min(h - 1, int(y)))
        nz = np.where(cloth_mask[row_y] > 0)[0]
        if len(nz) < 4:
            return 0.0
        row_left = float(nz.min())
        row_right = float(nz.max())
        return max(abs(cx - row_left), abs(row_right - cx))

    sh_y = landmarks["shoulder_left"][1]
    mid_y = landmarks["mid_left"][1]
    hem_y = landmarks["hem_left"][1]

    # Measure at each row, take max of left/right half-widths
    hw_sh = _halfwidth_at_y(sh_y)
    hw_mid = _halfwidth_at_y(mid_y)
    hw_hem = _halfwidth_at_y(hem_y)

    # Also measure shoulder_top row
    st_y = landmarks["shoulder_top_left"][1]
    hw_st = _halfwidth_at_y(st_y)

    # Measure side_torso row (35% height)
    side_y = landmarks["side_left"][1]
    hw_side = _halfwidth_at_y(side_y)

    # Measure chest row (22% height)
    chest_y = landmarks["chest_left"][1]
    hw_chest = _halfwidth_at_y(chest_y)

    # Measure waist row (42% height)
    waist_y = landmarks["waist_left"][1]
    hw_waist = _halfwidth_at_y(waist_y)

    # Measure armpit row (18% height)
    armpit_y = landmarks["armpit_left"][1]
    hw_armpit = _halfwidth_at_y(armpit_y)

    # Measure under_bust row (28% height)
    ub_y = landmarks["under_bust_left"][1]
    hw_ub = _halfwidth_at_y(ub_y)

    # Measure sleeve_tip row (15% height)
    sleeve_tip_y = landmarks["sleeve_tip_left"][1]
    hw_sleeve_tip = _halfwidth_at_y(sleeve_tip_y)

    return {
        "shoulder_top": hw_st,
        "shoulder": hw_sh,
        "chest": hw_chest,
        "armpit": hw_armpit,
        "under_bust": hw_ub,
        "side": hw_side,
        "waist": hw_waist,
        "mid": hw_mid,
        "hem": hw_hem,
        "sleeve_tip": hw_sleeve_tip,
    }


# ── Build body-side destination points (Graduated Body-Follow) ────────

# Per-row blend weights: graduated body following.
# Upper rows = garment-dominant (preserve shoulder seam position).
# Lower rows = body-dominant (follow body curvature for natural draping).
# Shifted more body-heavy vs v8 to fix rectangular garment shape.
# v15: Shifted weights toward more body-follow for natural draping.
# Previous weights were too garment-dominant at top → garment kept flat-lay shape.
# Body-dominant weights make garment conform to body contours (chest curve, waist taper).
_BLEND_WEIGHTS = {
    # (garment_weight, body_weight) — body-dominant for natural "wearing" look.
    # v16.8: Added sleeve_tip/sleeve_outer for short-sleeve arm conformity.
    "shoulder_top": (0.35, 0.65),  # neckline zone follows body
    "shoulder":     (0.25, 0.75),  # shoulder MUST follow body for sleeve conforming
    "chest":        (0.30, 0.70),  # chest follows body
    "armpit":       (0.25, 0.75),  # armpit junction follows body
    "under_bust":   (0.30, 0.70),  # body dominant
    "side":         (0.28, 0.72),  # sides hug body
    "waist":        (0.25, 0.75),  # waist follows body strongly
    "mid":          (0.25, 0.75),  # body-dominant at mid
    "hem":          (0.30, 0.70),  # hem follows hip
    "sleeve_tip":   (0.15, 0.85),  # sleeve tip STRONGLY follows arm contour
    "sleeve_outer": (0.10, 0.90),  # outer sleeve edge = near-pure body follow
}


def _compute_body_destinations(
    pose: dict[str, tuple[int, int]],
    fit_scale: float,
    y_offset_ratio: float,
    cloth_silhouette: dict[str, float] | None = None,
    prefit_scale: float = 1.0,
    sleeve_type: str = "short",
    garment_category: str = "top",
    h_out: int | None = None,
) -> np.ndarray:
    """Compute 24 body-side destination points — GRADUATED body-follow (v16.8).

    Order matches detect_cloth_landmarks():
      0-15: collar, sh_top_l/r, shoulder_l/r, chest_center, chest_l/r,
            side_l/r, waist_l/r, mid_l/r, hem_l/r
      16-19: armpit_l/r, under_bust_l/r
      20-23: sleeve_tip_l/r, sleeve_outer_l/r (v16.8)

    Sleeve_tip/sleeve_outer anchor the short-sleeve hem edge to follow the arm.
    """
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    lh = np.array(pose["left_hip"], dtype=np.float64)
    rh = np.array(pose["right_hip"], dtype=np.float64)

    sw = float(np.linalg.norm(ls - rs))
    hw = float(np.linalg.norm(lh - rh))
    dy = y_offset_ratio * 30.0

    # Vertical interpolation: torso starts at shoulder line
    torso_top_y = min(ls[1], rs[1])
    torso_bot_y = max(lh[1], rh[1])
    torso_h = max(1.0, torso_bot_y - torso_top_y)

    # Collar destination: at the NECKLINE, between nose and shoulders.
    # If nose is available, interpolate — neckline sits ~78% from nose to shoulder.
    # 78% instead of 82% places collar slightly higher on the neck for better
    # alignment with actual crew/round necklines. V-necks are handled by the
    # garment's own shape (the collar landmark is at the garment's actual top).
    neck = (ls + rs) / 2.0
    if "nose" in pose:
        nose_y = float(pose["nose"][1])
        shoulder_mid_y = float(neck[1])
        neck[1] = nose_y + (shoulder_mid_y - nose_y) * 0.78
    else:
        neck[1] -= torso_h * 0.07  # 7% above shoulder midpoint

    def _lerp_y(frac: float) -> float:
        return torso_top_y + torso_h * frac + dy

    # ── DRESS: compute hem destination at ankle/knee level ──
    # Default: hem at 95% torso (shoulder→hip). For dress, extend to ankle.
    if garment_category == "dress":
        _la = pose.get("left_ankle") or pose.get("left_knee")
        _ra = pose.get("right_ankle") or pose.get("right_knee")
        if _la and _ra:
            _dress_hem_y = (float(_la[1]) + float(_ra[1])) / 2.0
        else:
            _dress_hem_y = torso_bot_y + torso_h * 1.40  # estimate: ~40% below hip
        # Clamp to reasonable range (80%–280% below hip)
        _dress_hem_y = float(np.clip(
            _dress_hem_y,
            torso_bot_y + torso_h * 0.80,
            torso_bot_y + torso_h * 2.80,
        ))
        # v16.12: Hard-clamp to canvas bottom so ankle keypoints extrapolated
        # beyond the image frame (when person is cropped mid-thigh) don't push
        # the hem outside the sanity canvas bound. Prefer "hem at frame bottom"
        # over "hem out of frame" — remap handles it cleanly.
        if h_out is not None:
            _dress_hem_y = float(min(_dress_hem_y, h_out - 2.0))
        # Mid Y = 45% between shoulder-top and dress hem
        _dress_mid_y = torso_top_y + (_dress_hem_y - torso_top_y) * 0.55
    else:
        _dress_hem_y = None
        _dress_mid_y = None

    cx_top = (ls[0] + rs[0]) / 2.0
    cx_bot = (lh[0] + rh[0]) / 2.0
    cx_mid = (cx_top + cx_bot) / 2.0

    fit = float(np.clip(fit_scale, 0.90, 1.25))

    # ── BODY-BASED half-widths ──
    # v16.7f: Short sleeves extend beyond skeleton shoulders to cover upper arms.
    # Use elbow positions to estimate upper arm extent.
    le = np.array(pose.get("left_elbow", pose["left_shoulder"]), dtype=np.float64)
    re = np.array(pose.get("right_elbow", pose["right_shoulder"]), dtype=np.float64)

    if sleeve_type == "short":
        # v16.8: Short sleeve extends ~55% toward elbow (was 70%).
        # 70% was too wide — garment stuck out beyond arm contour at sleeve edge.
        # 55% covers the upper arm without looking boxy.
        arm_extend_l = (le[0] - ls[0]) * 0.55  # negative = arm extends left
        arm_extend_r = (re[0] - rs[0]) * 0.55  # positive = arm extends right
        body_hw_shoulder = (sw * 0.58 + max(0, -arm_extend_l) + max(0, arm_extend_r)) * fit
    else:
        body_hw_shoulder = (sw * 0.50) * fit
    body_hw_hem = (hw * 0.65) * fit
    body_hw_mid = (body_hw_shoulder * 0.55 + body_hw_hem * 0.45)
    # v16.7g: Shoulder_top narrower (0.50, was 0.65) → collar + st_left + st_right
    # form a tighter arc → preserves the garment's round collar curve during TPS.
    body_hw_st = body_hw_shoulder * 0.50

    # Side torso: close to shoulder width for natural body coverage
    body_hw_side = (body_hw_shoulder * 0.70 + body_hw_hem * 0.30)

    # Chest: match shoulder — short sleeves need wide chest coverage
    body_hw_chest = body_hw_shoulder * 1.0

    # Waist: noticeably narrower — this creates the visible body shape
    body_hw_waist = (body_hw_side * 0.48 + body_hw_mid * 0.52)

    # Armpit: narrower than shoulder for sleeve taper
    # v16.8: 0.90 (was 0.98) — creates visible narrowing from shoulder to armpit
    body_hw_armpit = body_hw_shoulder * 0.90

    # Under-bust: body taper between chest and waist
    body_hw_ub = (body_hw_chest * 0.55 + body_hw_side * 0.45)

    # Sleeve tip: for short sleeves, the sleeve end follows the upper arm contour.
    # At ~15% torso height below shoulders, the arm width is narrower than shoulder.
    # Use directional arm vector to position sleeve tip along arm axis.
    if sleeve_type == "short":
        body_hw_sleeve_tip = body_hw_shoulder * 0.80   # slightly narrower than shoulder
        body_hw_sleeve_outer = body_hw_shoulder * 0.92  # near-shoulder width at outer edge
    else:
        body_hw_sleeve_tip = body_hw_shoulder * 0.85
        body_hw_sleeve_outer = body_hw_shoulder * 0.95

    # ── GARMENT-BASED half-widths + GRADUATED BODY BLEND ──
    # Scale cloth silhouette widths by prefit_scale to match person space.
    # Blend ratio VARIES by row: shoulder is garment-dominant (65%),
    # lower rows become body-dominant (60%) for natural draping.
    if cloth_silhouette and cloth_silhouette.get("shoulder", 0) > 10:
        gar_hw_sh = cloth_silhouette["shoulder"] * prefit_scale
        gar_hw_mid = cloth_silhouette["mid"] * prefit_scale
        gar_hw_hem = cloth_silhouette["hem"] * prefit_scale
        gar_hw_st = cloth_silhouette.get("shoulder_top", gar_hw_sh * 0.55) * prefit_scale
        gar_hw_side = cloth_silhouette.get("side", (gar_hw_sh * 0.55 + gar_hw_mid * 0.45)) * prefit_scale
        gar_hw_chest = cloth_silhouette.get("chest", gar_hw_sh * 0.95) * prefit_scale
        gar_hw_waist = cloth_silhouette.get("waist", (gar_hw_side * 0.55 + gar_hw_mid * 0.45)) * prefit_scale
        gar_hw_armpit = cloth_silhouette.get("armpit", gar_hw_sh * 0.90) * prefit_scale
        gar_hw_ub = cloth_silhouette.get("under_bust", (gar_hw_chest * 0.60 + gar_hw_side * 0.40)) * prefit_scale
        gar_hw_sleeve_tip = cloth_silhouette.get("sleeve_tip", gar_hw_sh * 0.85) * prefit_scale
        gar_hw_sleeve_outer = gar_hw_sleeve_tip * 1.10  # outer slightly wider

        # GRADUATED BLEND: upper rows garment-heavy, lower rows body-heavy
        def _blend(gar_hw, body_hw, row_name):
            gw, bw = _BLEND_WEIGHTS[row_name]
            return gw * gar_hw + bw * body_hw

        hw_shoulder = _blend(gar_hw_sh, body_hw_shoulder, "shoulder")
        hw_mid = _blend(gar_hw_mid, body_hw_mid, "mid")
        hw_hem = _blend(gar_hw_hem, body_hw_hem, "hem")
        hw_st = _blend(gar_hw_st, body_hw_st, "shoulder_top")
        hw_side = _blend(gar_hw_side, body_hw_side, "side")
        hw_chest = _blend(gar_hw_chest, body_hw_chest, "chest")
        hw_waist = _blend(gar_hw_waist, body_hw_waist, "waist")
        hw_armpit = _blend(gar_hw_armpit, body_hw_armpit, "armpit")
        hw_ub = _blend(gar_hw_ub, body_hw_ub, "under_bust")
        hw_sleeve_tip = _blend(gar_hw_sleeve_tip, body_hw_sleeve_tip, "sleeve_tip")
        hw_sleeve_outer = _blend(gar_hw_sleeve_outer, body_hw_sleeve_outer, "sleeve_outer")
    else:
        # No cloth silhouette available — use body-only (legacy fallback)
        hw_shoulder = body_hw_shoulder
        hw_mid = body_hw_mid
        hw_hem = body_hw_hem
        hw_st = body_hw_st
        hw_side = body_hw_side
        hw_chest = body_hw_chest
        hw_waist = body_hw_waist
        hw_armpit = body_hw_armpit
        hw_ub = body_hw_ub
        hw_sleeve_tip = body_hw_sleeve_tip
        hw_sleeve_outer = body_hw_sleeve_outer

    # Clamp to reasonable range.
    # Short sleeves need wider clamps — garment must cover upper arm area.
    if sleeve_type == "short":
        hw_shoulder = float(np.clip(hw_shoulder, sw * 0.50, sw * 1.05))
        hw_armpit = float(np.clip(hw_armpit, hw_shoulder * 0.80, hw_shoulder * 1.05))
        hw_chest = float(np.clip(hw_chest, hw_shoulder * 0.80, hw_shoulder * 1.10))
    else:
        hw_shoulder = float(np.clip(hw_shoulder, sw * 0.40, sw * 0.85))
        hw_armpit = float(np.clip(hw_armpit, hw_shoulder * 0.70, hw_shoulder * 1.05))
        hw_chest = float(np.clip(hw_chest, hw_shoulder * 0.75, hw_shoulder * 1.10))
    hw_side = float(np.clip(hw_side, min(hw_shoulder, hw_mid) * 0.50, max(hw_shoulder, hw_mid) * 1.25))
    hw_mid = float(np.clip(hw_mid, min(hw_shoulder, hw_hem) * 0.50, max(hw_shoulder, hw_hem) * 1.25))
    # v16.11f: Dress skirt can flare wider than hips — allow up to 2.0x hip width
    if garment_category == "dress":
        hw_hem = float(np.clip(hw_hem, hw * 0.35, hw * 2.00))
        hw_mid = float(np.clip(hw_mid, min(hw_shoulder, hw_hem) * 0.45, max(hw_shoulder, hw_hem) * 1.50))
    else:
        hw_hem = float(np.clip(hw_hem, hw * 0.35, hw * 1.15))
    hw_st = float(np.clip(hw_st, hw_shoulder * 0.28, hw_shoulder * 0.72))
    hw_waist = float(np.clip(hw_waist, min(hw_side, hw_mid) * 0.55, max(hw_side, hw_mid) * 1.20))
    hw_ub = float(np.clip(hw_ub, min(hw_chest, hw_side) * 0.60, max(hw_chest, hw_side) * 1.15))
    hw_sleeve_tip = float(np.clip(hw_sleeve_tip, hw_shoulder * 0.55, hw_shoulder * 1.05))
    hw_sleeve_outer = float(np.clip(hw_sleeve_outer, hw_sleeve_tip * 0.90, hw_shoulder * 1.15))

    # v16.7g: Shoulder_top AT the shoulder line (not below).
    # Previous: 5% below → left a huge gap between collar and shoulder_top →
    # TPS interpolated freely in the neckline zone → collar curve flattened.
    # Now: shoulder_top is AT the shoulder line, creating tight anchor spacing:
    #   collar (above) → shoulder_top (at shoulder) → shoulder (just below)
    # This preserves the garment's original collar curve shape.
    st_y = _lerp_y(-0.01)  # 1% ABOVE shoulder line (at neckline-to-shoulder transition)

    # chest: ~15% torso — upper chest below shoulders
    # v16.7g: Keep at 15% (NOT 10%) to avoid crowding with armpit (9%).
    # Minimum 6% gap between adjacent anchor Y levels prevents TPS oscillation.
    cx_chest = (cx_top * 0.85 + cx_bot * 0.15)

    # side_torso: ~30% down torso — horizontal spread for anti-collinear
    cx_side = (cx_top * 0.65 + cx_bot * 0.35)

    # waist: ~42% down torso — fills gap between side and mid
    cx_waist = (cx_top * 0.55 + cx_bot * 0.45)

    # armpit: ~9% down torso — natural armpit junction
    # v16.7g: 9% (between shoulder 3% and chest 15%) for good spacing.
    cx_armpit = (cx_top * 0.90 + cx_bot * 0.10)

    # under_bust: ~22% down torso — prevents chest compression
    # v16.7g: 22% (between chest 15% and side 30%) for good spacing.
    cx_ub = (cx_top * 0.78 + cx_bot * 0.22)

    # ── ARM-ANGLE influence on shoulder/chest Y positions ──
    # When person's arms are asymmetric (one raised, one lowered), the
    # garment's shoulder edge tilts to follow. Uses DIFFERENTIAL shift
    # (deviation from average) so symmetric poses produce zero tilt.
    le = np.array(pose.get("left_elbow", pose["left_shoulder"]), dtype=np.float64)
    re = np.array(pose.get("right_elbow", pose["right_shoulder"]), dtype=np.float64)
    la_vec = le - ls
    ra_vec = re - rs
    la_len = float(np.linalg.norm(la_vec))
    ra_len = float(np.linalg.norm(ra_vec))
    # Normalized Y component: +1 when arm points straight down, 0 when horizontal
    arm_dy_l = la_vec[1] / max(1.0, la_len)
    arm_dy_r = ra_vec[1] / max(1.0, ra_len)
    # Differential shift: only asymmetry matters (symmetric arms → zero shift)
    arm_dy_avg = (arm_dy_l + arm_dy_r) / 2.0
    arm_y_max = torso_h * 0.03  # max ±3% of torso height
    sh_y_shift_l = arm_y_max * float(np.clip(arm_dy_l - arm_dy_avg, -1.0, 1.0))
    sh_y_shift_r = arm_y_max * float(np.clip(arm_dy_r - arm_dy_avg, -1.0, 1.0))

    # ── SLEEVE TIP destinations (v16.8) ──
    # For short sleeves, the sleeve tip should follow the upper arm direction.
    # Position at ~6% below shoulder line in the torso Y grid.
    # "tip" = mid-sleeve hem (inset), "outer" = outer edge of sleeve hem.

    # Sleeve tip Y (for the horizontal anchor row)
    sleeve_tip_y = _lerp_y(0.06)  # at 6% below shoulder line

    # cx for sleeve tip row — use body center like other rows
    cx_sleeve = (cx_top * 0.93 + cx_bot * 0.07)

    return np.array([
        [neck[0],               neck[1] + dy],                         # 0: collar
        [cx_top - hw_st,        st_y],                                 # 1: shoulder_top_left
        [cx_top + hw_st,        st_y],                                 # 2: shoulder_top_right
        [cx_top - hw_shoulder,  _lerp_y(0.03) + sh_y_shift_l],        # 3: shoulder_left (arm-tilted)
        [cx_top + hw_shoulder,  _lerp_y(0.03) + sh_y_shift_r],        # 4: shoulder_right (arm-tilted)
        [cx_chest,              _lerp_y(0.15)],                        # 5: chest_center (CENTERING)
        [cx_chest - hw_chest,   _lerp_y(0.15) + sh_y_shift_l * 0.5],  # 6: chest_left (half tilt)
        [cx_chest + hw_chest,   _lerp_y(0.15) + sh_y_shift_r * 0.5],  # 7: chest_right (half tilt)
        [cx_side - hw_side,     _lerp_y(0.30)],                       # 8: side_left  (ANTI-COLLINEAR)
        [cx_side + hw_side,     _lerp_y(0.30)],                       # 9: side_right (ANTI-COLLINEAR)
        [cx_waist - hw_waist,   _lerp_y(0.42)],                       # 10: waist_left
        [cx_waist + hw_waist,   _lerp_y(0.42)],                       # 11: waist_right
        [cx_mid - hw_mid,       _dress_mid_y + dy if _dress_mid_y is not None else _lerp_y(0.55)],  # 12: mid_left
        [cx_mid + hw_mid,       _dress_mid_y + dy if _dress_mid_y is not None else _lerp_y(0.55)],  # 13: mid_right
        [cx_bot - hw_hem,       _dress_hem_y + dy if _dress_hem_y is not None else _lerp_y(0.95)],  # 14: hem_left
        [cx_bot + hw_hem,       _dress_hem_y + dy if _dress_hem_y is not None else _lerp_y(0.95)],  # 15: hem_right
        [cx_armpit - hw_armpit, _lerp_y(0.09) + sh_y_shift_l * 0.3], # 16: armpit_left
        [cx_armpit + hw_armpit, _lerp_y(0.09) + sh_y_shift_r * 0.3], # 17: armpit_right
        [cx_ub - hw_ub,         _lerp_y(0.22)],                       # 18: under_bust_left
        [cx_ub + hw_ub,         _lerp_y(0.22)],                       # 19: under_bust_right
        # v16.8: Sleeve tip — inset from edge, follows arm direction
        [cx_sleeve - hw_sleeve_tip,  sleeve_tip_y + sh_y_shift_l * 0.5],  # 20: sleeve_tip_left
        [cx_sleeve + hw_sleeve_tip,  sleeve_tip_y + sh_y_shift_r * 0.5],  # 21: sleeve_tip_right
        # v16.8: Sleeve outer — at outer edge, near-pure body follow
        [cx_sleeve - hw_sleeve_outer, sleeve_tip_y + sh_y_shift_l * 0.5], # 22: sleeve_outer_left
        [cx_sleeve + hw_sleeve_outer, sleeve_tip_y + sh_y_shift_r * 0.5], # 23: sleeve_outer_right
    ], dtype=np.float64)


# ── Public API ──────────────────────────────────────────────────────────

def _diagnose_landmark_failure(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_shape: tuple[int, int],
) -> str:
    """Return a short string explaining why _landmarks_are_sane() failed (24-pt v16.8)."""
    if src_pts.shape != dst_pts.shape or src_pts.shape[0] < 16:
        return f"shape mismatch: src={src_pts.shape} dst={dst_pts.shape}"
    if not np.isfinite(src_pts).all():
        return "NaN/Inf in src_pts"
    if not np.isfinite(dst_pts).all():
        return "NaN/Inf in dst_pts"

    h_out, w_out = output_shape
    if (dst_pts[:, 0] < -0.25 * w_out).any() or (dst_pts[:, 0] > 1.25 * w_out).any():
        return f"dst x out of canvas: x range [{dst_pts[:,0].min():.0f}, {dst_pts[:,0].max():.0f}] vs w={w_out}"
    if (dst_pts[:, 1] < -0.25 * h_out).any() or (dst_pts[:, 1] > 1.25 * h_out).any():
        return f"dst y out of canvas: y range [{dst_pts[:,1].min():.0f}, {dst_pts[:,1].max():.0f}] vs h={h_out}"

    # 20-pt pairs: 3,4=shoulder 6,7=chest 8,9=side 10,11=waist 12,13=mid 14,15=hem 16,17=armpit 18,19=ub 20,21=sleeve_tip 22,23=sleeve_outer
    pair_ids = [(3, 4), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15), (16, 17), (18, 19), (20, 21), (22, 23)]
    for li, ri in pair_ids:
        if li < len(dst_pts) and ri < len(dst_pts):
            if dst_pts[li, 0] >= dst_pts[ri, 0]:
                return f"left-right inversion at pair ({li},{ri}): L.x={dst_pts[li,0]:.0f} >= R.x={dst_pts[ri,0]:.0f}"

    shoulder_w = float(dst_pts[4, 0] - dst_pts[3, 0])
    mid_w = float(dst_pts[13, 0] - dst_pts[12, 0]) if len(dst_pts) > 13 else 0.0
    hem_w = float(dst_pts[15, 0] - dst_pts[14, 0]) if len(dst_pts) > 15 else 0.0

    if min(shoulder_w, mid_w, hem_w) < 14.0:
        return f"width too narrow: sh={shoulder_w:.0f} mid={mid_w:.0f} hem={hem_w:.0f}"

    if not (0.35 * shoulder_w <= mid_w <= 1.70 * shoulder_w):
        return f"mid/shoulder ratio: mid={mid_w:.0f} sh={shoulder_w:.0f} ratio={mid_w/shoulder_w:.2f}"
    if not (0.30 * shoulder_w <= hem_w <= 2.50 * shoulder_w):  # wider for dress skirt flare
        return f"hem/shoulder ratio: hem={hem_w:.0f} sh={shoulder_w:.0f} ratio={hem_w/shoulder_w:.2f}"

    collar_y = float(dst_pts[0, 1])
    sh_y = float((dst_pts[3, 1] + dst_pts[4, 1]) * 0.5)
    mid_y = float((dst_pts[12, 1] + dst_pts[13, 1]) * 0.5) if len(dst_pts) > 13 else sh_y
    hem_y = float((dst_pts[14, 1] + dst_pts[15, 1]) * 0.5) if len(dst_pts) > 15 else mid_y

    if collar_y > sh_y + 28.0:
        return f"collar below shoulder: collar_y={collar_y:.0f} sh_y={sh_y:.0f}"
    if not (sh_y <= mid_y <= hem_y + 8.0):
        return f"y progression: sh={sh_y:.0f} mid={mid_y:.0f} hem={hem_y:.0f}"

    # Check for collinearity — all points on nearly the same vertical line
    x_spread = float(dst_pts[:, 0].max() - dst_pts[:, 0].min())
    y_spread = float(dst_pts[:, 1].max() - dst_pts[:, 1].min())
    if y_spread > 0 and x_spread / y_spread < 0.15:
        return f"near-collinear: x_spread={x_spread:.0f} y_spread={y_spread:.0f} ratio={x_spread/y_spread:.3f}"

    return "unknown"


def detect_pants_landmarks(cloth_mask: np.ndarray) -> dict[str, tuple[float, float]]:
    """Detect key landmarks on pants mask for TPS warping.

    v16.11c: Added for pants support.

    Returns:
      - 'waist_c': waist center (top middle)
      - 'hip_l', 'hip_r': hip left/right at ~30% height
      - 'knee_l', 'knee_r': knee left/right at ~65% height
      - 'ankle_l', 'ankle_r': ankle left/right at ~90% height
      - 'hem_c': hem center (bottom middle)
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 100:
        return {}

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    landmarks = {}

    # Waist: top center
    row_waist = y1
    xs_row = np.where(cloth_mask[row_waist] > 0)[0]
    if len(xs_row) > 0:
        landmarks["waist_c"] = (float((xs_row.min() + xs_row.max()) / 2), float(row_waist))

    # Hip: 30% down
    row_hip = y1 + int(h * 0.30)
    xs_row = np.where(cloth_mask[row_hip] > 0)[0]
    if len(xs_row) > 0:
        hip_l = float(xs_row.min())
        hip_r = float(xs_row.max())
        landmarks["hip_l"] = (hip_l, float(row_hip))
        landmarks["hip_r"] = (hip_r, float(row_hip))

    # Knee: 65% down
    row_knee = y1 + int(h * 0.65)
    xs_row = np.where(cloth_mask[row_knee] > 0)[0]
    if len(xs_row) > 0:
        knee_l = float(xs_row.min())
        knee_r = float(xs_row.max())
        landmarks["knee_l"] = (knee_l, float(row_knee))
        landmarks["knee_r"] = (knee_r, float(row_knee))

    # Ankle: 90% down
    row_ankle = y1 + int(h * 0.90)
    xs_row = np.where(cloth_mask[row_ankle] > 0)[0]
    if len(xs_row) > 0:
        ankle_l = float(xs_row.min())
        ankle_r = float(xs_row.max())
        landmarks["ankle_l"] = (ankle_l, float(row_ankle))
        landmarks["ankle_r"] = (ankle_r, float(row_ankle))

    # Hem: bottom center
    row_hem = y2
    xs_row = np.where(cloth_mask[row_hem] > 0)[0]
    if len(xs_row) > 0:
        landmarks["hem_c"] = (float((xs_row.min() + xs_row.max()) / 2), float(row_hem))

    return landmarks


def tps_warp_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict[str, tuple[int, int]],
    output_shape: tuple[int, int],
    fit_scale: float = 1.12,
    y_offset_ratio: float = 0.0,
    grid_step: int = 4,
    sleeve_type: str = "short",
    garment_category: str = "top",
) -> tuple[np.ndarray, np.ndarray]:
    """Warp cloth image via TPS — **Graduated Body-Follow** approach (v11: 20-pt).

    20-point TPS with armpit/under_bust for structural constraint.
    Shoulder pre-rotation aligns cloth tilt to body before TPS.
    Normalized TPS with Jacobian collapse guard.
    """
    # Prepare source cloth so warp interpolation around silhouette does not
    # sample pure background colors (white/black halo source).
    cloth_rgb = _prepare_cloth_for_warp(cloth_rgb, cloth_mask)

    # ── SHOULDER PRE-ROTATION (v11) ──────────────────────────────────
    # The flat-lay cloth has horizontal shoulders. If the person's shoulders
    # are tilted, pre-rotate the cloth to match BEFORE landmark detection.
    # This removes tilt burden from TPS → cleaner warp.
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    shoulder_vec = rs - ls
    shoulder_angle_deg = float(np.degrees(np.arctan2(shoulder_vec[1], shoulder_vec[0])))
    # Only rotate if tilt is noticeable (>2°) but not extreme (max ±15°)
    if abs(shoulder_angle_deg) > 2.0:
        rot_deg = float(np.clip(shoulder_angle_deg, -15.0, 15.0))
        h_c, w_c = cloth_rgb.shape[:2]
        center_c = (w_c / 2.0, h_c / 2.0)
        M_pre = cv2.getRotationMatrix2D(center_c, -rot_deg, 1.0)
        cloth_rgb = cv2.warpAffine(cloth_rgb, M_pre, (w_c, h_c),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        cloth_mask = cv2.warpAffine(cloth_mask, M_pre, (w_c, h_c),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    landmarks = detect_cloth_landmarks(cloth_mask)

    # ── Measure garment silhouette BEFORE warping ──
    # These widths become the primary shape reference for destination points.
    cloth_silhouette = _measure_cloth_silhouette(cloth_mask, landmarks)

    # ── SLEEVE ISOLATION (v12) ─────────────────────────────────────────
    # Mask out sleeve pixels from cloth BEFORE TPS. TPS only warps torso;
    # sleeves are warped independently by warp_sleeves_to_arms().
    # Without this, TPS stretches sleeve pixels into wrong positions →
    # curved/broken sleeves, shoulder blob, dark artifacts.
    # IMPORTANT: use copies to not modify caller's arrays (needed for sleeve warp).
    cloth_mask = cloth_mask.copy()
    cloth_rgb = cloth_rgb.copy()
    ys_c, xs_c = np.where(cloth_mask > 0)
    if len(xs_c) > 80:
        cy1, cy2 = int(ys_c.min()), int(ys_c.max())
        c_h = max(1, cy2 - cy1)

        sh_l_x = landmarks["shoulder_left"][0]
        sh_r_x = landmarks["shoulder_right"][0]
        mid_l_x = landmarks["mid_left"][0]
        mid_r_x = landmarks["mid_right"][0]

        # Torso boundary with margin
        # v16.7f: Short sleeves need wider margin — the sleeve IS the garment
        # at shoulder level. Cutting it too tight makes sleeves collapse inward.
        if sleeve_type == "short":
            margin = (sh_r_x - sh_l_x) * 0.20  # 20% margin for short (was 5%)
        else:
            margin = (sh_r_x - sh_l_x) * 0.05

        rows = np.arange(cloth_mask.shape[0])
        frac = np.clip((rows - cy1) / c_h, 0, 1)

        # Interpolated torso edges: shoulder seam → mid body
        left_bound = sh_l_x + (mid_l_x - sh_l_x) * frac - margin
        right_bound = sh_r_x + (mid_r_x - sh_r_x) * frac + margin

        # v16.7f: Short sleeves — NO isolation. The short sleeve IS part of the
        # garment shape and should be warped by TPS together with the torso.
        # Only long sleeves need isolation (warped separately by warp_sleeves_to_arms).
        if sleeve_type == "long":
            sleeve_iso_frac = 0.65
            upper = frac <= sleeve_iso_frac

            cols_2d = np.arange(cloth_mask.shape[1])[None, :]  # (1, W)
            outside_torso = upper[:, None] & (
                (cols_2d < left_bound[:, None]) | (cols_2d > right_bound[:, None])
            )

            # v13: Contextual inpaint fill for sleeve cutoff area.
            sleeve_cutoff_mask = outside_torso.astype(np.uint8) * 255
            sleeve_cutoff_mask = cv2.bitwise_and(
                sleeve_cutoff_mask,
                (cloth_mask > 0).astype(np.uint8) * 255,
            )
            cloth_mask[outside_torso] = 0

            if int(sleeve_cutoff_mask.sum()) > 255 * 30:
                edge_band = cv2.subtract(
                    cv2.dilate(cloth_mask, np.ones((5, 5), np.uint8), iterations=1),
                    cv2.erode(cloth_mask, np.ones((5, 5), np.uint8), iterations=1),
                )
                edge_pixels = cloth_rgb[edge_band > 0]
                if len(edge_pixels) > 50:
                    avg_color = edge_pixels.mean(axis=0).astype(np.uint8)
                    cloth_rgb[sleeve_cutoff_mask > 0] = avg_color

                bgr_tmp = cv2.cvtColor(cloth_rgb, cv2.COLOR_RGB2BGR)
                bgr_tmp = cv2.inpaint(bgr_tmp, sleeve_cutoff_mask,
                                       inpaintRadius=8, flags=cv2.INPAINT_TELEA)
                cloth_rgb = cv2.cvtColor(bgr_tmp, cv2.COLOR_BGR2RGB)

    # Source: cloth landmarks (cloth image space) — 24 points (v16.8)
    src_pts = np.array([
        landmarks["collar"],                # 0
        landmarks["shoulder_top_left"],     # 1
        landmarks["shoulder_top_right"],    # 2
        landmarks["shoulder_left"],         # 3
        landmarks["shoulder_right"],        # 4
        landmarks["chest_center"],          # 5  (CENTERING)
        landmarks["chest_left"],            # 6
        landmarks["chest_right"],           # 7
        landmarks["side_left"],             # 8
        landmarks["side_right"],            # 9
        landmarks["waist_left"],            # 10
        landmarks["waist_right"],           # 11
        landmarks["mid_left"],              # 12
        landmarks["mid_right"],             # 13
        landmarks["hem_left"],              # 14
        landmarks["hem_right"],             # 15
        landmarks["armpit_left"],           # 16
        landmarks["armpit_right"],          # 17
        landmarks["under_bust_left"],       # 18
        landmarks["under_bust_right"],      # 19
        landmarks["sleeve_tip_left"],       # 20 (v16.8)
        landmarks["sleeve_tip_right"],      # 21 (v16.8)
        landmarks["sleeve_outer_left"],     # 22 (v16.8)
        landmarks["sleeve_outer_right"],    # 23 (v16.8)
    ], dtype=np.float64)

    # Destination: body landmarks (person image space) — 24 points (v16.8)
    # Graduated body-follow: pass cloth silhouette for blended widths
    # prefit_scale accounts for the scaling applied before this function
    dst_pts = _compute_body_destinations(
        pose, fit_scale, y_offset_ratio,
        cloth_silhouette=cloth_silhouette,
        prefit_scale=1.0,  # silhouette already measured in cloth's current space
        sleeve_type=sleeve_type,
        garment_category=garment_category,
        h_out=output_shape[0],
    )

    # v16.11f: Clamp dress destination points to canvas bounds.
    # Ankle/knee keypoints can be beyond the image boundary (person is cropped).
    # v16.12: _compute_body_destinations already clamps dress hem to h_out-2;
    # here we just defensively clip any remaining overshoot from fit_scale.
    if garment_category == "dress":
        _canvas_h = float(output_shape[0])
        _canvas_w = float(output_shape[1])
        dst_pts[:, 0] = np.clip(dst_pts[:, 0], -0.18 * _canvas_w, 1.18 * _canvas_w)
        dst_pts[:, 1] = np.clip(dst_pts[:, 1], -0.18 * _canvas_h, 1.45 * _canvas_h)

    # v16.8b: For short sleeves, truncate to 20 points.
    # Points 20-23 (sleeve_tip/sleeve_outer) sit only 3% from armpit anchors
    # → competing TPS constraints → wavy sleeve deformation.
    # Post-TPS arm-contour shaping (in app.py) handles sleeve boundary better.
    if sleeve_type == "short":
        src_pts = src_pts[:20]
        dst_pts = dst_pts[:20]

    h_out, w_out = output_shape

    # ── POLYGON AREA PRE-CHECK (catch degenerate/collinear early) ──
    # If the convex hull of src or dst points has near-zero area,
    # TPS will produce a degenerate warp surface → area=0.
    for label, pts in [("src", src_pts), ("dst", dst_pts)]:
        hull = cv2.convexHull(pts.astype(np.float32))
        hull_area = cv2.contourArea(hull)
        if hull_area < 100.0:
            raise ValueError(
                f"TPS {label} points degenerate (hull_area={hull_area:.1f}<100): "
                f"x_range=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] "
                f"y_range=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}]"
            )

    if not _landmarks_are_sane(src_pts, dst_pts, output_shape, garment_category):
        # Detailed diagnostic for debugging
        _diag = _diagnose_landmark_failure(src_pts, dst_pts, output_shape)
        raise ValueError(f"Unstable TPS landmarks ({_diag}); fallback")

    # ── Affine pre-alignment (coarse) ──────────────────────────────
    # ROBUST: Use ALL 24 point pairs with estimateAffine2D (least-squares).
    # Previous approach used only 3 top points (collar, sh_top_l, sh_top_r)
    # which are nearly collinear → degenerate affine → all Y collapsed to
    # a single line (hull_area=0). Using all 16 points with good vertical
    # AND horizontal spread guarantees a stable affine estimate.
    M_affine, _inliers = cv2.estimateAffine2D(
        src_pts.astype(np.float32).reshape(-1, 1, 2),
        dst_pts.astype(np.float32).reshape(-1, 1, 2),
        method=cv2.LMEDS,  # Least Median of Squares — robust to outliers
    )
    if M_affine is None:
        # estimateAffine2D failed — try simpler partial (4-DOF: rotate+scale+translate)
        M_affine, _inliers = cv2.estimateAffinePartial2D(
            src_pts.astype(np.float32).reshape(-1, 1, 2),
            dst_pts.astype(np.float32).reshape(-1, 1, 2),
        )
    if M_affine is None:
        raise ValueError("Cannot compute affine pre-alignment from landmarks")

    aligned_cloth = cv2.warpAffine(
        cloth_rgb, M_affine, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # replicate edge pixels, not black
    )
    aligned_mask = cv2.warpAffine(
        cloth_mask, M_affine, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,  # mask: 0 outside is correct
    )

    # Transform all 16 source landmarks through affine → now in person space
    ones_col = np.ones((len(src_pts), 1), dtype=np.float64)
    src_hom = np.hstack([src_pts, ones_col])
    M64 = M_affine.astype(np.float64)
    src_affine = (M64 @ src_hom.T).T                  # Nx2

    # POST-AFFINE collinearity check: if affine collapsed the points,
    # bail out early rather than feeding degenerate points to TPS.
    hull_post = cv2.convexHull(src_affine.astype(np.float32))
    hull_post_area = cv2.contourArea(hull_post)
    if hull_post_area < 100.0:
        raise ValueError(
            f"Post-affine points collapsed (hull_area={hull_post_area:.1f}): "
            f"x_range=[{src_affine[:,0].min():.0f},{src_affine[:,0].max():.0f}] "
            f"y_range=[{src_affine[:,1].min():.0f},{src_affine[:,1].max():.0f}]"
        )

    # ── EXPLICIT CENTER ALIGNMENT (v10, enhanced v15) ─────────────
    # After affine, the garment center may still drift from body center.
    # v15: Align BOTH horizontal AND vertical center, using multiple anchor
    # pairs (shoulders + chest + waist) for robust centering.

    # Horizontal centering: average of shoulders, chest, and mid
    cloth_cx_top = float(src_affine[3, 0] + src_affine[4, 0]) / 2.0
    cloth_cx_mid = float(src_affine[12, 0] + src_affine[13, 0]) / 2.0
    body_cx_top = float(dst_pts[3, 0] + dst_pts[4, 0]) / 2.0
    body_cx_mid = float(dst_pts[12, 0] + dst_pts[13, 0]) / 2.0

    # Use weighted average (top matters more for visual alignment)
    cloth_cx = cloth_cx_top * 0.65 + cloth_cx_mid * 0.35
    body_cx = body_cx_top * 0.65 + body_cx_mid * 0.35
    center_offset_x = body_cx - cloth_cx
    if abs(center_offset_x) > 1.5:
        src_affine[:, 0] += center_offset_x

    # Vertical centering: align shoulder Y position
    cloth_sh_y = float(src_affine[3, 1] + src_affine[4, 1]) / 2.0
    body_sh_y = float(dst_pts[3, 1] + dst_pts[4, 1]) / 2.0
    center_offset_y = body_sh_y - cloth_sh_y
    if abs(center_offset_y) > 3.0:
        src_affine[:, 1] += center_offset_y

    # ── v16: SYMMETRY ENFORCEMENT ────────────────────────────────
    # Force left/right point pairs to be symmetric about the body center line.
    # This prevents asymmetric TPS warps (one side stretched, other compressed)
    # caused by noisy pose keypoints or affine estimation errors.
    # Also locks center points (collar, chest_center) to the center line X.
    body_center_x = (body_cx_top + body_cx_mid) / 2.0

    # Lock center-line points to center X (collar=0, chest_center=5)
    for ci in (0, 5):
        src_affine[ci, 0] = body_center_x
        dst_pts[ci, 0] = body_center_x

    # Symmetric pairs: (left_idx, right_idx) — enforce same Y and mirror X
    _SYM_PAIRS = [
        (1, 2),   # shoulder_top L/R
        (3, 4),   # shoulder L/R
        (6, 7),   # chest L/R
        (8, 9),   # side L/R
        (10, 11), # waist L/R
        (12, 13), # mid L/R
        (14, 15), # hem L/R
        (16, 17), # armpit L/R
        (18, 19), # under_bust L/R
        (20, 21), # sleeve_tip L/R
        (22, 23), # sleeve_outer L/R
    ]
    for li, ri in _SYM_PAIRS:
        # Skip pairs beyond truncated point count (short sleeves use 20pt)
        if li >= len(src_affine) or ri >= len(src_affine):
            continue
        # Average Y so both sides are at same level
        avg_y_src = (src_affine[li, 1] + src_affine[ri, 1]) / 2.0
        src_affine[li, 1] = avg_y_src
        src_affine[ri, 1] = avg_y_src

        avg_y_dst = (dst_pts[li, 1] + dst_pts[ri, 1]) / 2.0
        dst_pts[li, 1] = avg_y_dst
        dst_pts[ri, 1] = avg_y_dst

        # Mirror X about center line so distance from center is equal
        dx_src = (abs(src_affine[li, 0] - body_center_x) + abs(src_affine[ri, 0] - body_center_x)) / 2.0
        src_affine[li, 0] = body_center_x - dx_src
        src_affine[ri, 0] = body_center_x + dx_src

        dx_dst = (abs(dst_pts[li, 0] - body_center_x) + abs(dst_pts[ri, 0] - body_center_x)) / 2.0
        dst_pts[li, 0] = body_center_x - dx_dst
        dst_pts[ri, 0] = body_center_x + dx_dst

    # ── TPS refinement (residual deformation only, NORMALIZED v10) ──
    coeffs, dst_norm, dst_mean, dst_scale, src_mean, src_scale = _solve_tps(
        dst_pts, src_affine
    )

    # Build a coarse-grid query and upsample for speed
    gy, gx = np.mgrid[0:h_out:grid_step, 0:w_out:grid_step]
    pts = np.column_stack([gx.ravel().astype(np.float64),
                           gy.ravel().astype(np.float64)])
    mapped = _apply_tps(coeffs, dst_norm, dst_mean, dst_scale,
                        src_mean, src_scale, pts)

    mx = mapped[:, 0].reshape(gy.shape).astype(np.float32)
    my = mapped[:, 1].reshape(gy.shape).astype(np.float32)

    mx = cv2.resize(mx, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    my = cv2.resize(my, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

    # DEFORMATION CLAMP: Keep TPS as subtle body-contour adjustment only.
    # v16.7c: Reduced from 12% to 8%. ShoulderAlign already positions garment,
    # TPS should only do fine deformation. 12% still distorted garment form.
    iy, ix = np.mgrid[0:h_out, 0:w_out].astype(np.float32)
    max_disp = max(h_out, w_out) * 0.08
    mx = np.clip(mx, ix - max_disp, ix + max_disp)
    my = np.clip(my, iy - max_disp, iy + max_disp)

    # ── LOCAL JACOBIAN COLLAPSE GUARD (v10) ──────────────────────────
    # Compute local Jacobian determinant of the warp field. Negative or
    # near-zero det means the warp FOLDED space (mirrored/collapsed) → black blob.
    # Fix: blend collapsed regions back toward identity (no deformation).
    dmx_dx = np.gradient(mx, axis=1)  # d(map_x)/dx
    dmx_dy = np.gradient(mx, axis=0)  # d(map_x)/dy
    dmy_dx = np.gradient(my, axis=1)  # d(map_y)/dx
    dmy_dy = np.gradient(my, axis=0)  # d(map_y)/dy
    jacobian_det = dmx_dx * dmy_dy - dmx_dy * dmy_dx
    # Good warp: det ≈ 1.0 (identity). Collapsed: det ≤ 0. Extreme stretch: det >> 2
    # v12: Tighter thresholds — catch more distortion before it becomes visible
    collapse_mask = (jacobian_det < 0.25).astype(np.float32)
    # Also catch extreme stretch (det > 2.5) which causes blurring/blob
    collapse_mask = np.maximum(collapse_mask, (jacobian_det > 2.5).astype(np.float32))
    if collapse_mask.sum() > 0:
        # Smooth the mask so the fix blends gradually (no hard edges)
        collapse_mask = cv2.GaussianBlur(collapse_mask, (31, 31), 10.0)
        # Blend collapsed regions back to identity mapping
        mx = mx * (1.0 - collapse_mask) + ix * collapse_mask
        my = my * (1.0 - collapse_mask) + iy * collapse_mask

    # CHEST-CENTER DAMPENING: Reduce deformation in center-chest area
    # where logos/text/prints typically are. Keeps them undistorted.
    # v15: Reduced from 60% to 30% — previous value was too aggressive,
    # making the entire garment look flat/pasted instead of body-conforming.
    # 30% is enough to protect text while allowing body curvature.
    chest_cx = int(dst_pts[0, 0])  # collar x = garment center
    chest_cy = int((dst_pts[0, 1] + dst_pts[12, 1]) / 2)  # between collar and mid (idx 12)
    chest_w = int(abs(dst_pts[4, 0] - dst_pts[3, 0]) * 0.30)  # 30% of shoulder width (was 35%)
    chest_h = int(abs(dst_pts[12, 1] - dst_pts[0, 1]) * 0.40)  # 40% of upper torso height (was 50%)

    if chest_w > 10 and chest_h > 10:
        # Elliptical dampening zone
        chest_zone = np.zeros((h_out, w_out), dtype=np.float32)
        cv2.ellipse(chest_zone, (chest_cx, chest_cy), (chest_w, chest_h), 0, 0, 360, 1.0, -1)
        chest_zone = cv2.GaussianBlur(chest_zone, (21, 21), 8.0)
        # Dampen: blend deformed map back toward identity in chest zone
        dampen_strength = 0.30  # v15: 30% (was 60%) — gentler, allows body curvature
        dampen = (chest_zone * dampen_strength)[..., np.newaxis] if False else chest_zone * dampen_strength
        mx = mx * (1.0 - dampen) + ix * dampen
        my = my * (1.0 - dampen) + iy * dampen

    warped_cloth = cv2.remap(
        aligned_cloth, mx, my, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # replicate edge, not black
    )
    warped_mask = cv2.remap(
        aligned_mask, mx, my, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,  # mask: 0 outside is correct
    )

    # Refine mask: dilate → erode → feather (HR-VITON style)
    warped_mask = refine_warped_mask(warped_mask)

    # ── NECKLINE CLIP: prevent garment from extending above shoulder line ──
    # The garment hangs from shoulders — nothing should appear above them.
    # Use a gentle gradient fade (not hard cut) for natural neckline transition.
    shoulder_y = int(min(pose["left_shoulder"][1], pose["right_shoulder"][1]))
    # v16.7g: Wider margin above shoulders (22%, was 18%) for collar/neckline.
    # 18% was still too aggressive for some crew-neck/round-neck t-shirts.
    ls_arr = np.array(pose["left_shoulder"], dtype=np.float64)
    rs_arr = np.array(pose["right_shoulder"], dtype=np.float64)
    lh_arr = np.array(pose["left_hip"], dtype=np.float64)
    rh_arr = np.array(pose["right_hip"], dtype=np.float64)
    torso_h_clip = abs(float((lh_arr[1] + rh_arr[1]) / 2 - (ls_arr[1] + rs_arr[1]) / 2))
    neckline_y = max(0, shoulder_y - int(torso_h_clip * 0.22))
    # Fade from 0 at neckline_y to 1 at shoulder_y (gradient band)
    fade_band = max(1, shoulder_y - neckline_y)
    for row_y in range(0, min(shoulder_y + 1, h_out)):
        if row_y < neckline_y:
            warped_mask[row_y, :] = 0
            warped_cloth[row_y, :] = 0
        elif row_y < shoulder_y:
            # Gradual fade
            alpha = float(row_y - neckline_y) / fade_band
            warped_mask[row_y, :] = (warped_mask[row_y, :].astype(np.float32) * alpha).astype(np.uint8)

    # Reject pathological warps so caller can fallback to safer perspective mode.
    area_ratio = float((warped_mask > 20).sum()) / float(max(1, h_out * w_out))
    if area_ratio < 0.01 or area_ratio > 0.70:
        raise ValueError(f"TPS warp area={area_ratio:.3f} out of [0.01, 0.70]")

    return warped_cloth, warped_mask


def refine_warped_mask(mask: np.ndarray) -> np.ndarray:
    """v16: Clean up warped mask — erode to shrink, smooth edges.

    Previous versions either: (a) dilated too much → mask too fat, or
    (b) only closed holes → mask bleeds beyond garment edges → black borders.

    v16 approach:
      1. Threshold to remove fringe
      2. Erode 3px to shrink mask INSIDE garment boundary (prevents edge bleed)
      3. Close small internal holes
      4. Gaussian blur for smooth compositing edges (5px sigma)
    """
    # Threshold: remove semi-transparent fringe
    mask = (mask > 30).astype(np.uint8) * 255
    # Erode 2px: shrink mask slightly inward. Prevents mask from
    # extending beyond actual garment pixels, but less aggressive than 3px
    # to avoid white gaps at garment sides.
    k2 = np.ones((2, 2), np.uint8)
    mask = cv2.erode(mask, k2, iterations=1)
    # Close small internal holes (after erosion, some small gaps may appear)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    # Soft blur for smooth compositing (sigma ~1.5, kernel 5x5)
    mask = cv2.GaussianBlur(mask, (5, 5), 1.5)
    return mask


# ── Body-scaled fallback (when TPS fails) ─────────────────────────────

def simple_affine_warp_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict[str, tuple[int, int]],
    output_shape: tuple[int, int],
    garment_category: str = "top",
) -> tuple[np.ndarray, np.ndarray]:
    """Body-scaled affine warp — used when TPS fails.

    Better than uniform-scale affine: scales width and height INDEPENDENTLY
    to match person's shoulder width and torso height, then positions
    the garment at the shoulder midpoint.

    For garment_category="dress", the target height extends from shoulders to
    ankle/knee (or canvas bottom when the person is cropped), so the skirt
    actually reaches the bottom of the frame instead of being squashed into
    a top-sized box.

    Steps:
    1. Measure cloth shoulder width + cloth torso height from mask
    2. Measure person shoulder width + torso height from pose
    3. Compute separate scale_w and scale_h (clamped to safe range)
    4. Build non-uniform affine matrix: scale + translate
    5. Pre-fill background with edge colors to reduce black halo

    Returns: (warped_cloth, warped_mask)
    """
    cloth_rgb = _prepare_cloth_for_warp(cloth_rgb, cloth_mask)

    h_out, w_out = output_shape
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        return np.zeros((h_out, w_out, 3), dtype=np.uint8), np.zeros((h_out, w_out), dtype=np.uint8)

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    cloth_h = max(1, y2 - y1)
    cloth_w = max(1, x2 - x1)

    # Cloth shoulder width measurement
    h_mask = cloth_mask.shape[0]
    if garment_category == "dress":
        # v16.19: For dress, the 12%-row hits the neckline (narrow) not the actual
        # shoulder seam. Scan the top 30% for the widest row → true shoulder width.
        # Also use the overall bbox center_x for x-centering (more stable than a
        # single narrow neckline row, which causes large scale_w errors).
        # v18.9: For LONG-SLEEVE dresses, the top-30% widest row catches the
        # sleeve span (arms extended sideways) instead of body shoulder, which
        # inflates cloth_shoulder_w → underestimates scale_w → garment renders
        # too short on body. Cross-check against body width at hip level
        # (60-75%) — that band is sleeve-free. If the top-30% width is much
        # wider than hip-band body width, treat that as a sleeve artefact and
        # fall back to the hip-band width as the body reference.
        _top30_end = min(h_mask, y1 + int(cloth_h * 0.30))
        _best_w, _best_sx, _best_ex = 0, None, None
        for _row in range(max(0, y1 + int(cloth_h * 0.05)), _top30_end):
            _nz = np.where(cloth_mask[_row] > 0)[0]
            if len(_nz) > 4:
                _rw = int(_nz[-1]) - int(_nz[0])
                if _rw > _best_w:
                    _best_w = _rw
                    _best_sx = float(_nz[0])
                    _best_ex = float(_nz[-1])
        _hip_band_widths: list[int] = []
        _hip_band_start = y1 + int(cloth_h * 0.60)
        _hip_band_end = min(h_mask, y1 + int(cloth_h * 0.78))
        for _row in range(max(0, _hip_band_start), _hip_band_end):
            _nz = np.where(cloth_mask[_row] > 0)[0]
            if len(_nz) > 4:
                _hip_band_widths.append(int(_nz[-1]) - int(_nz[0]))
        _hip_band_w = float(np.median(_hip_band_widths)) if _hip_band_widths else 0.0
        if _best_sx is not None and _best_w > 10:
            # If the top-30% peak is >35% wider than the hip body band, it is
            # almost certainly a sleeve. Use the hip band as body width.
            if _hip_band_w > 10 and _best_w > _hip_band_w * 1.35:
                cloth_shoulder_w = _hip_band_w
            else:
                cloth_shoulder_w = float(_best_w)
            cloth_center_x = float(x1 + x2) / 2.0  # bbox center for stability
        elif _hip_band_w > 10:
            cloth_shoulder_w = _hip_band_w
            cloth_center_x = float(x1 + x2) / 2.0
        else:
            cloth_shoulder_w = float(cloth_w)
            cloth_center_x = float(x1 + x2) / 2.0
    else:
        shoulder_row = max(0, min(h_mask - 1, y1 + int(cloth_h * 0.12)))
        nz = np.where(cloth_mask[shoulder_row] > 0)[0]
        if len(nz) > 4:
            cloth_sx = float(nz.min())
            cloth_ex = float(nz.max())
            cloth_shoulder_w = cloth_ex - cloth_sx
            cloth_center_x = (cloth_sx + cloth_ex) / 2.0
        else:
            cloth_shoulder_w = float(cloth_w)
            cloth_center_x = float(x1 + x2) / 2.0

    cloth_top_y = float(y1)

    # Person body dimensions
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    lh = np.array(pose["left_hip"], dtype=np.float64)
    rh = np.array(pose["right_hip"], dtype=np.float64)
    person_sw = float(np.linalg.norm(ls - rs))
    person_center_x = float((ls[0] + rs[0]) / 2.0)
    person_shoulder_y = float(min(ls[1], rs[1]))
    person_torso_h = float(max(lh[1], rh[1]) - min(ls[1], rs[1]))

    # v18.8: Source aspect ratio dictates dress length. The old branch set
    # target_height = ankle_y - shoulder_y, forcing mid-thigh sources to
    # vertically stretch ~2x → ikat/print patterns rendered as vertical
    # streaks, and hems landed near ankles regardless of source. The new
    # behaviour scales height from the width-scale (uniform), so mid-thigh
    # stays mid-thigh and maxi stays maxi. The downstream scale_ratio_limit
    # still allows a small ±15% non-uniform nudge for fit.
    if garment_category == "dress":
        _person_hip_w_est = float(abs(rh[0] - lh[0]))
        _person_ref_w = max(person_sw, _person_hip_w_est)
        _w_scale_est = (_person_ref_w * 1.10) / max(cloth_shoulder_w, 10.0)
        _w_scale_est = float(np.clip(_w_scale_est, 0.5, 3.0))
        target_height = cloth_h * _w_scale_est
    else:
        target_height = person_torso_h * 1.05

    # Separate width and height scaling
    if cloth_shoulder_w > 10:
        if garment_category == "dress":
            # v16.56: Dresses must cover both shoulder and hip form. Scaling
            # only from shoulder width leaves a straight, pasted-looking tube
            # on wider hips; use the larger body reference with a small margin.
            person_hip_w = float(abs(rh[0] - lh[0]))
            person_ref_w = max(person_sw, person_hip_w)
            scale_w = (person_ref_w * 1.10) / cloth_shoulder_w
        else:
            scale_w = (person_sw * 1.05) / cloth_shoulder_w
    else:
        scale_w = 1.0

    if cloth_h > 20 and target_height > 20:
        scale_h = target_height / cloth_h
    else:
        scale_h = scale_w  # fallback to uniform

    # When scales differ wildly, use the smaller one to avoid distortion
    # but allow moderate non-uniform scaling for better fit
    # v18.8: Tightened from 2.4 → 1.18 for dress. Source aspect now drives
    # height (uniform with width). The 18% slack absorbs minor body-shape
    # mismatch without re-introducing vertical pattern streaks.
    scale_ratio_limit = 1.18 if garment_category == "dress" else 1.5
    scale_ratio = max(scale_w, scale_h) / max(min(scale_w, scale_h), 0.01)
    if scale_ratio > scale_ratio_limit:
        s_min = min(scale_w, scale_h)
        scale_w = min(scale_w, s_min * scale_ratio_limit)
        scale_h = min(scale_h, s_min * scale_ratio_limit)

    scale_w = float(np.clip(scale_w, 0.5, 3.0))
    scale_h = float(np.clip(scale_h, 0.5, 3.0 if garment_category == "dress" else 3.0))

    # Target: collar at shoulder midpoint, slightly above
    # v16.19: Dress neckline sits at neck level (~12% torso_h above shoulder_y).
    # Previous 3% was too low → upper chest not covered → gray rectangle artifact.
    if garment_category == "dress":
        target_top_y = person_shoulder_y - person_torso_h * 0.12
    else:
        target_top_y = person_shoulder_y - person_torso_h * 0.03

    tx = person_center_x - cloth_center_x * scale_w
    ty = target_top_y - cloth_top_y * scale_h

    # Non-uniform affine matrix: independent x/y scale + translate
    M = np.array([
        [scale_w, 0.0,     tx],
        [0.0,     scale_h, ty],
    ], dtype=np.float64)

    warped_cloth = cv2.warpAffine(
        cloth_rgb, M, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped_mask = cv2.warpAffine(
        cloth_mask, M, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    warped_mask = refine_warped_mask(warped_mask)
    return warped_cloth, warped_mask


# ── Piecewise dress warp (v18.24) ───────────────────────────────────

def piecewise_warp_dress_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict[str, tuple[int, int]],
    output_shape: tuple[int, int],
    silhouette: str = "a_line",
    length: str = "midi",
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise (per-row) dress warp — replacement for pure affine.

    Same vertical placement / overall scale as `simple_affine_warp_cloth`'s
    dress branch, but each output row is independently horizontally
    stretched so the garment silhouette follows shoulder → waist → hip →
    hem widths derived from body anthropometry + the named silhouette
    template (from ``src.garment_silhouettes``).

    The result is a body-following silhouette: bodycon/sheath stays ~shoulder
    width through the hip, a-line flares from waist down, mermaid keeps a
    narrow knee with a flared hem.  Diffusion then only has to do
    fold/shading work — the form is already correct.

    Falls back to ``simple_affine_warp_cloth`` on any failure (small mask,
    missing landmarks, numeric issues).
    """
    try:
        from src.garment_silhouettes import (
            DRESS_TEMPLATES,
            SAMPLE_FRACS_DRESS,
        )
    except Exception:
        return simple_affine_warp_cloth(
            cloth_rgb, cloth_mask, pose, output_shape, garment_category="dress"
        )

    cloth_rgb = _prepare_cloth_for_warp(cloth_rgb, cloth_mask)

    h_out, w_out = output_shape
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 200:
        return simple_affine_warp_cloth(
            cloth_rgb, cloth_mask, pose, output_shape, garment_category="dress"
        )

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    cloth_h = max(1, y2 - y1)
    cloth_cx_src = float(x1 + x2) / 2.0

    # ── Sample source half-widths at standard fractions ────────────
    h_mask = cloth_mask.shape[0]
    src_halves: list[float] = []
    for f in SAMPLE_FRACS_DRESS:
        row = max(0, min(h_mask - 1, y1 + int(cloth_h * f)))
        nz = np.where(cloth_mask[row] > 0)[0]
        if len(nz) >= 4:
            src_halves.append(float(nz[-1] - nz[0]) * 0.5)
        else:
            src_halves.append(0.0)

    # Anti-sleeve guard: for long-sleeve dresses with arms extended sideways,
    # the upper rows catch the SLEEVE span, not the body bust.  Cross-check
    # against a hip-band median (60-78% of garment height — always sleeve-free)
    # and treat oversized top widths as sleeve artefacts.
    _top30_end = min(h_mask, y1 + int(cloth_h * 0.30))
    _best_top_full = 0
    for _row in range(max(0, y1 + int(cloth_h * 0.05)), _top30_end):
        _nz = np.where(cloth_mask[_row] > 0)[0]
        if len(_nz) > 4:
            _rw = int(_nz[-1]) - int(_nz[0])
            if _rw > _best_top_full:
                _best_top_full = _rw
    _hip_widths: list[int] = []
    for _row in range(max(0, y1 + int(cloth_h * 0.60)),
                      min(h_mask, y1 + int(cloth_h * 0.78))):
        _nz = np.where(cloth_mask[_row] > 0)[0]
        if len(_nz) > 4:
            _hip_widths.append(int(_nz[-1]) - int(_nz[0]))
    _hip_full = float(np.median(_hip_widths)) if _hip_widths else 0.0
    # Bust half = max of upper rows (handles missing 0.12 row).  If top span
    # is >35% wider than hip span, treat as long-sleeve artefact and fall
    # back to hip half-width as the body reference.
    bust_half_src = max(src_halves[0], src_halves[1])
    if _hip_full > 10 and _best_top_full > _hip_full * 1.35:
        bust_half_src = _hip_full * 0.5
        # Override top-row half-widths so per-row scale doesn't over-shrink the
        # shoulder region using the (artefactual) sleeve span.
        src_halves[0] = bust_half_src * 0.92
        src_halves[1] = bust_half_src
    if bust_half_src < 6.0:
        return simple_affine_warp_cloth(
            cloth_rgb, cloth_mask, pose, output_shape, garment_category="dress"
        )

    # ── Body landmarks ─────────────────────────────────────────────
    try:
        ls = np.array(pose["left_shoulder"], dtype=np.float64)
        rs = np.array(pose["right_shoulder"], dtype=np.float64)
        lh = np.array(pose["left_hip"], dtype=np.float64)
        rh = np.array(pose["right_hip"], dtype=np.float64)
    except Exception:
        return simple_affine_warp_cloth(
            cloth_rgb, cloth_mask, pose, output_shape, garment_category="dress"
        )

    person_sw = float(np.linalg.norm(ls - rs))
    person_hip_w = float(abs(rh[0] - lh[0]))
    person_cx = float((ls[0] + rs[0]) / 2.0)
    person_shoulder_y = float(min(ls[1], rs[1]))
    person_torso_h = float(max(lh[1], rh[1]) - min(ls[1], rs[1]))
    if person_sw < 30 or person_torso_h < 40:
        return simple_affine_warp_cloth(
            cloth_rgb, cloth_mask, pose, output_shape, garment_category="dress"
        )

    # ── Vertical placement (mirror simple_affine_warp_cloth dress) ─
    person_ref_w = max(person_sw, person_hip_w)
    # Width-scale anchor: bust half-width should land near person_ref_w/2 * 1.10
    target_bust_half = person_ref_w * 0.55
    w_scale = float(np.clip(target_bust_half / bust_half_src, 0.5, 3.0))
    target_top_y = person_shoulder_y - person_torso_h * 0.12

    # v18.26: length-aware target_bot_y.  The aspect-uniform target
    # (cloth_h * w_scale) leaves long-sleeve sources too short on tall
    # models because their cloth_h is compressed by the arms-down pose.
    # Pin the hem to a body-anchored Y for the detected length:
    #   mini  → ~mid-thigh
    #   midi  → ~mid-calf
    #   maxi  → ~ankle
    # then take the larger of (aspect-uniform, body-anchored).
    hip_y = float((lh[1] + rh[1]) * 0.5)
    length_norm = (length or "midi").strip().lower()
    if length_norm == "mini":
        body_target_bot_y = hip_y + person_torso_h * 0.55
    elif length_norm == "maxi":
        body_target_bot_y = hip_y + person_torso_h * 2.05
    else:
        body_target_bot_y = hip_y + person_torso_h * 1.35
    # Clamp body-anchored hem to canvas to avoid drawing off-frame.
    body_target_bot_y = float(min(body_target_bot_y, h_out - 4))
    aspect_height = cloth_h * w_scale
    body_height = max(20.0, body_target_bot_y - target_top_y)
    # v18.27: tightened stretch cap from 1.35 → 1.20.  Beyond 20% vertical
    # stretch the dress texture starts producing vertical streak/scratch
    # artefacts (each source row gets smeared across multiple output rows
    # of the same pattern).  Within 20% the streaks are hidden by diffusion's
    # fold shading.  Midi/maxi dresses that need more length will simply use
    # the aspect_height anchor — better short-but-clean than long-but-streaky.
    target_height = float(min(body_height, aspect_height * 1.20))
    target_height = max(target_height, aspect_height)
    target_bot_y = target_top_y + target_height

    # ── Target half-width curve from silhouette template ───────────
    # v18.26: sheath/shift are straight tubes — disable per-row variation so the
    # 4% waist multiplier doesn't create a visible pinch after diffusion.
    uniform_scale = silhouette in {"sheath", "shift"}
    tmpl = DRESS_TEMPLATES.get(silhouette, DRESS_TEMPLATES["a_line"])
    target_halves: list[float] = []
    for f in SAMPLE_FRACS_DRESS:
        if uniform_scale:
            mult = 1.0
        else:
            mult = tmpl[f]
        half_px = target_bust_half * mult
        # Clamp per band to body anthropometry so we never paint past arms.
        if f <= 0.30:
            ceil_px = person_sw * 0.74
        elif f <= 0.50:
            ceil_px = max(person_sw * 0.74, person_hip_w * 0.85)
        elif f <= 0.75:
            ceil_px = max(person_sw * 0.92, person_hip_w * 1.10)
        else:
            ceil_px = max(person_sw * 1.05, person_hip_w * 1.45)
        floor_px = person_sw * 0.40
        target_halves.append(float(np.clip(half_px, floor_px, ceil_px)))

    # v18.26: also flatten source half-widths for sheath/shift so per-row
    # scale stays constant — otherwise a slightly tapered source waist
    # would still show up as a thinning ring in the output.
    if uniform_scale:
        src_halves = [bust_half_src] * len(SAMPLE_FRACS_DRESS)

    fracs_arr = np.array(SAMPLE_FRACS_DRESS, dtype=np.float32)
    src_halves_arr = np.array(src_halves, dtype=np.float32)
    tgt_halves_arr = np.array(target_halves, dtype=np.float32)

    # ── Build dense per-row scale + source-y maps ──────────────────
    y_out_idx = np.arange(h_out, dtype=np.float32)
    # frac within target garment span (clipped 0..1)
    frac = np.clip((y_out_idx - target_top_y) / max(target_height, 1.0), 0.0, 1.0)
    # source y for each output row
    src_y_per_row = y1 + frac * cloth_h
    # per-row half widths via interp
    src_half_per_row = np.interp(frac, fracs_arr, src_halves_arr)
    tgt_half_per_row = np.interp(frac, fracs_arr, tgt_halves_arr)
    # avoid zero-source half (fall back to neighbouring sample)
    src_half_per_row = np.where(src_half_per_row < 2.0,
                                np.maximum(bust_half_src, 2.0),
                                src_half_per_row)
    scale_per_row = src_half_per_row / np.maximum(tgt_half_per_row, 2.0)
    # v18.28: smooth scale curve so abrupt waist scale changes don't create
    # vertical streak artefacts ("xước") where adjacent rows pull from very
    # different x-ranges. Kernel ~3% of height keeps silhouette intact.
    _smooth_k = max(3, int(h_out * 0.03) | 1)
    scale_per_row = cv2.GaussianBlur(
        scale_per_row.reshape(-1, 1), (1, _smooth_k), 0
    ).reshape(-1)

    # ── Build remap fields ─────────────────────────────────────────
    x_out_idx = np.arange(w_out, dtype=np.float32)
    map_x = (x_out_idx[None, :] - person_cx) * scale_per_row[:, None] + cloth_cx_src
    map_y = np.broadcast_to(src_y_per_row[:, None], (h_out, w_out)).astype(np.float32)
    # Pixels outside the dress vertical span: send to -1 so cv2.remap leaves them
    # at borderValue 0.
    outside = (y_out_idx < target_top_y - 1) | (y_out_idx > target_bot_y + 1)
    map_x = np.ascontiguousarray(map_x, dtype=np.float32)
    map_y = np.ascontiguousarray(map_y, dtype=np.float32)
    if outside.any():
        map_x[outside, :] = -1.0
        map_y[outside, :] = -1.0

    # v18.28: INTER_CUBIC for cloth (sharper texture preservation) — mask stays
    # linear since binary upscaling doesn't benefit from cubic.
    warped_cloth = cv2.remap(
        cloth_rgb, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    warped_mask = cv2.remap(
        cloth_mask, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_mask = refine_warped_mask(warped_mask)
    return warped_cloth, warped_mask




def _warp_long_sleeve_tps(
    cloth_prepared: np.ndarray,
    side_mask: np.ndarray,
    cloth_mask: np.ndarray,
    cloth_sh_pt: tuple[float, float],
    person_sh: np.ndarray,
    person_el: np.ndarray,
    person_wr: np.ndarray,
    sh_l_x: float,
    sh_r_x: float,
    sh_y: float,
    landmarks: dict,
    side: str,
    output_shape: tuple[int, int],
    full_arm_len: float,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Warp a long sleeve using TPS with shoulder→elbow→wrist control points.

    Instead of a simple affine (rotate+scale+translate), this uses a thin-plate
    spline to map the flat-lay sleeve shape onto the actual arm pose. This allows
    the sleeve to follow arm bends naturally.

    Control points on the cloth sleeve:
      - Shoulder junction (top of sleeve)
      - Mid-sleeve (midpoint of sleeve extent) → maps to elbow
      - Sleeve tip (end of sleeve) → maps to wrist
      - Width control points at each level (left/right edges)

    Returns (warped_rgb, warped_mask_float) or (None, None) on failure.
    """
    h_out, w_out = output_shape

    # Find sleeve pixel extent
    sleeve_ys, sleeve_xs = np.where(side_mask > 0)
    if len(sleeve_xs) < 100:
        return None, None

    # Bounding box of sleeve
    sy_min, sy_max = int(sleeve_ys.min()), int(sleeve_ys.max())
    sx_min, sx_max = int(sleeve_xs.min()), int(sleeve_xs.max())
    sleeve_h = max(1, sy_max - sy_min)
    sleeve_w = max(1, sx_max - sx_min)

    # Cloth sleeve control points: sample at 3 levels along the sleeve
    # Level 0: shoulder junction (top)
    # Level 1: mid-sleeve → maps to elbow
    # Level 2: sleeve tip → maps to wrist
    def _sleeve_row_bounds(rel_y: float):
        """Get left/right bounds of sleeve at relative y position."""
        row = max(0, min(side_mask.shape[0] - 1, sy_min + int(sleeve_h * rel_y)))
        nz = np.where(side_mask[row] > 0)[0]
        if len(nz) < 3:
            # Scan nearby rows
            for delta in range(1, 8):
                for r in [row - delta, row + delta]:
                    r = max(0, min(side_mask.shape[0] - 1, r))
                    nz = np.where(side_mask[r] > 0)[0]
                    if len(nz) >= 3:
                        return float(nz.min()), float(nz.max()), float(r)
            return None, None, None
        return float(nz.min()), float(nz.max()), float(row)

    # Sample 3 levels
    top_l, top_r, top_y = _sleeve_row_bounds(0.05)
    mid_l, mid_r, mid_y = _sleeve_row_bounds(0.50)
    tip_l, tip_r, tip_y = _sleeve_row_bounds(0.92)

    if any(v is None for v in [top_l, top_r, mid_l, mid_r, tip_l, tip_r]):
        return None, None

    # Cloth source control points (in cloth image space)
    # Center + left edge + right edge at each level
    src_pts = np.array([
        [(top_l + top_r) / 2, top_y],   # 0: shoulder center
        [top_l, top_y],                  # 1: shoulder left edge
        [top_r, top_y],                  # 2: shoulder right edge
        [(mid_l + mid_r) / 2, mid_y],   # 3: mid-sleeve center (→elbow)
        [mid_l, mid_y],                  # 4: mid-sleeve left edge
        [mid_r, mid_y],                  # 5: mid-sleeve right edge
        [(tip_l + tip_r) / 2, tip_y],   # 6: sleeve tip center (→wrist)
        [tip_l, tip_y],                  # 7: tip left edge
        [tip_r, tip_y],                  # 8: tip right edge
    ], dtype=np.float64)

    # Person arm vectors
    sh_to_el = person_el - person_sh
    sh_to_wr = person_wr - person_sh

    # Arm perpendicular (for width offset)
    arm_dir = sh_to_wr / max(1.0, float(np.linalg.norm(sh_to_wr)))
    arm_perp = np.array([-arm_dir[1], arm_dir[0]])  # perpendicular

    # Sleeve widths at each level from cloth (for proportional mapping)
    top_half_w = (top_r - top_l) / 2.0
    mid_half_w = (mid_r - mid_l) / 2.0
    tip_half_w = (tip_r - tip_l) / 2.0

    # Scale widths: sleeve should narrow along arm realistically
    # Use the actual cloth proportions, scaled to arm length
    arm_scale = full_arm_len / max(1.0, float(np.sqrt(sleeve_h**2 + sleeve_w**2)))
    arm_scale = float(np.clip(arm_scale, 0.55, 1.30))

    top_w_scaled = top_half_w * arm_scale
    mid_w_scaled = mid_half_w * arm_scale
    tip_w_scaled = tip_half_w * arm_scale

    # Destination control points (in person image space)
    # Map centers along the arm skeleton, edges perpendicular
    p_top = person_sh.copy()
    p_mid = person_sh + sh_to_el * 0.95   # slightly before elbow for natural drape
    p_tip = person_sh + sh_to_wr * 0.92   # slightly before wrist

    dst_pts = np.array([
        p_top,                                  # 0: shoulder center
        p_top - arm_perp * top_w_scaled,       # 1: shoulder left edge
        p_top + arm_perp * top_w_scaled,       # 2: shoulder right edge
        p_mid,                                  # 3: elbow center
        p_mid - arm_perp * mid_w_scaled,       # 4: elbow left edge
        p_mid + arm_perp * mid_w_scaled,       # 5: elbow right edge
        p_tip,                                  # 6: wrist center
        p_tip - arm_perp * tip_w_scaled,       # 7: wrist left edge
        p_tip + arm_perp * tip_w_scaled,       # 8: wrist right edge
    ], dtype=np.float64)

    # Sanity: check hull area of both point sets
    for label, pts in [("sleeve_src", src_pts), ("sleeve_dst", dst_pts)]:
        hull = cv2.convexHull(pts.astype(np.float32))
        if cv2.contourArea(hull) < 50.0:
            return None, None

    # ── TPS warp ──
    # Step 1: Affine pre-alignment
    M_aff, _ = cv2.estimateAffine2D(
        src_pts.astype(np.float32).reshape(-1, 1, 2),
        dst_pts.astype(np.float32).reshape(-1, 1, 2),
        method=cv2.LMEDS,
    )
    if M_aff is None:
        M_aff, _ = cv2.estimateAffinePartial2D(
            src_pts.astype(np.float32).reshape(-1, 1, 2),
            dst_pts.astype(np.float32).reshape(-1, 1, 2),
        )
    if M_aff is None:
        return None, None

    # Prepare sleeve-only cloth image without neutral gray.  The warped sleeve
    # mask is soft, so any constant 128 background can leak into visible pixels.
    sleeve_cloth = _prepare_cloth_for_warp(cloth_prepared, side_mask)
    sleeve_pixels = cloth_prepared[side_mask > 0]
    sleeve_bg = (128, 128, 128)
    if len(sleeve_pixels) > 20:
        sleeve_bg = tuple(int(v) for v in np.median(sleeve_pixels, axis=0))

    # Affine warp
    aligned_s = cv2.warpAffine(
        sleeve_cloth, M_aff, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=sleeve_bg,
    )
    aligned_sm = cv2.warpAffine(
        side_mask, M_aff, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Step 2: TPS residual warp for fine adjustment (same as torso TPS)
    # _solve_tps does inverse mapping dst→src for cv2.remap
    ones_col = np.ones((len(src_pts), 1), dtype=np.float64)
    src_hom = np.hstack([src_pts, ones_col])
    M64 = M_aff.astype(np.float64)
    src_affine = (M64 @ src_hom.T).T  # src points after affine

    try:
        coeffs, dst_norm, dst_mean, dst_scale, src_mean, src_scale = _solve_tps(
            dst_pts, src_affine
        )
    except Exception:
        coeffs = None

    if coeffs is not None:
        grid_step = 4
        gy, gx = np.mgrid[0:h_out:grid_step, 0:w_out:grid_step]
        pts = np.column_stack([gx.ravel().astype(np.float64),
                               gy.ravel().astype(np.float64)])
        mapped = _apply_tps(coeffs, dst_norm, dst_mean, dst_scale,
                            src_mean, src_scale, pts)

        map_x = mapped[:, 0].reshape(gy.shape).astype(np.float32)
        map_y = mapped[:, 1].reshape(gy.shape).astype(np.float32)

        map_x = cv2.resize(map_x, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

        aligned_s = cv2.remap(
            aligned_s, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=sleeve_bg,
        )
        aligned_sm = cv2.remap(
            aligned_sm.astype(np.float32), map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).astype(np.uint8)

    # Post-process mask
    aligned_sm = (aligned_sm > 40).astype(np.uint8) * 255
    aligned_sm = cv2.erode(aligned_sm, np.ones((3, 3), np.uint8), iterations=1)

    # Don't cut at 0.6 arm_len for long sleeves — go to near-wrist
    max_y_long = int(person_wr[1] + 15)
    if max_y_long < h_out:
        aligned_sm[max_y_long:, :] = 0

    sm_f = aligned_sm.astype(np.float32) / 255.0
    sm_f = cv2.GaussianBlur(sm_f, (9, 9), 2.0)
    sm_f[sm_f < 0.08] = 0.0
    sm_f = np.clip(sm_f * 0.92, 0.0, 0.92)  # higher opacity for long sleeves

    if (sm_f > 0.05).sum() < 50:
        return None, None

    return aligned_s, sm_f


# ── Sleeve warping ─────────────────────────────────────────────────────

def warp_sleeves_to_arms(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict[str, tuple[int, int]],
    output_shape: tuple[int, int],
    sleeve_type: str = "long",
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Detect sleeves on cloth and warp each to follow the person's arm pose.

    v11: accepts sleeve_type parameter. Short sleeves get STRONGER rotation
    (±55° clamp, 0.80 dampening) because they need to clearly follow arm
    direction. Long sleeves keep gentler params (±40°, 0.70).
    """
    landmarks = detect_cloth_landmarks(cloth_mask)
    h_out, w_out = output_shape
    h_c, w_c = cloth_mask.shape[:2]

    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        return None

    y1, y2 = int(ys.min()), int(ys.max())
    ch = max(1, y2 - y1)

    def _row_lr(rel_y: float):
        row_y = max(0, min(h_c - 1, y1 + int(ch * rel_y)))
        nz = np.where(cloth_mask[row_y] > 0)[0]
        if len(nz) > 4:
            return float(nz.min()), float(nz.max())
        return None, None

    # Check if sleeves exist
    arm_l, arm_r = _row_lr(0.30)
    mid_l, mid_r = _row_lr(0.55)
    if arm_l is None or mid_l is None:
        return None
    if (arm_r - arm_l) < (mid_r - mid_l) * 1.15:
        return None

    # Torso width for row-by-row sleeve detection
    sh_l_x = landmarks["shoulder_left"][0]
    sh_r_x = landmarks["shoulder_right"][0]
    sh_y = landmarks["shoulder_left"][1]
    mid_l_x = landmarks["mid_left"][0]
    mid_r_x = landmarks["mid_right"][0]

    # Person pose
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    le = np.array(pose.get("left_elbow", pose["left_shoulder"]), dtype=np.float64)
    re = np.array(pose.get("right_elbow", pose["right_shoulder"]), dtype=np.float64)
    lw = np.array(pose.get("left_wrist", pose.get("left_elbow", pose["left_shoulder"])), dtype=np.float64)
    rw = np.array(pose.get("right_wrist", pose.get("right_elbow", pose["right_shoulder"])), dtype=np.float64)

    cloth_prepared = _prepare_cloth_for_warp(cloth_rgb, cloth_mask)

    # Build row-by-row sleeve masks: only pixels BEYOND torso at each row
    # v15: Extend limit for long sleeves (sleeves go much further down)
    rows = np.arange(h_c)
    max_row_frac = 0.65 if sleeve_type == "long" else 0.40
    max_row = y1 + int(ch * max_row_frac)
    row_frac = np.clip((rows - y1) / max(1, ch), 0, 1)
    torso_left_per_row = sh_l_x + (mid_l_x - sh_l_x) * row_frac
    torso_right_per_row = sh_r_x + (mid_r_x - sh_r_x) * row_frac

    cols_2d = np.broadcast_to(np.arange(w_c, dtype=np.float64)[None, :], (h_c, w_c))
    torso_left_2d = torso_left_per_row[:, None]
    torso_right_2d = torso_right_per_row[:, None]
    rows_2d = np.broadcast_to(rows[:, None], (h_c, w_c))

    in_upper = rows_2d <= max_row
    is_cloth = cloth_mask > 0

    # Sleeve masks: ONLY pixels OUTSIDE the torso boundary (strict separation).
    # v13: Reduced margin from 8px to 4px for tighter sleeve-torso junction.
    # The seam-aware blending in app.py handles the overlap smoothly.
    sleeve_margin = 4
    left_sleeve_mask_src = np.zeros((h_c, w_c), dtype=np.uint8)
    left_sleeve_mask_src[(is_cloth) & (in_upper) & (cols_2d < torso_left_2d - sleeve_margin)] = 255

    right_sleeve_mask_src = np.zeros((h_c, w_c), dtype=np.uint8)
    right_sleeve_mask_src[(is_cloth) & (in_upper) & (cols_2d > torso_right_2d + sleeve_margin)] = 255

    # Arm vectors for rotation angle calculation
    arm_vec_l = le - ls
    arm_vec_r = re - rs
    arm_len_l = float(np.linalg.norm(arm_vec_l))
    arm_len_r = float(np.linalg.norm(arm_vec_r))

    # Full arm length (shoulder→wrist) for long sleeves
    full_arm_len_l = float(np.linalg.norm(lw - ls))
    full_arm_len_r = float(np.linalg.norm(rw - rs))

    results = {}

    for side, side_mask, person_sh, person_el, person_wr, cloth_sh_x, arm_len, full_arm_len in [
        ("left",  left_sleeve_mask_src, ls, le, lw, sh_l_x, arm_len_l, full_arm_len_l),
        ("right", right_sleeve_mask_src, rs, re, rw, sh_r_x, arm_len_r, full_arm_len_r),
    ]:
        if side_mask.sum() < 255 * 150:  # need at least 150 sleeve pixels
            continue
        if arm_len < 10:
            continue

        # Cloth shoulder point (pivot for rotation)
        cloth_sh_pt = (float(cloth_sh_x), float(sh_y))

        # ── LONG SLEEVE: TPS WARP (shoulder→elbow→wrist) ──
        # For long sleeves, simple affine can't follow arm bends.
        # Use a multi-point TPS that maps cloth sleeve control points
        # to body arm keypoints for natural arm-following.
        if sleeve_type == "long" and full_arm_len > 30:
            warped_s, sm_f = _warp_long_sleeve_tps(
                cloth_prepared, side_mask, cloth_mask,
                cloth_sh_pt, person_sh, person_el, person_wr,
                sh_l_x, sh_r_x, sh_y, landmarks,
                side, (h_out, w_out), full_arm_len,
            )
            if warped_s is not None:
                results[side] = (warped_s, sm_f)
                continue

        # ── SHORT SLEEVE: AFFINE WARP (original logic) ──
        # Compute arm angle in degrees (from horizontal)
        arm_vec = person_el - person_sh
        arm_angle_deg = float(np.degrees(np.arctan2(arm_vec[1], arm_vec[0])))

        # Detect ACTUAL sleeve angle from cloth mask using PCA of sleeve pixels.
        sleeve_ys, sleeve_xs = np.where(side_mask > 0)
        if len(sleeve_xs) > 50:
            sx_centered = sleeve_xs - sleeve_xs.mean()
            sy_centered = sleeve_ys - sleeve_ys.mean()
            cov_xx = float(np.sum(sx_centered ** 2))
            cov_yy = float(np.sum(sy_centered ** 2))
            cov_xy = float(np.sum(sx_centered * sy_centered))
            theta = 0.5 * np.arctan2(2.0 * cov_xy, cov_xx - cov_yy + 1e-10)
            cloth_sleeve_angle = float(np.degrees(theta))
            if side == "left":
                if cloth_sleeve_angle > 0:
                    cloth_sleeve_angle -= 180.0
                default_angle = cloth_sleeve_angle
            else:
                if cloth_sleeve_angle < 0:
                    cloth_sleeve_angle += 180.0
                default_angle = cloth_sleeve_angle
        else:
            if side == "left":
                default_angle = 180.0
            else:
                default_angle = 0.0

        rotation_deg = arm_angle_deg - default_angle

        if sleeve_type == "short":
            rotation_deg = float(np.clip(rotation_deg, -55.0, 55.0))
            rotation_deg *= 0.80
        else:
            rotation_deg = float(np.clip(rotation_deg, -40.0, 40.0))
            rotation_deg *= 0.70

        # Scale sleeve to match arm length
        if len(sleeve_ys) > 50:
            sleeve_extent = float(np.sqrt((sleeve_xs.max() - sleeve_xs.min())**2 +
                                          (sleeve_ys.max() - sleeve_ys.min())**2))
            scale = float(np.clip(arm_len / max(1.0, sleeve_extent * 0.85), 0.80, 1.15))
        else:
            scale = float(np.clip(arm_len / max(1.0, float(ch) * 0.35), 0.85, 1.10))

        M_rot = cv2.getRotationMatrix2D(cloth_sh_pt, -rotation_deg, scale)
        M_rot[0, 2] += person_sh[0] - cloth_sh_pt[0]
        M_rot[1, 2] += person_sh[1] - cloth_sh_pt[1]

        sleeve_cloth = _prepare_cloth_for_warp(cloth_prepared, side_mask)
        sleeve_pixels = cloth_prepared[side_mask > 0]
        sleeve_bg = (128, 128, 128)
        if len(sleeve_pixels) > 20:
            sleeve_bg = tuple(int(v) for v in np.median(sleeve_pixels, axis=0))

        warped_s = cv2.warpAffine(
            sleeve_cloth, M_rot, (w_out, h_out),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=sleeve_bg,
        )
        warped_sm = cv2.warpAffine(
            side_mask, M_rot, (w_out, h_out),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        warped_sm = (warped_sm > 40).astype(np.uint8) * 255
        warped_sm = cv2.erode(warped_sm, np.ones((3, 3), np.uint8), iterations=1)

        max_sleeve_y = int(person_sh[1] + arm_len * 0.6)
        if max_sleeve_y < h_out:
            warped_sm[max_sleeve_y:, :] = 0

        sm_f = warped_sm.astype(np.float32) / 255.0
        sm_f = cv2.GaussianBlur(sm_f, (9, 9), 2.0)
        sm_f[sm_f < 0.08] = 0.0
        sm_f = np.clip(sm_f * 0.85, 0.0, 0.85)

        if (sm_f > 0.05).sum() < 50:
            continue

        results[side] = (warped_s, sm_f)

    return results if results else None
