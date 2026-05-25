"""Dress target silhouette builder — pose-driven, template-driven.

Given a pose dict + dress analysis (silhouette/length) + the source cloth
width-curve, draw a smooth dress shape onto a binary mask. This becomes the
authoritative footprint for the v2 pipeline — TPS is only used as a texture
seed inside this footprint, not as the source of truth.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.garment_silhouettes import DRESS_TEMPLATES, SAMPLE_FRACS_DRESS


_LENGTH_HEM_FRAC = {
    "mini": 0.55,
    "thigh": 0.62,
    "knee": 0.72,
    "midi": 0.86,
    "maxi": 0.94,
}


def _pose_xy(pose: dict, key: str, fallback: tuple[float, float]) -> tuple[float, float]:
    v = pose.get(key) if pose else None
    if v is None:
        return fallback
    return float(v[0]), float(v[1])


def _midpoint(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _euclid(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 2:
        return (mask > 0).astype(np.uint8) * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def _round_binary_edges(mask: np.ndarray) -> np.ndarray:
    """Round hard polygon corners without changing the template footprint."""
    if int(cv2.countNonZero(mask)) < 50:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    rounded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    rounded = cv2.morphologyEx(rounded, cv2.MORPH_CLOSE, kernel, iterations=1)
    rounded = cv2.GaussianBlur(rounded, (0, 0), 2.4)
    _, rounded = cv2.threshold(rounded, 88, 255, cv2.THRESH_BINARY)
    return _largest_component(rounded)


def _round_hem_corners(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) < 100:
        return mask
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    radius = int(np.clip(min(width * 0.055, height * 0.035), 7, 16))
    if radius <= 2:
        return mask
    out = mask.copy()
    y_start = max(y1, y2 - radius + 1)
    for y in range(y_start, y2 + 1):
        nz = np.where(out[y] > 0)[0]
        if len(nz) < radius * 2:
            continue
        t = (y - y_start) / max(1.0, float(y2 - y_start))
        shrink = int(round(radius * (1.0 - np.cos(t * np.pi * 0.5))))
        if shrink <= 0:
            continue
        left = int(nz.min())
        right = int(nz.max())
        out[y, left:left + shrink] = 0
        out[y, right - shrink + 1:right + 1] = 0
    return _largest_component(out)


def _shoulder_from_parsing(
    parsing: dict | None,
    shape: tuple[int, int],
    pose: dict | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Infer (left_shoulder, right_shoulder, garment_top_y) from upper_clothes
    + dress + shoulder region of parsing. The widest row in the upper band of
    the original garment is taken as the real shoulder line — far closer to the
    actual body than MediaPipe shoulder keypoints, which often sit *inside* the
    torso and underestimate width."""
    if not parsing:
        return None
    h, w = shape
    garment = np.zeros((h, w), dtype=np.uint8)
    upper = np.zeros((h, w), dtype=np.uint8)
    for key in ("upper_clothes", "dress"):
        v = parsing.get(key)
        if v is None:
            continue
        if v.shape[:2] != (h, w):
            v = cv2.resize(v, (w, h), interpolation=cv2.INTER_NEAREST)
        part = (v > 20).astype(np.uint8) * 255
        garment = cv2.bitwise_or(garment, part)
        if key == "upper_clothes":
            upper = cv2.bitwise_or(upper, part)
    if int(cv2.countNonZero(garment)) < 400:
        return None
    gys = np.where(garment > 0)[0]
    garment_top_y = float(gys.min()) if len(gys) else 0.0

    source = upper if int(cv2.countNonZero(upper)) >= 120 else garment
    source = _largest_component(source)
    ys, _xs = np.where(source > 0)
    if len(ys) < 50:
        return None
    y_top = int(ys.min())
    y_bot = int(ys.max())
    band_h = max(10, int((y_bot - y_top + 1) * 0.42))
    band = source[y_top:min(h, y_top + band_h), :]
    # Widest row in the top band approximates shoulder line of the worn garment.
    candidates = []
    for ry in range(band.shape[0]):
        row = np.where(band[ry] > 0)[0]
        if len(row) < 5:
            continue
        x_left = int(row.min())
        x_right = int(row.max())
        width = int(x_right - x_left + 1)
        fill_ratio = float(len(row)) / max(1.0, float(width))
        if fill_ratio < 0.28:
            continue
        preferred_y = band.shape[0] * 0.32
        score = float(width) - abs(float(ry) - preferred_y) * 0.55
        candidates.append((score, ry, x_left, x_right, width))
    if not candidates:
        return None
    candidates.sort(key=lambda t: -t[0])
    _score, ry, x_left, x_right, width = candidates[0]
    if width < 20:
        return None
    abs_y = float(y_top + ry)
    # Enforce symmetry around the body centerline. If parsing of one shoulder
    # is occluded (hand/bag) the widest row becomes one-sided and the dress
    # ends up cutting off the obscured shoulder. Centre on pose only (nose +
    # shoulder/hip midpoint); parsing x-median bends toward whichever side the
    # bag covers.
    full_ys, full_xs = np.where(source > 0)
    cx_candidates: list[float] = []
    pose_half_lower: float | None = None
    if pose:
        nose = pose.get("nose")
        if nose is not None:
            cx_candidates.append(float(nose[0]))
        ls_p = pose.get("left_shoulder")
        rs_p = pose.get("right_shoulder")
        if ls_p is not None and rs_p is not None:
            cx_candidates.append(float(ls_p[0] + rs_p[0]) * 0.5)
            # Pose shoulder distance is a hard lower bound on shoulder width:
            # the body cannot be narrower than its skeleton.
            pose_half_lower = abs(float(ls_p[0] - rs_p[0])) * 0.5
        lh_p = pose.get("left_hip")
        rh_p = pose.get("right_hip")
        if lh_p is not None and rh_p is not None:
            cx_candidates.append(float(lh_p[0] + rh_p[0]) * 0.5)
    if not cx_candidates and len(full_xs):
        cx_candidates.append(float(np.median(full_xs)))
    cx = float(np.median(cx_candidates)) if cx_candidates else float(x_left + x_right) * 0.5
    raw_half = (x_right - x_left) * 0.5
    half = max(raw_half, pose_half_lower or 0.0)
    x_left_sym = float(cx - half)
    x_right_sym = float(cx + half)
    return (x_left_sym, abs_y), (x_right_sym, abs_y), garment_top_y


def build_target_silhouette(
    pose: dict,
    dress_analysis: dict,
    frame_shape: tuple[int, int],
    parsing: dict | None = None,
) -> np.ndarray:
    """Return a binary uint8 mask (0/255) of the dress target footprint.

    Width is the source-cloth template scaled by torso/hip widths.
    """
    h, w = int(frame_shape[0]), int(frame_shape[1])
    out = np.zeros((h, w), dtype=np.uint8)

    ls = _pose_xy(pose, "left_shoulder", (w * 0.40, h * 0.18))
    rs = _pose_xy(pose, "right_shoulder", (w * 0.60, h * 0.18))
    pose_ls = ls
    pose_rs = rs
    lh = _pose_xy(pose, "left_hip", (w * 0.42, h * 0.52))
    rh = _pose_xy(pose, "right_hip", (w * 0.58, h * 0.52))
    lk = _pose_xy(pose, "left_knee", (w * 0.44, h * 0.72))
    rk = _pose_xy(pose, "right_knee", (w * 0.56, h * 0.72))
    la = _pose_xy(pose, "left_ankle", (w * 0.45, h * 0.93))
    ra = _pose_xy(pose, "right_ankle", (w * 0.55, h * 0.93))

    # Use parsing as a width/center hint only; keep pose shoulder height.
    # — fixes elongated-neck artifact caused by MediaPipe placing shoulder
    # keypoints below the actual garment shoulder.
    garment_top_y: float | None = None
    parsing_shoulder = _shoulder_from_parsing(parsing, (h, w), pose=pose)
    if parsing_shoulder is not None:
        ls_p, rs_p, garment_top_y = parsing_shoulder
        pose_c = _midpoint(pose_ls, pose_rs)
        pose_w = max(_euclid(pose_ls, pose_rs), w * 0.12)
        raw_cx = (ls_p[0] + rs_p[0]) * 0.5
        raw_w = abs(rs_p[0] - ls_p[0])
        cx = float(np.clip(raw_cx, pose_c[0] - pose_w * 0.30, pose_c[0] + pose_w * 0.30))
        shoulder_w_clamped = float(np.clip(raw_w, pose_w * 0.92, pose_w * 1.28))
        shoulder_y = float(np.clip((pose_ls[1] + pose_rs[1]) * 0.5, 0.0, h - 1.0))
        # Order: ls = right side of image when person faces camera in MP
        # convention. Keep ls = left-of-image vs rs = right-of-image consistent
        # with MediaPipe (ls.x ≤ rs.x is NOT guaranteed in MP — but for symmetry
        # of the silhouette it does not matter, we only use midpoint + width).
        ls = (cx - shoulder_w_clamped * 0.5, shoulder_y)
        rs = (cx + shoulder_w_clamped * 0.5, shoulder_y)

    shoulder_c = _midpoint(ls, rs)
    hip_c = _midpoint(lh, rh)
    knee_c = _midpoint(lk, rk)
    ankle_c = _midpoint(la, ra)

    shoulder_w = max(_euclid(ls, rs), 1.0)
    pose_shoulder_w = max(_euclid(pose_ls, pose_rs), shoulder_w * 0.75, 1.0)
    hip_w = max(_euclid(lh, rh), pose_shoulder_w * 0.78)

    silhouette = (dress_analysis.get("silhouette") or "a_line").lower()
    if silhouette not in DRESS_TEMPLATES:
        silhouette = "a_line"
    template = DRESS_TEMPLATES[silhouette]
    sleeve = (dress_analysis.get("sleeve") or "auto").lower()
    neckline = (dress_analysis.get("neckline") or "unknown").lower()

    # Extra hem boost for flared silhouettes — the source templates only carry
    # the relative shape; without this multiplier the target ends up looking
    # like a sheath because base_half already shrinks at bust. Kept modest so
    # the hem stays close enough to the body that the dress does not look
    # detached from the figure.
    _hem_boost = {
        "a_line": 1.14,
        "fit_and_flare": 1.20,
        "ball_gown": 1.38,
        "empire": 1.18,
        "sheath": 1.04,
        "shift": 1.10,
        "mermaid": 1.16,
    }.get(silhouette, 1.12)

    length = (dress_analysis.get("length") or "midi").lower()
    if length not in _LENGTH_HEM_FRAC:
        length = "midi"
    hem_y = _LENGTH_HEM_FRAC[length] * h

    # Hard-cap hem at pose ankle (with a small margin) so dress never extends
    # past the feet regardless of length classification or aspect override.
    _ankle_avail = pose and (pose.get("left_ankle") is not None or pose.get("right_ankle") is not None)
    if _ankle_avail:
        ankle_y = ankle_c[1]
        hem_y = min(hem_y, ankle_y + shoulder_w * 0.05)

    # Crew neck should sit below the chin at the base of the neck. The previous
    # high top_y overlapped the lower face, so agnostic masking carved the
    # neckline into fragments and diffusion hallucinated scarf/cowl shapes.
    nose = pose.get("nose") if pose else None
    nose_y = float(nose[1]) if nose is not None else shoulder_c[1] - shoulder_w * 0.55
    top_y = max(
        shoulder_c[1] - shoulder_w * 0.20,
        nose_y + shoulder_w * 0.36,
    )
    torso_h = max(1.0, max(lh[1], rh[1]) - min(ls[1], rs[1]))
    if neckline in {"strapless", "sweetheart"}:
        top_y = max(top_y, shoulder_c[1] + torso_h * 0.16)
    elif neckline in {"off_shoulder", "bardot"}:
        top_y = max(top_y, shoulder_c[1] + torso_h * 0.08)
    elif sleeve == "sleeveless":
        top_y = max(top_y, shoulder_c[1] + torso_h * 0.04)
    top_y = float(max(0.0, top_y))

    # Clamp top to original garment neckline — extending the dress above this
    # line is what stretches the model's neck visually.
    if garment_top_y is not None:
        top_y = max(top_y, float(garment_top_y) - shoulder_w * 0.05)

    n_rows = 64
    ys = np.linspace(top_y, hem_y, n_rows)

    pts_left: list[tuple[float, float]] = []
    pts_right: list[tuple[float, float]] = []

    if neckline in {"strapless", "sweetheart"}:
        bust_half = shoulder_w * 0.47
    elif neckline in {"off_shoulder", "bardot"}:
        bust_half = shoulder_w * 0.56
    elif sleeve == "sleeveless":
        bust_half = shoulder_w * 0.54
    else:
        bust_half = shoulder_w * 0.62
    hip_half = hip_w * 0.78

    fracs = list(SAMPLE_FRACS_DRESS)
    mults = [template[f] for f in fracs]

    for y in ys:
        t = (y - top_y) / max(1.0, hem_y - top_y)

        if t < 0.35:
            base_half = bust_half + (hip_half - bust_half) * (t / 0.35) * 0.25
        else:
            tt = (t - 0.35) / 0.65
            base_half = bust_half * (1 - tt) + hip_half * tt

        m = float(np.interp(t, fracs, mults))
        # Boost hem flare only in the lower half so bust/waist stay snug.
        if t > 0.45:
            m *= 1.0 + (_hem_boost - 1.0) * ((t - 0.45) / 0.55)
        half_w = base_half * m

        if t <= 0.45:
            tt = t / 0.45
            cx = shoulder_c[0] * (1 - tt) + hip_c[0] * tt
        elif t <= 0.78:
            tt = (t - 0.45) / 0.33
            cx = hip_c[0] * (1 - tt) + knee_c[0] * tt
        else:
            tt = (t - 0.78) / 0.22
            cx = knee_c[0] * (1 - tt) + ankle_c[0] * tt

        pts_left.append((cx - half_w, y))
        pts_right.append((cx + half_w, y))

    poly = np.array(pts_left + pts_right[::-1], dtype=np.int32)
    cv2.fillPoly(out, [poly], 255)

    if neckline in {"off_shoulder", "bardot"}:
        band = np.zeros_like(out)
        band_half = int(round(np.clip(shoulder_w * 0.62, pose_shoulder_w * 0.48, pose_shoulder_w * 0.76)))
        band_h = int(round(np.clip(pose_shoulder_w * 0.10, 7, 16)))
        band_y = int(round(top_y + band_h * 0.25))
        cx = int(round(shoulder_c[0]))
        cv2.ellipse(band, (cx, band_y), (band_half, band_h), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        out = cv2.bitwise_or(out, band)
    elif neckline in {"strapless", "sweetheart"}:
        band = np.zeros_like(out)
        band_half = int(round(np.clip(shoulder_w * 0.45, pose_shoulder_w * 0.36, pose_shoulder_w * 0.56)))
        band_h = int(round(np.clip(pose_shoulder_w * 0.075, 6, 12)))
        band_y = int(round(top_y + band_h * 0.25))
        cx = int(round(shoulder_c[0]))
        cv2.ellipse(band, (cx, band_y), (band_half, band_h), 0, 0, 360, 255, -1, lineType=cv2.LINE_AA)
        out = cv2.bitwise_or(out, band)

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out = cv2.GaussianBlur(out, (0, 0), 2.2)
    _, out = cv2.threshold(out, 96, 255, cv2.THRESH_BINARY)
    out = _round_binary_edges(out)
    return _round_hem_corners(out)
