"""Dress mask builder (v2): pose-driven target + agnostic + hair split."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DressMasks:
    target_mask: np.ndarray         # geometry-driven dress footprint
    agnostic_mask: np.ndarray       # paint zone for diffusion
    hair_front_mask: np.ndarray     # paste on top after diffusion
    hair_underlap_mask: np.ndarray  # diffusion paints dress here
    shoe_protect_mask: np.ndarray   # never paint dress here
    hand_protect_mask: np.ndarray   # hands outside dress kept from original
    old_clothes_mask: np.ndarray    # union of old upper/dress/skirt/pants


def _safe(parsing: dict | None, key: str, shape: tuple[int, int]) -> np.ndarray:
    if not parsing:
        return np.zeros(shape, dtype=np.uint8)
    v = parsing.get(key)
    if v is None:
        return np.zeros(shape, dtype=np.uint8)
    return (v > 0).astype(np.uint8) * 255


def _pose_xy(pose: dict | None, key: str) -> tuple[float, float] | None:
    if not pose:
        return None
    v = pose.get(key)
    if v is None:
        return None
    return float(v[0]), float(v[1])


def _stroke_segment(canvas: np.ndarray, a, b, radius: int) -> None:
    if a is None or b is None:
        return
    ax, ay = int(round(a[0])), int(round(a[1]))
    bx, by = int(round(b[0])), int(round(b[1]))
    cv2.line(canvas, (ax, ay), (bx, by), 255, thickness=max(2, radius * 2))
    cv2.circle(canvas, (ax, ay), radius, 255, thickness=-1)
    cv2.circle(canvas, (bx, by), radius, 255, thickness=-1)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 2:
        return (mask > 0).astype(np.uint8) * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


def _remove_small_components(mask: np.ndarray, min_area: int = 120) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return (mask > 0).astype(np.uint8) * 255
    keep = np.zeros(mask.shape, dtype=np.uint8)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            keep[labels == i] = 255
    return keep


def _round_mask_edges(mask: np.ndarray, radius: int = 9, sigma: float = 2.2) -> np.ndarray:
    if int(cv2.countNonZero(mask)) < 50:
        return mask
    radius = max(3, int(radius) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    rounded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    rounded = cv2.morphologyEx(rounded, cv2.MORPH_CLOSE, kernel, iterations=1)
    rounded = cv2.GaussianBlur(rounded, (0, 0), float(sigma))
    _, rounded = cv2.threshold(rounded, 88, 255, cv2.THRESH_BINARY)
    return _keep_largest_component(rounded)


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
    return _keep_largest_component(out)


def _build_long_sleeve_mask(pose: dict | None, shape: tuple[int, int]) -> np.ndarray:
    """Envelope around shoulder→elbow→wrist on both sides for long-sleeve dress.

    If pose only has keypoints for one side, mirror them to the other side so
    the silhouette still has both sleeves (occluded arms are common in flat
    studio shots).
    """
    out = np.zeros(shape, dtype=np.uint8)
    if not pose:
        return out
    ls = _pose_xy(pose, "left_shoulder")
    rs = _pose_xy(pose, "right_shoulder")
    le = _pose_xy(pose, "left_elbow")
    re = _pose_xy(pose, "right_elbow")
    lw = _pose_xy(pose, "left_wrist")
    rw = _pose_xy(pose, "right_wrist")

    cx = None
    if ls and rs:
        cx = (ls[0] + rs[0]) * 0.5
    elif ls:
        cx = ls[0]
    elif rs:
        cx = rs[0]

    def _mirror(p):
        if p is None or cx is None:
            return None
        return (2.0 * cx - p[0], p[1])

    if ls is None and rs is not None:
        ls = _mirror(rs)
    if rs is None and ls is not None:
        rs = _mirror(ls)
    if le is None and re is not None:
        le = _mirror(re)
    if re is None and le is not None:
        re = _mirror(le)
    if lw is None and rw is not None:
        lw = _mirror(rw)
    if rw is None and lw is not None:
        rw = _mirror(lw)

    shoulder_w = 60.0
    if ls and rs:
        shoulder_w = max(40.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    r_upper = max(14, int(shoulder_w * 0.22))
    r_lower = max(10, int(shoulder_w * 0.16))
    _stroke_segment(out, ls, le, r_upper)
    _stroke_segment(out, le, lw, r_lower)
    _stroke_segment(out, rs, re, r_upper)
    _stroke_segment(out, re, rw, r_lower)
    return cv2.GaussianBlur(out, (5, 5), 1.0)


def _build_short_sleeve_mask(pose: dict | None, shape: tuple[int, int]) -> np.ndarray:
    """Small cap/upper-arm envelope for short-sleeve dresses."""
    out = np.zeros(shape, dtype=np.uint8)
    if not pose:
        return out
    ls = _pose_xy(pose, "left_shoulder")
    rs = _pose_xy(pose, "right_shoulder")
    shoulder_w = 60.0
    if ls and rs:
        shoulder_w = max(40.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    r = max(9, int(shoulder_w * 0.13))
    for side in ("left", "right"):
        sh = _pose_xy(pose, f"{side}_shoulder")
        el = _pose_xy(pose, f"{side}_elbow")
        if sh is None:
            continue
        if el is None:
            cv2.circle(out, (int(round(sh[0])), int(round(sh[1]))), r, 255, thickness=-1)
            continue
        end = (
            sh[0] + (el[0] - sh[0]) * 0.38,
            sh[1] + (el[1] - sh[1]) * 0.38,
        )
        _stroke_segment(out, sh, end, r)
    return cv2.GaussianBlur(out, (5, 5), 1.0)


def _hand_only_mask(pose: dict | None, parsing: dict | None, shape: tuple[int, int]) -> np.ndarray:
    """A small half-disc beyond each wrist — protects the HAND only, never paints
    back into the forearm region (which would punch a circular hole inside the
    sleeve for long-sleeve dresses)."""
    out = np.zeros(shape, dtype=np.uint8)
    if not pose:
        return out
    ls = _pose_xy(pose, "left_shoulder")
    rs = _pose_xy(pose, "right_shoulder")
    shoulder_w = 60.0
    if ls and rs:
        shoulder_w = max(40.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
    # Shrink dramatically: 0.22 → 0.11 of shoulder width. A wrist disc that
    # reaches halfway up the forearm is what was creating the "circular gap"
    # in the sleeve when restore_occluders pasted bare-arm pixels back.
    r = max(8, int(shoulder_w * 0.11))
    for side in ("left", "right"):
        wrist = _pose_xy(pose, f"{side}_wrist")
        if wrist is None:
            continue
        elbow = _pose_xy(pose, f"{side}_elbow")
        wx, wy = int(round(wrist[0])), int(round(wrist[1]))
        if elbow is None:
            cv2.circle(out, (wx, wy), r, 255, thickness=-1)
            continue
        # Direction from elbow to wrist, then extend a short distance past the
        # wrist and draw a small disc. This keeps the protect region on the
        # HAND side of the wrist, never on the forearm side.
        dx = float(wrist[0] - elbow[0])
        dy = float(wrist[1] - elbow[1])
        n = max(1e-3, float(np.hypot(dx, dy)))
        ux, uy = dx / n, dy / n
        cx = int(round(wrist[0] + ux * r * 0.6))
        cy = int(round(wrist[1] + uy * r * 0.6))
        cv2.circle(out, (cx, cy), r, 255, thickness=-1)
    return out


def _split_hair(parsing: dict | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    hair = _safe(parsing, "hair", shape)
    if int(hair.sum()) == 0:
        zero = np.zeros(shape, dtype=np.uint8)
        return zero, zero
    face = _safe(parsing, "face", shape)
    if int(face.sum()) < 255 * 30:
        return hair.copy(), np.zeros(shape, dtype=np.uint8)
    face_neigh = cv2.dilate(face, np.ones((25, 25), np.uint8), iterations=2)
    hair_front = cv2.bitwise_and(hair, face_neigh)
    hair_under = cv2.subtract(hair, hair_front)
    return hair_front, hair_under


def _body_envelope(parsing: dict | None, pose: dict | None, shape: tuple[int, int]) -> np.ndarray:
    """Convex-hull body silhouette from parsing + pose sleeves, dilated.

    Used to clip the dress target so it never extends past the body outline
    (otherwise the dress looks detached and floats around the figure).

    Pose-derived sleeves are unioned in BEFORE the hull so a missing/occluded
    arm in parsing does not strip the dress sleeve away.
    """
    out = np.zeros(shape, dtype=np.uint8)
    body = np.zeros(shape, dtype=np.uint8)
    if parsing:
        for key in (
            "upper_clothes", "dress", "skirt", "pants", "belt",
            "left_arm", "right_arm", "left_leg", "right_leg",
            "face", "hair", "hat", "left_shoe", "right_shoe",
        ):
            body = cv2.bitwise_or(body, _safe(parsing, key, shape))
    sleeve_pose = _build_long_sleeve_mask(pose, shape)
    body = cv2.bitwise_or(body, sleeve_pose)
    if int(body.sum()) < 255 * 200:
        return out
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out
    pts = np.vstack(contours)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(out, hull, 255)
    out = cv2.dilate(out, np.ones((20, 20), np.uint8), iterations=1)
    return out


def build_dress_masks(
    target_silhouette: np.ndarray,
    parsing: dict | None,
    pose: dict | None,
    sleeve_type: str = "auto",
) -> DressMasks:
    """Assemble all masks the dress v2 pipeline needs from parsing + pose."""
    h, w = target_silhouette.shape[:2]
    shape = (h, w)
    target = (target_silhouette > 0).astype(np.uint8) * 255

    sleeve_type = (sleeve_type or "auto").lower()

    shoe_left = _safe(parsing, "left_shoe", shape)
    shoe_right = _safe(parsing, "right_shoe", shape)
    shoe = cv2.bitwise_or(shoe_left, shoe_right)
    shoe_protect = cv2.dilate(shoe, np.ones((9, 9), np.uint8), iterations=1)

    # Always union the long-sleeve pose envelope into target so diffusion
    # has room to paint sleeves even when sleeve detection mis-fires
    # (sleeveless/short detected on a long-sleeve reference dress).
    if sleeve_type in {"long", "auto"}:
        sleeve_target = _build_long_sleeve_mask(pose, shape)
        target = cv2.bitwise_or(target, sleeve_target)
    elif sleeve_type == "short":
        sleeve_target = _build_short_sleeve_mask(pose, shape)
        target = cv2.bitwise_or(target, sleeve_target)

    target = cv2.subtract(target, shoe_protect)

    silhouette_target = target.copy()

    # Clip only the upper torso/sleeve band to the body envelope. Clipping the
    # whole dress against the convex body hull makes the skirt collapse into the
    # gap between the legs, which encourages diffusion to draw a front slit.
    envelope = _body_envelope(parsing, pose, shape)
    if int(envelope.sum()) > 0:
        clip = cv2.dilate(envelope, np.ones((23, 23), np.uint8), iterations=1)
        upper_band = np.zeros(shape, dtype=np.uint8)
        ls = _pose_xy(pose, "left_shoulder")
        rs = _pose_xy(pose, "right_shoulder")
        lh = _pose_xy(pose, "left_hip")
        rh = _pose_xy(pose, "right_hip")
        if ls and rs and lh and rh:
            shoulder_w = max(40.0, float(np.hypot(ls[0] - rs[0], ls[1] - rs[1])))
            hip_y = int(round(max(lh[1], rh[1]) + shoulder_w * 0.20))
        else:
            ys = np.where(silhouette_target > 0)[0]
            hip_y = int(ys.min() + (ys.max() - ys.min()) * 0.42) if len(ys) else int(h * 0.48)
        hip_y = max(0, min(h - 1, hip_y))
        upper_band[:hip_y + 1, :] = 255
        upper_clipped = cv2.bitwise_and(target, clip)
        target = cv2.bitwise_or(
            cv2.bitwise_and(upper_clipped, upper_band),
            cv2.bitwise_and(silhouette_target, cv2.bitwise_not(upper_band)),
        )

    # Seal any vertical notch between the legs (parsing leaves a gap between
    # left_leg and right_leg; the convex hull keeps the gap closed but a later
    # erosion or AND with envelope can carve it open again). A wide horizontal
    # close kernel guarantees the dress reads as one continuous skirt.
    target = cv2.morphologyEx(target, cv2.MORPH_CLOSE, np.ones((31, 17), np.uint8), iterations=1)
    # One more horizontal seal across the lower skirt. This specifically closes
    # the leg valley in A-line/midi masks without inflating the dress vertically.
    lower_band = np.zeros(shape, dtype=np.uint8)
    ys = np.where(target > 0)[0]
    if len(ys):
        y1, y2 = int(ys.min()), int(ys.max())
        lower_band[int(y1 + (y2 - y1) * 0.48):y2 + 1, :] = 255
        lower = cv2.bitwise_and(target, lower_band)
        lower = cv2.morphologyEx(lower, cv2.MORPH_CLOSE, np.ones((41, 9), np.uint8), iterations=1)
        target = cv2.bitwise_or(cv2.bitwise_and(target, cv2.bitwise_not(lower_band)), lower)
    # Keep only the single largest connected component so disconnected satellite
    # bumps (from a mirrored sleeve landing far away) cannot bloom into extra
    # phantom dress layers during diffusion.
    target = _keep_largest_component(target)
    target = _round_mask_edges(target, radius=9, sigma=2.2)
    target = _round_hem_corners(target)

    old_clothes = np.zeros(shape, dtype=np.uint8)
    for k in ("dress", "skirt", "upper_clothes", "belt", "pants"):
        old_clothes = cv2.bitwise_or(old_clothes, _safe(parsing, k, shape))

    hair_front, hair_under = _split_hair(parsing, shape)

    face = _safe(parsing, "face", shape)
    face_protect = cv2.dilate(face, np.ones((3, 3), np.uint8), iterations=1)
    # Final alpha uses target_mask, not agnostic_mask. Remove face/front-hair
    # from target too so the neck/collar cannot blend dress pixels over the
    # chin or hairline after diffusion.
    target = cv2.subtract(target, face_protect)
    target = cv2.subtract(target, hair_front)
    target = _remove_small_components(target, min_area=max(80, int(h * w * 0.00025)))

    if sleeve_type == "long":
        # Long-sleeve dress: only protect HANDS, let diffusion paint sleeves
        # over the forearm.
        arm_outside = _hand_only_mask(pose, parsing, shape)
    elif sleeve_type == "auto":
        # Unknown sleeve: don't restore arm — let diffusion + cloth reference
        # decide sleeve length. Still protect hands.
        arm_outside = _hand_only_mask(pose, parsing, shape)
    elif sleeve_type == "sleeveless":
        arm = cv2.bitwise_or(_safe(parsing, "left_arm", shape), _safe(parsing, "right_arm", shape))
        arm_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        arm_outside = cv2.dilate(arm, arm_kernel, iterations=1)
    else:
        arm = cv2.bitwise_or(_safe(parsing, "left_arm", shape), _safe(parsing, "right_arm", shape))
        arm_outside = cv2.subtract(arm, cv2.dilate(target, np.ones((3, 3), np.uint8), iterations=1))

    agnostic = target.copy()
    agnostic = cv2.bitwise_or(agnostic, old_clothes)
    agnostic = cv2.bitwise_or(agnostic, hair_under)
    agnostic = cv2.subtract(agnostic, face_protect)
    agnostic = cv2.subtract(agnostic, hair_front)
    agnostic = cv2.subtract(agnostic, shoe_protect)
    agnostic = cv2.subtract(agnostic, arm_outside)

    agnostic = cv2.morphologyEx(agnostic, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    return DressMasks(
        target_mask=target,
        agnostic_mask=agnostic,
        hair_front_mask=hair_front,
        hair_underlap_mask=hair_under,
        shoe_protect_mask=shoe_protect,
        hand_protect_mask=arm_outside,
        old_clothes_mask=old_clothes,
    )
