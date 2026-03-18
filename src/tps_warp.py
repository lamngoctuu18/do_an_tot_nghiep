"""Thin-Plate-Spline (TPS) cloth warping.

Warps a flat garment image so its shape conforms to the person's body,
using control-point correspondences between cloth landmarks and body
landmarks derived from pose estimation.

Key design: we use exactly 9 paired landmarks (collar, shoulder, armpit,
mid-torso, hem — left and right) plus 2 sleeve-tip landmarks mapped to
elbow positions = 11 clean control points.
No overlapping grid to avoid TPS "explosion".
The sleeve-tip → elbow mapping ensures long-sleeve garments are not
compressed into the torso width (prevents crop-top / bra artefacts).
"""
from __future__ import annotations

import cv2
import numpy as np


# ── TPS math ────────────────────────────────────────────────────────────

def _solve_tps(dst_pts: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """Solve TPS coefficients for inverse mapping dst → src.

    dst_pts : Nx2  –  control points in output (person) space
    src_pts : Nx2  –  corresponding points in source (cloth) space
    Returns : (N+3)x2 coefficient matrix
    """
    n = len(dst_pts)
    diff = dst_pts[:, None, :] - dst_pts[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K = np.zeros_like(r2)
    pos = r2 > 0
    K[pos] = r2[pos] * np.log(r2[pos] + 1e-20)

    P = np.hstack([np.ones((n, 1)), dst_pts])

    # Higher regularisation (1e-2) prevents TPS from overfitting when
    # landmarks are noisy — avoids the "explosion" / hole artefacts.
    L = np.zeros((n + 3, n + 3), dtype=np.float64)
    L[:n, :n] = K + np.eye(n) * 5e-2
    L[:n, n:] = P
    L[n:, :n] = P.T

    rhs = np.zeros((n + 3, 2), dtype=np.float64)
    rhs[:n] = src_pts
    return np.linalg.solve(L, rhs)


def _apply_tps(
    coeffs: np.ndarray,
    dst_pts: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Map *query* points (in output space) back to source space."""
    n = len(dst_pts)
    diff = query[:, None, :] - dst_pts[None, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    K = np.zeros_like(r2)
    pos = r2 > 0
    K[pos] = r2[pos] * np.log(r2[pos] + 1e-20)

    P = np.hstack([np.ones((len(query), 1)), query])
    return K @ coeffs[:n] + P @ coeffs[n:]


# ── Cloth landmark detection ────────────────────────────────────────────

def detect_cloth_landmarks(cloth_mask: np.ndarray) -> dict[str, tuple[float, float]]:
    """Detect 11 landmarks on the garment from its mask silhouette.

    Returns: collar, shoulder_l/r, armpit_l/r, mid_l/r, hem_l/r,
             sleeve_tip_l/r (widest-row extremes for long-sleeve support).
    Order: top-to-bottom, left before right at each row.
    """
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        h, w = cloth_mask.shape[:2]
        return {
            "collar":           (w * 0.50, h * 0.05),
            "shoulder_left":    (w * 0.15, h * 0.12),
            "shoulder_right":   (w * 0.85, h * 0.12),
            "armpit_left":      (w * 0.12, h * 0.30),
            "armpit_right":     (w * 0.88, h * 0.30),
            "mid_left":         (w * 0.18, h * 0.55),
            "mid_right":        (w * 0.82, h * 0.55),
            "hem_left":         (w * 0.20, h * 0.92),
            "hem_right":        (w * 0.80, h * 0.92),
            "sleeve_tip_left":  (w * 0.05, h * 0.30),
            "sleeve_tip_right": (w * 0.95, h * 0.30),
        }

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    ch = max(1, y2 - y1)

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

    shl, shr = _row_lr(0.12)
    apl, apr = _row_lr(0.30)
    mdl, mdr = _row_lr(0.55)
    hml, hmr = _row_lr(0.88)

    # Sleeve tips: find the widest row (captures sleeve extent for long-sleeve shirts).
    # We scan the upper portion of the garment (20–45% height) where sleeves are
    # widest. For t-shirts the result will be equal to or very near the armpit row.
    max_width = 0.0
    sleeve_frac = 0.30
    for frac in np.arange(0.20, 0.50, 0.05):   # 0.20, 0.25, 0.30, 0.35, 0.40, 0.45
        sl, sr = _row_lr(float(frac))
        w_at_row = sr - sl
        if w_at_row > max_width:
            max_width = w_at_row
            sleeve_frac = float(frac)
    stl, str_ = _row_lr(sleeve_frac)

    return {
        "collar":           (collar_x,                   collar_y),
        "shoulder_left":    (shl, float(y1 + int(ch * 0.12))),
        "shoulder_right":   (shr, float(y1 + int(ch * 0.12))),
        "armpit_left":      (apl, float(y1 + int(ch * 0.30))),
        "armpit_right":     (apr, float(y1 + int(ch * 0.30))),
        "mid_left":         (mdl, float(y1 + int(ch * 0.55))),
        "mid_right":        (mdr, float(y1 + int(ch * 0.55))),
        "hem_left":         (hml, float(y1 + int(ch * 0.88))),
        "hem_right":        (hmr, float(y1 + int(ch * 0.88))),
        "sleeve_tip_left":  (stl, float(y1 + int(ch * sleeve_frac))),
        "sleeve_tip_right": (str_, float(y1 + int(ch * sleeve_frac))),
    }


# ── Build body-side destination points ──────────────────────────────────

def _compute_body_destinations(
    pose: dict[str, tuple[int, int]],
    fit_scale: float,
    y_offset_ratio: float,
) -> np.ndarray:
    """Compute 11 body-side destination points matching cloth landmarks.

    Order must be identical to detect_cloth_landmarks():
      collar, shoulder_l, shoulder_r, armpit_l, armpit_r,
      mid_l, mid_r, hem_l, hem_r, sleeve_tip_l, sleeve_tip_r.

    The two sleeve_tip points map to elbow positions so that
    long-sleeve garments are not compressed into the torso width.
    """
    ls = np.array(pose["left_shoulder"], dtype=np.float64)
    rs = np.array(pose["right_shoulder"], dtype=np.float64)
    lh = np.array(pose["left_hip"], dtype=np.float64)
    rh = np.array(pose["right_hip"], dtype=np.float64)
    le = np.array(pose.get("left_elbow", pose["left_shoulder"]), dtype=np.float64)
    re = np.array(pose.get("right_elbow", pose["right_shoulder"]), dtype=np.float64)

    sw = float(np.linalg.norm(ls - rs))
    hw = float(np.linalg.norm(lh - rh))
    dy = y_offset_ratio * 30.0

    # Neck / collar: midpoint above shoulders
    neck = (ls + rs) / 2.0
    neck[1] -= max(10.0, sw * 0.10)

    # Vertical interpolation helpers
    torso_top_y = min(ls[1], rs[1])
    torso_bot_y = max(lh[1], rh[1])
    torso_h = max(1.0, torso_bot_y - torso_top_y)

    def _lerp_y(frac: float) -> float:
        return torso_top_y + torso_h * frac + dy

    # Center X at each level
    cx_top = (ls[0] + rs[0]) / 2.0
    cx_bot = (lh[0] + rh[0]) / 2.0
    cx_mid = (cx_top + cx_bot) / 2.0

    # Garment Shape Constraint: half-widths at each level use the
    # wider of shoulder/hip so fabric never pinches inward.
    base_w = max(sw, hw) * fit_scale
    hw_shoulder = base_w * 0.56
    hw_armpit   = base_w * 0.58          # wider than shoulder for armholes
    hw_mid      = base_w * 0.54          # chest/waist keeps garment loose
    hw_hem      = base_w * 0.52

    # Sleeve tip destinations: map to elbow positions so that
    # long-sleeve garments are not squashed into the torso width.
    # We take the LEFTMOST X of (elbow X, armpit boundary) for the left
    # destination, and the RIGHTMOST X for the right destination.
    # ┌ Arms at sides:   elbow X ≈ shoulder X (rightward of armpit bound) →
    #   armpit boundary wins → same effective width as the armpit destination.
    # ┌ Arms spread wide: elbow X extends beyond armpit bound →
    #   elbow wins → sleeve tip destination pushes further out, preserving
    #   the sleeve shape against the actual arm position.
    sleeve_tip_l_x = min(float(le[0]), cx_top - hw_armpit)   # take the more-left X
    sleeve_tip_r_x = max(float(re[0]), cx_top + hw_armpit)   # take the more-right X
    sleeve_tip_y_l = float(le[1]) + dy
    sleeve_tip_y_r = float(re[1]) + dy

    return np.array([
        [neck[0],               neck[1] + dy],                # collar
        [cx_top - hw_shoulder,  _lerp_y(0.0)],               # shoulder_left
        [cx_top + hw_shoulder,  _lerp_y(0.0)],               # shoulder_right
        [cx_top - hw_armpit,    _lerp_y(0.25)],              # armpit_left
        [cx_top + hw_armpit,    _lerp_y(0.25)],              # armpit_right
        [cx_mid - hw_mid,       _lerp_y(0.55)],              # mid_left
        [cx_mid + hw_mid,       _lerp_y(0.55)],              # mid_right
        [cx_bot - hw_hem,       _lerp_y(0.95)],              # hem_left
        [cx_bot + hw_hem,       _lerp_y(0.95)],              # hem_right
        [sleeve_tip_l_x,        sleeve_tip_y_l],             # sleeve_tip_left  → left elbow
        [sleeve_tip_r_x,        sleeve_tip_y_r],             # sleeve_tip_right → right elbow
    ], dtype=np.float64)


# ── Public API ──────────────────────────────────────────────────────────

def tps_warp_cloth(
    cloth_rgb: np.ndarray,
    cloth_mask: np.ndarray,
    pose: dict[str, tuple[int, int]],
    output_shape: tuple[int, int],
    fit_scale: float = 1.12,
    y_offset_ratio: float = 0.0,
    grid_step: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp cloth image via TPS so it conforms to the body shape.

    Uses 11 clean landmark pairs (no overlapping grid) to avoid TPS
    distortion artefacts.  The extra 2 points (sleeve_tip_l/r → elbow)
    ensure long-sleeve garments are not compressed into the torso width.
    """
    # Pre-multiply cloth by mask BEFORE warp to eliminate white
    # background bleeding through anti-aliased mask edges.
    mask_f = (cloth_mask.astype(np.float32) / 255.0)[..., None]
    cloth_rgb = (cloth_rgb.astype(np.float32) * mask_f).clip(0, 255).astype(np.uint8)

    landmarks = detect_cloth_landmarks(cloth_mask)

    # Source: cloth landmarks (cloth image space) — 11 points
    src_pts = np.array([
        landmarks["collar"],
        landmarks["shoulder_left"],
        landmarks["shoulder_right"],
        landmarks["armpit_left"],
        landmarks["armpit_right"],
        landmarks["mid_left"],
        landmarks["mid_right"],
        landmarks["hem_left"],
        landmarks["hem_right"],
        landmarks.get("sleeve_tip_left",  landmarks["armpit_left"]),
        landmarks.get("sleeve_tip_right", landmarks["armpit_right"]),
    ], dtype=np.float64)

    # Destination: body landmarks (person image space) — 11 points
    dst_pts = _compute_body_destinations(pose, fit_scale, y_offset_ratio)

    h_out, w_out = output_shape

    # ── Affine pre-alignment (coarse) ──────────────────────────────
    # Map collar + shoulders from cloth space to body space.
    # Handles position, scale, and rotation so TPS only does fine local
    # deformation — prevents shoulder puffing and garment explosion.
    affine_src = src_pts[:3].astype(np.float32)   # collar, shoulder_l, shoulder_r
    affine_dst = dst_pts[:3].astype(np.float32)
    M_affine = cv2.getAffineTransform(affine_src, affine_dst)

    aligned_cloth = cv2.warpAffine(
        cloth_rgb, M_affine, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    aligned_mask = cv2.warpAffine(
        cloth_mask, M_affine, (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Transform all 11 source landmarks through affine → now in person space
    ones_col = np.ones((len(src_pts), 1), dtype=np.float64)
    src_hom = np.hstack([src_pts, ones_col])
    M64 = M_affine.astype(np.float64)
    src_affine = (M64 @ src_hom.T).T                  # Nx2

    # ── TPS refinement (residual deformation only) ─────────────────
    coeffs = _solve_tps(dst_pts, src_affine)

    # Build a coarse-grid query and upsample for speed
    gy, gx = np.mgrid[0:h_out:grid_step, 0:w_out:grid_step]
    pts = np.column_stack([gx.ravel().astype(np.float64),
                           gy.ravel().astype(np.float64)])
    mapped = _apply_tps(coeffs, dst_pts, pts)

    mx = mapped[:, 0].reshape(gy.shape).astype(np.float32)
    my = mapped[:, 1].reshape(gy.shape).astype(np.float32)

    mx = cv2.resize(mx, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
    my = cv2.resize(my, (w_out, h_out), interpolation=cv2.INTER_LINEAR)

    warped_cloth = cv2.remap(
        aligned_cloth, mx, my, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    warped_mask = cv2.remap(
        aligned_mask, mx, my, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Refine mask: dilate → erode → feather (HR-VITON style)
    warped_mask = refine_warped_mask(warped_mask)

    return warped_cloth, warped_mask


def refine_warped_mask(mask: np.ndarray) -> np.ndarray:
    """Clean up warped mask edges to prevent white halo and improve adhesion.

    Pipeline from HR-VITON / IDM-VTON:
      1. Dilate to fill tiny internal holes after warp and extend coverage
         so the garment adheres to the body without visible gaps
      2. Erode slightly to remove anti-aliased fringe at the boundary
         (cloth is pre-multiplied by mask so no white bleed, only dark fringe)
      3. Light Gaussian feather for smooth compositing edge
    Net effect: +1 dilation ensures garment mask fully covers the warped cloth.
    """
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, k3, iterations=2)   # fill holes + slight outward expansion
    mask = cv2.erode(mask, k3, iterations=1)    # remove dark anti-alias fringe
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask
