"""Garment silhouette templates + detection (v18.19).

Replaces hardcoded width_curve heuristics in app.py with a small library of
named silhouette templates derived from standard pattern-making references
(ASTM D5585 anthropometric data, ISO 8559, ASOS/Zara size charts).

Each template stores half-width multipliers at fixed Y-fractions of the
garment height.  Multipliers are RELATIVE to the bust half-width sampled
from the source cloth mask (template[0.12] is always 1.00).  This makes
the templates invariant to overall garment scale — we only encode the
*shape*.

Detection: sample the source cloth_mask half-widths at the same fractions,
normalize by the bust width, then pick the template with the lowest sum
of squared differences.  Falls back to ``a_line`` when the source is too
small or asymmetric to classify confidently.
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Dress silhouettes
# Multipliers are half-width / bust half-width at each Y-fraction.
# Source data references:
#   - ASOS dress fit guides (a-line, bodycon, skater, shift)
#   - ASTM D5585 women's misses size chart (bust→waist→hip ratios)
#   - "Patternmaking for Fashion Design" (Helen Joseph-Armstrong, 5th ed.)
# Y-fractions:  0.00 shoulder | 0.12 bust | 0.30 waist | 0.50 hip
#               0.75 mid-skirt | 1.00 hem
# ──────────────────────────────────────────────────────────────────────
DRESS_TEMPLATES: dict[str, dict[float, float]] = {
    "sheath": {
        0.00: 0.92, 0.12: 1.00, 0.30: 0.96, 0.50: 1.00,
        0.75: 0.98, 1.00: 0.95,
    },
    "shift": {
        0.00: 0.95, 0.12: 1.00, 0.30: 1.00, 0.50: 1.05,
        0.75: 1.05, 1.00: 1.05,
    },
    "a_line": {
        0.00: 0.88, 0.12: 1.00, 0.30: 0.95, 0.50: 1.10,
        0.75: 1.28, 1.00: 1.45,
    },
    "fit_and_flare": {
        0.00: 0.92, 0.12: 1.00, 0.30: 0.78, 0.50: 1.05,
        0.75: 1.30, 1.00: 1.55,
    },
    "empire": {
        0.00: 0.92, 0.12: 1.00, 0.30: 1.05, 0.50: 1.15,
        0.75: 1.20, 1.00: 1.30,
    },
    "mermaid": {
        0.00: 0.92, 0.12: 1.00, 0.30: 0.88, 0.50: 1.00,
        0.75: 0.92, 1.00: 1.30,
    },
    "ball_gown": {
        0.00: 0.92, 0.12: 1.00, 0.30: 0.82, 0.50: 1.30,
        0.75: 1.70, 1.00: 2.00,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Top silhouettes (reserved for future use in app.py top path)
# Y-fractions: 0.00 shoulder | 0.18 chest | 0.50 waist | 1.00 hem
# ──────────────────────────────────────────────────────────────────────
TOP_TEMPLATES: dict[str, dict[float, float]] = {
    "fitted":    {0.00: 0.92, 0.18: 1.00, 0.50: 0.92, 1.00: 0.96},
    "regular":   {0.00: 0.92, 0.18: 1.00, 0.50: 1.00, 1.00: 1.05},
    "oversized": {0.00: 1.05, 0.18: 1.10, 0.50: 1.18, 1.00: 1.20},
    "peplum":    {0.00: 0.92, 0.18: 1.00, 0.50: 0.85, 1.00: 1.35},
}

SAMPLE_FRACS_DRESS = (0.00, 0.12, 0.30, 0.50, 0.75, 1.00)


def _row_width_px(mask: np.ndarray, y1: int, ch: int, rel: float) -> float:
    h_mask = mask.shape[0]
    row = max(0, min(h_mask - 1, y1 + int(ch * rel)))
    nz = np.where(mask[row] > 0)[0]
    return float(nz.max() - nz.min()) if len(nz) >= 4 else 0.0


def detect_dress_type(
    cloth_mask: np.ndarray,
    cloth_rgb: np.ndarray | None = None,
) -> dict[str, str]:
    """Detect full dress type metadata (v18.22).

    Returns dict with length, silhouette, sleeve, neckline.
    Pure-geometry classifier; drives prompt/mask tuning per dress type.
    """
    out = {
        "length": "midi",
        "silhouette": "a_line",
        "sleeve": "sleeveless",
        "neckline": "unknown",
    }
    ys, xs = np.where(cloth_mask > 0)
    if len(xs) < 80:
        return out

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    ch = max(1, y2 - y1)
    cw = max(1, x2 - x1)

    sil_name, _conf = detect_dress_silhouette(cloth_mask)

    bust_w = max(
        _row_width_px(cloth_mask, y1, ch, 0.18),
        _row_width_px(cloth_mask, y1, ch, 0.22),
        _row_width_px(cloth_mask, y1, ch, 0.26),
        _row_width_px(cloth_mask, y1, ch, 0.30),
    )
    waist_w = _row_width_px(cloth_mask, y1, ch, 0.45)
    hem_w = _row_width_px(cloth_mask, y1, ch, 0.92)
    mid_skirt = _row_width_px(cloth_mask, y1, ch, 0.65)

    # v18.30: silhouette confidence guard — sheath at conf 0.00 is wrong for
    # flared dresses. Re-classify from raw widths when detector is uncertain.
    if _conf < 0.30 and bust_w > 8:
        if hem_w > bust_w * 1.35 and waist_w < bust_w * 0.90:
            sil_name = "fit_and_flare"
        elif hem_w > bust_w * 1.20:
            sil_name = "a_line"
        elif abs(hem_w - bust_w) < bust_w * 0.12:
            sil_name = "sheath"
        else:
            sil_name = "a_line"
    out["silhouette"] = sil_name

    if bust_w > 8:
        # v18.30: use effective width (max of bust, waist*1.10, hem*0.62) so
        # fit-and-flare / skater dresses with a narrow bust don't read as maxi
        # just because total height / bust_w is large.
        effective_w = max(bust_w, waist_w * 1.10, hem_w * 0.62, 8.0)
        aspect = ch / effective_w
        if aspect < 1.85:
            out["length"] = "mini"
        elif aspect < 2.60:
            out["length"] = "midi"
        else:
            out["length"] = "maxi"
        # Flare downgrade — strongly flared hems are almost never maxi.
        if hem_w > bust_w * 1.45 and out["length"] == "maxi":
            out["length"] = "midi"
        if hem_w > bust_w * 1.60 and out["length"] == "midi":
            out["length"] = "mini"

    upper_widths_pos = [
        w for w in (_row_width_px(cloth_mask, y1, ch, f)
                    for f in (0.04, 0.06, 0.08, 0.10, 0.12))
        if w > 0
    ]
    max_upper = max(upper_widths_pos) if upper_widths_pos else 0.0
    shoulder_top = _row_width_px(cloth_mask, y1, ch, 0.02)
    # waist_w / hem_w / mid_skirt already computed above (v18.30)

    sleeve = "sleeveless"
    if bust_w > 8:
        if (
            shoulder_top > 0
            and shoulder_top < bust_w * 0.55
            and max_upper < bust_w * 0.95
        ):
            sleeve = "sleeveless"
        elif hem_w > bust_w * 1.28 and max_upper < bust_w * 1.10:
            sleeve = "sleeveless"
        elif hem_w < bust_w * 1.12 and max_upper >= bust_w * 0.92:
            if waist_w >= bust_w * 0.85 and mid_skirt <= bust_w * 1.10:
                sleeve = "long"
            else:
                sleeve = "short"
        elif max_upper >= bust_w * 1.05:
            mid_lower = _row_width_px(cloth_mask, y1, ch, 0.33)
            if mid_lower > bust_w * 1.05 and max_upper > bust_w * 1.12:
                sleeve = "long"
            else:
                sleeve = "short"
    out["sleeve"] = sleeve

    neck_band_top = max(0, y1 + int(ch * 0.02))
    neck_band_bot = min(cloth_mask.shape[0] - 1, y1 + int(ch * 0.10))
    if neck_band_bot > neck_band_top + 2 and bust_w > 8:
        cx_lo = max(0, x1 + int(cw * 0.30))
        cx_hi = min(cloth_mask.shape[1] - 1, x1 + int(cw * 0.70))
        step = max(1, (cx_hi - cx_lo) // 32)
        col_tops = []
        for cx in range(cx_lo, cx_hi + 1, step):
            col = cloth_mask[neck_band_top:neck_band_bot + 1, cx]
            nz = np.where(col > 0)[0]
            if len(nz) > 0:
                col_tops.append(neck_band_top + int(nz[0]))
        if len(col_tops) >= 8:
            col_tops_np = np.array(col_tops, dtype=np.float32)
            depth = float(col_tops_np.max() - col_tops_np.min())
            mid_idx = len(col_tops_np) // 2
            mid_depth = float(col_tops_np[mid_idx] - col_tops_np.min())
            depth_ratio = mid_depth / max(1.0, depth)
            if depth < bust_w * 0.05:
                out["neckline"] = "square"
            elif depth_ratio > 0.78 and depth > bust_w * 0.18:
                out["neckline"] = "vneck"
            elif depth < bust_w * 0.14:
                out["neckline"] = "round"
            else:
                out["neckline"] = "vneck" if depth_ratio > 0.65 else "round"
    return out


def build_dress_prompt(dress_type: dict[str, str], color_anchor: str) -> str:
    """Build a dress prompt that names each axis explicitly so SD inpaint
    follows the detected length/silhouette/sleeve/neckline instead of
    falling back to a generic ``a-line midi`` default.
    """
    sil = dress_type.get("silhouette", "a_line")
    sleeve = dress_type.get("sleeve", "sleeveless")
    length = dress_type.get("length", "midi")
    neck = dress_type.get("neckline", "unknown")

    sil_phrase = {
        "sheath": "fitted sheath dress with straight hem",
        "shift": "loose shift dress",
        "a_line": "a-line dress with flared skirt",
        "fit_and_flare": "fit-and-flare dress with cinched waist and flared skirt",
        "empire": "empire-waist dress",
        "mermaid": "mermaid dress fitted through hip with flared hem",
        "ball_gown": "ball gown dress with full skirt",
    }.get(sil, "a-line dress")

    sleeve_phrase = {
        "sleeveless": "sleeveless, bare shoulders",
        "short": "short sleeves at upper arm",
        "long": "long sleeves to the wrist",
    }.get(sleeve, "")

    length_phrase = {
        "mini": "mini length above the knee",
        "midi": "midi length below the knee",
        "maxi": "maxi length to the ankle",
    }.get(length, "")

    neck_phrase = {
        "round": "round crew neckline",
        "vneck": "v-neckline",
        "square": "square neckline",
        "halter": "halter neckline",
    }.get(neck, "")

    parts = [
        f"single {sil_phrase} on the person",
        length_phrase,
        sleeve_phrase,
        neck_phrase,
        f"preserve the exact original {color_anchor} print layout from the reference garment",
        "natural body-following fold shadows, soft fabric drape",
        "do not redraw or reinterpret the print, no grey cast, not faded",
        "no scarf, no shawl, no cardigan, no coat",
        "clean shoulders, crisp hem, realistic fabric lighting",
    ]
    return ", ".join(p for p in parts if p)


def _sample_half_widths(
    mask: np.ndarray, fracs: tuple[float, ...]
) -> list[float] | None:
    """Sample mask half-widths at fractional Y positions of the bounding box."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 200:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    ch = max(1, y2 - y1)
    h = mask.shape[0]

    out: list[float] = []
    for f in fracs:
        row = max(0, min(h - 1, y1 + int(ch * f)))
        nz = np.where(mask[row] > 0)[0]
        if len(nz) >= 4:
            out.append(float(nz.max() - nz.min()) * 0.5)
        else:
            out.append(0.0)
    return out


def detect_dress_silhouette(cloth_mask: np.ndarray) -> tuple[str, float]:
    """Match a cloth mask to the closest dress silhouette template.

    Returns ``(name, confidence)``.  Confidence is in [0, 1] — 1 means a
    perfect template match, 0 means the source could not be classified.
    Callers should treat confidence < 0.35 as "uncertain — use default
    a_line".
    """
    halves = _sample_half_widths(cloth_mask, SAMPLE_FRACS_DRESS)
    if halves is None:
        return "a_line", 0.0

    bust_h = halves[1]
    if bust_h < 6.0:
        return "a_line", 0.0

    norm = [h / bust_h for h in halves]

    best_name = "a_line"
    best_err = float("inf")
    for name, tmpl in DRESS_TEMPLATES.items():
        tmpl_vec = [tmpl[f] for f in SAMPLE_FRACS_DRESS]
        # Skip shoulder point in error (templates often disagree there because
        # of strap geometry; that point is more about neckline than silhouette).
        err = sum((norm[i] - tmpl_vec[i]) ** 2 for i in range(1, len(norm)))
        if err < best_err:
            best_err = err
            best_name = name

    # err ranges roughly 0.0 (exact match) to ~1.0 (very different).
    confidence = float(max(0.0, 1.0 - best_err))
    return best_name, confidence


def build_dress_width_curve(
    silhouette: str,
    bust_half_width: float,
    sw: float,
    hip_w: float,
) -> np.ndarray:
    """Translate a silhouette template into an Nx2 width_curve.

    Args:
        silhouette: name in DRESS_TEMPLATES (falls back to "a_line").
        bust_half_width: bust half-width sampled from source cloth_mask.
        sw: model shoulder width (px).
        hip_w: model hip width (px).

    Returns:
        Nx2 numpy array of (Y-fraction, half-width-in-px) usable with
        ``np.interp``.  The half-width is clamped to a sane range relative
        to the body anthropometry so a wildly off cloth mask cannot push
        the dress envelope past the arms.
    """
    tmpl = DRESS_TEMPLATES.get(silhouette, DRESS_TEMPLATES["a_line"])

    # Anchor: the bust half-width in image space should sit near sw * 0.55
    # (typical bust-to-shoulder ratio in ASTM D5585 misses sizing). Trust
    # the source bust width only when it lies in [sw*0.42, sw*0.66] — clamp
    # otherwise to absorb TPS scale errors.
    anchor = float(np.clip(bust_half_width, sw * 0.42, sw * 0.66))

    rows = []
    for f in SAMPLE_FRACS_DRESS:
        mult = tmpl[f]
        half_px = anchor * mult
        # Per-fraction safety floor/ceiling tied to body width so we don't
        # paint outside the arms.
        if f <= 0.30:
            ceil_px = sw * 0.74
        elif f <= 0.50:
            ceil_px = max(sw * 0.74, hip_w * 0.78)
        elif f <= 0.75:
            ceil_px = max(sw * 0.88, hip_w * 1.05)
        else:
            ceil_px = max(sw * 1.05, hip_w * 1.40)
        floor_px = sw * 0.42
        half_px = float(np.clip(half_px, floor_px, ceil_px))
        rows.append([f, half_px])
    return np.array(rows, dtype=np.float32)
