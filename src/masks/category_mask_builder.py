"""Category-specific tryon mask builder.

Extracted 1:1 from `_build_human_tryon_prior_mask` in app.py so each
category (top / pants / dress / accessory) owns its allow/protect rules.

Helper functions (`parsing_union_mask`, `pose_envelope`, `neck_mask_fn`,
`fit_like`) are INJECTED to avoid circular imports — app.py keeps the
originals and just passes them in.

Public API:
    build_category_mask(category, ...) -> (mask, ok)

Dispatches to:
    build_top_mask
    build_pants_mask
    build_dress_mask
    build_accessory_mask
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from src.category_lock import apply_category_lock


# ---- Allow/Protect rules per category --------------------------------------

TOP_SEMANTIC = ("upper_clothes", "dress", "belt", "scarf", "left_arm", "right_arm")
TOP_PROTECT = (
    "face", "hair", "hat", "sunglasses", "pants", "skirt",
    "left_leg", "right_leg", "left_shoe", "right_shoe",
)

PANTS_SEMANTIC = ("pants", "skirt", "left_leg", "right_leg", "belt")
PANTS_PROTECT = (
    "face", "hair", "hat", "sunglasses", "upper_clothes",
    "dress", "left_arm", "right_arm",
)

DRESS_SEMANTIC_SLEEVED = (
    "upper_clothes", "dress", "skirt", "pants", "belt", "scarf",
    "left_leg", "right_leg", "left_arm", "right_arm",
)
DRESS_PROTECT_SLEEVED = (
    "face", "hair", "hat", "sunglasses", "left_shoe", "right_shoe",
)

DRESS_SEMANTIC_SLEEVELESS = (
    "upper_clothes", "dress", "skirt", "pants", "belt", "scarf",
    "left_leg", "right_leg",
)
DRESS_PROTECT_SLEEVELESS = (
    "face", "hair", "hat", "sunglasses",
    "left_shoe", "right_shoe", "left_arm", "right_arm",
)

ACCESSORY_PROTECT = (
    "face", "hair", "upper_clothes", "dress", "skirt", "pants",
    "left_arm", "right_arm", "left_leg", "right_leg",
    "left_shoe", "right_shoe",
)

SILHOUETTE_KEYS = (
    "hat", "hair", "sunglasses", "upper_clothes", "skirt", "pants",
    "dress", "belt", "left_shoe", "right_shoe", "face", "left_leg",
    "right_leg", "left_arm", "right_arm", "bag", "scarf",
)


# ---- Core builder ----------------------------------------------------------

def _common_compose(
    garment_mask: np.ndarray,
    parsing: Optional[dict],
    full_pose: Optional[dict],
    *,
    semantic_keys: tuple,
    protect_keys: tuple,
    category: str,
    parsing_union_mask: Callable,
    pose_envelope_fn: Callable,
    neck_mask_fn: Optional[Callable],
    fit_like: Callable,
    subtype: str = "",
    extra_allow_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """Shared composition: semantic ∪ envelope ∪ garment-support − protect.

    Mirrors `_build_human_tryon_prior_mask` exactly so behaviour is preserved.
    """
    garment_mask = (garment_mask > 20).astype(np.uint8) * 255
    h, w = garment_mask.shape[:2]
    shape = (h, w)

    semantic_mask = parsing_union_mask(parsing, semantic_keys, shape)
    silhouette = parsing_union_mask(parsing, SILHOUETTE_KEYS, shape)
    envelope = pose_envelope_fn(shape, full_pose, category, garment_mask)

    if int(silhouette.sum()) > 255 * 300:
        silhouette = cv2.morphologyEx(
            silhouette, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=1,
        )
        silhouette = cv2.dilate(silhouette, np.ones((5, 5), np.uint8), iterations=1)
        envelope = cv2.bitwise_and(envelope, silhouette)

    # Hoodie subtype: the pose envelope (torso half-width = 0.72×shoulder,
    # arm thickness = 0.13×shoulder) is generous and produces a baggy hoodie
    # silhouette with wide sleeves. Constrain the envelope to a tight band
    # around the TPS-warped garment so the final mask hugs the body and lets
    # diffusion paint natural folds inside the warped silhouette instead of
    # inventing fabric outside it.
    support = cv2.dilate(garment_mask, np.ones((7, 7), np.uint8), iterations=1)
    if category == "top" and subtype == "jacket":
        # Extend support downward so the ribbed hem + side pockets band that
        # the TPS warp tends to clip is still inside the diffusion mask.
        # Drop 5% of frame height below the lowest warp pixel (capped so we
        # don't reach the knees on long-frame inputs).
        ys_g, _ = np.where(garment_mask > 20)
        if len(ys_g) > 100:
            hem_extend = int(h * 0.06)
            hem_y = min(h - 1, int(ys_g.max()) + hem_extend)
            xs_g = np.where(garment_mask > 20)[1]
            x_lo = max(0, int(xs_g.min()) - 6)
            x_hi = min(w - 1, int(xs_g.max()) + 6)
            cv2.rectangle(support, (x_lo, int(ys_g.max())), (x_hi, hem_y), 255, thickness=-1)
        # Light close to seal any small gaps along the hem band.
        support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    if category == "top" and subtype == "hoodie":
        # Drop left_arm/right_arm from the parsing union — they cover the bare
        # arm well beyond the actual sleeve and inflate the shoulder/sleeve
        # mask, which is exactly what produces the "extra material at shoulder
        # and arm" artifact even with a good Gemini prompt.
        torso_keys = tuple(k for k in semantic_keys if k not in {"left_arm", "right_arm"})
        semantic_mask = parsing_union_mask(parsing, torso_keys, shape)
        # Clamp the wide pose envelope to a 5px dilation of the warped garment
        # so the mask cannot extend past the real garment silhouette.
        tight = cv2.dilate(garment_mask, np.ones((5, 5), np.uint8), iterations=1)
        envelope = cv2.bitwise_and(envelope, tight)
        # Smooth sleeve kinks: an opening before union erases thin TPS-warp
        # bumps along the arm so diffusion doesn't paint a "bent" sleeve.
        smooth_garment = cv2.morphologyEx(
            garment_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1,
        )
        support = cv2.dilate(smooth_garment, np.ones((3, 3), np.uint8), iterations=1)
    human_prior = cv2.bitwise_or(cv2.bitwise_or(semantic_mask, envelope), support)

    if extra_allow_mask is not None and int(cv2.countNonZero(extra_allow_mask)) > 0:
        human_prior = cv2.bitwise_or(human_prior, extra_allow_mask)

    if neck_mask_fn is not None and parsing is not None and category != "pants":
        # Hoodie: skip the neck-fill helper. It unions a wedge from chin → collar
        # which collides with the hood drape (`extra_allow_mask`) and forces
        # diffusion to paint a separate crew-neck collar under the hood, leaving
        # a hard ring around the neckline.
        if not (category == "top" and subtype == "hoodie"):
            neck = neck_mask_fn(parsing)
            if neck is not None:
                human_prior = cv2.bitwise_or(human_prior, fit_like(neck, human_prior, is_mask=True))

    protect_mask = parsing_union_mask(parsing, protect_keys, shape)
    if int(protect_mask.sum()) > 255 * 20:
        protect_mask = cv2.dilate(protect_mask, np.ones((5, 5), np.uint8), iterations=1)
        # Jacket subtype: a bomber jacket's ribbed hem + side pockets sit BELOW
        # the waist, overlapping the parsing "pants" region near the top of the
        # hip. Subtracting the dilated pants protect there clips the hem and
        # erases the pockets in the final result. Keep the protect mask, but
        # carve back the area that still lives INSIDE the warped garment so the
        # hem zone survives. This applies only to jacket, not generic top.
        if category == "top" and subtype == "jacket":
            jacket_keep = cv2.dilate(garment_mask, np.ones((9, 9), np.uint8), iterations=1)
            protect_mask = cv2.subtract(protect_mask, jacket_keep)
        human_prior = cv2.subtract(human_prior, protect_mask)

    if category == "dress":
        ys, _ = np.where(support > 20)
        if len(ys) > 100:
            hem_limit = min(h - 1, int(ys.max() + max(8, h * 0.018)))
            human_prior[hem_limit + 1:, :] = 0

    # Hoodie subtype: keep the mask tight to the garment support so the
    # generated hoodie hugs the body silhouette instead of inflating outward
    # (puffy oversized look) — and so the kangaroo pocket area isn't smoothed
    # over by a wider diffusion region.
    if category == "top" and subtype == "hoodie":
        human_prior = cv2.morphologyEx(
            human_prior, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1,
        )
        human_prior = cv2.dilate(human_prior, np.ones((3, 3), np.uint8), iterations=1)
        # Final hard clamp: nothing outside the warped garment + 11px slack.
        # Guarantees the mask cannot grow past the cloth silhouette regardless
        # of upstream envelope/silhouette unions.
        hoodie_cap = cv2.dilate(garment_mask, np.ones((7, 7), np.uint8), iterations=1)
        human_prior = cv2.bitwise_and(human_prior, hoodie_cap)
    else:
        human_prior = cv2.morphologyEx(
            human_prior, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1,
        )
        human_prior = cv2.dilate(human_prior, np.ones((5, 5), np.uint8), iterations=1)
    human_prior = (human_prior > 20).astype(np.uint8) * 255

    # v19.40: hard category lock — strict forbid pass on top of the softer
    # protect logic above. Stops cross-category bleed (e.g. pants diffusion
    # touching torso shirt) even when envelope/silhouette would allow it.
    human_prior = apply_category_lock(
        human_prior, category, parsing,
        parsing_union_mask=parsing_union_mask,
        fit_like=fit_like,
        garment_mask=garment_mask,
        subtype=subtype,
    )

    prior_area = int(cv2.countNonZero(human_prior))
    garment_area = int(cv2.countNonZero(garment_mask))
    if prior_area < max(500, int(garment_area * 0.65)):
        return garment_mask, False
    if prior_area > int(h * w * 0.72):
        return garment_mask, False
    return human_prior, True


# ---- Category-specific entry points ----------------------------------------

def _hood_extension_mask(
    parsing: Optional[dict],
    full_pose: Optional[dict],
    shape: Tuple[int, int],
    parsing_union_mask: Callable,
) -> np.ndarray:
    """Allow region behind/around the head where a hoodie hood can drape.

    Built from (a) the parsed hair/face bounding box, dilated outward, and
    (b) an ellipse sitting on the shoulder line that reaches up past the head.
    Combined with the existing semantic/envelope union, this gives the
    diffusion mask room to paint a hood — the legacy "top" mask never went
    above the shoulders so the hood was impossible to generate.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)

    head_box = None
    if parsing is not None:
        head = parsing_union_mask(parsing, ("face", "hair", "hat"), shape)
        if int(cv2.countNonZero(head)) > 255 * 30:
            ys, xs = np.where(head > 0)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            head_box = (x1, y1, x2, y2)
            pad_x = max(12, int((x2 - x1) * 0.55))
            pad_top = max(20, int((y2 - y1) * 0.45))
            pad_bot = max(16, int((y2 - y1) * 0.35))
            cv2.rectangle(
                out,
                (max(0, x1 - pad_x), max(0, y1 - pad_top)),
                (min(w - 1, x2 + pad_x), min(h - 1, y2 + pad_bot)),
                255, thickness=-1,
            )

    if full_pose:
        ls = full_pose.get("left_shoulder")
        rs = full_pose.get("right_shoulder")
        if ls is not None and rs is not None:
            sx = (float(ls[0]) + float(rs[0])) * 0.5
            sy = (float(ls[1]) + float(rs[1])) * 0.5
            sw = max(40.0, float(np.hypot(float(ls[0]) - float(rs[0]), float(ls[1]) - float(rs[1]))))
            # Tight hood ellipse: sits ABOVE the shoulder line and narrower
            # than the shoulders themselves so it cannot bleed sideways and
            # let SD paint an extra sleeve/arm next to the real shoulder.
            cx, cy = int(round(sx)), int(round(sy - sw * 0.70))
            ax = int(round(sw * 0.60))
            ay = int(round(sw * 0.95))
            cv2.ellipse(out, (cx, cy), (ax, ay), 0, 0, 360, 255, thickness=-1)
            # Clear anything that drifted at/below the shoulder line so the
            # hood extension never overlaps the shoulder/arm band.
            shoulder_y = int(round(sy - sw * 0.05))
            if 0 < shoulder_y < h:
                out[shoulder_y:, :] = 0

    if head_box is None and int(cv2.countNonZero(out)) == 0:
        return out

    # Soften the boundary with a close so the union with the body envelope is
    # smooth, not stepped.
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)
    return out


def build_top_mask(garment_mask, parsing, full_pose, top_subtype: str = "", **helpers):
    sub = (top_subtype or "").lower()
    protect = TOP_PROTECT
    extra_allow = None
    if sub == "hoodie":
        # Hoodie variant: keep face protected (no diffusion on the face) but
        # drop "hair" and "hat" so the hood can drape behind the head, and
        # widen the allow region above the shoulders.
        protect = tuple(k for k in TOP_PROTECT if k not in {"hair", "hat"})
        h, w = garment_mask.shape[:2]
        extra_allow = _hood_extension_mask(
            parsing, full_pose, (h, w),
            parsing_union_mask=helpers["parsing_union_mask"],
        )
    return _common_compose(
        garment_mask, parsing, full_pose,
        semantic_keys=TOP_SEMANTIC,
        protect_keys=protect,
        category="top",
        subtype=sub,
        extra_allow_mask=extra_allow,
        **helpers,
    )


def build_pants_mask(garment_mask, parsing, full_pose, **helpers):
    return _common_compose(
        garment_mask, parsing, full_pose,
        semantic_keys=PANTS_SEMANTIC,
        protect_keys=PANTS_PROTECT,
        category="pants",
        **helpers,
    )


def build_dress_mask(garment_mask, parsing, full_pose, sleeve_type="long", **helpers):
    if sleeve_type == "sleeveless":
        sem, prot = DRESS_SEMANTIC_SLEEVELESS, DRESS_PROTECT_SLEEVELESS
    else:
        sem, prot = DRESS_SEMANTIC_SLEEVED, DRESS_PROTECT_SLEEVED
    return _common_compose(
        garment_mask, parsing, full_pose,
        semantic_keys=sem,
        protect_keys=prot,
        category="dress",
        **helpers,
    )


def build_accessory_mask(garment_mask, parsing, full_pose, accessory_subtype: str = "", **helpers):
    # Accessories only paint inside the warped garment footprint;
    # the wide envelope is omitted so we don't recolour clothing.
    garment_mask = (garment_mask > 20).astype(np.uint8) * 255
    h, w = garment_mask.shape[:2]
    shape = (h, w)
    fit_like = helpers["fit_like"]
    parsing_union_mask = helpers["parsing_union_mask"]

    sub = (accessory_subtype or "").lower()

    # Per-subtype dilate (controls "edit reach" beyond the warped footprint)
    dilate_px = {
        "shoes": 7, "boots": 9, "hat": 9, "sunglasses": 3,
        "belt": 5, "bag": 9, "scarf": 9,
    }.get(sub, 5)
    support = cv2.dilate(garment_mask, np.ones((dilate_px, dilate_px), np.uint8), iterations=1)

    # Union the matching parsing region (where available) so we also overpaint
    # the original accessory the model is replacing.
    parsing_keys_by_sub = {
        "shoes":      ("left_shoe", "right_shoe"),
        "boots":      ("left_shoe", "right_shoe", "left_leg", "right_leg"),
        "hat":        ("hat",),
        "sunglasses": ("sunglasses",),
        "belt":       ("belt",),
        "bag":        ("bag",),
        "scarf":      ("scarf",),
    }
    keys = parsing_keys_by_sub.get(sub, ())
    if keys:
        existing = parsing_union_mask(parsing, keys, shape)
        if int(existing.sum()) > 255 * 20:
            # only union the part of `existing` near the warp footprint so we
            # don't recolour an unrelated shoe on the opposite foot.
            near = cv2.dilate(garment_mask, np.ones((35, 35), np.uint8), iterations=1)
            support = cv2.bitwise_or(support, cv2.bitwise_and(existing, near))

    # Protect: per-subtype
    protect_by_sub = {
        "shoes":      ("face", "hair", "upper_clothes", "dress", "left_arm", "right_arm",
                        "pants", "skirt"),
        "boots":      ("face", "hair", "upper_clothes", "dress", "left_arm", "right_arm"),
        "hat":        ("face", "left_arm", "right_arm", "upper_clothes", "dress",
                        "sunglasses", "pants", "skirt"),
        "sunglasses": ("hair", "hat", "upper_clothes", "dress",
                        "left_arm", "right_arm", "pants", "skirt"),
        "belt":       ("face", "hair", "left_arm", "right_arm",
                        "upper_clothes", "dress"),
        "bag":        ("face", "hair", "hat"),
        "scarf":      ("face", "hair", "hat"),
    }
    protect_keys = protect_by_sub.get(sub, ACCESSORY_PROTECT)
    protect = parsing_union_mask(parsing, protect_keys, shape)
    if int(protect.sum()) > 255 * 20:
        protect = cv2.erode(protect, np.ones((3, 3), np.uint8), iterations=1)
        support = cv2.subtract(support, protect)

    support = (support > 20).astype(np.uint8) * 255
    _ = fit_like  # signature parity
    support = apply_category_lock(
        support, "accessory", parsing,
        parsing_union_mask=parsing_union_mask,
        fit_like=fit_like,
        garment_mask=garment_mask,
    )
    ok = int(cv2.countNonZero(support)) > 200
    return support, ok


# ---- Dispatcher ------------------------------------------------------------

def build_category_mask(
    garment_mask: np.ndarray,
    parsing: Optional[dict],
    full_pose: Optional[dict],
    *,
    category: str,
    sleeve_type: str = "long",
    accessory_subtype: str = "",
    top_subtype: str = "",
    parsing_union_mask: Callable,
    pose_envelope_fn: Callable,
    neck_mask_fn: Optional[Callable] = None,
    fit_like: Callable,
) -> Tuple[np.ndarray, bool]:
    """Single entry point. Dispatches on `category`."""
    helpers = dict(
        parsing_union_mask=parsing_union_mask,
        pose_envelope_fn=pose_envelope_fn,
        neck_mask_fn=neck_mask_fn,
        fit_like=fit_like,
    )
    cat = (category or "top").lower()
    if cat == "pants":
        return build_pants_mask(garment_mask, parsing, full_pose, **helpers)
    if cat == "dress":
        return build_dress_mask(
            garment_mask, parsing, full_pose, sleeve_type=sleeve_type, **helpers,
        )
    if cat == "accessory":
        return build_accessory_mask(
            garment_mask, parsing, full_pose,
            accessory_subtype=accessory_subtype, **helpers,
        )
    return build_top_mask(garment_mask, parsing, full_pose, top_subtype=top_subtype, **helpers)


__all__ = [
    "build_category_mask",
    "build_top_mask",
    "build_pants_mask",
    "build_dress_mask",
    "build_accessory_mask",
]
