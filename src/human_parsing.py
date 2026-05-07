"""Human body parsing using SegFormer (mattmdjaga/segformer_b2_clothes).

Segments a person image into 18 body-part labels:
  background, hat, hair, sunglasses, upper_clothes, skirt, pants, dress,
  belt, left_shoe, right_shoe, face, left_leg, right_leg, left_arm,
  right_arm, bag, scarf.

Used to:
  1. Precisely locate the original clothing for erasure.
  2. Identify arms / face / hair so they stay on top of the new garment.
"""
from __future__ import annotations

import cv2
import numpy as np
import os

LABEL_MAP: dict[int, str] = {
    0: "background", 1: "hat", 2: "hair", 3: "sunglasses",
    4: "upper_clothes", 5: "skirt", 6: "pants", 7: "dress",
    8: "belt", 9: "left_shoe", 10: "right_shoe", 11: "face",
    12: "left_leg", 13: "right_leg", 14: "left_arm", 15: "right_arm",
    16: "bag", 17: "scarf",
}

_processor = None
_model = None
_ready: bool | None = None


def _ensure_model() -> bool:
    global _processor, _model, _ready
    if _ready is not None:
        return _ready
    try:
        import torch  # noqa: F401
        from transformers import (
            SegformerImageProcessor,
            SegformerForSemanticSegmentation,
        )
        cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
        print("[human_parsing] Downloading SegFormer model (first time only)...")
        _processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            cache_dir=cache_dir,
        )
        _model = SegformerForSemanticSegmentation.from_pretrained(
            "mattmdjaga/segformer_b2_clothes",
            cache_dir=cache_dir,
        )
        _model.eval()
        _ready = True
        print("[human_parsing] Model loaded")
    except Exception as exc:
        print(f"[human_parsing] Could not load model: {exc}")
        _ready = False
    return _ready


def parse_human(image_rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Return a dict mapping each label name to a binary mask (0 or 255)."""
    if not _ensure_model():
        return {}

    import torch
    from PIL import Image as PILImage

    h, w = image_rgb.shape[:2]
    pil = PILImage.fromarray(image_rgb)
    inputs = _processor(images=pil, return_tensors="pt")

    with torch.no_grad():
        logits = _model(**inputs).logits

    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False,
    )
    seg = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    masks: dict[str, np.ndarray] = {}
    for label_id, name in LABEL_MAP.items():
        masks[name] = ((seg == label_id).astype(np.uint8)) * 255
    return masks


# ── Helper extractors ──────────────────────────────────────────────────

def get_clothing_mask(parsing: dict[str, np.ndarray]) -> np.ndarray | None:
    """Combine upper-clothing regions from parsing into one mask.

    Includes upper_clothes, dress, scarf. Falls back to belt region
    if the mask is suspiciously small (possible mis-segmentation).
    """
    if not parsing:
        return None
    mask = parsing.get("upper_clothes")
    if mask is None:
        return None
    for key in ("dress", "scarf", "belt"):
        if key in parsing and parsing[key].shape == mask.shape:
            mask = cv2.bitwise_or(mask, parsing[key])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # If mask is tiny, segmentation likely missed the garment.
    if mask.sum() < 500 * 255:
        # Try to recover by adding any region between shoulders and hips.
        for fallback_key in ("skirt", "pants"):
            if fallback_key in parsing and parsing[fallback_key].shape == mask.shape:
                extra = parsing[fallback_key]
                if extra.sum() > mask.sum():
                    mask = cv2.bitwise_or(mask, extra)
    return mask


def get_foreground_keep_mask(parsing: dict[str, np.ndarray]) -> np.ndarray | None:
    """Mask for body parts that should stay ON TOP of the new garment (arms, face, hair)."""
    if not parsing:
        return None
    sample = next(iter(parsing.values()))
    h, w = sample.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for key in ("left_arm", "right_arm", "face", "hair", "hat", "sunglasses"):
        if key in parsing:
            mask = cv2.bitwise_or(mask, parsing[key])
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def get_arm_mask(parsing: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return union mask of both arms from parsing output."""
    if not parsing:
        return None
    left = parsing.get("left_arm")
    right = parsing.get("right_arm")
    if left is None and right is None:
        return None
    if left is None:
        return right.copy()
    if right is None:
        return left.copy()
    return cv2.bitwise_or(left, right)


def get_neck_mask(parsing: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return a mask covering the neck/collar area between face and upper_clothes.

    This is critical for erasing the old garment's neckline so the new
    garment collar is visible.
    """
    if not parsing:
        return None
    face = parsing.get("face")
    clothes = parsing.get("upper_clothes")
    if face is None:
        return None
    sample = next(iter(parsing.values()))
    h, w = sample.shape[:2]

    # Expand face downward aggressively to cover neck + upper chest.
    neck_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 55))
    neck_region = cv2.dilate(face, neck_kernel, iterations=2)

    # Keep only the portion BELOW the face center (neck area, not forehead).
    face_ys = np.where(face > 0)[0]
    if len(face_ys) > 10:
        face_bottom = int(face_ys.max())
        # Neck starts just above the chin and extends down.
        neck_top = max(0, face_bottom - int((face_ys.max() - face_ys.min()) * 0.15))
        neck_region[:neck_top, :] = 0

    # Remove anything that is hair, background, bag etc.
    for exclude in ("hair", "hat", "background", "bag", "sunglasses",
                    "left_shoe", "right_shoe", "left_leg", "right_leg"):
        if exclude in parsing:
            neck_region = cv2.subtract(neck_region, parsing[exclude])

    # Also include the top portion of upper_clothes (the old collar region).
    if clothes is not None:
        clothes_ys = np.where(clothes > 0)[0]
        if len(clothes_ys) > 10:
            clothes_top = int(clothes_ys.min())
            collar_depth = max(15, int((clothes_ys.max() - clothes_ys.min()) * 0.15))
            collar_band = np.zeros((h, w), dtype=np.uint8)
            collar_band[clothes_top:clothes_top + collar_depth, :] = clothes[
                clothes_top:clothes_top + collar_depth, :
            ]
            neck_region = cv2.bitwise_or(neck_region, collar_band)

    return neck_region if int(neck_region.sum()) > 255 * 20 else None


def get_pants_mask(parsing: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return a mask covering the lower-body region (pants/skirt/legs).

    v16.11c: Added for pants try-on support.
    Combines: pants, skirt, left_leg, right_leg — excluding shoes.
    """
    if not parsing:
        return None
    sample = next(iter(parsing.values()))
    h, w = sample.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for key in ("pants", "skirt", "left_leg", "right_leg"):
        if key in parsing:
            mask = cv2.bitwise_or(mask, parsing[key])
    # Preserve belt as part of pants waistband
    if "belt" in parsing:
        mask = cv2.bitwise_or(mask, parsing["belt"])
    # Exclude shoes — they stay on top
    for key in ("left_shoe", "right_shoe"):
        if key in parsing:
            mask = cv2.subtract(mask, parsing[key])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask if int(mask.sum()) > 255 * 100 else None


def get_skin_mask(parsing: dict[str, np.ndarray]) -> np.ndarray | None:
    """Mask of exposed skin (arms, face, legs) for skin colour sampling.

    Also infers neck/chest area as the gap between face and upper_clothes,
    since SegFormer has no explicit neck label.
    """
    if not parsing:
        return None
    sample = next(iter(parsing.values()))
    h, w = sample.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for key in ("left_arm", "right_arm", "face", "left_leg", "right_leg"):
        if key in parsing:
            mask = cv2.bitwise_or(mask, parsing[key])

    # Infer neck/chest: dilate face downward, subtract clothing + hair + bg.
    face = parsing.get("face")
    if face is not None:
        # Extend face region downward to approximate neck / upper chest.
        neck_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 45))
        neck_region = cv2.dilate(face, neck_kernel, iterations=2)
        # Remove regions that belong to other known labels.
        for exclude in ("upper_clothes", "hair", "hat", "background",
                        "bag", "scarf", "sunglasses", "dress"):
            if exclude in parsing:
                neck_region = cv2.subtract(neck_region, parsing[exclude])
        mask = cv2.bitwise_or(mask, neck_region)

    return mask
