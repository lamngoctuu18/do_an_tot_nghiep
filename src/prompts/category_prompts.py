"""Category-specific negative prompt registry.

Extracted from app.py to keep prompt rules separate from pipeline orchestration.
Each entry is meant to be APPENDED to the base GenConfig().negative_prompt
unless explicitly marked as a full override (dress).
"""
from __future__ import annotations


# Dress negative prompt — currently used as a FULL OVERRIDE in app.py.
# Strong rules: forbid second-layer garments, scarves/jackets, palette drift,
# and "old red sleeve" ghost from upper-body shirt.
DRESS_NEGATIVE = (
    "logo, text, graphic, animal face, different dress, changed print, "
    "leopard spots, spotted print, polka dots, dalmatian print, round spots, "
    "new pattern, geometric panels, vertical panels, triangular panels, "
    "wrong abstract print, smeared print, faded print, grey rewritten print, "
    "metallic panels, color block panels, glossy satin, armor, eyes, redesigned dress, "
    "plain fabric, inner dress, second dress, double layer, inner panel, "
    "scarf, shawl, cardigan, sweater, poncho, robe, coat, jacket, vest, wrap coat, "
    "blue denim, denim vest, red sash, blanket, cape, cloak, "
    "shoulder pads, puffy shoulders, bulky shoulders, cape shoulders, "
    "duplicate sleeves, extra sleeves, pasted sleeves, arm overlay, bare arm under sleeve, old red sleeve, "
    "cinched waist, tight waist, belt, sash, old red shirt, "
    "shorts visible, transparent skirt, missing skirt, open skirt, "
    "grey cast, silver wash, washed out fabric, desaturated fabric, low contrast dress, "
    "plain gray dress, gray base fabric, patchy gray panels, repaired fabric, dirty gray smudges, "
    "horizontal stripe, horizontal band, dashed horizontal line, scanline artifact, "
    "old clothing visible, cpu warp, geometric warp, flat pasted cloth, sticker effect, no folds, "
    "vertical streaks, scratched texture, torn fabric, grey patch, blurry, low quality, bad anatomy"
)

# Pants negative — APPENDED to the base negative.
# Locks the model away from the torso and the old jeans silhouette.
PANTS_NEGATIVE_APPEND = (
    "black rectangle, pasted cloth, flat patch, torso garment, apron, "
    "skirt, belt block, changed shirt, repainted shirt, "
    "covering stomach, covering chest, blue denim shorts visible, "
    "old jeans visible, missing leg opening, merged legs, "
    "no crotch seam, black belt only"
)

# Top negative — appended; forbid bottom-half changes.
TOP_NEGATIVE_APPEND = (
    "pants changed, skirt changed, lower body changed, "
    "extra sleeves, duplicate arms, "
    "old shirt visible, pasted cloth, flat texture"
)

# Hoodie subtype — extra negatives so SD does not collapse the hoodie into a
# crewneck/tee shape. Forces the model to keep a hood + drawstring silhouette
# rather than removing them to match the existing t-shirt outline.
TOP_HOODIE_NEGATIVE_APPEND = (
    "no hood, missing hood, removed hood, "
    "t-shirt, tee shirt, tank top, crewneck, scoop neck, v-neck, polo shirt, blouse, "
    "thin fabric, tight short sleeves, bare forearms, "
    "missing drawstring, missing kangaroo pocket, missing front pocket, "
    "flat collar, dress shirt collar, button-up shirt, "
    "third arm, extra arm, extra hand, duplicate hand, phantom arm, "
    "floating sleeve, extra sleeve next to shoulder, ghost limb, "
    "dropped shoulders, broad shoulder line, wide sleeves, boxy sleeves, "
    "loose sleeve tubes, sleeves wider than arms, "
    "sleeve fused with torso, arm fused into body, missing underarm seam, "
    "missing side seam, sleeve blended into torso, no sleeve separation, "
    "oversized puffy hoodie, baggy silhouette, balloon shape, inflated torso, "
    "bulging belly, protruding abdomen, darker belly panel, color block belly patch, "
    "contrasting lower torso panel, discolored pocket area, "
    "smooth blank front, no pocket, missing kangaroo pocket, "
    "split hem, vertical hem slit, gap at hem, open front hem, "
    "double collar, separate inner collar ring, crewneck under hood, "
    "hard neckline ring, choker collar, "
    "thick drawstrings, puffy drawstrings, rope-like drawstrings, inflated drawstring, "
    "melted drawstring tips, duplicated drawstrings, extra drawstring, third drawstring, "
    "multiple drawstrings, duplicate cord, extra cord, doubled cord, stray cord, "
    "knotted drawstring tangle, frayed drawstring, branched drawstring, "
    "drawstring fused to hood, painted drawstring shadow, drawstring blob, "
    "blurred neckline, muddy collar spot, "
    "dark vertical seam down torso, painted-on vertical seam, fake random stitched edge, "
    "missing waist ribbing, missing pocket seam, soft muddy hem, "
    "kinked sleeve, bent sleeve, broken sleeve line, fake sleeve fold artifact, "
    "zigzag sleeve edge, jagged sleeve outline, harsh sleeve crease, "
    "sleeve discontinuity, snapped sleeve, hooked sleeve seam, "
    "extra fabric bump on shoulder, shoulder pad bump, lump on shoulder, "
    "fabric flap at sleeve hem, extra cuff, doubled cuff, secondary sleeve edge"
)

# Hoodie positive — used as style_prompt when the user prompt is empty.
TOP_HOODIE_POSITIVE = (
    "a realistic photo of a person wearing a fitted hooded sweatshirt (hoodie), "
    "soft hood draped behind the head and neck, hood opening blending smoothly into the neckline with no separate inner collar, "
    "exactly two thin flat drawstring cords hanging straight down from the hood opening, "
    "drawstrings rendered as soft fabric ties with small straight tips, no knots or extra cords, "
    "long sleeves fully covering both arms down to the wrists, "
    "sleeves following the natural arm line with subtle lengthwise fabric folds and clear ribbed cuffs, "
    "ribbed cuffs, prominent kangaroo front pocket clearly visible across the lower torso, "
    "set-in shoulder seams aligned with the model shoulders, narrow tapered sleeves following the arms, "
    "medium-weight cotton fleece fabric matching the reference garment, "
    "even heather grey tone across the torso and kangaroo pocket, "
    "flat lower torso with natural pocket seam, continuous straight ribbed hem across the waist with no front slit or split, "
    "crisp hood opening, crisp shoulder seams, crisp waist pocket seam and ribbed hem, "
    "garment hugs the body silhouette, no oversized puffy shape, "
    "natural soft folds at the waist and elbows, realistic studio lighting"
)

TOP_HOODIE_CONSTRAINT = (
    "fitted regular hoodie matching the reference garment, two thin flat drawstrings only, "
    "no extra cords, narrow body silhouette, narrow tapered sleeves with subtle fabric grain, "
    "visible side seams, visible kangaroo pocket seam and straight ribbed hem, crisp ribbed cuffs"
)

# Accessory negative — appended; allow only the accessory region.
ACCESSORY_NEGATIVE_APPEND = (
    "changed clothing, repainted shirt, repainted pants, "
    "different body, modified face, modified hair, "
    "pasted cloth, flat texture"
)


def build_category_negative(category: str, base_negative: str = "", subtype: str = "") -> str:
    """Return the final negative prompt for a category.

    - "dress" → full override (legacy behaviour from app.py).
    - others → `base_negative` + ", " + the category-specific tail.
    - `subtype` (e.g. "hoodie") adds an extra tail on top of the category tail.
    """
    cat = (category or "top").lower()
    sub = (subtype or "").lower()
    if cat == "dress":
        return DRESS_NEGATIVE
    if cat == "pants":
        tail = PANTS_NEGATIVE_APPEND
    elif cat == "accessory":
        tail = ACCESSORY_NEGATIVE_APPEND
    else:
        tail = TOP_NEGATIVE_APPEND
        if sub == "hoodie":
            tail = f"{tail}, {TOP_HOODIE_NEGATIVE_APPEND}"
    base = (base_negative or "").strip()
    if not base:
        return tail
    return f"{base}, {tail}"


__all__ = [
    "DRESS_NEGATIVE",
    "PANTS_NEGATIVE_APPEND",
    "TOP_NEGATIVE_APPEND",
    "TOP_HOODIE_NEGATIVE_APPEND",
    "TOP_HOODIE_POSITIVE",
    "TOP_HOODIE_CONSTRAINT",
    "ACCESSORY_NEGATIVE_APPEND",
    "build_category_negative",
]
