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
    "no crotch seam, black belt only, "
    "dangling drawstring, hanging cord, hanging string, sash hanging below hem, "
    "long drawstring tail, drawstring tips below shorts, "
    "bare midriff above shorts, exposed belly gap, skin strip above waistband, "
    "white vertical stripe at center crotch, bright crotch seam, white drawstring line between legs, "
    "wavy waistband, jagged waistband, uneven waistband, bumpy waistband, "
    "scalloped waist edge, distorted waistband line"
)

# Shorts subtype (pants → shorts). Mirrors the shirt pattern: a clean POSITIVE
# template that REPLACES bloated user prompts, plus a short CONSTRAINT tail.
# Forces SD to redraw a fitted athletic short with two leg openings instead of
# preserving the warped reference blob (drawstring sash, hammerhead).
PANTS_SHORTS_POSITIVE = (
    "a realistic photo of a person wearing fitted athletic shorts, "
    "shorts ending at mid-thigh well above the knee, "
    "two clearly separated short leg openings with a visible center crotch gap between the thighs, "
    "narrow elastic waistband sitting flat on the hip line, "
    "shorts width following the natural hip line (NOT flared, NOT skirt-like), "
    "smooth fabric matching the reference garment colour and texture exactly, "
    "preserve reference contrast trim, white piping or side stripes along the side seams and hems when present, "
    "soft natural fabric folds at the waist and thighs, "
    "left leg and right leg fully separated by a clean inner-thigh gap, "
    "no drawstring sash hanging down, no fabric tail between the legs, "
    "no horizontal fabric bar across the waist, no second waistband, "
    "cover the previous bottom completely with the new shorts, "
    "preserve original shirt, preserve torso and arms, do not redraw upper body, "
    "realistic studio lighting, natural shadows at the waist and thighs"
)

PANTS_SHORTS_CONSTRAINT = (
    "fitted athletic shorts matching the reference garment, "
    "two separated leg openings with a clear center crotch gap, "
    "narrow elastic waistband at the hip line, mid-thigh hem, "
    "shorts width following the natural hips, no drawstring tail between the legs, "
    "main fabric colour and contrast trim matching the reference on outer side seams and leg hems only, "
    "no bright vertical stripe at the center crotch, smooth fabric texture"
)

PANTS_SHORTS_DENIM_CONSTRAINT = (
    "high-waisted denim shorts matching the reference garment, "
    "complete continuous denim waistband across the front, visible belt loops, "
    "front button and zip fly closure visible at the center, "
    "both front curved pocket openings visible, left and right front pockets visible symmetrically, "
    "pocket seam stitching visible on both hips, small coin pocket visible when present in the reference, "
    "classic blue denim texture and seams, fitted hip line, raw frayed hem, "
    "front denim panel fully visible with no shirt tail or scarf tail covering the shorts, "
    "waistband reaches and slightly tucks under the shirt hem, no exposed belly strip above the waistband, "
    "two separated leg openings with a clean center crotch gap, "
    "do not replace the denim waistband with skin, shirt, or an elastic band"
)


# Top negative — appended; forbid bottom-half changes.
TOP_NEGATIVE_APPEND = (
    "pants changed, skirt changed, lower body changed, "
    "extra sleeves, duplicate arms, "
    "old shirt visible, pasted cloth, flat texture, "
    "garbled text, jumbled letters, distorted text, mangled typography, "
    "illegible text, fake text, gibberish letters, melted letters, "
    "smeared print, smudged graphic, blurred graphic, doubled graphic, "
    "duplicate logo, extra logo, repeated graphic, mirrored letters, "
    "changed graphic, redesigned graphic, recolored graphic, recolored print, "
    "wrong logo, different logo, missing graphic, faded graphic, faded print, "
    "graphic moved, graphic stretched, graphic warped, distorted graphic, "
    "new pattern, replaced pattern, hallucinated motif, generated symbol, "
    "letters rearranged, font changed, font replaced, "
    "wrong color graphic, color drift on print, washed out print"
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

# Jacket subtype — bomber/zip jacket. Prevents SD from collapsing the jacket
# into a short cropped tee shape (cutting off the ribbed hem) and forces it
# to render both side pockets + full-length zip closure.
TOP_JACKET_NEGATIVE_APPEND = (
    "cropped hem, short hem, missing hem band, missing ribbed waistband, "
    "missing waistband, raw cut edge, jacket cut at chest, jacket cut at waist, "
    "jacket ending above waist, half-length jacket, vest, sleeveless jacket, "
    "missing pockets, no side pockets, pockets removed, sealed front pocket area, "
    "missing zipper, half zipper, broken zipper, zipper only on top, "
    "open neckline without collar, missing stand collar, "
    "shirt instead of jacket, t-shirt silhouette, sweater silhouette, "
    "fused with pants, jacket merged into trousers, hem clipped by belt, "
    "button-up jacket, button-up shirt, dress shirt, oxford shirt, polo shirt, "
    "row of buttons, visible buttons, front buttons, button placket, snap placket, "
    "shirt collar, pointed collar, dress shirt collar, blazer lapel, notched lapel, peaked lapel, "
    "open front jacket without zipper, missing center zipper teeth, "
    "darker upper body, darker chest, color shift, gradient torso color, "
    "two-tone jacket, lighter sleeves than torso, washed-out brown, muddy brown, "
    "color drift from reference, different color than reference garment"
)

TOP_JACKET_POSITIVE = (
    "a realistic photo of a person wearing a bomber-style zip jacket, "
    "small upright stand-up collar at the neck (NOT a shirt collar, NOT a lapel), "
    "single full-length metal zipper running straight down the front center from collar to hem, "
    "thin metal zipper teeth clearly visible along the front center, no buttons anywhere, "
    "smooth uninterrupted fabric panels on both sides of the zipper with no placket and no button row, "
    "two slanted side pockets clearly visible on the lower torso, "
    "ribbed waistband sitting at the hip with a clean horizontal hem line below the belt, "
    "long sleeves fully covering both arms down to the wrists with ribbed cuffs, "
    "set-in shoulder seams aligned with the model shoulders, "
    "smooth medium-weight suede-like fabric matching the reference garment, "
    "uniform camel tan colour identical across the torso, sleeves and hem band, "
    "even tone across the entire jacket exactly matching the reference garment colour, "
    "soft natural folds at the elbows and waistband, realistic studio lighting"
)

TOP_JACKET_CONSTRAINT = (
    "regular fit bomber jacket matching the reference garment, ribbed hem extending below the waist, "
    "two side pockets visible, single full-length front zipper with metal teeth, "
    "no buttons and no button placket, smooth fabric panels beside the zipper, "
    "ribbed cuffs, small stand collar (not a shirt collar), crisp shoulder seams, "
    "hem line clearly below the belt, "
    "uniform camel tan colour matching the reference, even tone across torso and sleeves"
)

# Shirt subtype — button-up dress shirt / casual shirt. Forces SD to keep the
# point collar + center button placket + buttoned cuffs and forbids the bomber
# jacket collapse (stand-up collar, zipper, ribbed hem) that the generic top
# pipeline tends to produce when a shirt mask resembles a jacket silhouette.
TOP_SHIRT_NEGATIVE_APPEND = (
    "zipper, front zipper, metal zipper teeth, zip closure, half zip, quarter zip, "
    "stand-up collar, mock neck, mandarin collar, turtleneck, hooded, hood, drawstring, "
    "ribbed hem, ribbed waistband, elastic hem, cropped hem, bomber jacket, varsity jacket, "
    "leather jacket, suede jacket, puffer, parka, blazer with lapel, notched lapel, peaked lapel, "
    "double-breasted, missing collar, missing buttons, smooth front panel, plain front placket, "
    "no buttons, no button row, sealed front, t-shirt, tee shirt, polo shirt, "
    "knit sweater, jumper, cardigan, fleece, hoodie silhouette, tank top, "
    "short sleeves on a long sleeve shirt, raglan sleeves, set-in puffy sleeves, "
    "wrong collar shape, blurred collar, melted collar, fused collar, "
    "duplicate placket, double placket, crooked placket, slanted button row, "
    "missing cuffs, ribbed cuffs, elastic cuffs, drawstring cuffs, "
    "different colour from reference, washed out colour, colour drift, "
    "vertical chest seam, vertical body seam, princess seam, dart seam line, "
    "two vertical seams on torso, parallel vertical lines on chest, parallel vertical lines on body, "
    "decorative vertical stitching down the chest, contrast stitching down the body, "
    "welt pocket on chest, chest welt pockets, two chest pockets, twin chest pockets, "
    "flap pockets on chest, slanted chest pockets, jacket pocket lines, "
    "safari shirt seams, military shirt panels, western shirt yoke seams, "
    "vertical pintucks, pintuck lines, vertical pleats down the front, "
    "extra panel lines on torso, jacket-style side seams visible on shirt"
)

TOP_SHIRT_POSITIVE = (
    "a realistic photo of a person wearing a button-up shirt (long sleeve dress shirt / casual shirt), "
    "classic pointed shirt collar at the neck (NOT a stand-up collar, NOT a hood), "
    "single straight vertical center front button placket running from the collar down to the hem, "
    "a clear row of small round buttons evenly spaced along the center placket, every button visible and aligned, "
    "long sleeves fully covering both arms down to the wrists, "
    "buttoned shirt cuffs at each wrist (NOT ribbed, NOT elastic), "
    "smooth woven fabric matching the reference garment, soft natural shirt folds at the elbows and waist, "
    "set-in shoulder seams aligned with the model shoulders, "
    "completely smooth and uninterrupted front torso fabric on both sides of the center placket, "
    "NO vertical seams on the chest, NO princess seams, NO darts visible on the front, "
    "NO chest pockets, NO welt pockets, NO flap pockets, NO decorative stitching down the body, "
    "the only vertical line on the torso is the center button placket itself, "
    "shirt hem ending at the hip with a clean straight or slightly curved edge (NO ribbed waistband, NO zipper), "
    "uniform fabric colour identical to the reference garment across the torso, sleeves and collar, "
    "realistic studio lighting, crisp collar edges, crisp placket and button row"
)

TOP_SHIRT_CONSTRAINT = (
    "regular fit button-up shirt matching the reference garment, pointed shirt collar, "
    "center button placket with a visible row of buttons, buttoned cuffs at the wrists, "
    "smooth woven fabric, no zipper and no ribbed hem, "
    "completely smooth front torso with no vertical chest seams, no princess seams, no chest pockets, "
    "the only vertical line on the body is the center button placket, "
    "uniform colour matching the reference, even tone across torso, sleeves and collar"
)

# T-shirt subtype — short-sleeve knit tee, often carries a front graphic /
# typography. Locks the chest print so SD does not regenerate it into garbled
# letters or a new pattern. Use whenever subtype == "tshirt".
TOP_TSHIRT_NEGATIVE_APPEND = (
    "long sleeves, three-quarter sleeves, button-up shirt, dress shirt, polo collar, "
    "stand collar, mock neck, turtleneck, hood, drawstring, ribbed waistband, "
    "zipper, front zipper, button placket, row of buttons, chest pocket, "
    "blazer, jacket, sweater, cardigan, fleece, hoodie silhouette, "
    "garbled text, jumbled letters, scrambled letters, illegible text, "
    "fake text, gibberish letters, melted letters, distorted typography, "
    "duplicate text, doubled text, mirrored text, repeated text, extra letters, "
    "missing letters, dropped letters, broken letters, fragmented letters, "
    "smeared print, blurred print, smudged graphic, faded print, "
    "redrawn print, regenerated print, hallucinated print, "
    "new graphic, different graphic, replaced graphic, additional graphic, extra logo, "
    "wrong color graphic, recolored letters, recolored graphic, color drift on print, "
    "graphic warped, graphic stretched, graphic moved, graphic resized, "
    "decorative seams across chest, vertical chest seams, princess seams, "
    "extra panel lines on torso, contrast stitching on chest, "
    "different color from reference, washed out fabric, color drift"
)

TOP_TSHIRT_POSITIVE = (
    "a realistic photo of a person wearing a fitted short-sleeve crew-neck t-shirt, "
    "round crew neckline at the collarbone, short sleeves ending at the upper arm, "
    "smooth soft cotton jersey fabric matching the reference garment, "
    "straight regular t-shirt hem ending at the hip, "
    "set-in shoulder seams aligned with the model shoulders, "
    "front chest graphic preserved exactly as in the reference garment: "
    "same wording, same letters, same font, same color, same position, same scale, "
    "all text on the chest must remain perfectly readable and identical to the reference, "
    "the chest graphic is reproduced verbatim with no added words and no removed words, "
    "no extra logos and no decorative motifs added beyond the reference graphic, "
    "uniform fabric color identical to the reference t-shirt across torso and sleeves, "
    "soft natural fabric folds at the waist and elbows, realistic studio lighting"
)

TOP_TSHIRT_CONSTRAINT = (
    "fitted short-sleeve crew-neck t-shirt matching the reference garment, "
    "preserve the front chest graphic exactly: identical wording, font, color, position and scale, "
    "all letters on the print remain readable and identical to the reference, "
    "no extra letters, no missing letters, no new logos beyond the reference graphic, "
    "smooth cotton jersey fabric with no zipper, no buttons, no ribbed hem, "
    "uniform color identical to the reference, even tone across torso and sleeves"
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
        elif sub == "jacket":
            tail = f"{tail}, {TOP_JACKET_NEGATIVE_APPEND}"
        elif sub == "shirt":
            tail = f"{tail}, {TOP_SHIRT_NEGATIVE_APPEND}"
        elif sub == "tshirt":
            tail = f"{tail}, {TOP_TSHIRT_NEGATIVE_APPEND}"
    base = (base_negative or "").strip()
    if not base:
        return tail
    return f"{base}, {tail}"


__all__ = [
    "DRESS_NEGATIVE",
    "PANTS_NEGATIVE_APPEND",
    "PANTS_SHORTS_POSITIVE",
    "PANTS_SHORTS_CONSTRAINT",
    "PANTS_SHORTS_DENIM_CONSTRAINT",
    "TOP_NEGATIVE_APPEND",
    "TOP_HOODIE_NEGATIVE_APPEND",
    "TOP_HOODIE_POSITIVE",
    "TOP_HOODIE_CONSTRAINT",
    "TOP_JACKET_NEGATIVE_APPEND",
    "TOP_JACKET_POSITIVE",
    "TOP_JACKET_CONSTRAINT",
    "TOP_SHIRT_NEGATIVE_APPEND",
    "TOP_SHIRT_POSITIVE",
    "TOP_SHIRT_CONSTRAINT",
    "TOP_TSHIRT_NEGATIVE_APPEND",
    "TOP_TSHIRT_POSITIVE",
    "TOP_TSHIRT_CONSTRAINT",
    "ACCESSORY_NEGATIVE_APPEND",
    "build_category_negative",
]
