"""Gemini Vision → structured try-on prompt.

Sends the person + garment images to Gemini and asks for a JSON envelope
that the local SD / cloud VTON pipeline can consume as
`positive_prompt` and `negative_prompt`. Gemini never edits pixels —
it only emits text. See `analyze_garment_prompt_with_gemini` for the
contract.

Activated by env `VTON_USE_GEMINI_PROMPT=1` and `GEMINI_API_KEY`.
Model id is overridable via `GEMINI_MODEL` (default
`gemini-2.5-flash`). On repeated 503/UNAVAILABLE responses the call
also tries `GEMINI_MODEL_FALLBACK` (default `gemini-2.5-flash-lite`)
before giving up. Results are cached by image hash under
`VTON_GEMINI_CACHE` so duplicate try-ons skip the network entirely.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from dotenv import load_dotenv


class GeminiPromptUnavailableError(RuntimeError):
    """Raised when Gemini auto-prompt can't run (disabled, missing key, SDK absent)."""


_TRUTHY = {"1", "true", "yes", "on"}
_TRANSIENT_TOKENS = (
    "503", "unavailable", "high demand", "overloaded",
    "rate", "quota", "resource_exhausted", "deadline",
    "timeout", "temporarily",
)


def _save_rgb_temp(image_rgb: np.ndarray, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return path


def _safe_json_loads(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


_DEFAULT_POSITIVE = (
    "realistic virtual try-on, preserve garment color and fabric texture, "
    "natural folds, sharp seams, single garment layer"
)
_DEFAULT_NEGATIVE = (
    "old clothing visible, pasted cloth, sticker effect, wrong garment category, "
    "duplicate sleeves, blurry fabric, distorted body, low quality"
)


# ---- Category-aware fallback ------------------------------------------------

def fallback_describe_garment(category: str) -> dict[str, Any]:
    """Return a structured prompt dict when Gemini is unreachable.

    Category-specific so the diffusion pipeline still gets useful hints
    (sleeve type, neckline, silhouette) instead of a generic phrase.
    """
    cat = (category or "auto").strip().lower()

    if cat in {"dress"}:
        positive = (
            "long sleeve A-line midi dress, round neckline, fitted bodice, "
            "defined waist seam, flared skirt, soft fabric, natural vertical folds, "
            "realistic cloth texture"
        )
        return {
            "category": "dress", "cloth_type": "overall",
            "sleeve_type": "long", "neckline": "crew neck",
            "fit": "regular", "silhouette": "a_line", "length": "midi",
            "fabric": "soft cotton blend", "color": "as in reference",
            "pattern": "plain",
            "positive_prompt": positive,
            "negative_prompt": _DEFAULT_NEGATIVE,
            "notes": ["fallback:dress"],
        }

    if cat in {"hoodie"}:
        positive = (
            "fitted hooded sweatshirt with soft hood, visible drawstring, "
            "long sleeves with ribbed cuffs, kangaroo front pocket, "
            "natural fabric folds, realistic fleece texture, "
            "garment hugs torso and follows arms"
        )
        return {
            "category": "hoodie", "cloth_type": "upper",
            "sleeve_type": "long", "neckline": "hood",
            "fit": "regular", "silhouette": "fitted", "length": "hip",
            "fabric": "cotton fleece", "color": "as in reference",
            "pattern": "plain",
            "positive_prompt": positive,
            "negative_prompt": _DEFAULT_NEGATIVE,
            "notes": ["fallback:hoodie"],
        }

    if cat in {"top", "tshirt", "shirt", "jacket", "outer", "generic"}:
        positive = (
            "upper body garment, natural shoulder fit, long or short sleeves "
            "matching the reference, visible seams, natural wrinkles, "
            "garment follows torso and arms, realistic fabric texture"
        )
        return {
            "category": cat if cat != "generic" else "top",
            "cloth_type": "upper",
            "sleeve_type": "unknown", "neckline": "unknown",
            "fit": "regular", "silhouette": "regular", "length": "hip",
            "fabric": "as in reference", "color": "as in reference",
            "pattern": "plain",
            "positive_prompt": positive,
            "negative_prompt": _DEFAULT_NEGATIVE,
            "notes": [f"fallback:{cat}"],
        }

    if cat in {"pants", "jeans", "trousers"}:
        positive = (
            "full-length pants, fitted waistband, natural hip and leg alignment, "
            "visible leg separation, realistic denim or cotton texture, "
            "natural folds and shadows"
        )
        return {
            "category": cat, "cloth_type": "lower",
            "sleeve_type": "none", "neckline": "unknown",
            "fit": "regular", "silhouette": "straight", "length": "ankle",
            "fabric": "denim", "color": "as in reference",
            "pattern": "denim wash" if cat == "jeans" else "plain",
            "positive_prompt": positive,
            "negative_prompt": _DEFAULT_NEGATIVE,
            "notes": [f"fallback:{cat}"],
        }

    if cat in {"shorts"}:
        return {
            "category": "shorts", "cloth_type": "lower",
            "sleeve_type": "none", "neckline": "unknown",
            "fit": "regular", "silhouette": "straight", "length": "thigh",
            "fabric": "as in reference", "color": "as in reference",
            "pattern": "plain",
            "positive_prompt": (
                "fitted shorts ending above the knee, two short leg openings, "
                "natural fabric folds, realistic texture"
            ),
            "negative_prompt": _DEFAULT_NEGATIVE,
            "notes": ["fallback:shorts"],
        }

    if cat in {"skirt"}:
        return {
            "category": "skirt", "cloth_type": "lower",
            "sleeve_type": "none", "neckline": "unknown",
            "fit": "regular", "silhouette": "a_line", "length": "knee",
            "fabric": "as in reference", "color": "as in reference",
            "pattern": "plain",
            "positive_prompt": (
                "knee-length skirt with natural drape, soft folds, "
                "realistic fabric texture, fitted waist"
            ),
            "negative_prompt": _DEFAULT_NEGATIVE,
            "notes": ["fallback:skirt"],
        }

    return {
        "category": cat or "top",
        "cloth_type": "upper",
        "sleeve_type": "unknown", "neckline": "unknown",
        "fit": "regular", "silhouette": "regular", "length": "unknown",
        "fabric": "as in reference", "color": "as in reference",
        "pattern": "plain",
        "positive_prompt": (
            "realistic garment, natural cloth folds, accurate fit on body, "
            "preserve garment color, texture, seams and silhouette"
        ),
        "negative_prompt": _DEFAULT_NEGATIVE,
        "notes": ["fallback:generic"],
    }


# ---- Cache + retry ----------------------------------------------------------

def _is_transient(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(tok in msg for tok in _TRANSIENT_TOKENS)


def _image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()[:24]


def _cache_dir() -> Path:
    base = os.getenv("VTON_GEMINI_CACHE", str(Path.home() / ".vton_cache" / "gemini"))
    p = Path(base) / "descriptions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(person_bytes: bytes, cloth_bytes: bytes,
               category_lock: str, user_prompt: str, model_id: str) -> str:
    h = hashlib.sha256()
    h.update(_image_hash(person_bytes).encode())
    h.update(b"|")
    h.update(_image_hash(cloth_bytes).encode())
    h.update(b"|")
    h.update((category_lock or "").encode())
    h.update(b"|")
    h.update((user_prompt or "").encode())
    h.update(b"|")
    h.update((model_id or "").encode())
    return h.hexdigest()[:32]


def _cache_load(key: str) -> dict[str, Any] | None:
    if os.getenv("VTON_GEMINI_CACHE_OFF", "0").strip().lower() in _TRUTHY:
        return None
    path = _cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_store(key: str, data: dict[str, Any]) -> None:
    if os.getenv("VTON_GEMINI_CACHE_OFF", "0").strip().lower() in _TRUTHY:
        return
    try:
        path = _cache_dir() / f"{key}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _call_with_retry(
    fn: Callable[[str], Any],
    *,
    primary_model: str,
    fallback_model: str | None,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> Any:
    """Run ``fn(model_id)`` with exponential backoff on transient errors.

    After exhausting retries on ``primary_model``, falls through to
    ``fallback_model`` (one attempt) if it's a different non-empty id.
    Raises the last exception when nothing succeeds.
    """
    models = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models.append(fallback_model)

    last_err: Exception | None = None
    for model_id in models:
        for attempt in range(max_retries):
            try:
                return fn(model_id)
            except Exception as exc:
                last_err = exc
                if not _is_transient(exc):
                    raise
                # exponential backoff + jitter, clamped
                sleep_s = min(base_delay * (2 ** attempt) + random.uniform(0, 0.8), max_delay)
                print(
                    f"[GEMINI] transient error on {model_id} (attempt "
                    f"{attempt + 1}/{max_retries}): {exc}. retry in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
        # next model gets one more shot at the next iteration's attempt 0
    assert last_err is not None
    raise last_err


def analyze_garment_prompt_with_gemini(
    person_rgb: np.ndarray,
    cloth_rgb: np.ndarray,
    category_lock: str = "auto",
    user_prompt: str = "",
    temp_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Return a JSON dict describing the garment + ready-to-use prompts.

    Raises GeminiPromptUnavailableError when the feature is disabled,
    the API key is missing, the SDK isn't installed, or the call fails
    after retry + fallback-model attempts. Callers should treat the
    exception as "skip Gemini, keep going" — typically by falling back
    to ``fallback_describe_garment(category_lock)``.

    Set ``force=True`` to bypass ``VTON_USE_GEMINI_PROMPT`` (e.g. when
    the user explicitly clicked the Gemini checkbox in the UI). The API
    key check is never bypassed.
    """
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

    if not force and os.getenv("VTON_USE_GEMINI_PROMPT", "0").strip().lower() not in _TRUTHY:
        raise GeminiPromptUnavailableError("Gemini prompt disabled (VTON_USE_GEMINI_PROMPT)")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiPromptUnavailableError("Missing GEMINI_API_KEY")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise GeminiPromptUnavailableError(
            "Missing google-genai. Install: pip install -U google-genai"
        ) from exc

    if temp_dir is None:
        temp_dir = os.getenv(
            "VTON_GEMINI_CACHE",
            str(Path.home() / ".vton_cache" / "gemini"),
        )

    temp_dir = Path(temp_dir)
    person_path = _save_rgb_temp(person_rgb, temp_dir / "person.jpg")
    cloth_path = _save_rgb_temp(cloth_rgb, temp_dir / "cloth.jpg")

    person_bytes = person_path.read_bytes()
    cloth_bytes = cloth_path.read_bytes()

    model_id = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    fallback_model = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash-lite").strip()

    # Cache check — same person + cloth + lock + user prompt returns instantly.
    ck = _cache_key(person_bytes, cloth_bytes, category_lock, user_prompt, model_id)
    cached = _cache_load(ck)
    if cached:
        print(f"[GEMINI] cache hit ({ck[:8]}…) — skipping API call")
        return cached

    instruction = f"""You are a virtual try-on prompt engineer for Stable Diffusion inpainting.

Image 1 = the person (target). Image 2 = the garment / accessory reference.
The user selected category_lock="{category_lock}".
User extra prompt="{user_prompt}".

Return ONLY valid JSON. No markdown fences. No commentary.

Rules:
- Do NOT identify the person or describe their face/identity.
- Focus on garment category, silhouette, fabric, color, pattern, sleeve, neckline, length.
- If category_lock is not "auto", obey it even if the image suggests otherwise.
- positive_prompt should help diffusion replace only the selected garment region.
- negative_prompt must prevent: old clothing showing, pasted/sticker look, duplicated
  sleeves, wrong category, blurry fabric, distorted body, extra clothing layers.

JSON schema:
{{
  "category": "top|tshirt|hoodie|jacket|outer|pants|jeans|shorts|dress|skirt|belt|bag|scarf|hat|sunglasses|shoes|boots|generic",
  "cloth_type": "upper|lower|overall|accessory",
  "sleeve_type": "none|sleeveless|short|long|unknown",
  "neckline": "crew neck|v neck|collar|hood|strapless|unknown",
  "fit": "slim|regular|loose|oversized|unknown",
  "silhouette": "fitted|regular|oversized|a_line|sheath|shift|fit_and_flare|straight|wide_leg|skinny|unknown",
  "length": "cropped|hip|thigh|knee|midi|maxi|ankle|unknown",
  "fabric": "short fabric phrase",
  "color": "short color phrase",
  "pattern": "plain|striped|floral|plaid|graphic|denim wash|other",
  "positive_prompt": "one concise English prompt, max 70 words",
  "negative_prompt": "one concise English negative prompt, max 80 words",
  "notes": ["short implementation notes"]
}}
"""

    client = genai.Client(api_key=api_key)
    _gen_config = None
    try:
        _gen_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
        )
    except Exception:
        _gen_config = None

    def _do_call(use_model: str):
        return client.models.generate_content(
            model=use_model,
            contents=[
                types.Part.from_bytes(data=person_bytes, mime_type="image/jpeg"),
                types.Part.from_bytes(data=cloth_bytes, mime_type="image/jpeg"),
                instruction,
            ],
            config=_gen_config,
        )

    try:
        max_retries = int(os.getenv("VTON_GEMINI_MAX_RETRIES", "3"))
    except ValueError:
        max_retries = 3
    try:
        base_delay = float(os.getenv("VTON_GEMINI_BASE_DELAY", "1.2"))
    except ValueError:
        base_delay = 1.2
    try:
        max_delay = float(os.getenv("VTON_GEMINI_MAX_DELAY", "12"))
    except ValueError:
        max_delay = 12.0

    try:
        response = _call_with_retry(
            _do_call,
            primary_model=model_id,
            fallback_model=fallback_model,
            max_retries=max(1, max_retries),
            base_delay=max(0.2, base_delay),
            max_delay=max(base_delay, max_delay),
        )
    except Exception as exc:
        raise GeminiPromptUnavailableError(f"Gemini call failed: {exc}") from exc

    raw_text = getattr(response, "text", "") or ""
    try:
        data = _safe_json_loads(raw_text)
    except Exception as exc:
        raise GeminiPromptUnavailableError(f"Gemini JSON parse failed: {exc}") from exc

    positive = str(data.get("positive_prompt", "")).strip() or _DEFAULT_POSITIVE
    negative = str(data.get("negative_prompt", "")).strip() or _DEFAULT_NEGATIVE
    data["positive_prompt"] = positive
    data["negative_prompt"] = negative

    _cache_store(ck, data)
    return data


__all__ = [
    "analyze_garment_prompt_with_gemini",
    "fallback_describe_garment",
    "GeminiPromptUnavailableError",
]
