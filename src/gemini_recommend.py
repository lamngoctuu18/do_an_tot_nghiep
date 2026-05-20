"""Gemini Vision → outfit styling recommendation.

Sends the *try-on result* image to Gemini and asks for a structured JSON
suggestion: main color, palette to pair with, colors to avoid, items to
combine, style summary and a one-line tip. Used by the
`POST /api/tryon/recommend` endpoint after a successful try-on so the user
gets stylist-level outfit advice on top of the generated image.

Mirrors the reliability story of `gemini_prompt`:
- SHA256 image-hash cache (per category+occasion+style+model)
- exponential backoff with jitter on 503 / quota / timeout
- automatic fallback to `GEMINI_MODEL_FALLBACK`
- category-aware local fallback dict when Gemini is unreachable, so the
  endpoint always returns something usable.
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


class GeminiRecommendUnavailableError(RuntimeError):
    """Raised when Gemini outfit recommend can't run (missing key, SDK absent)."""


_TRUTHY = {"1", "true", "yes", "on"}
_TRANSIENT_TOKENS = (
    "503", "unavailable", "high demand", "overloaded",
    "rate", "quota", "resource_exhausted", "deadline",
    "timeout", "temporarily",
)


_RECOMMEND_PROMPT = """Bạn là stylist thời trang cho hệ thống virtual try-on.

Hãy phân tích ảnh kết quả thử đồ và đưa ra gợi ý phối đồ cho người dùng.

Thông tin bổ sung:
- Category trang phục: {category}
- Occasion / ngữ cảnh sử dụng: {occasion}
- Style mong muốn: {style}

Yêu cầu:
- Nhận diện màu chính của trang phục.
- Gợi ý các màu nên phối cùng (3-6 màu).
- Gợi ý các màu nên tránh (2-4 màu).
- Gợi ý quần / áo / giày / túi / phụ kiện phù hợp (3-5 item).
- Nếu trang phục là váy/dress, ưu tiên giày, túi, áo khoác, phụ kiện.
- Nếu là áo/top/hoodie/jacket, ưu tiên quần/váy, giày, phụ kiện.
- Nếu là quần/jeans/shorts/skirt, ưu tiên áo, giày, phụ kiện.
- Trả lời bằng tiếng Việt, ngắn gọn, không markdown.
- CHỈ trả về JSON hợp lệ theo đúng schema. Không thêm giải thích, không thêm ```json fences.

Schema:
{{
  "main_color": "string",
  "garment_type": "string",
  "suitable_colors": ["string"],
  "avoid_colors": ["string"],
  "recommended_items": [
    {{"type": "string", "name": "string", "reason": "string"}}
  ],
  "style_summary": "string",
  "short_tip": "string"
}}
"""


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


def _encode_jpeg(image_rgb: np.ndarray, quality: int = 88) -> bytes:
    bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode result image to JPEG")
    return bytes(buf)


def _is_transient(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(tok in msg for tok in _TRANSIENT_TOKENS)


def _cache_dir() -> Path:
    base = os.getenv("VTON_GEMINI_CACHE", str(Path.home() / ".vton_cache" / "gemini"))
    p = Path(base) / "recommend"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(
    image_bytes: bytes,
    category: str,
    occasion: str,
    style: str,
    model_id: str,
) -> str:
    h = hashlib.sha256()
    h.update(hashlib.sha256(image_bytes).digest())
    h.update(b"|")
    h.update((category or "").encode())
    h.update(b"|")
    h.update((occasion or "").encode())
    h.update(b"|")
    h.update((style or "").encode())
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
                sleep_s = min(base_delay * (2 ** attempt) + random.uniform(0, 0.8), max_delay)
                print(
                    f"[GEMINI_RECOMMEND] transient on {model_id} "
                    f"({attempt + 1}/{max_retries}): {exc}. retry in {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


# ── Category-aware fallback ──────────────────────────────────────────

def fallback_recommendation(
    category: str = "garment",
    occasion: str = "casual",
    style: str = "minimal",
) -> dict[str, Any]:
    """Return a sensible recommendation dict when Gemini is unreachable."""
    cat = (category or "").strip().lower()

    if cat in {"dress", "skirt"}:
        return {
            "main_color": "màu trung tính",
            "garment_type": "váy liền thân" if cat == "dress" else "chân váy",
            "suitable_colors": ["trắng", "kem", "be", "nâu", "đen", "vàng nhạt"],
            "avoid_colors": ["xanh neon", "cam chói", "hồng neon"],
            "recommended_items": [
                {
                    "type": "shoes",
                    "name": "giày búp bê hoặc cao gót màu kem",
                    "reason": "giúp tổng thể nhẹ nhàng, nữ tính và hợp tông trang phục",
                },
                {
                    "type": "bag",
                    "name": "túi xách nhỏ màu nâu hoặc đen",
                    "reason": "tạo điểm nhấn vừa đủ mà không phá tông màu",
                },
                {
                    "type": "outerwear",
                    "name": "blazer màu kem hoặc đen",
                    "reason": "phù hợp khi đi làm hoặc sự kiện nhẹ",
                },
                {
                    "type": "accessory",
                    "name": "khuyên tai hoặc dây chuyền vàng nhạt",
                    "reason": "làm outfit mềm mại và sang trọng hơn",
                },
            ],
            "style_summary": "Phong cách thanh lịch, nhẹ nhàng, phù hợp đi làm, đi chơi hoặc gặp mặt.",
            "short_tip": "Phối với giày kem, túi nâu nhỏ và phụ kiện vàng nhạt.",
        }

    if cat in {"hoodie", "top", "tshirt", "shirt", "jacket", "outer"}:
        return {
            "main_color": "màu trung tính",
            "garment_type": cat,
            "suitable_colors": ["đen", "trắng", "xanh denim", "xám", "be"],
            "avoid_colors": ["màu neon", "màu quá tương phản"],
            "recommended_items": [
                {
                    "type": "bottom",
                    "name": "jeans xanh hoặc quần suông đen",
                    "reason": "dễ phối và tạo tổng thể cân đối",
                },
                {
                    "type": "shoes",
                    "name": "sneaker trắng",
                    "reason": "hợp phong cách casual, trẻ trung",
                },
                {
                    "type": "accessory",
                    "name": "mũ lưỡi trai hoặc túi đeo chéo nhỏ",
                    "reason": "tăng tính năng động cho outfit",
                },
            ],
            "style_summary": "Phong cách casual, dễ mặc hằng ngày, phù hợp đi học hoặc đi chơi.",
            "short_tip": "Phối jeans xanh, sneaker trắng và phụ kiện đơn giản.",
        }

    if cat in {"pants", "jeans", "shorts", "trousers"}:
        return {
            "main_color": "màu trung tính",
            "garment_type": cat,
            "suitable_colors": ["trắng", "đen", "xám", "be", "xanh navy"],
            "avoid_colors": ["màu quá chói", "màu trùng với quần"],
            "recommended_items": [
                {
                    "type": "top",
                    "name": "sơ mi trắng hoặc áo thun basic",
                    "reason": "giúp phần thân trên gọn và dễ phối",
                },
                {
                    "type": "shoes",
                    "name": "sneaker trắng hoặc loafer đen",
                    "reason": "giúp outfit sạch, gọn và dễ ứng dụng",
                },
                {
                    "type": "accessory",
                    "name": "thắt lưng da đen",
                    "reason": "tạo điểm nhấn ở eo và làm outfit chỉn chu hơn",
                },
            ],
            "style_summary": "Phong cách gọn gàng, dễ mặc, phù hợp đi học, đi làm hoặc đi chơi.",
            "short_tip": "Phối áo trắng, sneaker trắng và thắt lưng đen.",
        }

    return {
        "main_color": "màu trung tính",
        "garment_type": "trang phục",
        "suitable_colors": ["trắng", "đen", "be", "xám", "nâu"],
        "avoid_colors": ["màu neon", "màu quá chói"],
        "recommended_items": [
            {
                "type": "accessory",
                "name": "phụ kiện tối giản",
                "reason": "giúp tổng thể hài hòa, không bị rối",
            }
        ],
        "style_summary": "Phong cách đơn giản, dễ phối.",
        "short_tip": "Dùng các màu trung tính để outfit hài hòa hơn.",
    }


# ── Main entry ───────────────────────────────────────────────────────

def recommend_outfit_with_gemini(
    result_rgb: np.ndarray,
    category: str = "garment",
    occasion: str = "casual",
    style: str = "minimal",
) -> dict[str, Any]:
    """Send the try-on result image to Gemini and return a styling dict.

    Raises ``GeminiRecommendUnavailableError`` when the SDK or API key is
    missing, or when Gemini fails after retry + fallback-model attempts.
    Callers should typically catch that and fall back to
    ``fallback_recommendation(category, ...)``.
    """
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise GeminiRecommendUnavailableError("Missing GEMINI_API_KEY")

    try:
        from google import genai
        from google.genai import types
    except Exception as exc:
        raise GeminiRecommendUnavailableError(
            "Missing google-genai. Install: pip install -U google-genai"
        ) from exc

    image_bytes = _encode_jpeg(result_rgb)
    model_id = os.getenv("GEMINI_RECOMMEND_MODEL",
                         os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    fallback_model = os.getenv("GEMINI_MODEL_FALLBACK", "gemini-2.5-flash-lite").strip()

    ck = _cache_key(image_bytes, category, occasion, style, model_id)
    cached = _cache_load(ck)
    if cached:
        print(f"[GEMINI_RECOMMEND] cache hit ({ck[:8]}…)")
        cached["source"] = "cache"
        return cached

    prompt = _RECOMMEND_PROMPT.format(
        category=category or "garment",
        occasion=occasion or "casual",
        style=style or "minimal",
    )

    client = genai.Client(api_key=api_key)
    try:
        gen_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            temperature=0.5,
            max_output_tokens=1200,
        )
    except Exception:
        gen_config = None

    def _do_call(use_model: str):
        return client.models.generate_content(
            model=use_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
            config=gen_config,
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
        raise GeminiRecommendUnavailableError(f"Gemini recommend failed: {exc}") from exc

    raw_text = getattr(response, "text", "") or ""
    try:
        data = _safe_json_loads(raw_text)
    except Exception as exc:
        raise GeminiRecommendUnavailableError(f"Gemini JSON parse failed: {exc}") from exc

    # Defensive defaults so the frontend never crashes on missing keys.
    data.setdefault("main_color", "")
    data.setdefault("garment_type", category or "")
    data.setdefault("suitable_colors", [])
    data.setdefault("avoid_colors", [])
    data.setdefault("recommended_items", [])
    data.setdefault("style_summary", "")
    data.setdefault("short_tip", "")
    data["source"] = "gemini"

    _cache_store(ck, data)
    return data


__all__ = [
    "recommend_outfit_with_gemini",
    "fallback_recommendation",
    "GeminiRecommendUnavailableError",
]
