"""Unified server – serves React frontend + try-on API from a single process."""

from __future__ import annotations

import asyncio
import io
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent
ENV_FILE = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

# Import the try_on function from the existing Gradio app.
# The Gradio Blocks object is created at module level but launch()
# is guarded by __name__ == "__main__", so it won't start.
from app import try_on
try:
    from src.gemini_prompt import (
        analyze_garment_prompt_with_gemini,
        fallback_describe_garment,
        GeminiPromptUnavailableError,
    )
except Exception:  # pragma: no cover
    analyze_garment_prompt_with_gemini = None
    fallback_describe_garment = None
    GeminiPromptUnavailableError = RuntimeError
try:
    from src.gemini_recommend import (
        recommend_outfit_with_gemini,
        fallback_recommendation,
        GeminiRecommendUnavailableError,
    )
except Exception:  # pragma: no cover
    recommend_outfit_with_gemini = None
    fallback_recommendation = None
    GeminiRecommendUnavailableError = RuntimeError

TIMEOUT_SECONDS = int(os.getenv("VTON_TIMEOUT_SECONDS", os.getenv("VTON_CLOUD_TIMEOUT", "1200")))
WEB_SHOP_DIR = ROOT_DIR / "web-shop"
DIST_DIR = WEB_SHOP_DIR / "dist"
NEST_BASE_URL = os.getenv("NEST_BASE_URL", "http://127.0.0.1:3000")

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="Virtual Try-On")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Info", "X-Pipeline-Info", "X-Backend", "X-Warning"],
)


def _read_upload(contents: bytes) -> np.ndarray:
    """Decode uploaded image bytes -> RGB numpy array."""
    arr = np.frombuffer(contents, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _encode_png(rgb: np.ndarray) -> bytes:
    """Encode RGB numpy array -> PNG bytes."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _safe_header(val: str) -> str:
    """Make header value safe for h11.

    h11 field_value regex: ``[^\\x00\\s]+(?:[ \\t]+[^\\x00\\s]+)*``
    → value must NOT have leading/trailing whitespace, and non-NUL
      non-whitespace chars between optional space/tab gaps.
    We keep only printable ASCII, collapse runs of whitespace, and strip.
    """
    cleaned = []
    for ch in val:
        code = ord(ch)
        if 0x21 <= code <= 0x7E:          # visible ASCII (no space/tab)
            cleaned.append(ch)
        elif code == 0x20 or code == 0x09: # space or tab → single space
            cleaned.append(" ")
        else:                              # non-ASCII / control → skip
            pass
    # Collapse multiple spaces and strip leading/trailing whitespace
    result = re.sub(r" +", " ", "".join(cleaned)).strip()
    # h11 rejects empty or trailing-space header values. Keep enough room for
    # late pipeline tags such as DressFullErase/DressFullGenMask/Diffusion so
    # the UI can confirm which generation path actually ran.
    result = result[:1000].strip()
    if not result:
        return "ok"
    return result


def _extract_info_field(info: str, field: str) -> str:
    """Extract a specific field value from the info string."""
    for line in info.split("\n"):
        stripped = line.strip()
        if stripped.startswith(f"{field}:"):
            return stripped[len(field) + 1:].strip()
    return ""


def _extract_warning(info: str) -> str:
    """Extract warning lines from info string."""
    warnings = []
    warning_markers = (
        "warning",
        "cloud vton unavailable",
        "local diffusion failed",
        "falling back",
        "diffusion unstable",
        "nearly black",
    )
    for line in info.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if any(marker in lower for marker in warning_markers) or stripped.startswith("?"):
            warnings.append(stripped)
    return " ".join(warnings)


def _reload_env_file() -> None:
    """Load newly-added .env values for long-running dev processes."""
    load_dotenv(dotenv_path=ENV_FILE, override=False)


def _fallback_describe_payload(category_lock: str, user_prompt: str = "", error: str = "") -> dict:
    """Return a valid prompt payload when Gemini is unavailable."""
    category = (category_lock or "auto").strip().lower() or "auto"
    if category in {"jeans", "pants", "shorts"}:
        positive = (
            "realistic denim pants matching the reference garment, correct waist placement, "
            "two separated pant legs, visible crotch seam, natural denim texture and folds"
        )
        negative = (
            "old pants visible, blue paint spill, merged legs, missing crotch gap, skirt shape, "
            "floating fabric, extra cloth outside legs, blurry denim"
        )
    elif category in {"dress", "skirt"}:
        positive = "realistic garment matching the reference, clean silhouette, natural fabric folds"
        negative = "old clothing visible, extra garment layer, distorted body, blurry fabric"
    else:
        positive = "realistic virtual try-on matching the reference garment, clean edges, natural fabric folds"
        negative = "old clothing visible, pasted cloth, wrong category, extra clothing layer, blurry fabric"
    if user_prompt.strip():
        positive = f"{positive}, {user_prompt.strip()}"
    return {
        "category": category if category != "auto" else "generic",
        "cloth_type": "lower" if category in {"jeans", "pants", "shorts", "skirt"} else "upper",
        "positive_prompt": positive,
        "negative_prompt": negative,
        "provider": "fallback",
        "gemini_error": error,
        "notes": ["Gemini describe unavailable; using local fallback prompt."],
    }


def _frontend_needs_build() -> bool:
    index_file = DIST_DIR / "index.html"
    if not index_file.is_file():
        return True

    built_at = index_file.stat().st_mtime
    watched_paths = [
        WEB_SHOP_DIR / "src",
        WEB_SHOP_DIR / "package.json",
        WEB_SHOP_DIR / "package-lock.json",
        WEB_SHOP_DIR / "vite.config.ts",
        WEB_SHOP_DIR / "tsconfig.json",
        WEB_SHOP_DIR / "tsconfig.app.json",
        WEB_SHOP_DIR / "index.html",
    ]
    for path in watched_paths:
        if path.is_file() and path.stat().st_mtime > built_at:
            return True
        if path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.stat().st_mtime > built_at:
                    return True
    return False


def _ensure_frontend_build() -> None:
    """Build the React shop when the production bundle is missing or stale."""
    if not _frontend_needs_build():
        return
    if not WEB_SHOP_DIR.is_dir():
        return

    print("Building React frontend...")
    npm_cmd = "npm.cmd" if sys.platform.startswith("win") else "npm"
    subprocess.run([npm_cmd, "run", "build"], cwd=str(WEB_SHOP_DIR), check=True)
    print("Frontend built successfully")


def _copy_request_headers(request: Request) -> dict[str, str]:
    """Forward useful client headers while leaving hop-by-hop headers behind."""
    blocked = {
        "host",
        "connection",
        "content-length",
        "accept-encoding",
        "transfer-encoding",
    }
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in blocked
    }


def _copy_response_headers(headers) -> dict[str, str]:
    blocked = {
        "connection",
        "content-encoding",
        "content-length",
        "transfer-encoding",
    }
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in blocked
    }


async def _proxy_to_nest(request: Request, path: str) -> Response:
    """Proxy shop API/upload traffic to the NestJS backend on port 3000."""
    query = request.url.query
    target = f"{NEST_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    if query:
        target = f"{target}?{query}"

    body = await request.body()
    proxied = urllib.request.Request(
        target,
        data=body if body else None,
        method=request.method,
        headers=_copy_request_headers(request),
    )
    try:
        with urllib.request.urlopen(proxied, timeout=TIMEOUT_SECONDS) as upstream:
            return Response(
                content=upstream.read(),
                status_code=upstream.status,
                headers=_copy_response_headers(upstream.headers),
                media_type=upstream.headers.get("content-type"),
            )
    except urllib.error.HTTPError as exc:
        return Response(
            content=exc.read(),
            status_code=exc.code,
            headers=_copy_response_headers(exc.headers),
            media_type=exc.headers.get("content-type"),
        )
    except urllib.error.URLError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": "NestJS API is not running",
                "detail": str(exc.reason),
                "expected": NEST_BASE_URL,
            },
        )


# ── Gemini describe-only endpoint ──────────────────────────────────
# Runs Gemini Vision and returns the structured prompt JSON without
# triggering the heavy try-on pipeline. The React UI calls this as the
# "Mô tả thử đồ" step to auto-fill the prompt textbox before submitting
# the main /api/tryon request.
@app.post("/api/tryon/describe")
async def api_tryon_describe(
    person: UploadFile = File(...),
    cloth: UploadFile = File(...),
    category_lock: str = Form("auto"),
    user_prompt: str = Form(""),
):
    _reload_env_file()
    if analyze_garment_prompt_with_gemini is None:
        error = "Gemini SDK not installed (pip install google-genai)"
        print(f"[GEMINI] describe fallback: {error}")
        return JSONResponse(content=_fallback_describe_payload(category_lock, user_prompt, error))

    try:
        person_rgb = _read_upload(await person.read())
        cloth_rgb = _read_upload(await cloth.read())
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    try:
        loop = asyncio.get_event_loop()
        data = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: analyze_garment_prompt_with_gemini(
                    person_rgb=person_rgb,
                    cloth_rgb=cloth_rgb,
                    category_lock=category_lock,
                    user_prompt=user_prompt,
                    force=True,
                ),
            ),
            timeout=120.0,
        )
    except GeminiPromptUnavailableError as exc:
        error = str(exc)
        print(f"[GEMINI] describe fallback: {error}")
        return JSONResponse(content=_fallback_describe_payload(category_lock, user_prompt, error))
    except asyncio.TimeoutError:
        error = "Gemini timed out"
        print(f"[GEMINI] describe fallback: {error}")
        return JSONResponse(content=_fallback_describe_payload(category_lock, user_prompt, error))
    except Exception as exc:
        error = f"Gemini failed: {exc}"
        print(f"[GEMINI] describe fallback: {error}")
        return JSONResponse(content=_fallback_describe_payload(category_lock, user_prompt, error))

    return JSONResponse(content=data)


# ── Try-on endpoint ────────────────────────────────────────────────
@app.post("/api/tryon")
async def api_tryon(
    person: UploadFile = File(...),
    cloth: UploadFile = File(...),
    fit_scale: float = Form(1.12),
    alpha: float = Form(0.65),
    y_offset: float = Form(-0.01),
    use_gen: bool = Form(True),
    style_prompt: str = Form(""),
    gen_steps: int = Form(24),
    gen_guidance: float = Form(5.2),
    preserve_strength: float = Form(0.82),
    quality_preset: str = Form("hq"),
    refiner_mode: str = Form("dpm++"),
    cloth_type: str = Form("auto"),
    use_catvton_cloud: bool = Form(True),
    use_gemini_prompt: bool = Form(True),
    prompt_mode: str = Form("auto"),
):
    """Run the full try-on pipeline and return the result image as PNG."""
    person_bytes = await person.read()
    cloth_bytes = await cloth.read()

    try:
        person_rgb = _read_upload(person_bytes)
        cloth_rgb = _read_upload(cloth_bytes)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # Keep the public API on AI refinement. Older built frontends may still send
    # use_gen=false from the removed CPU-only button; treat that as Local SD.
    effective_use_gen = True

    try:
        loop = asyncio.get_event_loop()
        output, info = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: try_on(
                    person_img=person_rgb,
                    cloth_img=cloth_rgb,
                    fit_scale=fit_scale,
                    alpha=alpha,
                    y_offset=y_offset,
                    use_gen=effective_use_gen,
                    style_prompt=style_prompt,
                    gen_steps=gen_steps,
                    gen_guidance=gen_guidance,
                    preserve_strength=preserve_strength,
                    quality_preset=quality_preset,
                    refiner_mode=refiner_mode,
                    cloth_type=cloth_type,
                    use_catvton_cloud=use_catvton_cloud,
                    use_gemini_prompt=use_gemini_prompt,
                    prompt_mode=prompt_mode,
                ),
            ),
            timeout=float(TIMEOUT_SECONDS),
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": f"Processing timed out after {TIMEOUT_SECONDS}s"},
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()  # Print full traceback to terminal for debugging
        err_msg = str(exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Pipeline error: {err_msg}"},
            headers={"X-Warning": _safe_header(err_msg)},
        )

    png_bytes = _encode_png(output)

    # Parse info string into structured headers for frontend
    pipeline = _extract_info_field(info, "Pipeline")
    backend = _extract_info_field(info, "Backend")
    warning = _extract_warning(info)
    safe_info = _safe_header(info.replace("\n", " | "))

    return StreamingResponse(
        io.BytesIO(png_bytes),
        media_type="image/png",
        headers={
            "X-Info": safe_info,
            "X-Pipeline-Info": _safe_header(pipeline),
            "X-Backend": _safe_header(backend),
            "X-Warning": _safe_header(warning),
        },
    )


# ── Health endpoint ────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ── Outfit recommend endpoint ──────────────────────────────────────
@app.post("/api/tryon/recommend")
async def api_tryon_recommend(
    result: UploadFile = File(...),
    category: str = Form("garment"),
    occasion: str = Form("casual"),
    style: str = Form("minimal"),
):
    """Send the try-on result image to Gemini for outfit styling advice.

    Always returns 200 — falls back to a category-aware recommendation when
    Gemini is unavailable so the UI never breaks.
    """
    _reload_env_file()
    try:
        result_rgb = _read_upload(await result.read())
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"success": False, "error": str(exc)})

    def _fallback_payload(reason: str) -> dict:
        rec = (
            fallback_recommendation(category, occasion, style)
            if fallback_recommendation is not None
            else {
                "main_color": "",
                "garment_type": category,
                "suitable_colors": [],
                "avoid_colors": [],
                "recommended_items": [],
                "style_summary": "",
                "short_tip": "",
            }
        )
        rec["source"] = "fallback"
        return {"success": True, "recommendation": rec, "warning": reason}

    if recommend_outfit_with_gemini is None:
        return JSONResponse(content=_fallback_payload("Gemini SDK not installed"))

    try:
        loop = asyncio.get_event_loop()
        data = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: recommend_outfit_with_gemini(
                    result_rgb=result_rgb,
                    category=category,
                    occasion=occasion,
                    style=style,
                ),
            ),
            timeout=90.0,
        )
        return JSONResponse(content={"success": True, "recommendation": data})
    except GeminiRecommendUnavailableError as exc:
        print(f"[RECOMMEND] fallback: {exc}")
        return JSONResponse(content=_fallback_payload(str(exc)))
    except asyncio.TimeoutError:
        print("[RECOMMEND] fallback: timeout")
        return JSONResponse(content=_fallback_payload("Gemini timed out"))
    except Exception as exc:
        print(f"[RECOMMEND] fallback: {exc}")
        return JSONResponse(content=_fallback_payload(f"Recommend failed: {exc}"))


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_api(request: Request, path: str):
    return await _proxy_to_nest(request, f"api/{path}")


@app.api_route(
    "/uploads/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_uploads(request: Request, path: str):
    return await _proxy_to_nest(request, f"uploads/{path}")


# ── Serve React build (static files) ──────────────────────────────
app.mount(
    "/assets",
    StaticFiles(directory=str(DIST_DIR / "assets"), check_dir=False),
    name="assets",
)


@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Never serve index.html for API paths.
    if full_path.startswith("api/"):
        return JSONResponse(status_code=404, content={"error": "Not found"})
    file_path = DIST_DIR / full_path
    if full_path and file_path.is_file():
        headers = {}
        if file_path.name == "index.html":
            headers["Cache-Control"] = "no-store"
        return FileResponse(str(file_path), headers=headers)
    index_file = DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file), headers={"Cache-Control": "no-store"})
    return JSONResponse(
        status_code=503,
        content={
            "error": "Frontend build is missing",
            "hint": "Run: cd web-shop && npm.cmd run build",
        },
    )


if __name__ == "__main__":
    _ensure_frontend_build()

    print("Starting unified server at http://localhost:8000")
    print("   Frontend:  http://localhost:8000")
    print("   Try-on:    http://localhost:8000/try-on")
    print("   API (POST): http://localhost:8000/api/tryon")
    print(f"   Shop API proxy: {NEST_BASE_URL}/api")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=600,
        timeout_graceful_shutdown=5,
    )
