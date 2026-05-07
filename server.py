"""Unified server – serves React frontend + try-on API from a single process."""

from __future__ import annotations

import asyncio
import io
import os
import re
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Import the try_on function from the existing Gradio app.
# The Gradio Blocks object is created at module level but launch()
# is guarded by __name__ == "__main__", so it won't start.
from app import try_on

TIMEOUT_SECONDS = int(os.getenv("VTON_CLOUD_TIMEOUT", "600"))

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
    for line in info.split("\n"):
        stripped = line.strip()
        if "warning" in stripped.lower() or stripped.startswith("?"):
            warnings.append(stripped)
    return " ".join(warnings)


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
    gen_steps: int = Form(20),
    gen_guidance: float = Form(2.5),
    preserve_strength: float = Form(0.90),
    quality_preset: str = Form("balanced"),
    refiner_mode: str = Form("dpm++"),
    cloth_type: str = Form("auto"),
    use_catvton_cloud: bool = Form(True),
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


# ── Serve React build (static files) ──────────────────────────────
DIST_DIR = Path(__file__).resolve().parent / "web-shop" / "dist"

if DIST_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Never serve index.html for API paths
        if full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"error": "Not found"})
        file_path = DIST_DIR / full_path
        if full_path and file_path.is_file():
            headers = {}
            if file_path.name == "index.html":
                headers["Cache-Control"] = "no-store"
            return FileResponse(str(file_path), headers=headers)
        return FileResponse(
            str(DIST_DIR / "index.html"),
            headers={"Cache-Control": "no-store"},
        )


if __name__ == "__main__":
    import subprocess
    import sys

    web_shop_dir = Path(__file__).resolve().parent / "web-shop"
    if web_shop_dir.is_dir() and not DIST_DIR.is_dir():
        print("Building React frontend...")
        subprocess.run(
            ["npm", "run", "build"],
            cwd=str(web_shop_dir),
            check=True,
            shell=True,
        )
        print("Frontend built successfully")

    print("Starting unified server at http://localhost:8000")
    print("   Frontend:  http://localhost:8000")
    print("   Try-on:    http://localhost:8000/try-on")
    print("   API (POST): http://localhost:8000/api/tryon")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)
