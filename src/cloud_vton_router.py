from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path

try:
    from gradio_client import Client, handle_file
    from gradio_client.exceptions import AppError
    _HAS_GRADIO_CLIENT = True
except ImportError:
    _HAS_GRADIO_CLIENT = False
    Client = None  # type: ignore[assignment,misc]
    handle_file = None  # type: ignore[assignment]
    AppError = Exception  # type: ignore[assignment,misc]


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
COOLDOWN_SECONDS = int(os.getenv("VTON_BACKEND_COOLDOWN_SECONDS", "300"))

_BACKEND_STATE: dict[str, float] = {}
_CLIENT_CACHE: dict[str, Client] = {}


class CloudVTONUnavailableError(RuntimeError):
    pass


def _get_client(space: str) -> Client:
    if not _HAS_GRADIO_CLIENT:
        raise CloudVTONUnavailableError("gradio_client not installed")
    key = f"{space}:{HF_TOKEN or ''}"
    if key not in _CLIENT_CACHE:
        if HF_TOKEN:
            _CLIENT_CACHE[key] = Client(space, hf_token=HF_TOKEN)
        else:
            _CLIENT_CACHE[key] = Client(space)
    return _CLIENT_CACHE[key]


def _in_cooldown(backend: str) -> bool:
    until = _BACKEND_STATE.get(backend, 0)
    return time.time() < until


def _mark_cooldown(backend: str) -> None:
    _BACKEND_STATE[backend] = time.time() + COOLDOWN_SECONDS


def _should_cooldown(message: str) -> bool:
    keys = [
        "No GPU was available",
        "RUNTIME_ERROR",
        "PAUSED",
        "BUILD_ERROR",
        "invalid state",
    ]
    return any(k in message for k in keys)


# ── Tier 1: CatVTON (multi-space failover) ─────────────────────────

def _catvton_tryon(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    steps: int,
    guidance: float,
    seed: int,
    cloth_type: str = "upper",
) -> Path:
    """CatVTON via multi-space client with automatic failover."""
    backend = "catvton"
    if _in_cooldown(backend):
        raise CloudVTONUnavailableError(f"{backend} in cooldown")

    try:
        from src.catvton_client import (
            generate_with_catvton,
            CatVTONUnavailableError as CatErr,
        )
        return generate_with_catvton(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            steps=steps,
            guidance=guidance,
            seed=seed,
            cloth_type=cloth_type,
        )
    except Exception as exc:
        message = str(exc)
        if _should_cooldown(message):
            _mark_cooldown(backend)
        raise CloudVTONUnavailableError(f"{backend}: {message}") from exc


# ── Tier 2: IDM-VTON ──────────────────────────────────────────────

def _idm_tryon(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    garment_description: str,
    denoise_steps: int,
    seed: int,
) -> Path:
    space = os.getenv("IDMVTON_SPACE", "yisol/IDM-VTON")
    backend = f"idm:{space}"

    if _in_cooldown(backend):
        raise CloudVTONUnavailableError(f"{backend} in cooldown")

    person_path = str(Path(person_image_path).resolve())
    cloth_path = str(Path(cloth_image_path).resolve())

    person_editor_value = {
        "background": handle_file(person_path),
        "layers": [],
        "composite": handle_file(person_path),
    }

    if not garment_description.strip():
        garment_description = "a realistic top garment"

    try:
        output_path, _ = _get_client(space).predict(
            **{
                "dict": person_editor_value,
                "garm_img": handle_file(cloth_path),
                "garment_des": garment_description,
                "is_checked": True,
                "is_checked_crop": False,
                "denoise_steps": float(max(20, min(40, denoise_steps))),
                "seed": float(seed),
                "api_name": "/tryon",
            }
        )
        return Path(output_path)
    except Exception as exc:
        message = str(exc)
        if _should_cooldown(message):
            _mark_cooldown(backend)
        raise CloudVTONUnavailableError(f"{backend}: {message}") from exc


# ── Tier 3: Fal.ai FLUX Klein 9B ──────────────────────────────────

def _fal_flux_tryon(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    style_prompt: str,
    steps: int,
    guidance: float,
) -> Path:
    """Fal.ai FLUX Klein 9B try-on backend (paid API, most reliable)."""
    backend = "fal_flux"
    if _in_cooldown(backend):
        raise CloudVTONUnavailableError(f"{backend} in cooldown")

    try:
        from src.fal_flux_client import (
            generate_with_fal_flux_tryon,
            FalFluxUnavailableError,
        )
        output_url = generate_with_fal_flux_tryon(
            person_image_path=str(person_image_path),
            top_image_path=str(cloth_image_path),
            bottom_image_path=None,
            user_prompt=style_prompt,
            steps=steps,
            guidance=guidance,
        )
        # Download URL to local temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        urllib.request.urlretrieve(output_url, tmp.name)
        return Path(tmp.name)
    except Exception as exc:
        message = str(exc)
        if _should_cooldown(message):
            _mark_cooldown(backend)
        raise CloudVTONUnavailableError(f"{backend}: {message}") from exc


# ── Tier 4: Replicate (placeholder for future) ────────────────────

def _replicate_tryon(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    style_prompt: str,
    steps: int,
    seed: int,
) -> Path:
    """Replicate.com backend (requires REPLICATE_API_TOKEN)."""
    api_token = os.getenv("REPLICATE_API_TOKEN", "").strip()
    if not api_token:
        raise CloudVTONUnavailableError("REPLICATE_API_TOKEN not set")

    backend = "replicate"
    if _in_cooldown(backend):
        raise CloudVTONUnavailableError(f"{backend} in cooldown")

    try:
        import replicate
        output = replicate.run(
            os.getenv("REPLICATE_VTON_MODEL", "cuuupid/idm-vton:latest"),
            input={
                "human_img": open(str(person_image_path), "rb"),
                "garm_img": open(str(cloth_image_path), "rb"),
                "garment_des": style_prompt or "a realistic top garment",
                "seed": seed,
                "steps": max(20, min(40, steps)),
            },
        )
        output_url = str(output) if isinstance(output, str) else str(output[0])
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        urllib.request.urlretrieve(output_url, tmp.name)
        return Path(tmp.name)
    except Exception as exc:
        message = str(exc)
        if _should_cooldown(message):
            _mark_cooldown(backend)
        raise CloudVTONUnavailableError(f"{backend}: {message}") from exc


# ── Public router: 3-tier fallback (CatVTON -> IDM-VTON -> Fal.ai)

def generate_with_cloud_router(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    style_prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> tuple[Path, str]:
    errors: list[str] = []

    # Tier 1: CatVTON (fastest, free HF Spaces, multi-space failover)
    try:
        return _catvton_tryon(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            steps=steps,
            guidance=guidance,
            seed=seed,
            cloth_type="upper",
        ), "CatVTON Cloud"
    except CloudVTONUnavailableError as exc:
        errors.append(str(exc))

    # Tier 2: IDM-VTON (higher quality, free HF Spaces)
    try:
        return _idm_tryon(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            garment_description=style_prompt,
            denoise_steps=steps,
            seed=seed,
        ), "IDM-VTON Cloud"
    except CloudVTONUnavailableError as exc:
        errors.append(str(exc))

    # Tier 3: Fal.ai FLUX Klein 9B (paid API, most reliable)
    try:
        return _fal_flux_tryon(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            style_prompt=style_prompt,
            steps=steps,
            guidance=guidance,
        ), "Fal.ai FLUX"
    except CloudVTONUnavailableError as exc:
        errors.append(str(exc))

    # Tier 4: Replicate (if configured)
    try:
        return _replicate_tryon(
            person_image_path=person_image_path,
            cloth_image_path=cloth_image_path,
            style_prompt=style_prompt,
            steps=steps,
            seed=seed,
        ), "Replicate"
    except CloudVTONUnavailableError as exc:
        errors.append(str(exc))

    raise CloudVTONUnavailableError(" | ".join(errors[:3]))
