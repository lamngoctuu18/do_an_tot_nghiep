from __future__ import annotations

import os
import time
from pathlib import Path

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError


HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
COOLDOWN_SECONDS = int(os.getenv("VTON_BACKEND_COOLDOWN_SECONDS", "300"))

_BACKEND_STATE: dict[str, float] = {}
_CLIENT_CACHE: dict[str, Client] = {}


class CloudVTONUnavailableError(RuntimeError):
    pass


def _get_client(space: str) -> Client:
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


def _catvton_tryon(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    steps: int,
    guidance: float,
    seed: int,
    cloth_type: str = "upper",
) -> Path:
    space = os.getenv("CATVTON_SPACE", "FIT-Check/CatVTON")
    backend = f"catvton:{space}"

    if _in_cooldown(backend):
        raise CloudVTONUnavailableError(f"{backend} đang cooldown")

    person_path = str(Path(person_image_path).resolve())
    cloth_path = str(Path(cloth_image_path).resolve())

    person_editor_value = {
        "background": handle_file(person_path),
        "layers": [],
        "composite": handle_file(person_path),
    }

    try:
        result_path = _get_client(space).predict(
            person_image=person_editor_value,
            cloth_image=handle_file(cloth_path),
            cloth_type=cloth_type,
            num_inference_steps=float(max(10, min(100, steps))),
            guidance_scale=float(max(0.0, min(7.5, guidance))),
            seed=float(seed),
            show_type="result only",
            api_name="/submit_function",
        )
        return Path(result_path)
    except Exception as exc:
        message = str(exc)
        if _should_cooldown(message):
            _mark_cooldown(backend)
        raise CloudVTONUnavailableError(f"{backend}: {message}") from exc


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
        raise CloudVTONUnavailableError(f"{backend} đang cooldown")

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


def generate_with_cloud_router(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    style_prompt: str,
    steps: int,
    guidance: float,
    seed: int,
) -> tuple[Path, str]:
    errors: list[str] = []

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

    raise CloudVTONUnavailableError(" | ".join(errors[:2]))
