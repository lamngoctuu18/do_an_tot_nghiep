from __future__ import annotations

import os
from pathlib import Path

from gradio_client import Client, handle_file
from gradio_client.exceptions import AppError


DEFAULT_SPACES = os.getenv(
    "CATVTON_SPACES",
    # v18.23: expanded space list. Many older spaces have been paused; we
    # try a wider net of public CatVTON forks so at least one is alive.
    "zhengchong/CatVTON,"
    "FIT-Check/CatVTON,"
    "Nymbo/CatVTON,"
    "Shad0ws/CatVTON,"
    "hungdang1610/CatVTON,"
    "PrasannaKumar1812/CatVTON,"
    "vilarin/CatVTON",
)


class CatVTONUnavailableError(RuntimeError):
    pass


_CLIENT_CACHE: dict[str, Client] = {}


def _spaces() -> list[str]:
    return [space.strip() for space in DEFAULT_SPACES.split(",") if space.strip()]


def _get_client(space: str) -> Client:
    if space not in _CLIENT_CACHE:
        _CLIENT_CACHE[space] = Client(space)
    return _CLIENT_CACHE[space]


def generate_with_catvton(
    person_image_path: str | Path,
    cloth_image_path: str | Path,
    steps: int,
    guidance: float,
    seed: int,
    cloth_type: str = "upper",
) -> Path:
    person_path = str(Path(person_image_path).resolve())
    cloth_path = str(Path(cloth_image_path).resolve())

    person_editor_value = {
        "background": handle_file(person_path),
        "layers": [],
        "composite": handle_file(person_path),
    }

    errors: list[str] = []
    for space in _spaces():
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
        except AppError as exc:
            message = str(exc)
            errors.append(f"{space}: {message}")
            if "No GPU was available" in message:
                continue
            continue
        except Exception as exc:
            errors.append(f"{space}: {exc}")
            continue

    raise CatVTONUnavailableError(
        "Không lấy được kết quả từ CatVTON Cloud. "
        f"Chi tiết: {' | '.join(errors[:2])}"
    )
