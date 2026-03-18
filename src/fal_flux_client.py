from __future__ import annotations

import os
from pathlib import Path

MODEL_ID = "fal-ai/flux-2/klein/9b/base/edit/lora"
DEFAULT_LORA = "https://huggingface.co/fal/flux-klein-9b-virtual-tryon-lora/resolve/main/flux-klein-tryon.safetensors"


class FalFluxUnavailableError(RuntimeError):
    pass


def _resolve_fal_key() -> str:
    # Accept common variable names to avoid misconfiguration.
    for key_name in ("FAL_KEY", "FAL_API_KEY", "FAIL_KEY", "fail_key"):
        value = os.getenv(key_name, "").strip()
        if value:
            return value
    return ""


def _build_tryon_prompt(user_prompt: str) -> str:
    prompt = (user_prompt or "").strip()
    if not prompt:
        prompt = (
            "TRYON full body person. Replace the outfit with the provided garment references. "
            "Preserve identity, realistic fabric folds, natural lighting."
        )
    if not prompt.upper().startswith("TRYON"):
        prompt = f"TRYON {prompt}"
    if "full body" not in prompt.lower():
        prompt = f"{prompt}. The final image is a full body shot."
    return prompt


def _upload_if_needed(client, path_or_url: str) -> str:
    p = Path(path_or_url)
    if p.exists():
        # fal-client uploads local files and returns a public URL.
        return client.upload_file(str(p))
    return path_or_url


def generate_with_fal_flux_tryon(
    person_image_path: str,
    top_image_path: str,
    bottom_image_path: str | None,
    user_prompt: str,
    steps: int = 28,
    guidance: float = 2.5,
    lora_scale: float = 1.0,
) -> str:
    """Run FLUX Klein 9B edit+lora try-on via fal.ai.

    Returns output image URL.
    """
    fal_key = _resolve_fal_key()
    if not fal_key:
        raise FalFluxUnavailableError("Thiếu FAL_KEY/FAL_API_KEY (hoặc bạn đang đặt nhầm FAIL_KEY).")

    try:
        import fal_client
    except Exception as exc:
        raise FalFluxUnavailableError(
            "Thiếu thư viện fal-client. Hãy cài: pip install fal-client"
        ) from exc

    os.environ["FAL_KEY"] = fal_key

    top_ref = _upload_if_needed(fal_client, top_image_path)
    person_ref = _upload_if_needed(fal_client, person_image_path)
    bottom_ref = _upload_if_needed(fal_client, bottom_image_path or top_image_path)

    lora_path = os.getenv("FAL_FLUX_LORA_PATH", DEFAULT_LORA).strip() or DEFAULT_LORA

    arguments = {
        "prompt": _build_tryon_prompt(user_prompt),
        "image_urls": [person_ref, top_ref, bottom_ref],
        "loras": [{"path": lora_path, "scale": float(lora_scale)}],
        "num_inference_steps": int(max(8, min(steps, 40))),
        "guidance_scale": float(max(0.0, min(guidance, 8.0))),
    }

    try:
        result = fal_client.submit(MODEL_ID, arguments=arguments).get()
        images = result.get("images") or []
        if not images:
            raise FalFluxUnavailableError("FAL Flux trả về rỗng.")
        url = images[0].get("url")
        if not url:
            raise FalFluxUnavailableError("FAL Flux không có URL output.")
        return str(url)
    except Exception as exc:
        raise FalFluxUnavailableError(f"FAL Flux lỗi: {exc}") from exc
