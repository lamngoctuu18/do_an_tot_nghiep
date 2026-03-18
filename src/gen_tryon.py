from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os

import numpy as np
from PIL import Image


MODEL_ID = "runwayml/stable-diffusion-inpainting"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
HYPER_SD_REPO = "ByteDance/Hyper-SD"


@dataclass
class GenConfig:
    num_inference_steps: int = 18
    guidance_scale: float = 1.3
    refiner_mode: str = "dpm++"  # one of: lcm, hypersd, dpm++, euler, base
    strength: float = 0.35
    cloth_type: str = "auto"
    use_cloth_lora: bool = True
    negative_prompt: str = (
        "different clothing, wrong color, wrong pattern, changed design, "
        "deformed body, duplicate arms, duplicate torso, bad anatomy, blurry, low quality, artifacts"
    )


def _to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")


def _to_pil_mask(mask_gray: np.ndarray) -> Image.Image:
    return Image.fromarray(mask_gray.astype(np.uint8), mode="L")


@lru_cache(maxsize=1)
def _get_pipeline(refiner_mode: str):
    try:
        import torch
        from diffusers import LCMScheduler, StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện cho chế độ Gen. Hãy cài: pip install torch diffusers transformers accelerate safetensors"
        ) from exc

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # VRAM optimizations for weak GPUs / CPU fallback.
    pipe.enable_attention_slicing()
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if refiner_mode == "lcm":
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        try:
            pipe.load_lora_weights(LCM_LORA_ID)
            pipe.fuse_lora()
        except Exception:
            pass
    elif refiner_mode == "dpm++":
        from diffusers import DPMSolverMultistepScheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
        )
    elif refiner_mode == "euler":
        from diffusers import EulerDiscreteScheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif refiner_mode == "hypersd":
        try:
            from huggingface_hub import hf_hub_download

            ckpt_name = os.getenv("HYPERSD_CKPT", "Hyper-SD15-8steps-CFG-lora.safetensors")
            lora_path = hf_hub_download(HYPER_SD_REPO, ckpt_name)
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=0.125)
        except Exception:
            pass

    if use_cuda:
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe


def _parse_lora_map() -> dict[str, str]:
    """Read optional cloth-type LoRA map from env.

    Format:
      CLOTH_LORA_MAP='{"tshirt":"owner/repo","hoodie":"owner/repo","jacket":"owner/repo"}'
    """
    raw = os.getenv("CLOTH_LORA_MAP", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _auto_detect_cloth_type(user_prompt: str) -> str:
    text = (user_prompt or "").lower()
    if any(k in text for k in ["hoodie", "sweater", "pullover"]):
        return "hoodie"
    if any(k in text for k in ["jacket", "blazer", "coat"]):
        return "jacket"
    if any(k in text for k in ["dress", "gown"]):
        return "dress"
    if any(k in text for k in ["shirt", "tee", "t-shirt", "top"]):
        return "tshirt"
    return "generic"


def _maybe_apply_cloth_lora(pipe, cloth_type: str, user_prompt: str, enabled: bool) -> bool:
    """Load a pretrained fashion LoRA by cloth type if configured."""
    if not enabled:
        return False
    lora_map = _parse_lora_map()
    if not lora_map:
        return False

    key = (cloth_type or "auto").lower()
    if key == "auto":
        key = _auto_detect_cloth_type(user_prompt)

    target = lora_map.get(key) or lora_map.get("generic")
    if not target:
        return False

    current = getattr(pipe, "_vto_active_cloth_lora", None)
    if current == target:
        return True

    try:
        if hasattr(pipe, "unfuse_lora"):
            try:
                pipe.unfuse_lora()
            except Exception:
                pass
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        pipe.load_lora_weights(target)
        scale = float(os.getenv("CLOTH_LORA_SCALE", "0.65"))
        pipe.fuse_lora(lora_scale=scale)
        pipe._vto_active_cloth_lora = target
        return True
    except Exception:
        return False


def _build_prompt(user_prompt: str) -> str:
    base = (
        "person wearing the exact same garment from reference image, "
        "preserve original color, pattern, logo, fabric texture exactly, "
        "natural folds, realistic lighting"
    )
    text = (user_prompt or "").strip()
    if not text:
        return base
    # Avoid descriptive garment prompts that override the reference image.
    return f"{base}, {text}"


def generate_tryon_image(
    init_tryon_rgb: np.ndarray,
    inpaint_mask_gray: np.ndarray,
    user_prompt: str,
    config: GenConfig | None = None,
) -> np.ndarray:
    config = config or GenConfig()
    mode = (config.refiner_mode or "lcm").strip().lower()
    if mode not in {"lcm", "hypersd", "dpm++", "euler", "base"}:
        mode = "lcm"
    pipe = _get_pipeline(mode)

    init_image = _to_pil(init_tryon_rgb)
    mask_image = _to_pil_mask(inpaint_mask_gray)

    target_h, target_w = init_tryon_rgb.shape[:2]
    infer_w = max(256, (target_w // 8) * 8)
    infer_h = max(256, (target_h // 8) * 8)

    # Optional cloth-type LoRA (pretrained external adapters, no retraining).
    _maybe_apply_cloth_lora(
        pipe,
        cloth_type=config.cloth_type,
        user_prompt=user_prompt,
        enabled=config.use_cloth_lora,
    )

    steps = int(max(4, min(config.num_inference_steps, 30)))
    guidance = float(config.guidance_scale)
    if mode == "lcm":
        guidance = float(np.clip(guidance, 0.5, 2.0))
        steps = int(max(4, min(steps, 14)))
    elif mode == "hypersd":
        guidance = float(np.clip(guidance, 1.5, 5.0))
        steps = int(max(4, min(steps, 14)))
    elif mode in ("dpm++", "euler"):
        guidance = float(np.clip(guidance, 1.0, 7.5))
        steps = int(max(8, min(steps, 30)))

    output = pipe(
        prompt=_build_prompt(user_prompt),
        negative_prompt=config.negative_prompt,
        image=init_image,
        mask_image=mask_image,
        width=infer_w,
        height=infer_h,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=float(np.clip(config.strength, 0.15, 0.5)),
    ).images[0]
    output_np = np.array(output)

    if output_np.shape[:2] != (target_h, target_w):
        output_np = np.array(output.resize((target_w, target_h), Image.BILINEAR))

    return output_np
