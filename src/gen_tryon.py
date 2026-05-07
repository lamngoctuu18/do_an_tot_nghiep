from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os

import cv2
import numpy as np
from PIL import Image


MODEL_ID = "runwayml/stable-diffusion-inpainting"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
HYPER_SD_REPO = "ByteDance/Hyper-SD"


@dataclass
class GenConfig:
    num_inference_steps: int = 20
    guidance_scale: float = 2.5
    refiner_mode: str = "lcm"  # v16.10e: LCM for speed on 4GB
    strength: float = 0.82  # v16.10e: folds without destroying sleeve shape
    cloth_type: str = "auto"
    use_cloth_lora: bool = True
    infer_size: int = 0  # 0 = auto (512); set to 640/768 for higher-res dress inference
    negative_prompt: str = (
        "deformed garment, wrinkled mess, wrong sleeve length, sleeveless, "
        "distorted fabric, pasted on, flat texture, sticker effect, "
        "deformed body, duplicate arms, bad anatomy, blurry, low quality, artifacts, "
        "multiple garments, layered clothing, old clothing visible"
    )


def _to_pil(image_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")


def _to_pil_mask(mask_gray: np.ndarray) -> Image.Image:
    return Image.fromarray(mask_gray.astype(np.uint8), mode="L")


def _patch_huggingface_cached_download() -> None:
    """Compat for older diffusers with newer huggingface_hub."""
    try:
        import huggingface_hub as _hf_hub
        if hasattr(_hf_hub, "cached_download"):
            return

        def _cached_download(*args, **kwargs):
            return _hf_hub.hf_hub_download(*args, **kwargs)

        _hf_hub.cached_download = _cached_download
    except Exception:
        return


@lru_cache(maxsize=1)
def _get_pipeline(refiner_mode: str):
    try:
        import torch
        _patch_huggingface_cached_download()
        from diffusers import LCMScheduler, StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện cho chế độ Gen. Hãy cài: pip install torch diffusers transformers accelerate safetensors"
        ) from exc

    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_label = "cuda" if use_cuda else "cpu"
    print(f"[gen_tryon] Diffusion backend: {device_label} ({dtype})")

    cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")

    common_kwargs = {
        "safety_checker": None,
        "requires_safety_checker": False,
        "cache_dir": cache_dir,
    }

    if use_cuda:
        # v16.10: Load fp16 safetensors (available on HF) then cast to fp32.
        # torch 2.5 blocks .bin loading (CVE-2025-32434). The repo only has
        # fp16 safetensors, not fp32 safetensors. So: load fp16 → .to(fp32).
        # With cpu_offload, only 1 module on GPU at a time → fp32 fits 4GB.
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                **common_kwargs,
            )
        except Exception:
            # Some mirrors may not have variant metadata
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                **common_kwargs,
            )
        force_fp32 = (
            os.getenv("VTON_FORCE_FP32", "0").strip() == "1"
            or refiner_mode in {"dpm++", "euler", "base"}
        )
        if force_fp32:
            pipe = pipe.to(dtype=torch.float32)
            print(f"[gen_tryon] Loaded fp16 safetensors -> fp32 for {refiner_mode}")
        else:
            pipe = pipe.to(dtype=torch.float16)
            print("[gen_tryon] Loaded fp16 safetensors on CUDA")
    else:
        # CPU: also load fp16 safetensors → cast to fp32 to avoid .bin loading
        # (torch < 2.6 blocks .bin due to CVE-2025-32434)
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                **common_kwargs,
            )
            pipe = pipe.to(dtype=torch.float32)
            print("[gen_tryon] CPU: loaded fp16 safetensors -> cast to fp32")
        except Exception as exc:
            raise RuntimeError(
                "Khong the load local diffusion. "
                "Hay nang cap torch >= 2.6 hoac dam bao co safetensors trong cache."
            ) from exc

    # VRAM optimizations for 4GB GPU.
    pipe.enable_attention_slicing()
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass
    try:
        pipe.vae.enable_tiling()
        print("[gen_tryon] VAE tiling enabled")
    except Exception:
        pass

    # v16.10: Pipeline is fp32 — no NaN issues, no force_upcast needed.

    # v16.10: xformers memory-efficient attention (saves ~30% VRAM)
    if use_cuda:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[gen_tryon] xformers enabled")
        except Exception:
            print("[gen_tryon] xformers not available, using default attention")

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
            lora_path = hf_hub_download(HYPER_SD_REPO, ckpt_name, cache_dir=cache_dir)
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=0.125)
        except Exception:
            pass

    if use_cuda:
        # v16.10: cpu_offload is critical for 4GB GPU — moves modules to GPU only when needed
        try:
            if os.getenv("VTON_CPU_OFFLOAD", "1").strip() == "0":
                pipe = pipe.to("cuda")
                print("[gen_tryon] full CUDA pipeline enabled")
            else:
                pipe.enable_model_cpu_offload()
                print("[gen_tryon] model_cpu_offload enabled")
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
                print("[gen_tryon] sequential_cpu_offload enabled")
            except Exception:
                pipe = pipe.to("cuda")
        # channels_last memory format for ~10% speedup on GPU
        try:
            pipe.unet.to(memory_format=torch.channels_last)
            print("[gen_tryon] UNet channels_last enabled")
        except Exception:
            pass
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
        "photorealistic try-on, single garment layer, natural folds, sharp seams"
    )
    text = (user_prompt or "").strip()
    if text:
        words = text.split()
        if len(words) > 46:
            text = " ".join(words[:46])
    return f"{base}, {text}" if text else base


def generate_tryon_image(
    init_tryon_rgb: np.ndarray,
    inpaint_mask_gray: np.ndarray,
    user_prompt: str,
    config: GenConfig | None = None,
) -> np.ndarray:
    # Clamp and sanitize inputs before entering diffusion.
    init_tryon_rgb = np.nan_to_num(
        init_tryon_rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0,
    )
    init_tryon_rgb = np.clip(init_tryon_rgb, 0.0, 255.0).astype(np.uint8)

    inpaint_mask_gray = np.nan_to_num(
        inpaint_mask_gray.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0,
    )
    inpaint_mask_gray = np.clip(inpaint_mask_gray, 0.0, 255.0).astype(np.uint8)

    config = config or GenConfig()
    mode = (config.refiner_mode or "lcm").strip().lower()
    if mode not in {"lcm", "hypersd", "dpm++", "euler", "base"}:
        mode = "lcm"
    pipe = _get_pipeline(mode)

    init_image = _to_pil(init_tryon_rgb)
    mask_image = _to_pil_mask(inpaint_mask_gray)

    target_h, target_w = init_tryon_rgb.shape[:2]
    # v16.10: Default 512x512 for SD-inpaint on 4GB GPU.
    # config.infer_size overrides this (e.g. 640 for dress on higher-VRAM GPUs).
    INFER_SIZE = int(config.infer_size) if (config.infer_size and config.infer_size >= 512) else 512
    init_image = init_image.resize((INFER_SIZE, INFER_SIZE), Image.LANCZOS)
    mask_image = mask_image.resize((INFER_SIZE, INFER_SIZE), Image.NEAREST)
    infer_w = INFER_SIZE
    infer_h = INFER_SIZE

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
        # v16.11b: LCM needs guidance ~4-7 for detail. 2.0 was too low → flat output.
        guidance = float(np.clip(guidance, 1.0, 7.5))
        steps = int(max(4, min(steps, 14)))
    elif mode == "hypersd":
        guidance = float(np.clip(guidance, 1.5, 5.0))
        steps = int(max(4, min(steps, 14)))
    elif mode in ("dpm++", "euler"):
        guidance = float(np.clip(guidance, 1.0, 7.5))
        steps = int(max(8, min(steps, 30)))

    call_kwargs = dict(
        prompt=_build_prompt(user_prompt),
        negative_prompt=config.negative_prompt,
        image=init_image,
        mask_image=mask_image,
        width=infer_w,
        height=infer_h,
        num_inference_steps=steps,
        guidance_scale=guidance,
        strength=float(np.clip(config.strength, 0.15, 0.90)),
    )

    # v16.10: fp32 pipeline — no NaN guard needed. Run directly.
    result = pipe(**call_kwargs)
    output = result.images[0]

    # Convert PIL → numpy with NaN safety
    output_np = np.array(output, dtype=np.float32)
    _raw_max = float(np.nanmax(output_np))
    _raw_min = float(np.nanmin(output_np))
    _raw_mean = float(np.nanmean(output_np))
    _nan_count = int(np.isnan(np.array(output, dtype=np.float32)).sum())
    print(f"[gen_tryon] Raw output: shape={output_np.shape}, min={_raw_min:.2f}, max={_raw_max:.2f}, mean={_raw_mean:.2f}, NaN={_nan_count}")

    # Handle NaN
    output_np = np.nan_to_num(output_np, nan=128.0, posinf=255.0, neginf=0.0)

    if _raw_max <= 1.5 and _raw_max > 0.01:
        # Float [0,1] range
        output_np = np.clip(output_np, 0.0, 1.0) * 255.0
    elif _raw_max <= 0.01:
        # Nearly all zeros means diffusion failed. Do not silently return the
        # init image, because that makes the UI report Local Diffusion while the
        # visible result is CPU-only.
        raise RuntimeError("Diffusion produced near-zero output")

    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    # v16.7f: Resize to target FIRST — diffusion may output different size (e.g. 256x256)
    if output_np.shape[:2] != (target_h, target_w):
        output_np = cv2.resize(output_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Check if output is mostly black (NaN → 0 artifact)
    _mean_brightness = float(output_np.mean())
    if _mean_brightness < 5:
        raise RuntimeError(f"Diffusion output nearly black (mean={_mean_brightness:.1f})")

    # v16.11: Simple mask-based composite — diffusion inside mask, init outside.
    # No sleeve-specific attenuation here; app.py handles sleeve preservation if needed.
    if inpaint_mask_gray.shape[:2] != (target_h, target_w):
        safe_mask = cv2.resize(inpaint_mask_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        safe_mask = inpaint_mask_gray.copy()
    safe_mask = cv2.dilate(safe_mask, np.ones((3, 3), np.uint8), iterations=1)
    safe_mask = cv2.GaussianBlur(safe_mask, (5, 5), 0)
    safe_alpha = (safe_mask.astype(np.float32) / 255.0)[..., None]

    output_np = (
        output_np.astype(np.float32) * safe_alpha
        + init_tryon_rgb.astype(np.float32) * (1.0 - safe_alpha)
    )
    output_np = np.nan_to_num(output_np, nan=0.0, posinf=255.0, neginf=0.0)
    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    return output_np
