from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.storage import resolve_storage_config


MODEL_ID = "runwayml/stable-diffusion-inpainting"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
HYPER_SD_REPO = "ByteDance/Hyper-SD"


def _debug_enabled() -> bool:
    return os.getenv("VTON_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _debug_dir() -> Path | None:
    if not _debug_enabled():
        return None
    try:
        if os.getenv("VTON_DEBUG_DIR", "").strip():
            base = Path(os.getenv("VTON_DEBUG_DIR", "")).expanduser().resolve()
        else:
            base = resolve_storage_config().base_dir / "debug"
        base.mkdir(parents=True, exist_ok=True)
        return base
    except OSError:
        return None


def _debug_save_rgb(name: str, image_rgb: np.ndarray, run_id: str) -> None:
    d = _debug_dir()
    if d is None:
        return
    arr = np.nan_to_num(image_rgb.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(str(d / f"{run_id}_{name}.png"), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def _debug_save_mask(name: str, mask_gray: np.ndarray, run_id: str) -> None:
    d = _debug_dir()
    if d is None:
        return
    arr = np.nan_to_num(mask_gray.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    cv2.imwrite(str(d / f"{run_id}_{name}.png"), arr)


def _debug_save_text(name: str, text: str, run_id: str) -> None:
    d = _debug_dir()
    if d is None:
        return
    try:
        (d / f"{run_id}_{name}.txt").write_text(text, encoding="utf-8")
    except OSError:
        return


@dataclass
class GenConfig:
    num_inference_steps: int = 20
    guidance_scale: float = 2.5
    refiner_mode: str = "lcm"  # v16.10e: LCM for speed on 4GB
    strength: float = 0.82  # v16.10e: folds without destroying sleeve shape
    cloth_type: str = "auto"
    use_cloth_lora: bool = True
    infer_size: int = 0  # 0 = auto (512); set to 640/768 for higher-res dress inference
    reference_image_rgb: np.ndarray | None = None
    ip_adapter_scale: float = 0.46
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
    require_cuda = os.getenv("VTON_REQUIRE_CUDA", "1").strip() == "1"
    allow_cpu = os.getenv("VTON_ALLOW_CPU_DIFFUSION", "0").strip() == "1"
    if require_cuda and not allow_cpu and not use_cuda:
        raise RuntimeError(
            "VTON_REQUIRE_CUDA=1 but torch.cuda.is_available() is False. "
            "Install a CUDA PyTorch wheel or set VTON_ALLOW_CPU_DIFFUSION=1 for explicit CPU testing."
        )
    dtype = torch.float16 if use_cuda else torch.float32
    device_label = "cuda" if use_cuda else "cpu"
    print(f"[gen_tryon] Diffusion backend: {device_label} ({dtype})")
    if use_cuda:
        try:
            print(f"[gen_tryon] CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

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
            try:
                pipe.vae = pipe.vae.to(dtype=torch.float32)
                if hasattr(pipe.vae.config, "force_upcast"):
                    pipe.vae.config.force_upcast = True
                print("[gen_tryon] VAE upcast to fp32 (NaN-safe on GTX 16xx)")
            except Exception as _exc:
                print(f"[gen_tryon] VAE upcast skipped: {_exc}")
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

    # VRAM optimizations for 4GB GPU. Attention slicing is enabled lazily in
    # generate_tryon_image after optional IP-Adapter loading. Loading IP-Adapter
    # into a pipeline that already uses SlicedAttnProcessor can fail on some
    # diffusers versions.
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


def _maybe_enable_attention_slicing(pipe, *, allow_with_ip_adapter: bool = False) -> bool:
    if os.getenv("VTON_ATTENTION_SLICING", "1").strip().lower() in {"0", "false", "no"}:
        return False
    if getattr(pipe, "_vto_attention_slicing_enabled", False):
        return True
    if getattr(pipe, "_vto_ip_adapter_loaded", False) and not allow_with_ip_adapter:
        return False
    try:
        pipe.enable_attention_slicing()
        pipe._vto_attention_slicing_enabled = True
        print("[gen_tryon] attention slicing enabled")
        return True
    except Exception as exc:
        print(f"[gen_tryon] attention slicing unavailable: {exc}")
        return False


def _maybe_disable_attention_slicing(pipe) -> None:
    if not getattr(pipe, "_vto_attention_slicing_enabled", False):
        return
    if not hasattr(pipe, "disable_attention_slicing"):
        return
    try:
        pipe.disable_attention_slicing()
        pipe._vto_attention_slicing_enabled = False
    except Exception:
        pass


def _mark_ip_adapter_unavailable(pipe) -> None:
    try:
        if hasattr(pipe, "unload_ip_adapter"):
            pipe.unload_ip_adapter()
    except Exception:
        pass
    pipe._vto_ip_adapter_loaded = False
    pipe._vto_ip_adapter_failed = True


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


def _ip_adapter_enabled(reference_image_rgb: np.ndarray | None) -> bool:
    if reference_image_rgb is None:
        return False
    return os.getenv("VTON_USE_IP_ADAPTER", "1").strip().lower() not in {"0", "false", "no"}


def _align_ip_adapter_device(pipe) -> None:
    """Move IP-Adapter modules (image_encoder + UNet attn procs) to pipe device/dtype.

    Why: load_ip_adapter() leaves image_encoder/proj layers on CPU fp32 while the
    rest of the pipeline is on CUDA fp16, causing
    "Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)"
    at generate-time. We re-pin everything to match pipe.unet.
    """
    try:
        import torch  # local import; torch already imported elsewhere
        target_device = getattr(pipe.unet, "device", None) or getattr(pipe, "_execution_device", None)
        target_dtype = getattr(pipe.unet, "dtype", None) or torch.float16
        if target_device is None:
            return
        encoder = getattr(pipe, "image_encoder", None)
        if encoder is not None:
            encoder.to(device=target_device, dtype=target_dtype)
        # UNet attention processors include IP-Adapter cross-attn modules
        unet = getattr(pipe, "unet", None)
        if unet is not None:
            for proc in unet.attn_processors.values():
                if hasattr(proc, "to"):
                    try:
                        proc.to(device=target_device, dtype=target_dtype)
                    except Exception:
                        pass
    except Exception as exc:
        print(f"[gen_tryon] IP-Adapter device align warning: {exc}")


def _load_ip_adapter_image_encoder(pipe, repo: str, image_encoder_folder: str, cache_dir: str | None) -> bool:
    """Load CLIP image encoder for IP-Adapter.

    The h94/IP-Adapter HF repo no longer ships image_encoder/config.json on the
    main branch (404), so we fall back to laion/CLIP-ViT-H-14-laion2B-s32B-b79K,
    which is the encoder the SD1.5 IP-Adapter weights were trained against
    (1280-wide). The original h94 path is still tried first for backward compat.
    """
    if getattr(pipe, "image_encoder", None) is not None:
        return True

    import torch
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    dtype = getattr(pipe, "dtype", None) or getattr(pipe.unet, "dtype", None) or torch.float32

    candidates: list[tuple[str, str | None]] = []
    if image_encoder_folder:
        candidates.append((repo, image_encoder_folder))
    fallback_repo = os.getenv(
        "VTON_IP_ADAPTER_ENCODER_REPO",
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ).strip()
    if fallback_repo:
        candidates.append((fallback_repo, None))

    last_exc: Exception | None = None
    for cand_repo, cand_folder in candidates:
        try:
            load_kwargs: dict = {"cache_dir": cache_dir}
            if cand_folder:
                load_kwargs["subfolder"] = cand_folder
            try:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    cand_repo,
                    dtype=dtype,
                    **load_kwargs,
                )
            except TypeError:
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    cand_repo,
                    torch_dtype=dtype,
                    **load_kwargs,
                )

            pipe.register_modules(image_encoder=image_encoder.to("cpu"))
            if getattr(pipe, "feature_extractor", None) is None:
                clip_size = int(getattr(image_encoder.config, "image_size", 224))
                pipe.register_modules(
                    feature_extractor=CLIPImageProcessor(size=clip_size, crop_size=clip_size)
                )
            print(
                f"[gen_tryon] IP-Adapter image encoder loaded: "
                f"{cand_repo}/{cand_folder or ''} hidden={image_encoder.config.hidden_size}"
            )
            return True
        except Exception as exc:
            last_exc = exc
            print(f"[gen_tryon] IP-Adapter encoder candidate failed ({cand_repo}/{cand_folder}): {exc}")

    print(f"[gen_tryon] IP-Adapter image encoder unavailable: {last_exc}")
    return False


def _refresh_cpu_offload_after_ip_adapter(pipe) -> None:
    """Reinstall accelerate hooks after adding optional IP-Adapter modules."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        if os.getenv("VTON_CPU_OFFLOAD", "1").strip() == "0":
            _align_ip_adapter_device(pipe)
            return
        if hasattr(pipe, "maybe_free_model_hooks"):
            pipe.maybe_free_model_hooks()
        pipe.enable_model_cpu_offload()
        pipe._vto_cpu_offload_refreshed_after_ip_adapter = True
        print("[gen_tryon] model_cpu_offload refreshed for IP-Adapter")
    except Exception as exc:
        print(f"[gen_tryon] IP-Adapter offload refresh warning: {exc}")
        _align_ip_adapter_device(pipe)


def _maybe_apply_ip_adapter(pipe, reference_image_rgb: np.ndarray | None, scale: float) -> bool:
    """Load SD1.5 IP-Adapter for garment-reference conditioning when available."""
    if not _ip_adapter_enabled(reference_image_rgb):
        return False
    if not hasattr(pipe, "load_ip_adapter"):
        return False
    if getattr(pipe, "_vto_ip_adapter_failed", False):
        return False

    if not getattr(pipe, "_vto_ip_adapter_loaded", False):
        _maybe_disable_attention_slicing(pipe)
        repo = os.getenv("VTON_IP_ADAPTER_REPO", "h94/IP-Adapter").strip()
        subfolder = os.getenv("VTON_IP_ADAPTER_SUBFOLDER", "models").strip()
        image_encoder_folder = os.getenv("VTON_IP_ADAPTER_IMAGE_ENCODER", "image_encoder").strip()
        configured_weight = os.getenv("VTON_IP_ADAPTER_WEIGHT", "").strip()
        weight_candidates = [
            configured_weight,
            "ip-adapter_sd15.bin",
            "ip-adapter_sd15.safetensors",
            "ip-adapter_sd15_light.bin",
            "ip-adapter_sd15_light.safetensors",
        ]
        loaded = False
        last_exc: Exception | None = None
        cache_dir = os.getenv("HF_HUB_CACHE") or os.getenv("HUGGINGFACE_HUB_CACHE")
        if image_encoder_folder:
            _load_ip_adapter_image_encoder(pipe, repo, image_encoder_folder, cache_dir)
        for weight_name in [w for w in weight_candidates if w]:
            try:
                kwargs = {
                    "subfolder": subfolder,
                    "weight_name": weight_name,
                    "image_encoder_folder": (
                        None if getattr(pipe, "image_encoder", None) is not None
                        else ("../image_encoder" if subfolder else "image_encoder")
                    ),
                }
                if cache_dir:
                    kwargs["cache_dir"] = cache_dir
                pipe.load_ip_adapter(repo, **kwargs)
                _refresh_cpu_offload_after_ip_adapter(pipe)
                pipe._vto_ip_adapter_loaded = True
                pipe._vto_ip_adapter_weight = weight_name
                loaded = True
                print(f"[gen_tryon] IP-Adapter loaded: {repo}/{subfolder}/{weight_name}")
                break
            except Exception as exc:
                last_exc = exc
                print(f"[gen_tryon] IP-Adapter candidate failed ({weight_name}): {exc}")
                try:
                    if hasattr(pipe, "unload_ip_adapter"):
                        pipe.unload_ip_adapter()
                except Exception:
                    pass
        if not loaded:
            pipe._vto_ip_adapter_failed = True
            print(f"[gen_tryon] IP-Adapter unavailable, continuing without it: {last_exc}")
            return False

    try:
        if hasattr(pipe, "set_ip_adapter_scale"):
            pipe.set_ip_adapter_scale(float(np.clip(scale, 0.05, 1.0)))
        return True
    except Exception as exc:
        print(f"[gen_tryon] IP-Adapter scale failed, continuing without it: {exc}")
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
    debug_run_id = dt.datetime.now().strftime("%H%M%S")
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
    _debug_save_rgb("gen_tryon_00_init", init_tryon_rgb, debug_run_id)
    _debug_save_mask("gen_tryon_01_inpaint_mask", inpaint_mask_gray, debug_run_id)
    _debug_save_text(
        "gen_tryon_00_meta",
        json.dumps(
            {
                "mode": mode,
                "cloth_type": config.cloth_type,
                "steps": int(config.num_inference_steps),
                "guidance": float(config.guidance_scale),
                "strength": float(config.strength),
                "infer_size": int(config.infer_size or 0),
                "ip_adapter_scale": float(config.ip_adapter_scale),
                "use_cloth_lora": bool(config.use_cloth_lora),
                "prompt": user_prompt or "",
                "negative_prompt": config.negative_prompt,
            },
            ensure_ascii=False,
            indent=2,
        ),
        debug_run_id,
    )
    pipe = _get_pipeline(mode)

    init_image = _to_pil(init_tryon_rgb)
    mask_image = _to_pil_mask(inpaint_mask_gray)

    target_h, target_w = init_tryon_rgb.shape[:2]
    # v16.10: Default 512x512 for SD-inpaint on 4GB GPU.
    # config.infer_size overrides this (e.g. 640 for dress on higher-VRAM GPUs).
    INFER_SIZE = int(config.infer_size) if (config.infer_size and config.infer_size >= 512) else 512
    # v18.6: aspect-preserving inference. Forcing portrait into a square canvas
    # stretched bodies horizontally and produced vertical gray strips at the
    # L/R edges after the back-resize. Compute infer_w/infer_h matching input
    # aspect ratio (snapped to multiples of 8 — SD UNet hard requirement).
    aspect = float(target_w) / float(max(target_h, 1))
    if aspect <= 1.0:
        infer_h = INFER_SIZE
        infer_w = max(384, int(round(INFER_SIZE * aspect / 8.0)) * 8)
    else:
        infer_w = INFER_SIZE
        infer_h = max(384, int(round(INFER_SIZE / aspect / 8.0)) * 8)
    init_image = init_image.resize((infer_w, infer_h), Image.LANCZOS)
    mask_image = mask_image.resize((infer_w, infer_h), Image.NEAREST)

    # Optional cloth-type LoRA (pretrained external adapters, no retraining).
    _maybe_apply_cloth_lora(
        pipe,
        cloth_type=config.cloth_type,
        user_prompt=user_prompt,
        enabled=config.use_cloth_lora,
    )
    ip_adapter_active = _maybe_apply_ip_adapter(
        pipe,
        config.reference_image_rgb,
        config.ip_adapter_scale,
    )
    _maybe_enable_attention_slicing(
        pipe,
        allow_with_ip_adapter=(
            not ip_adapter_active
            or os.getenv("VTON_IP_ADAPTER_ATTENTION_SLICING", "0").strip().lower() in {"1", "true", "yes"}
        ),
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
    if ip_adapter_active and config.reference_image_rgb is not None:
        reference_rgb = np.nan_to_num(
            config.reference_image_rgb.astype(np.float32),
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )
        reference_rgb = np.clip(reference_rgb, 0, 255).astype(np.uint8)
        call_kwargs["ip_adapter_image"] = _to_pil(reference_rgb).resize(
            (INFER_SIZE, INFER_SIZE),
            Image.LANCZOS,
        )

    # v16.10: fp32 pipeline — no NaN guard needed. Run directly.
    try:
        result = pipe(**call_kwargs)
    except RuntimeError as exc:
        _debug_save_text("gen_tryon_02_pipe_exception", repr(exc), debug_run_id)
        retry_without_ip = (
            ip_adapter_active
            and os.getenv("VTON_IP_ADAPTER_RETRY_WITHOUT", "1").strip().lower() not in {"0", "false", "no"}
        )
        if not retry_without_ip:
            raise
        print(f"[gen_tryon] IP-Adapter generation failed, retrying without it: {exc}")
        _mark_ip_adapter_unavailable(pipe)
        call_kwargs.pop("ip_adapter_image", None)
        _maybe_enable_attention_slicing(pipe)
        try:
            result = pipe(**call_kwargs)
        except RuntimeError as retry_exc:
            _debug_save_text("gen_tryon_02_retry_exception", repr(retry_exc), debug_run_id)
            raise
    output = result.images[0]

    # Convert PIL → numpy with NaN safety
    output_np = np.array(output, dtype=np.float32)
    _raw_max = float(np.nanmax(output_np))
    _raw_min = float(np.nanmin(output_np))
    _raw_mean = float(np.nanmean(output_np))
    _nan_count = int(np.isnan(np.array(output, dtype=np.float32)).sum())
    print(f"[gen_tryon] Raw output: shape={output_np.shape}, min={_raw_min:.2f}, max={_raw_max:.2f}, mean={_raw_mean:.2f}, NaN={_nan_count}")
    _debug_save_rgb("gen_tryon_03_raw_output", output_np, debug_run_id)
    _debug_save_text(
        "gen_tryon_03_raw_stats",
        f"shape={output_np.shape}\nmin={_raw_min:.4f}\nmax={_raw_max:.4f}\nmean={_raw_mean:.4f}\nNaN={_nan_count}\n",
        debug_run_id,
    )

    # Handle NaN
    output_np = np.nan_to_num(output_np, nan=128.0, posinf=255.0, neginf=0.0)

    if _raw_max <= 1.5 and _raw_max > 0.01:
        # Float [0,1] range
        output_np = np.clip(output_np, 0.0, 1.0) * 255.0
    elif _raw_max <= 0.01:
        # Nearly all zeros means diffusion failed. Do not silently return the
        # init image, because that makes the UI report Local Diffusion while the
        # visible result is CPU-only.
        _debug_save_rgb("gen_tryon_03b_near_zero_raw", output_np, debug_run_id)
        raise RuntimeError("Diffusion produced near-zero output")

    output_np = np.clip(output_np, 0, 255).astype(np.uint8)

    # v16.7f: Resize to target FIRST — diffusion may output different size (e.g. 256x256)
    if output_np.shape[:2] != (target_h, target_w):
        output_np = cv2.resize(output_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    _debug_save_rgb("gen_tryon_04_resized_output", output_np, debug_run_id)

    # Check if output is mostly black (NaN → 0 artifact)
    _mean_brightness = float(output_np.mean())
    if _mean_brightness < 5:
        _debug_save_rgb("gen_tryon_04b_black_output", output_np, debug_run_id)
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
    _debug_save_rgb("gen_tryon_05_composited_output", output_np, debug_run_id)

    return output_np
