from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageConfig:
    base_dir: Path
    inputs_dir: Path
    outputs_dir: Path
    cache_dir: Path


DEFAULT_BASE_DIR = Path("E:/virtual_try_on_data")


def resolve_storage_config() -> StorageConfig:
    base_dir = Path(os.getenv("VTO_BASE_DIR", str(DEFAULT_BASE_DIR))).expanduser().resolve()
    inputs_dir = base_dir / "inputs"
    outputs_dir = base_dir / "outputs"
    cache_dir = base_dir / "cache"

    for directory in (base_dir, inputs_dir, outputs_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _export_cache_vars(base_dir)

    return StorageConfig(
        base_dir=base_dir,
        inputs_dir=inputs_dir,
        outputs_dir=outputs_dir,
        cache_dir=cache_dir,
    )


def _export_cache_vars(base_dir: Path) -> None:
    # Force model/cache downloads to the configured storage root (default on drive E).
    hf_home = base_dir / "huggingface"
    huggingface_hub_cache = hf_home / "hub"
    torch_home = base_dir / "torch"

    for directory in (hf_home, huggingface_hub_cache, torch_home):
        directory.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    # Newer Transformers deprecates TRANSFORMERS_CACHE in favor of HF_HOME.
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(huggingface_hub_cache)
    os.environ["HF_HUB_CACHE"] = str(huggingface_hub_cache)
    os.environ["TORCH_HOME"] = str(torch_home)
