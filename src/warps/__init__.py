"""Per-category warp facades over `src/tps_warp.py`.

These modules don't reimplement geometry — they're thin re-exports so
callers can write `from src.warps.pants_warp import warp_pants` instead of
reaching directly into the monolithic `tps_warp` module. This keeps the
public surface organised by category and makes future per-category tuning
(e.g. dress-only hem snap) a local change.
"""

from .top_warp import warp_top, refine_top_mask, warp_sleeves
from .pants_warp import warp_pants, detect_pants_landmarks
from .dress_warp import warp_dress

__all__ = [
    "warp_top",
    "warp_sleeves",
    "refine_top_mask",
    "warp_pants",
    "detect_pants_landmarks",
    "warp_dress",
]
