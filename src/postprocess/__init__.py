"""Per-category postprocess.

These modules extract category-specific cleanup steps (pants colour reseed,
dress print lock, upper-body restore) from `app.py`. Helpers that live in
`app.py` (`_fit_like`, `_safe_uint8`, `build_cloth_mask`) are passed in as
callables to avoid circular imports.
"""
