"""Landmark schema package.

See `landmark_schema.py` for the unified lookup. Body landmarks live in
`body_landmarks` (MediaPipe 33 + derived points). Each garment category
has its own module so the schema can be extended without touching others.
"""

from .landmark_schema import (
    GARMENT_LANDMARK_SCHEMA,
    BODY_TARGET_SCHEMA,
    MIN_SCHEMA,
    PRIORITY_FOR_VTON,
    schema_for,
    body_targets_for,
    all_body_landmark_names,
)
from . import body_landmarks

__all__ = [
    "GARMENT_LANDMARK_SCHEMA",
    "BODY_TARGET_SCHEMA",
    "MIN_SCHEMA",
    "PRIORITY_FOR_VTON",
    "schema_for",
    "body_targets_for",
    "all_body_landmark_names",
    "body_landmarks",
]
