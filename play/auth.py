"""Lightweight screen-name identity, matching the Splendor play experience."""
from __future__ import annotations
import re


def normalize_username(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", value.strip()):
        raise ValueError("Enter a screen name with 1–32 letters, numbers, underscores, or hyphens")
    return value.strip()
