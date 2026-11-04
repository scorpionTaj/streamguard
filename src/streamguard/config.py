"""Config loading + window parsing (implemented).

`configs/features.yaml` is the single source of truth for features; the Spark
online job and the polars offline twin must both build from it (PLAN.md §6).
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_WINDOW_RE = re.compile(r"^(\d+)([smhd])$")
_WINDOW_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_window(window: str) -> int:
    """Parse a velocity window like '5m', '1h', '24h', '7d' into seconds.

    >>> parse_window("24h")
    86400
    """
    match = _WINDOW_RE.match(window.strip())
    if match is None:
        raise ValueError(f"invalid window {window!r}: expected <int><s|m|h|d>, e.g. '24h'")
    value, unit = int(match.group(1)), match.group(2)
    return value * _WINDOW_SECONDS[unit]


def load_yaml_file(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config(path: str | Path = REPO_ROOT / "configs" / "pipeline.yaml") -> dict:
    """Topic names, partitions, trigger, watermark, alert bands (pipeline.yaml)."""
    return load_yaml_file(path)


def load_feature_spec(path: str | Path = REPO_ROOT / "configs" / "features.yaml") -> dict:
    """Feature contract: static list, m_flags, c_features, v_subset, velocity windows."""
    spec = load_yaml_file(path)
    if not spec.get("key"):
        raise ValueError("features.yaml must define 'key' (velocity grouping key)")
    if not spec.get("velocity", {}).get("windows"):
        raise ValueError("features.yaml must define velocity.windows")
    return spec


def velocity_window_seconds(spec: dict) -> dict[str, int]:
    """Velocity windows parsed to seconds, e.g. {'5m': 300, '24h': 86400, ...}."""
    return {w: parse_window(w) for w in spec["velocity"]["windows"]}


def redis_card_key(prefix: str, card_key: str) -> str:
    """Redis hash key for a card, e.g. 'sg:card:<card_key>' (prefix from pipeline.yaml)."""
    return f"{prefix}{card_key}"
