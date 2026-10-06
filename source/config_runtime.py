"""Helpers for serializing and restoring runtime config snapshots."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import config


def snapshot_config() -> dict[str, object]:
    """Return a deep-copied snapshot of all uppercase config values."""
    values = {}
    for key in dir(config):
        if key.startswith("_") or not key[:1].isupper():
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        values[key] = copy.deepcopy(value)
    return values


def recompute_derived() -> None:
    """Recompute config values that are derived from other parameters."""
    config.FIR_GROUP_DELAY = (config.FIR_NUM_TAPS - 1) // 2
    config.CHUNK_SAMPLES = int(config.SAMPLE_RATE * config.CHUNK_DURATION_MS / 1000.0)
    if hasattr(config, "CLUST_WAVEFORM_COLLECT_N_NEARBY"):
        config.CLUST_WAVEFORM_COLLECT_N_NEARBY = config.CLUST_N_NEARBY


def apply_config_snapshot(values: dict[str, object]) -> None:
    """Apply a serialized config snapshot onto the live config module."""
    if ("PRECISION_KMEANS" in values and
            "PRECISION_ASSIGN" not in values and
            hasattr(config, "PRECISION_ASSIGN")):
        values = dict(values)
        values["PRECISION_ASSIGN"] = values["PRECISION_KMEANS"]
    for key, value in values.items():
        if not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(config, key, value)
    recompute_derived()


def save_config_snapshot(path: str | Path) -> str:
    """Write the current config snapshot to JSON and return the path."""
    path = str(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot_config(), f, indent=2, allow_nan=True)
    return path


def load_config_snapshot(path: str | Path) -> dict[str, object]:
    """Load a config snapshot JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
