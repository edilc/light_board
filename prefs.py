"""Persisted user preferences for the dashboard.

Stored at `data/preferences.json` (the `data/` dir is gitignored so user
settings don't leak into commits). Loaded once at module import via
`load()`; saved imperatively via `save()` from change handlers in main.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("light_board.prefs")

PREFS_FILE = Path(__file__).parent / "data" / "preferences.json"

DEFAULTS: dict[str, Any] = {
    "bright_white": [244, 218, 182],
    "morning_target_volume": 50,
    "night_target_volume": 100,
    "internal_audio_volume": 1.0,
    "dark_mode": False,
    # Calibration results from the spacebar-tap calibration. None means
    # "use the auto-detected/assumed value" (audio: AudioContext.outputLatency;
    # hue: HUE_LATENCY_S constant).
    "audio_latency_override_ms": None,
    "hue_latency_override_ms": None,
}


def load() -> dict[str, Any]:
    """Return persisted prefs, falling back to defaults for missing keys."""
    if not PREFS_FILE.exists():
        return dict(DEFAULTS)
    try:
        with open(PREFS_FILE) as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("failed to load %s: %s", PREFS_FILE, exc)
        return dict(DEFAULTS)
    merged = dict(DEFAULTS)
    # Only accept keys we know about so a stale file with extra junk doesn't poison state.
    merged.update({k: v for k, v in data.items() if k in DEFAULTS})
    return merged


def save(prefs: dict[str, Any]) -> None:
    """Atomically write prefs to disk."""
    try:
        PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(prefs, f, indent=2)
        tmp.replace(PREFS_FILE)
    except Exception as exc:
        logger.warning("failed to save %s: %s", PREFS_FILE, exc)
