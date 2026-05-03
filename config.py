"""Persistent user configuration.

One source of truth for everything that should survive across runs:
the day-target color, Spotify volume targets, internal audio volume,
dark-mode preference, and the audio/Hue latency overrides.

Stored at `data/preferences.json`. Load once at startup with
`Config.load(path)`; mutate via `config.update(field=value, ...)` which
sets the field(s) and writes the JSON atomically. Effects read fields
live (e.g. `config.bright_white`) on every frame, so a mid-effect change
takes hold on the next iteration without any signaling plumbing.

Latencies are stored in milliseconds (matches the JSON shape and the
dashboard input fields); `audio_latency_override_s` /
`hue_latency_override_s` properties expose the seconds form used by
`run_effect`.

This module replaces the previous `prefs.py` + `effects.BRIGHT_WHITE`
module-global pair. Effects no longer need to remember "read the global
on every frame"; that's now just `config.bright_white`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger("light_board.config")

RGB = tuple[int, int, int]


@dataclass
class Config:
    bright_white: RGB = (244, 218, 182)
    morning_target_volume: int = 50
    night_target_volume: int = 100
    internal_audio_volume: float = 1.0
    dark_mode: bool = False
    # Latency overrides are stored in milliseconds (matches JSON + UI);
    # use the `_s` properties for the seconds form.
    audio_latency_override_ms: float | None = None
    hue_latency_override_ms: float | None = None

    # Where to persist. None means in-memory only — the test default, so
    # tests can construct `Config()` without touching the filesystem.
    path: Path | None = field(default=None, repr=False, compare=False)

    @property
    def audio_latency_override_s(self) -> float | None:
        ms = self.audio_latency_override_ms
        return ms / 1000 if ms is not None else None

    @property
    def hue_latency_override_s(self) -> float | None:
        ms = self.hue_latency_override_ms
        return ms / 1000 if ms is not None else None

    @classmethod
    def load(cls, path: Path) -> Config:
        """Read the JSON at `path` (falling back to defaults on missing or
        malformed file) and return a Config bound to that path."""
        if not path.exists():
            return cls(path=path)
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("failed to load %s: %s — using defaults", path, exc)
            return cls(path=path)

        # JSON arrays come back as lists; bright_white wants a tuple.
        if isinstance(data.get("bright_white"), list):
            data["bright_white"] = tuple(data["bright_white"])

        # Filter unknown keys so a stale file with extra junk doesn't
        # poison the constructor.
        valid = {f.name for f in fields(cls) if f.name != "path"}
        kept = {k: v for k, v in data.items() if k in valid}
        return cls(path=path, **kept)

    def save(self) -> None:
        """Write the current state atomically. No-op if `path` is None."""
        if self.path is None:
            return
        data: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "path":
                continue
            value = getattr(self, f.name)
            # Tuples → lists for JSON friendliness.
            if isinstance(value, tuple):
                value = list(value)
            data[f.name] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.path)
        except Exception as exc:
            logger.warning("failed to save %s: %s", self.path, exc)

    def update(self, **kwargs: Any) -> None:
        """Set one or more fields and persist. Raises AttributeError for
        unknown field names so typos surface immediately."""
        valid = {f.name for f in fields(type(self)) if f.name != "path"}
        for k in kwargs:
            if k not in valid:
                raise AttributeError(f"Config has no field {k!r}")
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.save()
