"""Signal-shaping primitives for effects.

Three families:
  - **Easings** (smoothstep, cosine, quadratic_in): pure 1-D shapers used
    by every effect's settle and fade phases.
  - **Envelope**: a wrapper around the per-frame RMS arrays produced by
    `analysis.py`. Replaces the inline `_envelope_at(rms, hop, t)` helper
    so effects don't need to know the array shape.
  - **Audio shapers** (quiver, shaped, candle): map an envelope value (or
    just time) into a brightness multiplier with named parameters
    instead of inline magic numbers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ────────────────────────── easings ──────────────────────────────────


def smoothstep(f: float) -> float:
    """3rd-order ease-in-out, standard `f*f*(3-2f)`. Used by morning and
    night phase transitions."""
    f = max(0.0, min(1.0, f))
    return f * f * (3 - 2 * f)


def cosine(f: float) -> float:
    """Cosine-shaped ease-in-out, `0.5 - 0.5*cos(pi*f)`. Used by gavel,
    lightning, gong settles. Smoother at the endpoints than smoothstep."""
    f = max(0.0, min(1.0, f))
    return 0.5 - 0.5 * math.cos(f * math.pi)


def quadratic_in(f: float) -> float:
    """`f*f` ease-in. Used by gavel/lightning flash → fade transitions —
    the curve sticks high then drops fast, making strikes feel punchy."""
    f = max(0.0, min(1.0, f))
    return f * f


# ────────────────────────── envelope ─────────────────────────────────


@dataclass
class Envelope:
    """A piecewise-constant audio envelope (typically RMS) as a function of
    time. `at(t)` returns 0 outside `[0, duration)`."""

    samples: np.ndarray   # 1-D, normalized 0..1
    hop_seconds: float

    @property
    def duration(self) -> float:
        return len(self.samples) * self.hop_seconds

    def at(self, t: float) -> float:
        if t < 0:
            return 0.0
        idx = int(t / self.hop_seconds)
        if idx >= len(self.samples):
            return 0.0
        return float(self.samples[idx])


# ──────────────────── audio-derived shapers ──────────────────────────


def quiver(env_value: float, low: float = 0.85, high: float = 1.15, gamma: float = 0.5) -> float:
    """Map an envelope value to a brightness multiplier in [low, high].
    `gamma < 1` lifts quiet rumbles so they're still visibly pulsing."""
    return low + (high - low) * (max(0.0, env_value) ** gamma)


def shaped(env_value: float, gamma: float = 0.55, floor: float = 0.08) -> float:
    """Apply gamma-shaping with a floor to an envelope value. Used by
    gong so the bulb stays slightly lit even during sub-threshold tail."""
    return max(floor, max(0.0, env_value) ** gamma)


def candle(t: float, channel: int, amp: float = 0.20) -> float:
    """Deterministic candle-flicker multiplier around 1.0. Each channel
    gets its own phase offset so the lights flicker out of sync, like a
    real flame casting different shadows."""
    offset = channel * 1.7
    s1 = math.sin((t + offset) * 6.3)
    s2 = math.sin((t + offset) * 11.7 + 1.2)
    s3 = math.sin((t + offset) * 17.5 + 2.5)
    s4 = math.sin((t + offset) * 3.4 + 0.8)
    noise = (s1 + 0.7 * s2 + 0.5 * s3 + 0.4 * s4) / 2.6
    return 1.0 + amp * noise
