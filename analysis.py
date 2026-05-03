"""Audio analysis for sound-synced effects.

Owns the librosa dependency. Effects consume the cached analyses via
`get_thunder()` / `get_gong()`; `warm_cache()` runs both off-thread at
startup so the first effect click doesn't pay analysis latency.

Per-sound analysis funcs (`_analyze_*`) are intentionally not async — they
are CPU-bound. Wrap them with `asyncio.to_thread` when you need them off
the event loop.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SOUNDS_DIR = Path(__file__).parent / "sounds"


@dataclass
class ThunderAnalysis:
    duration: float
    peaks: list[tuple[float, float]]  # (time, normalized strength 0..1)
    rms: np.ndarray  # normalized 0..1
    rms_hop_seconds: float


@dataclass
class GongAnalysis:
    duration: float
    rms: np.ndarray  # normalized 0..1
    hop_seconds: float


_thunder: ThunderAnalysis | None = None
_gong: GongAnalysis | None = None


def _analyze_thunder() -> ThunderAnalysis:
    import librosa

    path = str(SOUNDS_DIR / "thunder.wav")
    y, sr = librosa.load(path, sr=22050, mono=True)
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    peak_frames = librosa.util.peak_pick(
        env, pre_max=10, post_max=10, pre_avg=20, post_avg=20, delta=0.2, wait=10
    )
    times = librosa.frames_to_time(peak_frames, sr=sr, hop_length=hop)
    strengths = env[peak_frames]
    max_strength = float(strengths.max()) if len(strengths) else 1.0
    peaks = [
        (float(t), float(s) / max_strength)
        for t, s in zip(times, strengths)
        if float(s) / max_strength > 0.04
    ]
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = rms / (float(rms.max()) + 1e-9)
    return ThunderAnalysis(
        duration=float(len(y) / sr),
        peaks=peaks,
        rms=rms.astype(np.float32),
        rms_hop_seconds=hop / sr,
    )


def _analyze_gong() -> GongAnalysis:
    import librosa

    path = str(SOUNDS_DIR / "gong.wav")
    y, sr = librosa.load(path, sr=22050, mono=True)
    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms = rms / (float(rms.max()) + 1e-9)
    return GongAnalysis(
        duration=float(len(y) / sr),
        rms=rms.astype(np.float32),
        hop_seconds=hop / sr,
    )


def get_thunder() -> ThunderAnalysis:
    global _thunder
    if _thunder is None:
        _thunder = _analyze_thunder()
    return _thunder


def get_gong() -> GongAnalysis:
    global _gong
    if _gong is None:
        _gong = _analyze_gong()
    return _gong


async def warm_cache() -> None:
    """Pre-compute analyses off the event loop so first click is snappy."""
    await asyncio.gather(
        asyncio.to_thread(get_thunder),
        asyncio.to_thread(get_gong),
    )
