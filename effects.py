"""Audio-synced lighting effects.

Each effect is an async coroutine that drives a `LightController` (see
`control.py`). Effects also take an optional `Clock` so tests can replace real
time with a `VirtualClock` and capture the full event trace.

Spectral analysis (librosa) is lazy and cached so first click warms up,
subsequent clicks are instant.
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from control import Clock, LightController, VirtualClock

SOUNDS_DIR = Path(__file__).parent / "sounds"

ClockLike = Clock | VirtualClock


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


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    f = max(0.0, min(1.0, f))
    return tuple(int(round(a[i] + (b[i] - a[i]) * f)) for i in range(3))  # type: ignore[return-value]


def _scale(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(c * factor)))) for c in color)  # type: ignore[return-value]


def _envelope_at(rms: np.ndarray, hop_seconds: float, t: float) -> float:
    if t < 0:
        return 0.0
    idx = int(t / hop_seconds)
    if idx >= len(rms):
        return 0.0
    return float(rms[idx])


FRAME = 1 / 60

# Shared "settled" end state for every effect except Night. Soft warm white,
# roughly 100% brightness on a daylight-leaning bulb. Mutable at runtime via
# `set_bright_white(...)` — the dashboard's config panel updates it live.
BRIGHT_WHITE: tuple[int, int, int] = (244, 218, 182)


def set_bright_white(r: int, g: int, b: int) -> None:
    """Update the BRIGHT_WHITE constant. Effects look it up by name on every
    frame, so the new value takes effect on the next iteration."""
    global BRIGHT_WHITE
    BRIGHT_WHITE = (
        max(0, min(255, int(r))),
        max(0, min(255, int(g))),
        max(0, min(255, int(b))),
    )


def _resolve_clock(clock: ClockLike | None) -> ClockLike:
    return clock if clock is not None else Clock()


async def gavel_effect(ctl: LightController, *, clock: ClockLike | None = None) -> None:
    """Two synced flashes timed to gavel.mp3: yellow on the first hit,
    red on the second. Hit times come from offline onset analysis. After
    the audio finishes, the lights settle to BRIGHT_WHITE."""
    clock = _resolve_clock(clock)

    AMBIENT = (40, 30, 15)
    YELLOW = (255, 220, 0)
    RED = (255, 30, 20)

    # Onset analysis of sounds/gavel.wav (shortened, sample-accurate WAV):
    # two clean peaks, no spurious file-attack transient.
    hits = [
        (0.070, YELLOW),
        (0.557, RED),
    ]
    audio_duration = 1.4
    flash_dur = 0.06
    fade_dur = 0.40
    settle_dur = 1.5

    start = clock.now()
    last_color: tuple[int, int, int] = AMBIENT
    while True:
        t = clock.now() - start
        if t >= audio_duration:
            break

        recent = None
        for hit_t, hit_color in hits:
            if hit_t > t:
                break
            recent = (hit_t, hit_color)

        if recent is None:
            color = AMBIENT
        else:
            hit_t, hit_color = recent
            d = t - hit_t
            if d < flash_dur:
                color = hit_color
            elif d < flash_dur + fade_dur:
                f = (d - flash_dur) / fade_dur
                f = f * f
                color = _lerp(hit_color, AMBIENT, f)
            else:
                color = AMBIENT

        ctl.set_color(*color)
        last_color = color
        await clock.sleep(FRAME)

    t0 = clock.now()
    while True:
        t = clock.now() - t0
        if t >= settle_dur:
            break
        f = 0.5 - 0.5 * math.cos((t / settle_dur) * math.pi)
        ctl.set_color(*_lerp(last_color, BRIGHT_WHITE, f))
        await clock.sleep(FRAME)
    ctl.set_color(*BRIGHT_WHITE)


async def lightning_effect(
    ctl: LightController, *, clock: ClockLike | None = None
) -> None:
    """Lightning with per-channel ambient + RMS-driven quiver.

    Between strikes each channel shows its own shade of blue/purple, with
    brightness pulsing subtly with the audio's RMS envelope (quiet rumble
    → ~85% of base; loud rumble → ~115%). At each strike time, all 3
    channels flash max-white, then ease back down to their (still
    quivering) base color. After the audio, settles to BRIGHT_WHITE.

    Strike times are from peak waveform amplitude analysis of thunder.wav."""
    clock = _resolve_clock(clock)
    a = get_thunder()

    BASE_COLORS = [
        (10, 30, 110),    # ch0: deep blue
        (40, 15, 120),    # ch1: indigo
        (80, 20, 110),    # ch2: violet
    ]
    WHITE = (255, 255, 255)
    STRIKES = [0.300, 0.900]

    FLASH_DUR = 0.08
    FADE_DUR = 0.30
    SETTLE_DUR = 2.0

    settle_start = a.duration
    duration = settle_start + SETTLE_DUR

    def quiver_factor(t: float) -> float:
        # 0.85 → 1.15 driven by RMS. ^0.5 lifts quiet rumbles so they're
        # still visible as gentle pulsing rather than dead-flat.
        env = _envelope_at(a.rms, a.rms_hop_seconds, t)
        return 0.85 + 0.30 * (env ** 0.5)

    def base_snapshot(t: float) -> list[tuple[int, int, int]]:
        f = quiver_factor(t)
        return [_scale(c, f) for c in BASE_COLORS]

    last_snapshot: tuple[tuple[int, int, int], ...] = tuple(BASE_COLORS)

    start = clock.now()
    while True:
        t = clock.now() - start
        if t >= duration:
            break

        if t < settle_start:
            base = base_snapshot(t)
            recent = None
            for s_t in STRIKES:
                if s_t > t:
                    break
                recent = s_t
            if recent is not None and t - recent < FLASH_DUR + FADE_DUR:
                d = t - recent
                if d < FLASH_DUR:
                    snap = [WHITE, WHITE, WHITE]
                else:
                    f = (d - FLASH_DUR) / FADE_DUR
                    f = f * f
                    snap = [_lerp(WHITE, base[ch], f) for ch in range(3)]
            else:
                snap = base
            last_snapshot = tuple(snap)
        else:
            f = (t - settle_start) / SETTLE_DUR
            f = 0.5 - 0.5 * math.cos(f * math.pi)
            snap = [_lerp(last_snapshot[ch], BRIGHT_WHITE, f) for ch in range(3)]

        ctl.set_colors(snap)
        await clock.sleep(FRAME)
    ctl.set_color(*BRIGHT_WHITE)


async def gong_effect(ctl: LightController, *, clock: ClockLike | None = None) -> None:
    clock = _resolve_clock(clock)
    a = get_gong()
    GOLD = (255, 165, 30)

    start = clock.now()
    last_color: tuple[int, int, int] = (0, 0, 0)
    while True:
        t = clock.now() - start
        if t > a.duration:
            break
        env = _envelope_at(a.rms, a.hop_seconds, t)
        shaped = max(0.08, env ** 0.55)
        color = _scale(GOLD, shaped)
        ctl.set_color(*color)
        last_color = color
        await clock.sleep(FRAME)

    transition = 1.5
    t0 = clock.now()
    while True:
        t = clock.now() - t0
        if t >= transition:
            break
        f = 0.5 - 0.5 * math.cos((t / transition) * math.pi)
        ctl.set_color(*_lerp(last_color, BRIGHT_WHITE, f))
        await clock.sleep(FRAME)
    ctl.set_color(*BRIGHT_WHITE)


async def morning_effect(ctl: LightController, *, clock: ClockLike | None = None) -> None:
    """Quick 1s smoothstep crossfade from whatever the lights are currently
    showing into BRIGHT_WHITE. Pairs with `night_effect` — pressing Morning
    after Night gracefully lifts out of the candle state."""
    clock = _resolve_clock(clock)
    duration = 1.0

    start_colors = list(ctl.last_colors[:3])
    while len(start_colors) < 3:
        start_colors.append(BRIGHT_WHITE)

    start = clock.now()
    while True:
        t = clock.now() - start
        if t >= duration:
            break
        f = t / duration
        f = f * f * (3 - 2 * f)
        snap = [_lerp(start_colors[ch], BRIGHT_WHITE, f) for ch in range(3)]
        ctl.set_colors(snap)
        await clock.sleep(FRAME)
    ctl.set_color(*BRIGHT_WHITE)


def _candle_factor(t: float, channel: int) -> float:
    """Deterministic candle-flicker multiplier around 1.0, ±~20%. Each
    channel gets its own phase offset so the three lights flicker out of
    sync, like a real flame casting different shadows."""
    offset = channel * 1.7
    s1 = math.sin((t + offset) * 6.3)
    s2 = math.sin((t + offset) * 11.7 + 1.2)
    s3 = math.sin((t + offset) * 17.5 + 2.5)
    s4 = math.sin((t + offset) * 3.4 + 0.8)
    noise = (s1 + 0.7 * s2 + 0.5 * s3 + 0.4 * s4) / 2.6
    return 1.0 + 0.20 * noise


async def night_effect(ctl: LightController, *, clock: ClockLike | None = None) -> None:
    """Persistent Night state. Three phases, the last of which sustains
    forever until cancelled (typically by pressing Morning):
      1. (1s)  Crossfade from current state to 20% brightness blue / purple /
               dark blue (one color per channel).
      2. (2s)  Crossfade those into 60%-brightness blue / purple / orange,
               with the candle flicker amplitude ramping up over the same window.
      3. (∞)   Sustained candle flicker on the 60% colors. Runs until the
               task is cancelled — Night does NOT auto-fade to BRIGHT_WHITE.
    """
    clock = _resolve_clock(clock)

    P1_DUR = 1.0
    P2_DUR = 2.0
    P2_END = P1_DUR + P2_DUR

    p1_targets = [
        _scale((0, 100, 255), 0.2),     # ch0 blue
        _scale((140, 70, 220), 0.2),    # ch1 purple
        _scale((40, 80, 200), 0.2),     # ch2 dark blue
    ]
    p2_bases = [
        _scale((0, 100, 255), 0.6),     # ch0 blue
        _scale((140, 70, 220), 0.6),    # ch1 purple
        _scale((255, 110, 20), 0.6),    # ch2 orange
    ]

    start_colors = list(ctl.last_colors[:3])
    while len(start_colors) < 3:
        start_colors.append(BRIGHT_WHITE)

    start = clock.now()
    while True:
        t = clock.now() - start
        if t < P1_DUR:
            f = t / P1_DUR
            f = f * f * (3 - 2 * f)
            snap = [_lerp(start_colors[ch], p1_targets[ch], f) for ch in range(3)]
        elif t < P2_END:
            local = t - P1_DUR
            f = local / P2_DUR
            f = f * f * (3 - 2 * f)
            snap = []
            for ch in range(3):
                base = _lerp(p1_targets[ch], p2_bases[ch], f)
                if ch == 2:  # only the orange "candle" channel flickers
                    cf = _candle_factor(t, ch)
                    cf_blend = 1.0 + (cf - 1.0) * f
                    snap.append(_scale(base, cf_blend))
                else:
                    snap.append(base)
        else:
            snap = []
            for ch in range(3):
                if ch == 2:
                    cf = _candle_factor(t, ch)
                    snap.append(_scale(p2_bases[ch], cf))
                else:
                    snap.append(p2_bases[ch])

        ctl.set_colors(snap)
        await clock.sleep(FRAME)
