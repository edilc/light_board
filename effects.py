"""Audio-synced lighting effects.

Each effect is an async coroutine that drives a `LightController` and
takes a `Config` for live persistent settings. The dashboard's "Save as
Day target" mutates `config.bright_white` in place; effects read it on
every frame, so transitions land on the current value without any
signaling plumbing.

Effects also accept an optional `Clock` so tests can swap in
`VirtualClock` and capture the full event trace via
`RecordingController`.

This module is choreography only. Time-walking lives in `runner.py`,
color math in `colors.py`, signal shaping (easings, candle, RMS-driven
quiver) in `signals.py`, audio analysis in `analysis.py`. Re-exports the
analysis accessors so existing import paths keep working.
"""
from __future__ import annotations

import asyncio

from analysis import (
    GongAnalysis,
    ThunderAnalysis,
    get_gong,
    get_thunder,
    warm_cache,
)
from colors import lerp, scale
from config import Config
from control import Clock, ClockLike, LightController, VolumeController
from runner import frame_loop
from signals import (
    Envelope,
    candle,
    cosine,
    quadratic_in,
    quiver,
    shaped,
    smoothstep,
)

__all__ = [
    # Re-exports
    "GongAnalysis", "ThunderAnalysis", "get_gong", "get_thunder", "warm_cache",
    # Effects
    "gavel_effect", "gong_effect", "lightning_effect",
    "morning_effect", "night_effect",
]


def _resolve_clock(clock: ClockLike | None) -> ClockLike:
    return clock if clock is not None else Clock()


async def _settle_to_bright_white(
    ctl: LightController,
    config: Config,
    clock: ClockLike,
    duration: float,
    ease=cosine,
) -> None:
    """Lerp from `ctl.last_colors` into `config.bright_white` over
    `duration`. Reads `config.bright_white` on every frame so a
    mid-effect dashboard update shifts the destination on the fly."""
    start_colors = list(ctl.last_colors[:3])
    while len(start_colors) < 3:
        start_colors.append(config.bright_white)

    def frame(t: float):
        f = ease(t / duration)
        target = config.bright_white
        return [lerp(c, target, f) for c in start_colors]

    await frame_loop(ctl, clock, duration, frame)
    ctl.set_colors([config.bright_white] * len(start_colors))


# ────────────────────────── Gavel ───────────────────────────────────


async def gavel_effect(
    ctl: LightController,
    config: Config,
    *,
    clock: ClockLike | None = None,
) -> None:
    """Two synced flashes timed to gavel.wav: yellow on the first hit,
    red on the second. Hit times come from offline onset analysis. After
    the audio finishes, the lights settle to BRIGHT_WHITE."""
    clock = _resolve_clock(clock)

    AMBIENT = (40, 30, 15)
    YELLOW = (255, 220, 0)
    RED = (255, 30, 20)
    HITS = [(0.070, YELLOW), (0.557, RED)]
    AUDIO_DURATION = 1.4
    FLASH_DUR = 0.06
    FADE_DUR = 0.40

    def frame(t: float):
        recent = None
        for hit_t, hit_color in HITS:
            if hit_t > t:
                break
            recent = (hit_t, hit_color)
        if recent is None:
            return AMBIENT
        hit_t, hit_color = recent
        d = t - hit_t
        if d < FLASH_DUR:
            return hit_color
        if d < FLASH_DUR + FADE_DUR:
            f = quadratic_in((d - FLASH_DUR) / FADE_DUR)
            return lerp(hit_color, AMBIENT, f)
        return AMBIENT

    await frame_loop(ctl, clock, AUDIO_DURATION, frame)
    await _settle_to_bright_white(ctl, config, clock, duration=1.5)


# ────────────────────────── Lightning ───────────────────────────────


async def lightning_effect(
    ctl: LightController,
    config: Config,
    *,
    clock: ClockLike | None = None,
) -> None:
    """Lightning with per-channel ambient + RMS-driven quiver.

    Between strikes each channel shows its own shade of blue/purple, with
    brightness pulsing subtly with the audio's RMS envelope. At each
    strike time, all 3 channels flash max-white, then ease back down to
    their (still quivering) base color. After the audio, settles to
    BRIGHT_WHITE.

    Strike times are the leading edges of each high-frequency cluster in
    thunder.wav (the HF-flux "Edge" detection method)."""
    clock = _resolve_clock(clock)
    a = get_thunder()
    env = Envelope(samples=a.rms, hop_seconds=a.rms_hop_seconds)

    BASE_COLORS = [
        (10, 30, 110),    # ch0: deep blue
        (40, 15, 120),    # ch1: indigo
        (80, 20, 110),    # ch2: violet
    ]
    WHITE = (255, 255, 255)
    STRIKES = [0.046, 0.580]
    FLASH_DUR = 0.08
    FADE_DUR = 0.30
    SETTLE_DUR = 2.0

    # Don't start the settle until the last strike has had time to fully
    # flash AND fade — otherwise a strike near the file's tail gets
    # visually eaten by the lerp to BRIGHT_WHITE.
    last_strike = max(STRIKES) if STRIKES else 0.0
    main_duration = max(a.duration, last_strike + FLASH_DUR + FADE_DUR + 0.1)

    def frame(t: float):
        q = quiver(env.at(t))
        base = [scale(c, q) for c in BASE_COLORS]
        recent = None
        for s_t in STRIKES:
            if s_t > t:
                break
            recent = s_t
        if recent is not None and t - recent < FLASH_DUR + FADE_DUR:
            d = t - recent
            if d < FLASH_DUR:
                return [WHITE, WHITE, WHITE]
            f = quadratic_in((d - FLASH_DUR) / FADE_DUR)
            return [lerp(WHITE, base[ch], f) for ch in range(3)]
        return base

    await frame_loop(ctl, clock, main_duration, frame)
    await _settle_to_bright_white(ctl, config, clock, duration=SETTLE_DUR)


# ────────────────────────── Gong ────────────────────────────────────


async def gong_effect(
    ctl: LightController,
    config: Config,
    *,
    clock: ClockLike | None = None,
) -> None:
    """RMS-envelope-driven gold pulse, then a 1.5s settle to BRIGHT_WHITE."""
    clock = _resolve_clock(clock)
    a = get_gong()
    env = Envelope(samples=a.rms, hop_seconds=a.hop_seconds)
    GOLD = (255, 165, 30)

    def frame(t: float):
        return scale(GOLD, shaped(env.at(t)))

    await frame_loop(ctl, clock, a.duration, frame)
    await _settle_to_bright_white(ctl, config, clock, duration=1.5)


# ────────────────────────── Morning ─────────────────────────────────


async def morning_effect(
    ctl: LightController,
    config: Config,
    *,
    clock: ClockLike | None = None,
) -> None:
    """1s smoothstep crossfade from current state into BRIGHT_WHITE.
    Pairs with `night_effect` — pressing Morning after Night gracefully
    lifts the lights out of the candle state."""
    clock = _resolve_clock(clock)
    await _settle_to_bright_white(ctl, config, clock, duration=1.0, ease=smoothstep)


# ────────────────────────── Night ───────────────────────────────────


async def night_effect(
    ctl: LightController,
    config: Config,
    *,
    spotify: VolumeController | None = None,
    prev_volume: int | None = None,
    clock: ClockLike | None = None,
) -> None:
    """Persistent Night state. Three phases, the last sustains forever
    until cancelled (typically by pressing Morning):
      1. (1s)  Crossfade from current state to 20% blue / purple / dark blue.
      2. (2s)  Crossfade those into 60% blue / purple / orange, with the
               candle flicker amplitude ramping up over the same window.
      3. (∞)   Sustained candle flicker on ch2 (orange); ch0/ch1 hold
               steady. Runs until cancelled — does NOT auto-fade to
               BRIGHT_WHITE.

    If `spotify` and `prev_volume` are provided, the music volume is
    faded from `prev_volume` to `config.night_target_volume` over the
    full phase 1+2 transition (3s) — synced with the lights' settle
    into the candle state. Volume steps are spaced ~300ms apart because
    each set_volume call is a slow osascript invocation.
    """
    clock = _resolve_clock(clock)

    P1_DUR = 1.0
    P2_DUR = 2.0

    p1_targets = [
        scale((0, 100, 255), 0.2),     # ch0 blue
        scale((140, 70, 220), 0.2),    # ch1 purple
        scale((40, 80, 200), 0.2),     # ch2 dark blue
    ]
    p2_bases = [
        scale((0, 100, 255), 0.6),     # ch0 blue
        scale((140, 70, 220), 0.6),    # ch1 purple
        scale((255, 110, 20), 0.6),    # ch2 orange
    ]

    start_colors = list(ctl.last_colors[:3])
    while len(start_colors) < 3:
        start_colors.append(config.bright_white)

    # Volume choreography runs concurrent with phase 1+2 so the music
    # crests up to the night setpoint as the lights settle in. Cancelled
    # in `finally` if night_effect itself is cancelled.
    volume_task: asyncio.Task | None = None
    if spotify is not None and prev_volume is not None:
        volume_task = asyncio.create_task(
            _fade_volume_synced(
                spotify, clock,
                start=prev_volume,
                end=config.night_target_volume,
                duration=P1_DUR + P2_DUR,
                steps=10,
            )
        )

    try:
        def phase1(t: float):
            f = smoothstep(t / P1_DUR)
            return [lerp(start_colors[ch], p1_targets[ch], f) for ch in range(3)]

        await frame_loop(ctl, clock, P1_DUR, phase1)

        def phase2(t: float):
            f = smoothstep(t / P2_DUR)
            snap = []
            for ch in range(3):
                base = lerp(p1_targets[ch], p2_bases[ch], f)
                if ch == 2:
                    # Use absolute time for candle so the waveform is
                    # continuous across phase boundaries.
                    cf = candle(P1_DUR + t, ch)
                    cf_blend = 1.0 + (cf - 1.0) * f
                    snap.append(scale(base, cf_blend))
                else:
                    snap.append(base)
            return snap

        await frame_loop(ctl, clock, P2_DUR, phase2)

        # Phase 3 is "forever": frame_loop with infinite duration, broken
        # by the task being cancelled (which propagates from `clock.sleep`).
        def phase3(t: float):
            absolute_t = t + P1_DUR + P2_DUR
            snap = []
            for ch in range(3):
                if ch == 2:
                    cf = candle(absolute_t, ch)
                    snap.append(scale(p2_bases[ch], cf))
                else:
                    snap.append(p2_bases[ch])
            return snap

        await frame_loop(ctl, clock, float("inf"), phase3)
    finally:
        if volume_task is not None and not volume_task.done():
            volume_task.cancel()


async def _fade_volume_synced(
    spotify: VolumeController,
    clock: ClockLike,
    *,
    start: int,
    end: int,
    duration: float,
    steps: int = 10,
) -> None:
    """Stepped volume fade tied to the effect's clock. Each step is
    ~`duration / steps` seconds; throttled because `set_volume` is slow
    (~50–100ms per osascript call on macOS)."""
    if steps <= 0 or duration <= 0:
        await spotify.set_volume(end)
        return
    step_dur = duration / steps
    for i in range(1, steps + 1):
        f = i / steps
        v = int(round(start + (end - start) * f))
        await spotify.set_volume(v)
        await clock.sleep(step_dur)
