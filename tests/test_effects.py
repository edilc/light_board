"""Trace-and-assert tests for every effect in `effects.py`.

Each test:
  1. Builds a `VirtualClock` so the 20-25s effects run in milliseconds.
  2. Runs the effect against a `RecordingController` that captures every
     controller call as `(t, snapshot)` (one color per channel).
  3. Asserts hard invariants (event count, monotonic time, hold color).
  4. Prints a sparse trace + summary so `pytest -s` reads like a debug log.

If something looks off, run `pytest -s tests/test_effects.py::test_morning`
and inspect the printed trace.
"""
from __future__ import annotations

import asyncio

import pytest

import effects
from config import Config
from control import RecordingController, RecordingVolumeController, VirtualClock
from tests.trace import sparse_table, summarize


@pytest.fixture(scope="session", autouse=True)
def _warm_audio_cache():
    """Load thunder + gong analyses once per session — pytest-asyncio cannot
    use this fixture in async form, so we call the sync getters directly."""
    effects.get_thunder()
    effects.get_gong()


@pytest.fixture
def cfg() -> Config:
    """A default in-memory Config (no `path`, so save() is a no-op)."""
    return Config()


def _check_invariants(name: str, rec: RecordingController, expected_seconds: float) -> None:
    events = rec.events
    assert events, f"{name} produced no events"
    times = [e[0] for e in events]
    assert all(t2 >= t1 for t1, t2 in zip(times, times[1:])), f"{name} timestamps not monotonic"
    span = times[-1] - times[0]
    expected_frames = expected_seconds * 60
    assert 0.9 * expected_frames <= len(events) <= 1.15 * expected_frames, (
        f"{name} frame count {len(events)} outside expected {expected_frames:.0f} ±10–15%"
    )
    assert span <= expected_seconds + 0.5, f"{name} span {span:.2f}s exceeds budget"


def _print_trace(name: str, rec: RecordingController, every: int) -> None:
    print()
    print(summarize(name, rec.events))
    print(sparse_table(rec.events, every))


def _last_color(rec: RecordingController, channel: int = 0):
    return rec.events[-1][1][channel]


BRIGHT_WHITE = (244, 218, 182)  # matches Config default — kept as a literal here
                                # so the test's expected value is locked in even
                                # if the Config default ever drifts.


async def test_morning(cfg: Config):
    """Morning is a 1s lerp from current state to BRIGHT_WHITE.
    Pre-seed last_colors to a known non-bright-white state to verify the
    starting-point logic."""
    clock = VirtualClock()
    rec = RecordingController(clock)
    rec.last_colors = ((50, 30, 80), (40, 60, 100), (90, 60, 20))
    await effects.morning_effect(rec, cfg, clock=clock)
    _check_invariants("morning", rec, expected_seconds=1.0)
    _print_trace("morning", rec, every=10)
    # First frame should be near the seeded state (smoothstep at f=0 → 0).
    first = rec.events[0][1]
    assert first[0] == (50, 30, 80), f"morning ch0 first frame {first[0]}, expected seeded color"
    assert first[1] == (40, 60, 100), f"morning ch1 first frame {first[1]}"
    assert first[2] == (90, 60, 20), f"morning ch2 first frame {first[2]}"
    # Final frame should be BRIGHT_WHITE on every channel.
    assert rec.events[-1][1] == (BRIGHT_WHITE,) * 3, (
        f"morning should settle at BRIGHT_WHITE on all channels, got {rec.events[-1][1]}"
    )


async def test_night(cfg: Config):
    """Night runs forever — phases 1-2 transition into the candle state,
    phase 3 sustains candle flicker until the task is cancelled. The test
    pumps the virtual clock past phase 3 and then cancels."""
    clock = VirtualClock()
    rec = RecordingController(clock)
    rec.last_colors = (BRIGHT_WHITE, BRIGHT_WHITE, BRIGHT_WHITE)

    task = asyncio.create_task(effects.night_effect(rec, cfg, clock=clock))
    # Pump until ~9s of virtual time has elapsed (phase 1+2 = 3s, plus 6s of candle).
    target = 9.0
    safety = int(target * 60 * 3)  # 3x frames as a hang fuse
    iterations = 0
    while clock.now() < target and not task.done() and iterations < safety:
        await asyncio.sleep(0)
        iterations += 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert clock.now() >= target, f"virtual clock only reached {clock.now():.2f}s"
    _print_trace("night", rec, every=80)

    # First frame: smoothstep at f=0 → starts at seeded BRIGHT_WHITE.
    assert rec.events[0][1] == (BRIGHT_WHITE,) * 3, (
        f"night should start from BRIGHT_WHITE, got {rec.events[0][1]}"
    )
    # Effect should NOT settle at BRIGHT_WHITE — it sustains candle flicker.
    assert rec.events[-1][1] != (BRIGHT_WHITE,) * 3, (
        "night must persist in candle state; it should not auto-fade to BRIGHT_WHITE"
    )
    # During phases 1-3, channels should differ (per-channel colors).
    non_uniform_count = sum(1 for _, snap in rec.events if len(set(snap)) != 1)
    assert non_uniform_count > 100, (
        f"night should have many non-uniform frames; got {non_uniform_count}"
    )
    # Phase 3 (sustained, t=3..target): only channel 2 (orange candle)
    # should flicker; channels 0 and 1 hold steady.
    p3_ch0 = [snap[0] for t, snap in rec.events if 3.5 <= t <= target]
    p3_ch1 = [snap[1] for t, snap in rec.events if 3.5 <= t <= target]
    p3_ch2 = [sum(snap[2]) / 3 for t, snap in rec.events if 3.5 <= t <= target]
    assert len(set(p3_ch0)) == 1, (
        f"channel 0 should be steady during candle phase; saw {len(set(p3_ch0))} distinct colors"
    )
    assert len(set(p3_ch1)) == 1, (
        f"channel 1 should be steady during candle phase; saw {len(set(p3_ch1))} distinct colors"
    )
    assert max(p3_ch2) - min(p3_ch2) > 5, (
        f"channel 2 candle should flicker visibly; range was {max(p3_ch2) - min(p3_ch2):.1f}"
    )


async def test_gong(cfg: Config):
    clock = VirtualClock()
    rec = RecordingController(clock)
    await effects.gong_effect(rec, cfg, clock=clock)
    gong = effects.get_gong()
    _check_invariants("gong", rec, expected_seconds=gong.duration + 1.5)
    _print_trace("gong", rec, every=40)
    assert _last_color(rec) == BRIGHT_WHITE, "gong should settle at BRIGHT_WHITE"
    assert rec.is_uniform()


async def test_gavel(cfg: Config):
    clock = VirtualClock()
    rec = RecordingController(clock)
    await effects.gavel_effect(rec, cfg, clock=clock)
    # 1.4s of flash logic + 1.5s settle = ~2.9s
    _check_invariants("gavel", rec, expected_seconds=2.9)
    _print_trace("gavel", rec, every=20)
    yellow_frame = next(
        (i for i, (_, snap) in enumerate(rec.events) if snap[0] == (255, 220, 0)),
        None,
    )
    red_frame = next(
        (i for i, (_, snap) in enumerate(rec.events) if snap[0] == (255, 30, 20)),
        None,
    )
    assert yellow_frame is not None, "yellow flash never appeared in gavel trace"
    assert red_frame is not None, "red flash never appeared in gavel trace"
    yellow_t = rec.events[yellow_frame][0]
    red_t = rec.events[red_frame][0]
    assert abs(yellow_t - 0.070) < 0.05, f"yellow flash at {yellow_t:.3f}s, expected ~0.070s"
    assert abs(red_t - 0.557) < 0.05, f"red flash at {red_t:.3f}s, expected ~0.557s"
    assert _last_color(rec) == BRIGHT_WHITE, "gavel should settle at BRIGHT_WHITE"
    assert rec.is_uniform()


async def test_lightning(cfg: Config):
    clock = VirtualClock()
    rec = RecordingController(clock)
    await effects.lightning_effect(rec, cfg, clock=clock)
    thunder = effects.get_thunder()
    _check_invariants("lightning", rec, expected_seconds=thunder.duration + 2.0)
    _print_trace("lightning", rec, every=120)

    # Each strike should produce a max-white frame on all 3 channels within ~0.1s.
    STRIKES = [0.046, 0.580]
    for strike_t in STRIKES:
        white_frames = [
            (t, snap)
            for t, snap in rec.events
            if abs(t - strike_t) < 0.15 and all(min(c) >= 240 for c in snap)
        ]
        assert white_frames, f"no max-white frame within 150ms of strike at {strike_t}s"

    # In-between (after both strikes' fades, before settle): channels must
    # show distinct shades — not uniform — and brightness must vary
    # (the RMS-driven quiver).
    in_between = [
        (t, snap)
        for t, snap in rec.events
        if 2.0 < t < 4.0
    ]
    assert in_between, "no frames captured in the in-between window"
    # Channels are distinct (different blue/purple shades).
    assert all(len(set(snap)) > 1 for _, snap in in_between), (
        "in-between frames should have varying shades across channels, not uniform"
    )
    # Channel-0 brightness varies across the window (quiver). The amplitude
    # should be visible — not just int-rounding noise. Quiver factor is
    # 0.85→1.15 (30% range) on a base average ~50, so we expect ≥5 units of
    # spread across the ~120-frame window. Older test allowed >1, which
    # would pass even on a totally broken quiver via quantization noise.
    ch0_brightness = [sum(snap[0]) / 3 for _, snap in in_between]
    spread = max(ch0_brightness) - min(ch0_brightness)
    assert spread > 5, (
        f"channel 0 should quiver with RMS; range was {spread:.1f}, expected >5"
    )

    # Settle: last frame is BRIGHT_WHITE on all channels.
    assert rec.events[-1][1] == (BRIGHT_WHITE,) * 3, (
        f"lightning should settle at BRIGHT_WHITE on all channels, got {rec.events[-1][1]}"
    )


async def test_bright_white_live_update():
    """Effects must read `config.bright_white` on every frame, not capture
    it once at function start. The dashboard's "Save as Day target"
    mutates the config in place — a regression to capturing-at-start
    would silently break that flow.

    We probe this by changing config.bright_white *mid-effect* and
    asserting the final frame reflects the new value. A capture-at-start
    regression would fail this even though it'd pass a "set before, run,
    check final" style of test."""
    cfg = Config(bright_white=(50, 50, 50))
    clock = VirtualClock()
    rec = RecordingController(clock)
    rec.last_colors = ((10, 10, 10),) * 3

    task = asyncio.create_task(effects.morning_effect(rec, cfg, clock=clock))
    # Pump until ~halfway through morning (1.0s total).
    target_t = 0.5
    safety = int(target_t * 60 * 3)
    i = 0
    while clock.now() < target_t and not task.done() and i < safety:
        await asyncio.sleep(0)
        i += 1
    # Mutate the config *while the effect is running*.
    cfg.bright_white = (200, 220, 240)
    await task

    assert rec.events[-1][1] == ((200, 220, 240),) * 3, (
        f"morning should land on the *later* config.bright_white, "
        f"got {rec.events[-1][1]} (pre-change value was 50,50,50)"
    )


async def test_night_cancellation_preserves_state(cfg: Config):
    """Effects do NOT zero the lights on cancel — that lets the next effect
    (typically morning) lerp continuously from the prior state. If an
    effect adds a `finally: ctl.set_color(0,0,0)` it would break that
    contract silently."""
    clock = VirtualClock()
    rec = RecordingController(clock)
    rec.last_colors = (BRIGHT_WHITE,) * 3

    task = asyncio.create_task(effects.night_effect(rec, cfg, clock=clock))
    target = 4.0  # past phase 1+2, well into the candle sustain
    safety = int(target * 60 * 3)
    iterations = 0
    while clock.now() < target and not task.done() and iterations < safety:
        await asyncio.sleep(0)
        iterations += 1
    last_before_cancel = rec.last_colors
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert rec.last_colors == last_before_cancel, (
        "controller state must not change after cancellation — "
        f"was {last_before_cancel}, became {rec.last_colors}"
    )
    assert rec.last_colors != ((0, 0, 0),) * 3, (
        "night must NOT zero on cancel — Morning relies on lerping from prior color"
    )
    assert rec.last_colors != (BRIGHT_WHITE,) * 3, (
        "night should have moved last_colors away from the seeded BRIGHT_WHITE by t=4s"
    )


async def test_night_drives_volume_synced_with_lights(cfg: Config):
    """If `night_effect` is given a VolumeController + prev_volume, it must
    fade the volume from prev_volume to `config.night_target_volume` while
    the lights settle into the candle state — not as an external task."""
    cfg.night_target_volume = 80
    clock = VirtualClock()
    rec = RecordingController(clock)
    vol = RecordingVolumeController(clock)

    task = asyncio.create_task(
        effects.night_effect(rec, cfg, spotify=vol, prev_volume=20, clock=clock)
    )
    # Let the effect run well past the phase 1+2 transition (3s).
    target_t = 8.0
    safety = int(target_t * 60 * 3)
    i = 0
    while clock.now() < target_t and not task.done() and i < safety:
        await asyncio.sleep(0)
        i += 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert vol.events, "night_effect must emit volume events when given a controller"
    volumes = [v for _, v in vol.events]
    # First volume should be between start and target (a fade step), and
    # the last should hit the target.
    assert 20 < volumes[0] <= 80, f"first step {volumes[0]} should be a fade step between 20 and 80"
    assert volumes[-1] == 80, f"final volume should hit night_target_volume=80, got {volumes[-1]}"
    # Fade is monotonic upward toward target.
    assert volumes == sorted(volumes), f"volume fade should be monotonic; got {volumes}"


async def test_night_volume_skipped_without_spotify(cfg: Config):
    """When `spotify` or `prev_volume` is None, night_effect must not
    attempt any volume work — used by tests that don't care about audio,
    and as a defensive fallback if SpotifyController isn't available."""
    clock = VirtualClock()
    rec = RecordingController(clock)
    # No spotify → no volume task spawned, no exception.
    task = asyncio.create_task(effects.night_effect(rec, cfg, clock=clock))
    safety = 60 * 3 * 3  # ~3s of frames
    i = 0
    while clock.now() < 3.5 and not task.done() and i < safety:
        await asyncio.sleep(0)
        i += 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # If we got here without an exception and produced light events, OK.
    assert rec.events, "lights still drive when spotify is omitted"


