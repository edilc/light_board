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
from control import RecordingController, VirtualClock
from tests.trace import sparse_table, summarize


@pytest.fixture(scope="session", autouse=True)
def _warm_audio_cache():
    """Load thunder + gong analyses once per session — pytest-asyncio cannot
    use this fixture in async form, so we call the sync getters directly."""
    effects.get_thunder()
    effects.get_gong()


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


def _first_color(rec: RecordingController, channel: int = 0):
    return rec.events[0][1][channel]


BRIGHT_WHITE = (244, 218, 182)


async def test_morning():
    """Morning is now a 1s lerp from current state to BRIGHT_WHITE.
    Pre-seed last_colors to a known non-bright-white state to verify the
    starting-point logic."""
    clock = VirtualClock()
    rec = RecordingController(clock)
    rec.last_colors = ((50, 30, 80), (40, 60, 100), (90, 60, 20))
    await effects.morning_effect(rec, clock=clock)
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


async def test_night():
    """Night runs forever — phases 1-2 transition into the candle state,
    phase 3 sustains candle flicker until the task is cancelled. The test
    pumps the virtual clock past phase 3 and then cancels."""
    clock = VirtualClock()
    rec = RecordingController(clock)
    rec.last_colors = (BRIGHT_WHITE, BRIGHT_WHITE, BRIGHT_WHITE)

    task = asyncio.create_task(effects.night_effect(rec, clock=clock))
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


async def test_gong():
    clock = VirtualClock()
    rec = RecordingController(clock)
    await effects.gong_effect(rec, clock=clock)
    gong = effects.get_gong()
    _check_invariants("gong", rec, expected_seconds=gong.duration + 1.5)
    _print_trace("gong", rec, every=40)
    assert _last_color(rec) == BRIGHT_WHITE, "gong should settle at BRIGHT_WHITE"
    assert rec.is_uniform()


async def test_gavel():
    clock = VirtualClock()
    rec = RecordingController(clock)
    await effects.gavel_effect(rec, clock=clock)
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


async def test_lightning():
    clock = VirtualClock()
    rec = RecordingController(clock)
    await effects.lightning_effect(rec, clock=clock)
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
    # Channel-0 brightness varies across the window (quiver).
    ch0_brightness = [sum(snap[0]) / 3 for _, snap in in_between]
    assert max(ch0_brightness) - min(ch0_brightness) > 1, (
        f"channel 0 should quiver with RMS; range was {max(ch0_brightness) - min(ch0_brightness):.1f}"
    )

    # Settle: last frame is BRIGHT_WHITE on all channels.
    assert rec.events[-1][1] == (BRIGHT_WHITE,) * 3, (
        f"lightning should settle at BRIGHT_WHITE on all channels, got {rec.events[-1][1]}"
    )
