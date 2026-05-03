"""Unit tests for runner primitives (frame_loop, settle_to, hold)."""
from __future__ import annotations

import pytest

from control import RecordingController, VirtualClock
from runner import frame_loop, hold, settle_to


class TestFrameLoop:
    async def test_uniform_color_dispatches_to_set_color(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        await frame_loop(rec, clock, duration=0.1, frame_fn=lambda t: (100, 50, 25))
        assert rec.events, "no events recorded"
        # Uniform: every channel should match.
        assert rec.is_uniform()
        assert rec.events[0][1][0] == (100, 50, 25)

    async def test_per_channel_dispatches_to_set_colors(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        per_channel = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
        await frame_loop(rec, clock, duration=0.05, frame_fn=lambda t: per_channel)
        assert rec.events
        assert rec.events[0][1] == tuple(per_channel)

    async def test_frame_count_matches_duration(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        await frame_loop(rec, clock, duration=1.0, frame_fn=lambda t: (1, 2, 3))
        # 1.0s at 1/60 step = 60 frames, give-or-take one for the boundary.
        assert 59 <= len(rec.events) <= 61, f"got {len(rec.events)} frames"

    async def test_frame_fn_receives_monotonic_t(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        ts: list[float] = []

        def fn(t: float):
            ts.append(t)
            return (10, 10, 10)

        await frame_loop(rec, clock, duration=0.2, frame_fn=fn)
        assert ts[0] == pytest.approx(0.0)
        assert all(t2 > t1 for t1, t2 in zip(ts, ts[1:])), "t not strictly increasing"
        assert ts[-1] < 0.2, "last t should be strictly less than duration"

    async def test_zero_duration_no_events(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        await frame_loop(rec, clock, duration=0.0, frame_fn=lambda t: (1, 2, 3))
        assert rec.events == []


class TestSettleTo:
    async def test_lands_exactly_on_target(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        rec.last_colors = ((10, 10, 10),) * 3
        target = (200, 150, 100)
        await settle_to(rec, clock, target=target, duration=0.5)
        # Last frame must be EXACTLY target on every channel — the runner
        # explicitly snaps to the endpoint after the loop.
        assert rec.events[-1][1] == (target,) * 3

    async def test_monotonic_progress_per_channel(self):
        # Cosine ease is monotonic non-decreasing for each channel between
        # start and target. Also pins the start frame, since seeded → eased
        # progression → exact target endpoint is one continuous contract.
        clock = VirtualClock()
        rec = RecordingController(clock)
        seed = ((0, 0, 0),) * 3
        rec.last_colors = seed
        target = (255, 255, 255)
        await settle_to(rec, clock, target=target, duration=0.5)
        assert rec.events[0][1] == seed, "first frame should equal seeded state"
        for ch in range(3):
            channel_r = [snap[ch][0] for _, snap in rec.events]
            assert channel_r == sorted(channel_r), (
                f"ch{ch} R not monotonically increasing during settle"
            )


class TestHold:
    async def test_single_event_then_sleep(self):
        clock = VirtualClock()
        rec = RecordingController(clock)
        await hold(rec, clock, color=(50, 50, 50), duration=0.5)
        # hold sets once, sleeps; only one event recorded.
        assert len(rec.events) == 1
        assert rec.events[0][1][0] == (50, 50, 50)
        # Clock advanced by the hold duration.
        assert clock.now() == pytest.approx(0.5)
