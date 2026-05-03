"""Unit tests for signals (easings, Envelope, audio shapers)."""
from __future__ import annotations

import numpy as np
import pytest

from signals import Envelope, candle, cosine, quadratic_in, quiver, shaped, smoothstep


class TestEasings:
    @pytest.mark.parametrize("ease", [smoothstep, cosine, quadratic_in])
    def test_endpoints(self, ease):
        assert ease(0.0) == pytest.approx(0.0)
        assert ease(1.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("ease", [smoothstep, cosine, quadratic_in])
    def test_clamps(self, ease):
        # Out-of-range inputs clamp to the endpoint values.
        assert ease(-0.5) == pytest.approx(0.0)
        assert ease(1.5) == pytest.approx(1.0)

    @pytest.mark.parametrize("ease", [smoothstep, cosine, quadratic_in])
    def test_monotonic(self, ease):
        prev = ease(0.0)
        for f in [i / 100 for i in range(1, 101)]:
            v = ease(f)
            assert v >= prev - 1e-9, f"{ease.__name__} not monotonic at f={f}"
            prev = v

    def test_quadratic_in_starts_slow(self):
        # Distinguishes ease-in from ease-out shape: f^2 at f=0.1 → 0.01,
        # well below the symmetric eases' midpoints.
        assert quadratic_in(0.1) == pytest.approx(0.01)


class TestEnvelope:
    def setup_method(self):
        self.env = Envelope(
            samples=np.array([0.1, 0.5, 0.9, 0.3], dtype=np.float32),
            hop_seconds=0.1,
        )

    def test_duration(self):
        assert self.env.duration == pytest.approx(0.4)

    def test_at_zero(self):
        assert self.env.at(0.0) == pytest.approx(0.1)

    def test_at_within_first_bucket(self):
        # t=0.05 falls in bucket 0 (0..0.1).
        assert self.env.at(0.05) == pytest.approx(0.1)

    def test_at_bucket_boundary(self):
        # t=0.1 → idx=1 → samples[1].
        assert self.env.at(0.1) == pytest.approx(0.5)

    def test_at_negative_returns_zero(self):
        assert self.env.at(-0.5) == 0.0

    def test_at_past_end_returns_zero(self):
        assert self.env.at(1.0) == 0.0
        assert self.env.at(0.4) == 0.0  # exactly at duration → idx=4 → past end


class TestQuiver:
    @pytest.mark.parametrize(
        "env, low, high, expected",
        [
            (0.0, 0.85, 1.15, 0.85),   # default low bound
            (1.0, 0.85, 1.15, 1.15),   # default high bound
            (0.0, 0.5, 2.0, 0.5),      # custom bounds
            (1.0, 0.5, 2.0, 2.0),
        ],
    )
    def test_bounds(self, env, low, high, expected):
        assert quiver(env, low=low, high=high) == pytest.approx(expected)

    def test_negative_clamps(self):
        # Negative env should not produce negative quiver — should land at low.
        assert quiver(-0.5) == pytest.approx(0.85)

    def test_gamma_lifts_quiet(self):
        # With gamma=0.5, env=0.04 → 0.2, so quiver = 0.85 + 0.30*0.2 = 0.91.
        # With gamma=1.0, env=0.04 → 0.04, so quiver = 0.85 + 0.30*0.04 = 0.862.
        # Default gamma=0.5 should be HIGHER than gamma=1 for the same low env.
        assert quiver(0.04, gamma=0.5) > quiver(0.04, gamma=1.0)


class TestShaped:
    def test_floor(self):
        assert shaped(0.0) == pytest.approx(0.08)
        assert shaped(-1.0) == pytest.approx(0.08)

    def test_at_one(self):
        # 1^0.55 = 1, above floor.
        assert shaped(1.0) == pytest.approx(1.0)

    def test_custom_floor(self):
        assert shaped(0.0, floor=0.5) == 0.5


class TestCandle:
    def test_amplitude_within_bound(self):
        # With amp=0.2, |candle(t) - 1| <= 0.2 (the noise term is normalized
        # so |noise| <= 1).
        for t in [i / 60 for i in range(600)]:
            v = candle(t, 0)
            assert 0.79 <= v <= 1.21, f"candle({t}, 0)={v} outside ±0.2 bound"

    def test_channels_differ(self):
        # Different channels have different phase offsets — at any given
        # time, the three channels' candle factors should not all be equal.
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            v0, v1, v2 = candle(t, 0), candle(t, 1), candle(t, 2)
            assert not (v0 == v1 == v2), f"channels identical at t={t}"

    def test_deterministic(self):
        assert candle(1.234, 0) == candle(1.234, 0)

    def test_amp_zero_is_constant(self):
        # amp=0 → always 1.0 regardless of t/channel.
        for t in [0.0, 0.5, 1.0, 1.5]:
            for ch in range(3):
                assert candle(t, ch, amp=0.0) == 1.0
