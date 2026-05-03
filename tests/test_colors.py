"""Unit tests for colors.lerp / scale / clamp."""
from __future__ import annotations

import pytest

from colors import clamp, lerp, scale


class TestLerp:
    @pytest.mark.parametrize(
        "f, expected",
        [
            (-1.0, (10, 20, 30)),    # below 0 clamps to a
            (0.0, (10, 20, 30)),     # f=0 → a
            (1.0, (200, 100, 50)),   # f=1 → b
            (5.0, (200, 100, 50)),   # above 1 clamps to b
        ],
    )
    def test_endpoints_and_clamps(self, f, expected):
        assert lerp((10, 20, 30), (200, 100, 50), f) == expected

    def test_midpoint(self):
        assert lerp((0, 0, 0), (100, 200, 50), 0.5) == (50, 100, 25)


class TestScale:
    def test_clamps_above_255(self):
        assert scale((200, 200, 200), 2.0) == (255, 255, 255)

    def test_clamps_below_zero(self):
        assert scale((100, 50, 25), -1.0) == (0, 0, 0)

    def test_rounds(self):
        # 100 * 0.123 = 12.3 → 12
        assert scale((100, 0, 0), 0.123) == (12, 0, 0)


class TestClamp:
    def test_clamps_above_255(self):
        assert clamp((300, 256, 1000)) == (255, 255, 255)

    def test_clamps_negative(self):
        assert clamp((-10, -1, 50)) == (0, 0, 50)
