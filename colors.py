"""Color arithmetic for effects.

Colors are bare `(r, g, b)` int tuples to stay compatible with
`LightController.set_color(*c)` and the recorded snapshots tests assert
on. These helpers replace the per-effect `_lerp` / `_scale` lambdas that
used to live in effects.py.
"""
from __future__ import annotations

RGB = tuple[int, int, int]


def lerp(a: RGB, b: RGB, f: float) -> RGB:
    """Linear interpolate between two colors. f is clamped to [0, 1]."""
    f = max(0.0, min(1.0, f))
    return (
        int(round(a[0] + (b[0] - a[0]) * f)),
        int(round(a[1] + (b[1] - a[1]) * f)),
        int(round(a[2] + (b[2] - a[2]) * f)),
    )


def scale(color: RGB, factor: float) -> RGB:
    """Scale a color by a multiplicative factor and clamp to [0, 255]."""
    return (
        max(0, min(255, int(round(color[0] * factor)))),
        max(0, min(255, int(round(color[1] * factor)))),
        max(0, min(255, int(round(color[2] * factor)))),
    )


def clamp(color: RGB) -> RGB:
    """Clamp each channel to [0, 255] without scaling."""
    return (
        max(0, min(255, int(color[0]))),
        max(0, min(255, int(color[1]))),
        max(0, min(255, int(color[2]))),
    )
