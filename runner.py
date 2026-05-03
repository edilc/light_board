"""Effect-runtime primitives.

The frame-loop and settle-to patterns were duplicated in every effect.
These helpers keep all time-walking and `await clock.sleep` in one place
so effect bodies focus on choreography (color-at-time computations).

Effects pass `frame_fn(t) -> RGB | Sequence[RGB]` to `frame_loop`. Returning
a single tuple means "broadcast to all channels"; returning a sequence
means "per-channel". The runner dispatches to `set_color` vs `set_colors`
to match the existing controller protocol.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from colors import RGB, lerp
from control import ClockLike, LightController
from signals import cosine

FRAME = 1 / 60

FrameOutput = RGB | Sequence[RGB]
FrameFn = Callable[[float], FrameOutput]


def _apply(ctl: LightController, out: FrameOutput) -> None:
    """Push a frame to the controller, dispatching uniform vs per-channel.
    A bare `(r, g, b)` int tuple goes to `set_color`; a sequence of
    tuples goes to `set_colors`."""
    if isinstance(out, tuple) and len(out) == 3 and all(isinstance(c, int) for c in out):
        rgb = cast(RGB, out)
        ctl.set_color(*rgb)
    else:
        ctl.set_colors(cast("Sequence[RGB]", out))


async def frame_loop(
    ctl: LightController,
    clock: ClockLike,
    duration: float,
    frame_fn: FrameFn,
) -> None:
    """Walks `t` from 0 to `duration` in FRAME-sized steps, applying
    `frame_fn(t)` to the controller each tick. Effects import this in
    place of the per-effect `start = clock.now(); while True: ...` boilerplate."""
    start = clock.now()
    while True:
        t = clock.now() - start
        if t >= duration:
            break
        _apply(ctl, frame_fn(t))
        await clock.sleep(FRAME)


async def settle_to(
    ctl: LightController,
    clock: ClockLike,
    target: RGB,
    duration: float,
    ease: Callable[[float], float] = cosine,
) -> None:
    """Smoothly lerp from `ctl.last_colors` into `target` over `duration`.
    Reads the per-channel start state at entry — that's the shared "settle
    to BRIGHT_WHITE" tail used by gavel, lightning, gong, morning."""
    start_colors = list(ctl.last_colors)
    if not start_colors:
        start_colors = [target]

    def frame(t: float) -> Sequence[RGB]:
        f = ease(t / duration)
        return [lerp(c, target, f) for c in start_colors]

    await frame_loop(ctl, clock, duration, frame)
    # Land exactly on target — frame_loop's `t < duration` check can leave
    # the last frame one FRAME short of the endpoint.
    ctl.set_colors([target] * len(start_colors))


async def hold(
    ctl: LightController,
    clock: ClockLike,
    color: RGB,
    duration: float,
) -> None:
    """Set a color and sleep. Used by phases that just need to hold a value."""
    ctl.set_color(*color)
    await clock.sleep(duration)
