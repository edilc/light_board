"""Light control layer.

Separates the "how do we drive the lights" plumbing from the "what should the
lights do" choreography in `effects.py`. Both real (`HueController`) and test
(`RecordingController`) implementations satisfy the `LightController` protocol,
so effects can be exercised without a bridge by injecting the recorder + a
virtual clock.
"""
from __future__ import annotations

import asyncio
import time
from typing import Protocol, Sequence, runtime_checkable

from hue_entertainment_pykit import (
    Discovery,
    Entertainment,
    EntertainmentConfiguration,
    Streaming,
)
from hue_entertainment_pykit.models.bridge import Bridge


Color = tuple[int, int, int]
Snapshot = tuple[Color, ...]


@runtime_checkable
class LightController(Protocol):
    """Anything that can accept color commands for the active lights.

    `set_color` broadcasts one color to every channel; `set_colors` lets
    effects drive each channel independently (cycles if shorter than the
    channel count). `last_colors` gives effects a way to read the current
    per-channel state so they can lerp continuously from wherever the
    previous effect (or manual control) left the lights."""

    last_colors: tuple[Color, ...]

    def set_color(self, r: int, g: int, b: int) -> None: ...
    def set_colors(self, colors: Sequence[Color]) -> None: ...


class Clock:
    """Real wall-clock. Effects use `now()` and `await sleep(s)`; tests swap in `VirtualClock`."""

    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualClock:
    """Deterministic clock for tests. `sleep` advances the clock and yields once
    so the event loop can cycle, but no real time passes."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def now(self) -> float:
        return self._t

    async def sleep(self, seconds: float) -> None:
        self._t += seconds
        await asyncio.sleep(0)


class HueController:
    """Drives a Hue Entertainment streaming session. Owns the channel list so
    effects don't need to know about it."""

    def __init__(
        self,
        bridge: Bridge,
        entertainment: Entertainment,
        streaming: Streaming,
        config: EntertainmentConfiguration,
        channel_ids: list[int],
    ) -> None:
        self.bridge = bridge
        self.entertainment = entertainment
        self.streaming = streaming
        self.config = config
        self.channel_ids = list(channel_ids)
        self.last_colors: tuple[Color, ...] = tuple((0, 0, 0) for _ in self.channel_ids)

    def set_color(self, r: int, g: int, b: int) -> None:
        for ch in self.channel_ids:
            self.streaming.set_input((r, g, b, ch))
        self.last_colors = tuple((r, g, b) for _ in self.channel_ids)

    def set_colors(self, colors: Sequence[Color]) -> None:
        if not colors:
            return
        per_channel = tuple(colors[i % len(colors)] for i in range(len(self.channel_ids)))
        for ch, (r, g, b) in zip(self.channel_ids, per_channel):
            self.streaming.set_input((r, g, b, ch))
        self.last_colors = per_channel

    def status(self) -> str:
        return (
            f"Connected to {self.bridge.get_name()} • "
            f"area “{self.config.name}” • {len(self.channel_ids)} channel(s)"
        )


class RecordingController:
    """Captures every controller call as `(t, snapshot)` where `snapshot` is
    one `(r, g, b)` tuple per channel. Drives `t` from an injected clock so
    traces line up with whatever clock the effect was using.

    For uniform-color effects (which only call `set_color`), use
    `channel(idx)` to get a per-event `(t, r, g, b)` view of one channel."""

    def __init__(self, clock: Clock | VirtualClock, channel_count: int = 3) -> None:
        self._clock = clock
        self.channel_count = channel_count
        self.events: list[tuple[float, Snapshot]] = []
        self.last_colors: tuple[Color, ...] = tuple((0, 0, 0) for _ in range(channel_count))

    def set_color(self, r: int, g: int, b: int) -> None:
        snap: Snapshot = tuple((r, g, b) for _ in range(self.channel_count))
        self.events.append((self._clock.now(), snap))
        self.last_colors = snap

    def set_colors(self, colors: Sequence[Color]) -> None:
        if not colors:
            return
        snap: Snapshot = tuple(colors[i % len(colors)] for i in range(self.channel_count))
        self.events.append((self._clock.now(), snap))
        self.last_colors = snap

    def channel(self, idx: int) -> list[tuple[float, int, int, int]]:
        """View of one channel's history as `(t, r, g, b)` per event."""
        return [(t, *snap[idx]) for t, snap in self.events]

    def is_uniform(self) -> bool:
        """True if every recorded snapshot has the same color across channels."""
        return all(len(set(snap)) == 1 for _, snap in self.events)


def connect() -> HueController:
    """Discover the first bridge, pick the first entertainment area, start
    streaming, and return a controller. Blocking — call from a thread."""
    bridges = Discovery().discover_bridges()
    if not bridges:
        raise RuntimeError("No Hue bridge found on the network.")

    bridge = next(iter(bridges.values()))
    entertainment = Entertainment(bridge)
    configs = entertainment.get_entertainment_configs()
    if not configs:
        raise RuntimeError("Bridge has no entertainment areas configured.")

    config = next(iter(configs.values()))
    streaming = Streaming(bridge, config, entertainment.get_ent_conf_repo())
    streaming.start_stream()
    streaming.set_color_space("rgb")

    channel_ids = [ch.channel_id for ch in config.channels]
    return HueController(bridge, entertainment, streaming, config, channel_ids)
