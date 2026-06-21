"""Light control layer.

Separates the "how do we drive the lights" plumbing from the "what should the
lights do" choreography in `effects.py`. Both real (`HueController`) and test
(`RecordingController`) implementations satisfy the `LightController` protocol,
so effects can be exercised without a bridge by injecting the recorder + a
virtual clock.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from hue_entertainment_pykit import (
    Discovery,
    Entertainment,
    EntertainmentConfiguration,
    Streaming,
)
from hue_entertainment_pykit.models.bridge import Bridge
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

logger = logging.getLogger("light_board.control")

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


@runtime_checkable
class VolumeController(Protocol):
    """Anything that can accept media-volume commands. Effects use this to
    choreograph music volume alongside lights — `night_effect` fades the
    volume up to the night setpoint over the same window the lights
    transition into the candle state. `SpotifyController` satisfies this;
    tests use `RecordingVolumeController`."""

    async def set_volume(self, percent: int) -> None: ...


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


ClockLike = Clock | VirtualClock


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

    def stop(self) -> None:
        """Release the active Hue Entertainment stream."""
        self.streaming.stop_stream()


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


class RecordingVolumeController:
    """Captures every `set_volume(percent)` call as `(t, percent)`. Drives
    `t` from an injected clock so volume traces line up with light traces
    when both are recorded in the same test."""

    def __init__(self, clock: Clock | VirtualClock) -> None:
        self._clock = clock
        self.events: list[tuple[float, int]] = []

    async def set_volume(self, percent: int) -> None:
        self.events.append((self._clock.now(), int(percent)))


# The DTLS handshake hangs roughly half the time on this network —
# succeeds in ~600ms when it works, hangs forever when it doesn't.
# Discovery and entertainment-config selection are fast and reliable, so we
# do them once up front and retry only `start_stream` (the PUT + handshake)
# with progressively longer per-attempt budgets, each on a fresh socket.
_CONNECT_TIMEOUTS_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


def _discover() -> tuple[Bridge, Entertainment, EntertainmentConfiguration]:
    """Discover the first bridge and pick its first entertainment area.
    Blocking but reliable — no DTLS handshake here, so it's run once and not
    retried for timeouts. Raises if there's no bridge or no entertainment
    area (those surface immediately rather than being retried)."""
    bridges = Discovery().discover_bridges()
    if not bridges:
        raise RuntimeError("No Hue bridge found on the network.")

    bridge = next(iter(bridges.values()))
    entertainment = Entertainment(bridge)
    configs = entertainment.get_entertainment_configs()
    if not configs:
        raise RuntimeError("Bridge has no entertainment areas configured.")

    config = next(iter(configs.values()))
    return bridge, entertainment, config


def _safe_stop(streaming: Streaming) -> None:
    """Best-effort teardown of a streaming session; never raises."""
    try:
        streaming.stop_stream()
    except Exception:
        logger.warning("failed to tear down abandoned Hue stream", exc_info=True)


def _start_stream_with_timeout(streaming: Streaming, timeout_s: float) -> None:
    """Run `streaming.start_stream()` (PUT + DTLS handshake) in a daemon
    thread; raise TimeoutError if it doesn't finish in `timeout_s`.

    Python can't safely kill a thread, so a hung handshake leaks the worker.
    The Hue bridge allows only one active entertainment stream, so a worker
    that completes *after* its attempt timed out would hold that single slot
    and hang every later attempt. We flag such a worker abandoned and have it
    tear its own just-opened session down once the handshake finally returns.

    A `finished` event guarded by a lock closes the boundary race: if the
    worker completes at the same instant we time out, whichever side takes
    the lock first wins — we either adopt the just-finished session (we see
    `finished` set) or the worker sees `abandoned` set and stops the stream."""
    err: list[BaseException] = []
    finished = threading.Event()
    lock = threading.Lock()
    abandoned = False

    def worker() -> None:
        nonlocal abandoned
        try:
            streaming.start_stream()
        except BaseException as e:
            # Forward everything (incl. KeyboardInterrupt) to the caller.
            with lock:
                err.append(e)
                finished.set()
            return
        with lock:
            finished.set()
            if abandoned:
                _safe_stop(streaming)

    t = threading.Thread(target=worker, name="hue-start-stream", daemon=True)
    t.start()
    finished.wait(timeout_s)
    with lock:
        if not finished.is_set():
            abandoned = True
            raise TimeoutError(f"Hue handshake hung past {timeout_s:.1f}s")
        if err:
            raise err[0]


def connect() -> HueController:
    """Discover the first bridge, pick the first entertainment area, start
    streaming, and return a controller. Blocking — call from a thread.

    Discovery + area selection happen once; only `start_stream` (the DTLS
    handshake that hangs) is retried, with the `_CONNECT_TIMEOUTS_S` budgets
    — 1s, 2s, 4s, 8s — each on a fresh `Streaming`/socket. Non-timeout
    errors (no bridge, no entertainment area) surface immediately."""
    bridge, entertainment, config = _discover()

    for attempt in Retrying(
        stop=stop_after_attempt(len(_CONNECT_TIMEOUTS_S)),
        wait=wait_fixed(0.3),
        retry=retry_if_exception_type(TimeoutError),
        reraise=True,
    ):
        with attempt:
            i = attempt.retry_state.attempt_number - 1
            timeout = _CONNECT_TIMEOUTS_S[i]
            logger.info(
                "connect attempt %d/%d (timeout=%.1fs)",
                i + 1, len(_CONNECT_TIMEOUTS_S), timeout,
            )
            # Fresh Streaming each attempt: a timed-out attempt's worker may
            # still own its old socket, so the retry must not reuse it.
            streaming = Streaming(bridge, config, entertainment.get_ent_conf_repo())
            try:
                _start_stream_with_timeout(streaming, timeout)
            except TimeoutError as e:
                logger.warning("connect attempt %d/%d timed out: %s",
                               i + 1, len(_CONNECT_TIMEOUTS_S), e)
                raise
            streaming.set_color_space("rgb")
            channel_ids = [ch.channel_id for ch in config.channels]
            return HueController(bridge, entertainment, streaming, config, channel_ids)
    raise RuntimeError("unreachable: Retrying with reraise=True must raise on exhaustion")
