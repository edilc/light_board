import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nicegui import app, ui

from hue_entertainment_pykit import setup_logs

import control
import effects
import spotify as spotify_mod


# Write app logs to logs/light_board.log AND stderr. Run with `uv run main.py`
# as before — the file gets appended to automatically.
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "light_board.log"

logger = logging.getLogger("light_board")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    _fh = logging.FileHandler(LOG_FILE)
    _fh.setFormatter(_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)
    logger.propagate = False  # don't double-log via root

setup_logs(level=logging.INFO)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("zeroconf").setLevel(logging.WARNING)
# Streaming logs every set_color call — quiet it down, otherwise the file is
# flooded with thousands of lines per effect.
logging.getLogger("hue_entertainment_pykit.services.streaming_service").setLevel(
    logging.WARNING
)

SOUNDS_DIR = Path(__file__).parent / "sounds"
app.add_static_files("/sounds", str(SOUNDS_DIR))

# Approximate latency from `streaming.set_input(...)` to the bulb actually
# changing color over WiFi+DTLS. Detecting it precisely would require
# timing something visible at the bulb itself, so we assume a fixed value.
# Combined with the browser's reported audio output latency, we compute the
# light-side delay as `max(0, audio_latency - HUE_LATENCY_S)` so audio and
# lights land at the user roughly together.
HUE_LATENCY_S = 0.050


@dataclass
class AppState:
    controller: Optional[control.HueController] = None
    effect_task: Optional[asyncio.Task] = None
    audio_latency_s: float = 0.0  # browser-reported output latency
    spotify: spotify_mod.SpotifyController = None  # type: ignore[assignment]
    last_volume: int = 50
    volume_task: Optional[asyncio.Task] = None


state = AppState()
state.spotify = spotify_mod.SpotifyController()


async def detect_audio_latency() -> float:
    """Probe `AudioContext.outputLatency` in the browser. Returns seconds.

    Built-in speakers report ~5-15ms; Bluetooth/AirPods ~150-250ms. Used to
    delay light commands so they reach the bulbs as the audio reaches the
    user's ears.
    """
    js = """
    try {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return 0;
        const ctx = new Ctx();
        const latency = ctx.outputLatency || ctx.baseLatency || 0;
        try { ctx.close(); } catch (e) {}
        return latency;
    } catch (e) {
        return 0;
    }
    """
    try:
        # NiceGUI's default timeout is 1.0s — bump it so a slow round-trip
        # doesn't silently fall back to 0 (and waste a full second per click).
        result = await ui.run_javascript(js, timeout=3.0)
        return float(result) if result else 0.0
    except Exception as exc:
        logger.warning("detect_audio_latency: %s", exc)
        return 0.0


def set_lights(r: int, g: int, b: int) -> None:
    if state.controller is not None:
        state.controller.set_color(r, g, b)


async def stop_effect() -> None:
    task = state.effect_task
    state.effect_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    vtask = state.volume_task
    state.volume_task = None
    if vtask and not vtask.done():
        vtask.cancel()
        try:
            await vtask
        except (asyncio.CancelledError, Exception):
            pass


@ui.page("/")
def home() -> None:
    with ui.column().classes("w-full min-h-screen items-center justify-center gap-4 p-6"):
        status = ui.label("Connecting…").classes("text-xs text-gray-500")

        thunder_audio = ui.audio("/sounds/thunder.wav").props("preload=auto").style("display: none")
        gong_audio = ui.audio("/sounds/gong.wav").props("preload=auto").style("display: none")
        gavel_audio = ui.audio("/sounds/gavel.wav").props("preload=auto").style("display: none")
        rooster_audio = ui.audio("/sounds/rooster.wav").props("preload=auto").style("display: none")
        all_audio = (thunder_audio, gong_audio, gavel_audio, rooster_audio)

        async def run_effect(coro_factory, audio_el=None, volume_choreography=None) -> None:
            t_click = time.monotonic()
            name = coro_factory.__name__
            logger.info("[%s] run_effect start", name)
            await stop_effect()
            for a in all_audio:
                a.pause()
                a.seek(0)
            if state.controller is None:
                logger.info("[%s] aborting: no controller", name)
                return
            if audio_el is not None:
                t_pre_play = time.monotonic()
                audio_el.play()
                logger.info(
                    "[%s] audio.play() dispatched at +%.0fms",
                    name, (t_pre_play - t_click) * 1000,
                )
                t_pre_detect = time.monotonic()
                try:
                    state.audio_latency_s = await detect_audio_latency()
                    detect_ms = (time.monotonic() - t_pre_detect) * 1000
                    logger.info(
                        "[%s] outputLatency=%.3fs (probe took %.0fms)",
                        name, state.audio_latency_s, detect_ms,
                    )
                except Exception as exc:
                    logger.warning("[%s] latency probe failed: %s", name, exc)
                delay = max(0.0, state.audio_latency_s - HUE_LATENCY_S)
                logger.info(
                    "[%s] computed light delay = max(0, %.3fs - %.3fs) = %.0fms",
                    name, state.audio_latency_s, HUE_LATENCY_S, delay * 1000,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            state.effect_task = asyncio.create_task(coro_factory(state.controller))
            logger.info(
                "[%s] light task created at +%.0fms after click",
                name, (time.monotonic() - t_click) * 1000,
            )
            if volume_choreography is not None:
                prev_vol = state.last_volume
                state.volume_task = asyncio.create_task(
                    volume_choreography(state.spotify, prev_vol)
                )
                logger.info("[%s] volume choreography started (prev_vol=%d)", name, prev_vol)

        async def on_lightning_onset() -> None:
            await run_effect(effects.lightning_onset_effect, thunder_audio, spotify_mod.lightning_volume)

        async def on_lightning_hf() -> None:
            await run_effect(effects.lightning_hf_effect, thunder_audio, spotify_mod.lightning_volume)

        async def on_lightning_amp() -> None:
            await run_effect(effects.lightning_amp_effect, thunder_audio, spotify_mod.lightning_volume)

        async def on_gong() -> None:
            await run_effect(effects.gong_effect, gong_audio, spotify_mod.gong_volume)

        async def on_gavel() -> None:
            await run_effect(effects.gavel_effect, gavel_audio, spotify_mod.gavel_volume)

        async def on_rooster() -> None:
            # Plays the rooster crow but uses the simple morning lerp animation.
            await run_effect(effects.morning_effect, rooster_audio, spotify_mod.morning_volume)

        async def on_morning() -> None:
            await run_effect(effects.morning_effect, None, spotify_mod.morning_volume)

        async def on_night() -> None:
            await run_effect(effects.night_effect, None, spotify_mod.night_volume)

        async def on_stop() -> None:
            await stop_effect()
            for a in all_audio:
                a.pause()
                a.seek(0)
            set_lights(0, 0, 0)

        # flex-wrap means buttons fit in one row when the window is wide
        # enough, and wrap to a column when it's not.
        with ui.row().classes(
            "flex flex-wrap justify-center items-center gap-2 max-w-3xl"
        ):
            l_onset_btn = ui.button("L: Onset", on_click=on_lightning_onset).props("color=indigo")
            l_hf_btn = ui.button("L: HF", on_click=on_lightning_hf).props("color=deep-purple")
            l_amp_btn = ui.button("L: Amp", on_click=on_lightning_amp).props("color=blue")
            gong_btn = ui.button("Gong", on_click=on_gong).props("color=amber-9")
            gavel_btn = ui.button("Gavel", on_click=on_gavel).props("color=brown")
            rooster_btn = ui.button("Rooster", on_click=on_rooster).props("color=orange")
            morning_btn = ui.button("Morning", on_click=on_morning).props("color=yellow-8")
            night_btn = ui.button("Night", on_click=on_night).props("color=blue-9")
            ui.button("Stop", on_click=on_stop).props("flat color=grey")

        effect_buttons = (
            l_onset_btn, l_hf_btn, l_amp_btn,
            gong_btn, gavel_btn, rooster_btn, morning_btn, night_btn,
        )
        for btn in effect_buttons:
            btn.disable()

        # ── Spotify widget ──────────────────────────────────────────
        with ui.column().classes("items-center gap-1 w-full max-w-md"):
            track_label = ui.label("Spotify: not detected").classes(
                "text-xs text-gray-500 truncate max-w-full"
            )
            with ui.row().classes("items-center gap-3 w-full"):
                ui.label("♫").classes("text-sm")
                volume_slider = (
                    ui.slider(min=0, max=100, value=50)
                    .props("label-always color=green")
                    .classes("flex-1")
                )

        # User dragging the slider → push to Spotify, update last_volume.
        async def on_volume_change(e) -> None:
            v = int(e.value)
            state.last_volume = v
            await state.spotify.set_volume(v)

        volume_slider.on("change", on_volume_change)

        async def refresh_spotify() -> None:
            """Sync the widget with the actual Spotify state. Runs every 3s and
            picks up changes made externally (the Spotify app, other devices,
            or our own volume choreographies)."""
            track = await state.spotify.get_track()
            track_label.text = track if track else "Spotify: not playing"
            vol = await state.spotify.get_volume()
            if vol is not None:
                # Only reset the slider if Spotify drifted from our last
                # known value — avoids fighting the user mid-drag.
                if vol != state.last_volume:
                    state.last_volume = vol
                    volume_slider.value = vol

        ui.timer(3.0, refresh_spotify)

        async def startup() -> None:
            try:
                ctl = await asyncio.to_thread(control.connect)
            except Exception as exc:
                status.text = f"Connect failed: {exc}"
                return
            state.controller = ctl
            status.text = "Analyzing audio…"
            try:
                await effects.warm_cache()
            except Exception as exc:
                status.text = f"Audio analysis failed: {exc}"
                return
            try:
                state.audio_latency_s = await detect_audio_latency()
            except Exception as exc:
                logger.warning("startup latency probe failed: %s", exc)
                state.audio_latency_s = 0.0
            logger.info(
                "startup outputLatency=%.3fs → light delay = %.0fms (HUE_LATENCY_S=%.0fms)",
                state.audio_latency_s,
                max(0.0, state.audio_latency_s - HUE_LATENCY_S) * 1000,
                HUE_LATENCY_S * 1000,
            )
            await refresh_spotify()
            logger.info("Spotify initial: track=%r volume=%d", track_label.text, state.last_volume)
            status.set_visibility(False)
            for btn in effect_buttons:
                btn.enable()

        ui.timer(0.1, startup, once=True)


ui.run(
    title="Light Board",
    port=8080,
    reload=False,
    native=True,
    window_size=(360, 720),
)
