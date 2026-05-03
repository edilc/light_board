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
import prefs
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
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    _app_fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    _fh = logging.FileHandler(LOG_FILE)
    _fh.setFormatter(_app_fmt)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_app_fmt)
    logger.addHandler(_fh)
    logger.addHandler(_sh)
    logger.propagate = False  # don't double-log via root

    # Catch WARNING+ from anywhere else (NiceGUI event errors, asyncio task
    # failures, third-party libraries) and append them to the same log file.
    # Includes the logger name so it's clear where each line came from.
    _root_fh = logging.FileHandler(LOG_FILE)
    _root_fh.setFormatter(_fmt)
    _root_fh.setLevel(logging.WARNING)
    _root = logging.getLogger()
    _root.addHandler(_root_fh)
    if _root.level == logging.WARNING - 1 or _root.level == 0:
        _root.setLevel(logging.WARNING)

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
    # Manual color override (config panel)
    manual_override_active: bool = False
    # Latency overrides (None = use auto-detected / constant); seconds
    audio_latency_override_s: Optional[float] = None
    hue_latency_override_s: Optional[float] = None
    # Spotify volume targets for morning/night (defaults match plan)
    morning_target_volume: int = 50
    night_target_volume: int = 100
    # 0.0-1.0; applied to all <audio> elements via JS
    internal_audio_volume: float = 1.0
    # Cursor into logs/light_board.log for the live tail viewer
    log_offset: int = 0
    # Toggle for the expanded UI (config panel + log viewer). Tailwind
    # responsive breakpoints don't fire reliably inside pywebview, so we
    # gate visibility on this boolean instead.
    expanded_mode: bool = False


state = AppState()
state.spotify = spotify_mod.SpotifyController()

# ── Load persisted preferences and apply to state + effects ──────────
_loaded_prefs = prefs.load()
state.morning_target_volume = int(_loaded_prefs["morning_target_volume"])
state.night_target_volume = int(_loaded_prefs["night_target_volume"])
state.internal_audio_volume = float(_loaded_prefs["internal_audio_volume"])
_bw = _loaded_prefs["bright_white"]
effects.set_bright_white(_bw[0], _bw[1], _bw[2])


def save_prefs() -> None:
    """Persist current settings. Called from change handlers."""
    prefs.save({
        "bright_white": list(effects.BRIGHT_WHITE),
        "morning_target_volume": state.morning_target_volume,
        "night_target_volume": state.night_target_volume,
        "internal_audio_volume": state.internal_audio_volume,
        "dark_mode": _loaded_prefs.get("dark_mode", False),
    })


def save_prefs_with_dark(dark: bool) -> None:
    """Update the dark-mode preference and save. Called from the toggle."""
    _loaded_prefs["dark_mode"] = bool(dark)
    save_prefs()


def _ensure_click_wav() -> None:
    """Generate sounds/click.wav once if it doesn't exist. Used by the
    latency-test button. ~50ms 800Hz sine burst with fast decay envelope."""
    path = SOUNDS_DIR / "click.wav"
    if path.exists():
        return
    import numpy as np
    import soundfile as sf
    sr = 22050
    duration = 0.05
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    env = np.exp(-t * 60)
    y = (np.sin(2 * np.pi * 800 * t) * env * 0.6).astype(np.float32)
    sf.write(str(path), y, sr)


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
    # ─── PAGE LAYOUT ────────────────────────────────────────────
    # Outer column: a top row of (controls | config) over a log strip.
    # The config panel + log are toggled via state.expanded_mode (button
    # in the header), since Tailwind responsive breakpoints don't fire
    # reliably inside pywebview's WKWebView.
    page_wrapper = ui.column().classes("w-full min-h-screen p-0 m-0 gap-0")
    with page_wrapper:
      top_row = ui.row().classes("w-full flex-1 items-stretch gap-0 m-0 p-0 flex-nowrap")
      with top_row:
       # `[&>*]:max-w-xs` caps every direct child at 320px so the controls
       # stay clustered in the center of the column even when the column
       # is widened to 50% of the screen in expanded mode.
       controls_col = ui.column().classes(
        "w-full items-center gap-3 p-3 [&>*]:max-w-xs"
       )
       with controls_col:
        # Dark mode element (no UI of its own — driven by the header button).
        dark_mode = ui.dark_mode(value=bool(_loaded_prefs.get("dark_mode", False)))

        with ui.row().classes("w-full items-center gap-1 -mb-1"):
            status = ui.label("Connecting…").classes("text-xs text-gray-500 flex-1")
            dark_btn = ui.button(icon="dark_mode").props("flat dense size=sm")
            dark_btn.tooltip("Toggle dark mode")
            toggle_btn = ui.button(icon="unfold_more").props("flat dense size=sm")
            toggle_btn.tooltip("Show config + logs")

        def toggle_dark() -> None:
            dark_mode.toggle()
            save_prefs_with_dark(bool(dark_mode.value))
            dark_btn.props(f'icon={"light_mode" if dark_mode.value else "dark_mode"}')

        dark_btn.on_click(toggle_dark)
        # Reflect persisted state in the icon at first paint.
        if dark_mode.value:
            dark_btn.props("icon=light_mode")

        thunder_audio = ui.audio("/sounds/thunder.wav").props("preload=auto").style("display: none")
        gong_audio = ui.audio("/sounds/gong.wav").props("preload=auto").style("display: none")
        gavel_audio = ui.audio("/sounds/gavel.wav").props("preload=auto").style("display: none")
        rooster_audio = ui.audio("/sounds/rooster.wav").props("preload=auto").style("display: none")
        click_audio = ui.audio("/sounds/click.wav").props("preload=auto").style("display: none")
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
            if state.manual_override_active:
                logger.info("[%s] aborting: manual override active", name)
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
                # Apply config-panel overrides if set, else fall back to detected/constant.
                audio_lat = (
                    state.audio_latency_override_s
                    if state.audio_latency_override_s is not None
                    else state.audio_latency_s
                )
                hue_lat = (
                    state.hue_latency_override_s
                    if state.hue_latency_override_s is not None
                    else HUE_LATENCY_S
                )
                delay = max(0.0, audio_lat - hue_lat)
                logger.info(
                    "[%s] computed light delay = max(0, %.3fs - %.3fs) = %.0fms",
                    name, audio_lat, hue_lat, delay * 1000,
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

        async def on_lightning() -> None:
            await run_effect(effects.lightning_effect, thunder_audio, spotify_mod.lightning_volume)

        async def on_gong() -> None:
            await run_effect(effects.gong_effect, gong_audio, spotify_mod.gong_volume)

        async def on_gavel() -> None:
            await run_effect(effects.gavel_effect, gavel_audio, spotify_mod.gavel_volume)

        async def on_rooster() -> None:
            # Plays the rooster crow but uses the simple morning lerp animation.
            await run_effect(
                effects.morning_effect,
                rooster_audio,
                lambda s, p: spotify_mod.morning_volume(s, p, target=state.morning_target_volume),
            )

        async def on_morning() -> None:
            await run_effect(
                effects.morning_effect,
                None,
                lambda s, p: spotify_mod.morning_volume(s, p, target=state.morning_target_volume),
            )

        async def on_night() -> None:
            await run_effect(
                effects.night_effect,
                None,
                lambda s, p: spotify_mod.night_volume(s, p, target=state.night_target_volume),
            )

        async def on_stop() -> None:
            await stop_effect()
            for a in all_audio:
                a.pause()
                a.seek(0)
            set_lights(0, 0, 0)

        # ╭─ buttons grid (2 columns) ────────────────────────────────╮
        with ui.grid(columns=2).classes("gap-2"):
            lightning_btn = (
                ui.button(icon="bolt", on_click=on_lightning)
                .props("color=indigo")
                .classes("w-16 h-16 text-3xl")
            )
            lightning_btn.tooltip("Lightning")
            gong_btn = (
                ui.button(icon="surround_sound", on_click=on_gong)
                .props("color=amber-9")
                .classes("w-16 h-16 text-3xl")
            )
            gong_btn.tooltip("Gong")
            gavel_btn = (
                ui.button(icon="gavel", on_click=on_gavel)
                .props("color=brown")
                .classes("w-16 h-16 text-3xl")
            )
            gavel_btn.tooltip("Gavel")
            rooster_btn = (
                ui.button(icon="wb_twilight", on_click=on_rooster)
                .props("color=red")
                .classes("w-16 h-16 text-3xl")
            )
            rooster_btn.tooltip("Rooster")
            morning_btn = (
                ui.button(icon="wb_sunny", on_click=on_morning)
                .props("color=yellow-8")
                .classes("w-16 h-16 text-3xl")
            )
            morning_btn.tooltip("Morning")
            night_btn = (
                ui.button(icon="bedtime", on_click=on_night)
                .props("color=blue-9")
                .classes("w-16 h-16 text-3xl")
            )
            night_btn.tooltip("Night")

        # Stop bar — small, full-width
        ui.button("Stop", icon="stop", on_click=on_stop).props(
            "flat color=grey"
        ).classes("w-full text-xs")

        effect_buttons = (lightning_btn, gong_btn, gavel_btn, rooster_btn, morning_btn, night_btn)
        for btn in effect_buttons:
            btn.disable()

        # ╭─ Spotify widget — track + artist on own lines, wide slider ╮
        with ui.column().classes("items-center gap-0 w-full mt-2"):
            track_label = ui.label("Spotify: not detected").classes(
                "text-sm font-medium text-center w-full truncate"
            )
            artist_label = ui.label("").classes(
                "text-xs text-gray-500 text-center w-full truncate"
            )
            with ui.row().classes("items-center gap-2 w-full mt-1"):
                ui.icon("music_note").classes("text-xs text-gray-400")
                volume_slider = (
                    ui.slider(min=0, max=100, value=50)
                    .props("label-always color=green dense")
                    .classes("flex-1")
                )

        async def on_volume_change(_e) -> None:
            v = int(volume_slider.value)
            state.last_volume = v
            await state.spotify.set_volume(v)

        volume_slider.on("change", on_volume_change)

        async def refresh_spotify() -> None:
            """Sync widget with actual Spotify state. Runs every 3s."""
            full = await state.spotify.get_track()
            if full and " — " in full:
                track, artist = full.split(" — ", 1)
                track_label.text = track
                artist_label.text = artist
            elif full:
                track_label.text = full
                artist_label.text = ""
            else:
                track_label.text = "Spotify: not detected"
                artist_label.text = ""
            vol = await state.spotify.get_volume()
            if vol is not None and vol != state.last_volume:
                state.last_volume = vol
                volume_slider.value = vol

        ui.timer(3.0, refresh_spotify)

       # ╭───────────────────────── CONFIG PANEL ──────────────────────────────╮
       # Visible only when state.expanded_mode is True (toggled via the
       # header button). w-1/2 makes it take exactly half the screen in
       # expanded mode; items-center + [&>*]:max-w-2xl center the content
       # within that half so it doesn't sprawl when the window is huge.
       config_col = ui.column().classes(
           "w-1/2 items-center gap-1 px-6 py-4 overflow-y-auto"
           " border-l border-gray-200 dark:border-gray-700"
           " [&>*]:max-w-2xl"
       )
       with config_col:
        _hdr = "font-semibold text-xs uppercase tracking-wide text-gray-700 dark:text-gray-300"
        _hdr_break = _hdr + " mt-5"  # subsequent section headers — extra top margin
        _sub = "text-xs text-gray-500 dark:text-gray-400"

        # ── MANUAL COLOR ─────────────────────────────────────────────
        ui.label("Manual color").classes(_hdr)
        override_switch = ui.switch("Override effects", value=False).props("dense")
        ui.label("Channel").classes(_sub)
        channel_toggle = (
            ui.toggle(["All", "Ch 0", "Ch 1", "Ch 2"], value="All")
            .props("dense no-caps")
            .classes("w-full")
        )
        ui.label("R / G / B / Brightness").classes(_sub)
        r_slider = ui.slider(min=0, max=255, value=0).props("label-always color=red")
        g_slider = ui.slider(min=0, max=255, value=0).props("label-always color=green")
        b_slider = ui.slider(min=0, max=255, value=0).props("label-always color=blue")
        bri_slider = ui.slider(min=0, max=100, value=100).props("label-always")

        def push_manual_color() -> None:
            if not state.manual_override_active or state.controller is None:
                return
            bri = bri_slider.value / 100
            r = max(0, min(255, int(r_slider.value * bri)))
            g = max(0, min(255, int(g_slider.value * bri)))
            b = max(0, min(255, int(b_slider.value * bri)))
            ch = channel_toggle.value
            if ch == "All":
                state.controller.set_color(r, g, b)
            else:
                try:
                    idx = int(ch.split()[-1])
                except (ValueError, IndexError):
                    state.controller.set_color(r, g, b)
                    return
                colors = list(state.controller.last_colors)
                while len(colors) <= idx:
                    colors.append((0, 0, 0))
                colors[idx] = (r, g, b)
                state.controller.set_colors(colors)

        for s in (r_slider, g_slider, b_slider, bri_slider):
            s.on("change", lambda _e: push_manual_color())

        async def on_override_toggle(_e) -> None:
            state.manual_override_active = bool(override_switch.value)
            if state.manual_override_active:
                await stop_effect()
                push_manual_color()

        override_switch.on("update:model-value", on_override_toggle)

        def sync_sliders_from_state() -> None:
            if state.manual_override_active or state.controller is None:
                return
            ch = channel_toggle.value
            if ch == "All":
                idx = 0
            else:
                try:
                    idx = int(ch.split()[-1])
                except (ValueError, IndexError):
                    idx = 0
            if idx < len(state.controller.last_colors):
                cr, cg, cb = state.controller.last_colors[idx]
                r_slider.value = cr
                g_slider.value = cg
                b_slider.value = cb

        ui.timer(0.5, sync_sliders_from_state)

        def save_as_day_target() -> None:
            """Snapshot the current manual-color slider state into BRIGHT_WHITE
            and persist it. The new value applies to morning/rooster/gong/gavel
            on their next run."""
            bri = bri_slider.value / 100
            r = max(0, min(255, int(r_slider.value * bri)))
            g = max(0, min(255, int(g_slider.value * bri)))
            b = max(0, min(255, int(b_slider.value * bri)))
            effects.set_bright_white(r, g, b)
            save_prefs()
            ui.notify(f"Day target saved: ({r}, {g}, {b})")

        ui.button(
            "Save as Day target", icon="save", on_click=save_as_day_target
        ).props("size=sm flat color=primary").classes("mt-1")

        # ── LATENCY ──────────────────────────────────────────────────
        ui.label("Latency").classes(_hdr_break)
        detected_label = ui.label(
            f"Detected audio: {state.audio_latency_s * 1000:.0f}ms"
        ).classes(_sub)
        audio_override_input = ui.number(
            "Audio override (ms)", value=None, format="%.0f", min=0, max=2000
        ).classes("w-full")
        ui.label(f"Hue assumed: {HUE_LATENCY_S * 1000:.0f}ms").classes(_sub + " mt-1")
        hue_override_input = ui.number(
            "Hue override (ms)", value=None, format="%.0f", min=0, max=500
        ).classes("w-full")
        with ui.row().classes("items-center gap-3 mt-2"):
            light_indicator = (
                ui.icon("lightbulb").classes("text-3xl").style("color: #ddd")
            )
            sound_indicator = (
                ui.icon("volume_up").classes("text-3xl").style("color: #ddd")
            )
            ui.button(
                "Test",
                icon="play_arrow",
                on_click=lambda: asyncio.create_task(on_latency_test()),
            ).props("size=sm color=primary")

        click_audio = (
            ui.audio("/sounds/click.wav").props("preload=auto").style("display: none")
        )

        def _audio_override_changed(_e) -> None:
            v = audio_override_input.value
            state.audio_latency_override_s = (v / 1000) if v not in (None, "") else None

        def _hue_override_changed(_e) -> None:
            v = hue_override_input.value
            state.hue_latency_override_s = (v / 1000) if v not in (None, "") else None

        audio_override_input.on("change", _audio_override_changed)
        hue_override_input.on("change", _hue_override_changed)

        async def _flash_indicator(indicator, delay_s: float, color: str) -> None:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            indicator.style(f"color: {color}")
            await asyncio.sleep(0.2)
            indicator.style("color: #ddd")

        async def on_latency_test() -> None:
            audio_lat = (
                state.audio_latency_override_s
                if state.audio_latency_override_s is not None
                else state.audio_latency_s
            )
            hue_lat = (
                state.hue_latency_override_s
                if state.hue_latency_override_s is not None
                else HUE_LATENCY_S
            )
            light_delay = max(0.0, audio_lat - hue_lat)
            logger.info(
                "[latency_test] audio_lat=%.3fs hue_lat=%.3fs light_delay=%.3fs",
                audio_lat, hue_lat, light_delay,
            )
            click_audio.play()

            async def _fire_light() -> None:
                if light_delay > 0:
                    await asyncio.sleep(light_delay)
                if state.controller is not None and not state.manual_override_active:
                    saved = state.controller.last_colors
                    state.controller.set_color(255, 255, 255)
                    await asyncio.sleep(0.2)
                    if saved:
                        state.controller.set_colors(list(saved))
                    else:
                        state.controller.set_color(0, 0, 0)

            asyncio.create_task(_fire_light())
            asyncio.create_task(
                _flash_indicator(light_indicator, light_delay + hue_lat, "gold")
            )
            asyncio.create_task(_flash_indicator(sound_indicator, audio_lat, "#0a8"))

        # ── SPOTIFY VOLUME ───────────────────────────────────────────
        ui.label("Spotify volume").classes(_hdr_break)
        with ui.row().classes("items-center gap-2 w-full"):
            day_input = (
                ui.number("Day target %", value=state.morning_target_volume,
                          min=0, max=100, step=1)
                .bind_value(state, "morning_target_volume")
                .classes("flex-1")
            )
            day_input.on("change", lambda _e: save_prefs())

            async def test_day() -> None:
                await spotify_mod.morning_volume(
                    state.spotify, state.last_volume, target=state.morning_target_volume
                )

            ui.button(icon="play_arrow", on_click=test_day).props("size=sm flat")
        with ui.row().classes("items-center gap-2 w-full"):
            night_input = (
                ui.number("Night target %", value=state.night_target_volume,
                          min=0, max=100, step=1)
                .bind_value(state, "night_target_volume")
                .classes("flex-1")
            )
            night_input.on("change", lambda _e: save_prefs())

            async def test_night() -> None:
                await spotify_mod.night_volume(
                    state.spotify, state.last_volume, target=state.night_target_volume
                )

            ui.button(icon="play_arrow", on_click=test_night).props("size=sm flat")

        # ── INTERNAL AUDIO ───────────────────────────────────────────
        ui.label("Internal audio").classes(_hdr_break)
        ui.label("Effects volume %").classes(_sub)
        audio_vol_slider = ui.slider(
            min=0, max=100, value=int(state.internal_audio_volume * 100)
        ).props("label-always")

        def _push_audio_vol(persist: bool = True) -> None:
            v = audio_vol_slider.value / 100
            state.internal_audio_volume = v
            for au in (*all_audio, click_audio):
                ui.run_javascript(
                    f'const el = document.getElementById("c{au.id}");'
                    f' if (el) el.volume = {v};'
                )
            if persist:
                save_prefs()

        audio_vol_slider.on("change", lambda _e: _push_audio_vol())

        with ui.row().classes("gap-1 flex-wrap mt-1"):
            ui.button("Thunder", on_click=thunder_audio.play).props("size=sm flat")
            ui.button("Gong", on_click=gong_audio.play).props("size=sm flat")
            ui.button("Gavel", on_click=gavel_audio.play).props("size=sm flat")
            ui.button("Rooster", on_click=rooster_audio.play).props("size=sm flat")

      # ╭────────────────────── LOG VIEWER ────────────────────────────╮
      # Visible only when state.expanded_mode is True.
      log_viewer = ui.log(max_lines=80).classes(
        "w-full h-40 font-mono text-xs p-2 border-t border-gray-200 dark:border-gray-700"
      )

    def tail_log() -> None:
        try:
            with open(LOG_FILE, "r") as f:
                f.seek(state.log_offset)
                new_data = f.read()
                state.log_offset = f.tell()
        except FileNotFoundError:
            return
        for line in new_data.splitlines():
            if line.strip():
                log_viewer.push(line)

    ui.timer(1.0, tail_log)

    # ─── Expanded-mode toggle ────────────────────────────────────
    # Hidden by default; click the header icon to flip.
    config_col.set_visibility(False)
    log_viewer.set_visibility(False)

    def toggle_expanded() -> None:
        state.expanded_mode = not state.expanded_mode
        config_col.set_visibility(state.expanded_mode)
        log_viewer.set_visibility(state.expanded_mode)
        if state.expanded_mode:
            # Each column takes half the row; their internal max-w + items-center
            # keep content clustered in the middle of each half.
            controls_col.classes(remove="w-full", add="w-1/2")
            toggle_btn.props("icon=unfold_less")
            toggle_btn.tooltip("Hide config + logs")
            try:
                app.native.main_window.resize(960, 720)
            except Exception as exc:
                logger.info("window resize unavailable: %s", exc)
        else:
            controls_col.classes(remove="w-1/2", add="w-full")
            toggle_btn.props("icon=unfold_more")
            toggle_btn.tooltip("Show config + logs")
            try:
                app.native.main_window.resize(360, 720)
            except Exception as exc:
                logger.info("window resize unavailable: %s", exc)

    toggle_btn.on_click(toggle_expanded)

    async def startup() -> None:
        await asyncio.to_thread(_ensure_click_wav)
        try:
            ctl = await asyncio.to_thread(control.connect)
        except Exception as exc:
            status.text = f"Connect failed: {exc}"
            return
        state.controller = ctl
        # Update the channel toggle to match the actual channel count.
        channel_toggle.options = ["All"] + [f"Ch {i}" for i in range(len(ctl.channel_ids))]
        channel_toggle.update()
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
        detected_label.text = f"Detected audio: {state.audio_latency_s * 1000:.0f}ms"
        logger.info(
            "startup outputLatency=%.3fs → light delay = %.0fms (HUE_LATENCY_S=%.0fms)",
            state.audio_latency_s,
            max(0.0, state.audio_latency_s - HUE_LATENCY_S) * 1000,
            HUE_LATENCY_S * 1000,
        )
        await refresh_spotify()
        logger.info(
            "Spotify initial: track=%r artist=%r volume=%d",
            track_label.text, artist_label.text, state.last_volume,
        )
        # Apply persisted internal-audio volume to the <audio> elements now
        # that the page is fully connected (run_javascript needs a client).
        _push_audio_vol(persist=False)
        # Skip past existing log content so the viewer only shows new lines.
        try:
            state.log_offset = LOG_FILE.stat().st_size
        except FileNotFoundError:
            state.log_offset = 0
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
