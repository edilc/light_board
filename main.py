import asyncio
import colorsys
import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean as _mean
from statistics import stdev

from hue_entertainment_pykit import setup_logs
from nicegui import app, background_tasks, ui

import control
import effects
import spotify as spotify_mod
from config import Config

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

# Swallow the default action of space when nothing editable is focused.
# Otherwise WKWebView (and macOS in general) emits the system "donk" on
# every space tap during calibration because no element accepts the
# keystroke. Capture phase so we beat any other listeners. The Python
# `ui.keyboard` handler still runs — it goes through `on('key')` which
# isn't the keystroke's default action, so preventing default doesn't
# block our tap-recording.
ui.add_head_html("""
<script>
document.addEventListener('keydown', (e) => {
    if (e.key !== ' ') return;
    const el = document.activeElement;
    const tag = (el && el.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if (el && el.isContentEditable) return;
    e.preventDefault();
}, true);
</script>
""", shared=True)

# Approximate latency from `streaming.set_input(...)` to the bulb actually
# changing color over WiFi+DTLS. Detecting it precisely would require
# timing something visible at the bulb itself, so we assume a fixed value.
# Combined with the browser's reported audio output latency, we compute the
# light-side delay as `max(0, audio_latency - HUE_LATENCY_S)` so audio and
# lights land at the user roughly together.
HUE_LATENCY_S = 0.050

# Persistent user prefs. One source of truth — `config.update(field=value)`
# mutates the field and writes the JSON atomically. Effects read live
# fields (e.g. `config.bright_white`) every frame.
PREFS_FILE = Path(__file__).parent / "data" / "preferences.json"
config = Config.load(PREFS_FILE)


@dataclass
class AudioClip:
    """A NiceGUI <audio> element + the volume offset to apply to it.

    `volume_offset` is in percentage points (signed). It's added to the
    "Effects volume %" slider value and the result is clamped to
    [0, 100] before being pushed to the underlying <audio> element.
    Lets quieter clips (gavel.wav) match the perceived loudness of the
    others without retouching the audio file.

    Construction creates the <audio> as a child of whichever NiceGUI
    container is currently open — call from inside a `with column:`
    block, same constraint as a bare `ui.audio(...)`.
    """
    path: str
    volume_offset: int = 0
    element: ui.audio = field(init=False)

    def __post_init__(self) -> None:
        self.element = ui.audio(self.path).props("preload=auto").style("display: none")


@dataclass
class AppState:
    """Ephemeral runtime state. Persistent preferences live on `config`."""
    controller: control.HueController | None = None
    effect_task: asyncio.Task | None = None
    audio_latency_s: float = 0.0  # browser-reported output latency (auto-detected)
    spotify: spotify_mod.SpotifyController = field(
        default_factory=spotify_mod.SpotifyController
    )
    last_volume: int = 50
    volume_task: asyncio.Task | None = None
    # Manual color override (config panel)
    manual_override_active: bool = False
    # Cursor into logs/light_board.log for the live tail viewer
    log_offset: int = 0
    # Toggle for the expanded UI (config panel + log viewer). Tailwind
    # responsive breakpoints don't fire reliably inside pywebview, so we
    # gate visibility on this boolean instead.
    expanded_mode: bool = False
    # Spacebar-tap calibration runtime state. `mode` is "audio" or "light"
    # while a calibration session is active (None otherwise). `taps` is a
    # rolling window of the last 5 tap-time-minus-beat-time offsets.
    calibration_mode: str | None = None
    calibration_taps: list = field(default_factory=list)
    calibration_beat_t: float = 0.0
    calibration_beat_count: int = 0
    calibration_pending: tuple | None = None  # (mode, mean_seconds) when locked in


state = AppState()


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
        dark_mode = ui.dark_mode(value=config.dark_mode)

        with ui.row().classes("w-full items-center gap-1 -mb-1"):
            status = ui.label("Connecting…").classes("text-xs text-gray-500 flex-1")
            dark_btn = ui.button(icon="dark_mode").props("flat dense size=sm")
            dark_btn.tooltip("Toggle dark mode")
            toggle_btn = ui.button(icon="unfold_more").props("flat dense size=sm")
            toggle_btn.tooltip("Show config + logs")

        def toggle_dark() -> None:
            dark_mode.toggle()
            config.update(dark_mode=bool(dark_mode.value))
            dark_btn.props(f'icon={"light_mode" if dark_mode.value else "dark_mode"}')

        dark_btn.on_click(toggle_dark)
        # Reflect persisted state in the icon at first paint.
        if dark_mode.value:
            dark_btn.props("icon=light_mode")

        thunder_audio = AudioClip("/sounds/thunder.wav")
        gong_audio = AudioClip("/sounds/gong.wav")
        # Gavel.wav was recorded markedly quieter than the other clips; +30 brings
        # the bang in line without retouching the audio file (which already peaks
        # at full scale, so amplitude-boosting it would clip the transient).
        gavel_audio = AudioClip("/sounds/gavel.wav", volume_offset=30)
        rooster_audio = AudioClip("/sounds/rooster.wav")
        good_victory_audio = AudioClip("/sounds/good_victory.wav")
        evil_victory_audio = AudioClip("/sounds/evil_victory.wav")
        click_audio = AudioClip("/sounds/click.wav")
        all_audio = (
            thunder_audio, gong_audio, gavel_audio, rooster_audio,
            good_victory_audio, evil_victory_audio,
        )

        async def run_effect(coro_factory, audio_clip: AudioClip | None = None, volume_choreography=None) -> None:
            t_click = time.monotonic()
            name = coro_factory.__name__
            logger.info("[%s] run_effect start", name)
            await stop_effect()
            for a in all_audio:
                a.element.pause()
                a.element.seek(0)
            if state.controller is None:
                logger.info("[%s] aborting: no controller", name)
                return
            if state.manual_override_active:
                logger.info("[%s] aborting: manual override active", name)
                return
            if audio_clip is not None:
                # Probe outputLatency only when there's no calibrated override.
                if config.audio_latency_override_s is None:
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
                audio_lat = (
                    config.audio_latency_override_s
                    if config.audio_latency_override_s is not None
                    else state.audio_latency_s
                )
                hue_lat = (
                    config.hue_latency_override_s
                    if config.hue_latency_override_s is not None
                    else HUE_LATENCY_S
                )
                # Delay whichever side is faster so both arrive together.
                audio_lead = max(0.0, hue_lat - audio_lat)
                light_lead = max(0.0, audio_lat - hue_lat)
                logger.info(
                    "[%s] audio_lead=%.0fms light_lead=%.0fms (audio=%.3fs hue=%.3fs)",
                    name, audio_lead * 1000, light_lead * 1000, audio_lat, hue_lat,
                )
                if audio_lead > 0:
                    await asyncio.sleep(audio_lead)
                audio_clip.element.play()
                logger.info(
                    "[%s] audio.play() dispatched at +%.0fms",
                    name, (time.monotonic() - t_click) * 1000,
                )
                if light_lead > 0:
                    await asyncio.sleep(light_lead)
            # Effects that drive volume internally (currently just `night_effect`)
            # accept `spotify` and `prev_volume` kwargs. Detected by signature.
            effect_kwargs: dict = {}
            if "spotify" in inspect.signature(coro_factory).parameters:
                effect_kwargs["spotify"] = state.spotify
                effect_kwargs["prev_volume"] = state.last_volume
                logger.info(
                    "[%s] driving volume internally (prev_vol=%d → target)",
                    name, state.last_volume,
                )
            state.effect_task = asyncio.create_task(
                coro_factory(state.controller, config, **effect_kwargs)
            )
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
                lambda s, p: spotify_mod.morning_volume(s, p, target=config.morning_target_volume),
            )

        async def on_morning() -> None:
            await run_effect(
                effects.morning_effect,
                None,
                lambda s, p: spotify_mod.morning_volume(s, p, target=config.morning_target_volume),
            )

        async def on_night() -> None:
            # Night drives its own volume internally so the fade is synced
            # with the lights' settle into the candle state. No separate
            # volume_choreography needed.
            await run_effect(effects.night_effect, None, None)

        async def on_good_victory() -> None:
            await run_effect(effects.good_victory_effect, good_victory_audio, spotify_mod.good_victory_volume)

        async def on_evil_victory() -> None:
            await run_effect(effects.evil_victory_effect, evil_victory_audio, spotify_mod.evil_victory_volume)

        async def on_stop() -> None:
            await stop_effect()
            for a in all_audio:
                a.element.pause()
                a.element.seek(0)
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
            good_victory_btn = (
                ui.button(icon="celebration", on_click=on_good_victory)
                .props("color=blue")
                .classes("w-16 h-16 text-3xl")
            )
            good_victory_btn.tooltip("Good Victory")
            evil_victory_btn = (
                ui.button(icon="whatshot", on_click=on_evil_victory)
                .props("color=red")
                .classes("w-16 h-16 text-3xl")
            )
            evil_victory_btn.tooltip("Evil Victory")

        # Stop bar — small, full-width
        ui.button("Stop", icon="stop", on_click=on_stop).props(
            "flat color=grey"
        ).classes("w-full text-xs")

        effect_buttons = (
            lightning_btn, gong_btn, gavel_btn, rooster_btn, morning_btn, night_btn,
            good_victory_btn, evil_victory_btn,
        )
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

        def manual_rgb() -> tuple[int, int, int]:
            """Combine the R/G/B sliders (hue + saturation) with the
            brightness slider (the V in HSV) into the color the lights
            should show. Brightness scales the chosen color's level
            independently, so it can both brighten and dim it."""
            h, s, _ = colorsys.rgb_to_hsv(
                r_slider.value / 255, g_slider.value / 255, b_slider.value / 255
            )
            fr, fg, fb = colorsys.hsv_to_rgb(h, s, bri_slider.value / 100)
            return round(fr * 255), round(fg * 255), round(fb * 255)

        def push_manual_color() -> None:
            if not state.manual_override_active or state.controller is None:
                return
            r, g, b = manual_rgb()
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
                # Split the live color back into the same representation
                # push uses: hue+saturation at full value in the R/G/B
                # sliders, level in the brightness slider.
                h, s, v = colorsys.rgb_to_hsv(cr / 255, cg / 255, cb / 255)
                fr, fg, fb = colorsys.hsv_to_rgb(h, s, 1.0)
                r_slider.value = round(fr * 255)
                g_slider.value = round(fg * 255)
                b_slider.value = round(fb * 255)
                bri_slider.value = round(v * 100)

        ui.timer(0.5, sync_sliders_from_state)

        def save_as_day_target() -> None:
            """Snapshot the current manual-color slider state into
            `config.bright_white` and persist it. The new value applies to
            morning/rooster/gong/gavel on their next run."""
            r, g, b = manual_rgb()
            config.update(bright_white=(r, g, b))
            ui.notify(f"Day target saved: ({r}, {g}, {b})")

        ui.button(
            "Save as Day target", icon="save", on_click=save_as_day_target
        ).props("size=sm flat color=primary").classes("mt-1")

        # ── LATENCY ──────────────────────────────────────────────────
        ui.label("Latency").classes(_hdr_break)
        detected_label = ui.label(
            f"Detected audio: {state.audio_latency_s * 1000:.0f}ms"
        ).classes(_sub)
        with ui.row().classes("items-center gap-2 w-full"):
            audio_override_input = ui.number(
                "Audio override (ms)",
                value=config.audio_latency_override_ms,
                format="%.0f", min=0, max=2000,
            ).classes("flex-1")
            ui.button(
                "Calibrate", icon="straighten",
                on_click=lambda: start_calibration("audio"),
            ).props("size=sm color=primary")
        ui.label(f"Hue assumed: {HUE_LATENCY_S * 1000:.0f}ms").classes(_sub + " mt-1")
        with ui.row().classes("items-center gap-2 w-full"):
            hue_override_input = ui.number(
                "Hue override (ms)",
                value=config.hue_latency_override_ms,
                format="%.0f", min=0, max=500,
            ).classes("flex-1")
            ui.button(
                "Calibrate", icon="straighten",
                on_click=lambda: start_calibration("light"),
            ).props("size=sm color=primary")

        click_audio = AudioClip("/sounds/click.wav")

        def _audio_override_changed(_e) -> None:
            v = audio_override_input.value
            ms = float(v) if v not in (None, "") else None
            config.update(audio_latency_override_ms=ms)

        def _hue_override_changed(_e) -> None:
            v = hue_override_input.value
            ms = float(v) if v not in (None, "") else None
            config.update(hue_latency_override_ms=ms)

        audio_override_input.on("change", _audio_override_changed)
        hue_override_input.on("change", _hue_override_changed)

        # ── Calibration dialog ──────────────────────────────────────
        with ui.dialog() as calibrate_dialog, ui.card().classes("gap-2 items-center w-96"):
            cal_title = ui.label("Calibrate").classes("text-base font-semibold")
            # No transition-colors — we want the indicator to flash crisply
            # in lockstep with the audio click / light pulse.
            cal_indicator = ui.icon("circle").classes("text-6xl").style(
                "color: #d1d5db"
            )
            cal_tap_btn = (
                ui.button("Tap", icon="touch_app")
                .props("color=primary size=xl unelevated")
                .classes("w-40")
            )
            cal_status = ui.label(
                "Click Tap (or press Enter) on each beat. Locks in after 5 consistent taps."
            ).classes("text-xs text-gray-500 text-center")
            ui.label("100 BPM • 600ms / beat").classes("text-xs text-gray-400")

            # Live feed: each beat appends a "Press!" line; on tap we annotate
            # the most recent un-tapped line with the offset. Newest at the
            # bottom, scrolls into view as more beats arrive.
            cal_feed = ui.column().classes(
                "w-72 h-28 overflow-y-auto items-start gap-0 px-2 py-1"
                " border border-gray-200 dark:border-gray-700 rounded text-sm font-mono"
            )

            cal_taps_label = ui.label("Taps: 0/5").classes("text-xs text-gray-500")
            cal_stddev_label = ui.label("stddev: —").classes("text-xs text-gray-500")
            cal_result = ui.label("").classes("text-base font-medium mt-1")
            with ui.row().classes("gap-2 mt-1"):
                cal_apply_btn = ui.button("Apply", icon="check").props("color=primary size=sm")
                cal_cancel_btn = ui.button("Cancel", icon="close").props("flat size=sm")

        cal_apply_btn.set_visibility(False)

        # Recent feed entries: each is [label_element, has_press_recorded].
        cal_recent_entries: list[list] = []
        CAL_FEED_MAX = 8

        def add_beat_entry() -> None:
            with cal_feed:
                lbl = ui.label("Press!").classes(
                    "text-gray-600 dark:text-gray-300 leading-tight"
                )
            cal_recent_entries.append([lbl, False])
            while len(cal_recent_entries) > CAL_FEED_MAX:
                old_lbl, _ = cal_recent_entries.pop(0)
                old_lbl.delete()
            ui.run_javascript(
                f'const el = document.getElementById("c{cal_feed.id}");'
                f' if (el) el.scrollTop = el.scrollHeight;'
            )

        def record_press_in_feed(offset_s: float) -> None:
            for entry in reversed(cal_recent_entries):
                if not entry[1]:
                    offset_ms = offset_s * 1000
                    entry[0].text = f"Press!  ✓  +{offset_ms:.0f}ms"
                    entry[0].classes(
                        remove="text-gray-600 dark:text-gray-300",
                        add="text-emerald-600 dark:text-emerald-400",
                    )
                    entry[1] = True
                    return

        def reset_feed() -> None:
            for lbl, _ in cal_recent_entries:
                lbl.delete()
            cal_recent_entries.clear()

        BEAT_INTERVAL = 60.0 / 100.0  # 100 BPM
        STDDEV_THRESHOLD = 0.030       # 30 ms
        WINDOW = 5
        # 250ms flash hold — long enough that the Hue bulb's ~50ms WiFi
        # transit doesn't eat the entire visible window.
        FLASH_HOLD = 0.25

        FLASH_COLORS = {"audio": "#facc15", "light": "#38bdf8"}  # yellow / blue

        # The metronome runs as a `ui.timer` rather than an asyncio task.
        # The previous design used `asyncio.create_task(loop)` and died after
        # one iteration — exceptions in the loop were swallowed by the task,
        # and the dialog's `hide` handler could flip `calibration_mode` to
        # None before the second beat. ui.timer runs in the page's UI
        # context, surfaces errors, and is start/stoppable as a unit.
        def beat_tick() -> None:
            mode = state.calibration_mode
            if mode is None:
                return
            state.calibration_beat_count += 1
            state.calibration_beat_t = time.monotonic()
            if mode == "audio":
                # HTML5 <audio> won't replay a finished clip without seeking
                # to the start first.
                click_audio.element.seek(0)
                click_audio.element.play()
            elif mode == "light":
                if state.controller is not None:
                    state.controller.set_color(255, 255, 255)
                    logger.info(
                        "[calibrate] beat %d: light on", state.calibration_beat_count
                    )
                else:
                    logger.warning(
                        "[calibrate] beat %d: no controller — flash skipped",
                        state.calibration_beat_count,
                    )
            cal_indicator.style(f"color: {FLASH_COLORS[mode]}")
            add_beat_entry()
            ui.timer(FLASH_HOLD, beat_reset, once=True)

        def beat_reset() -> None:
            if state.calibration_mode == "light" and state.controller is not None:
                state.controller.set_color(0, 0, 0)
            cal_indicator.style("color: #d1d5db")

        beat_timer = ui.timer(BEAT_INTERVAL, beat_tick, active=False)

        def update_calibration_status() -> None:
            n = len(state.calibration_taps)
            cal_taps_label.text = f"Taps: {min(n, WINDOW)}/{WINDOW}"
            if n >= 2:
                s = stdev(state.calibration_taps)
                cal_stddev_label.text = f"stddev: {s * 1000:.0f}ms"
            else:
                cal_stddev_label.text = "stddev: —"
            if n >= WINDOW:
                s = stdev(state.calibration_taps[-WINDOW:])
                if s < STDDEV_THRESHOLD:
                    # Floor at 0: a negative signed offset means the user is
                    # anticipating the beat faster than the actual latency,
                    # which is real but doesn't translate to a meaningful
                    # "latency" value. Treat it as "≈0ms latency".
                    m = max(0.0, _mean(state.calibration_taps[-WINDOW:]))
                    state.calibration_pending = (state.calibration_mode, m)
                    cal_result.text = f"Calibrated: {m * 1000:.0f}ms"
                    cal_apply_btn.set_visibility(True)
                    cal_status.text = "Press Apply to save, or Cancel to retry."
                    # Stop the metronome now that we've locked in.
                    beat_timer.deactivate()
                    state.calibration_mode = None

        def stop_calibration() -> None:
            state.calibration_mode = None
            beat_timer.deactivate()
            if state.controller is not None:
                # Make sure the bulb isn't left on if we cancel mid-flash.
                state.controller.set_color(0, 0, 0)
            cal_indicator.style("color: #d1d5db")
            calibrate_dialog.close()

        def start_calibration(mode: str) -> None:
            background_tasks.create(stop_effect(), name="calibration-stop-effect")
            state.calibration_mode = mode
            state.calibration_taps = []
            state.calibration_pending = None
            state.calibration_beat_count = 0
            state.calibration_beat_t = 0.0
            reset_feed()
            cal_title.text = f"Calibrate {mode}"
            cal_status.text = "Tap SPACE on each beat. Locks in after 5 consistent taps."
            cal_taps_label.text = "Taps: 0/5"
            cal_stddev_label.text = "stddev: —"
            cal_result.text = ""
            cal_apply_btn.set_visibility(False)
            calibrate_dialog.open()
            # Fire the first beat immediately so the user doesn't wait 600ms
            # staring at a silent dialog.
            beat_tick()
            beat_timer.activate()

        def on_cal_cancel() -> None:
            state.calibration_pending = None
            stop_calibration()

        def on_cal_apply() -> None:
            if state.calibration_pending is None:
                stop_calibration()
                return
            mode, mean_s = state.calibration_pending
            mean_ms = mean_s * 1000
            if mode == "audio":
                config.update(audio_latency_override_ms=mean_ms)
                audio_override_input.value = round(mean_ms)
            else:
                config.update(hue_latency_override_ms=mean_ms)
                hue_override_input.value = round(mean_ms)
            ui.notify(f"{mode.title()} latency calibrated: {mean_ms:.0f}ms")
            logger.info(
                "[calibrate] %s mean=%.3fs (%d taps)", mode, mean_s,
                len(state.calibration_taps),
            )
            state.calibration_pending = None
            stop_calibration()

        cal_cancel_btn.on_click(on_cal_cancel)
        cal_apply_btn.on_click(on_cal_apply)
        # Only treat a `hide` event as a cancel if we're actually mid-session.
        # Without this guard a spurious early hide (e.g. Quasar settling
        # initial state) would flip calibration_mode to None and kill the
        # metronome after one beat.
        calibrate_dialog.on(
            "hide",
            lambda _e: on_cal_cancel() if state.calibration_mode is not None else None,
        )

        def register_tap() -> None:
            """Record a tap. Called by both the Tap button and the keyboard
            handler — single source of truth for the tap-recording logic."""
            if state.calibration_mode is None or state.calibration_beat_t == 0.0:
                return
            raw = time.monotonic() - state.calibration_beat_t
            # Map to nearest beat: a tap that lands in the second half of
            # the interval is treated as anticipating the *next* beat (signed
            # negative). This lets users who tap slightly early — or who
            # have a fast auditory reaction — still get counted instead of
            # silently dropped as in the old `offset <= 0` reject.
            offset = raw if raw < BEAT_INTERVAL / 2 else raw - BEAT_INTERVAL
            if abs(offset) > BEAT_INTERVAL / 2:
                return
            state.calibration_taps.append(offset)
            if len(state.calibration_taps) > WINDOW:
                state.calibration_taps = state.calibration_taps[-WINDOW:]
            record_press_in_feed(offset)
            update_calibration_status()

        def on_tap_click() -> None:
            register_tap()
            # Blur so a follow-up Enter keypress doesn't synthetic-click the
            # button AND hit the keyboard handler (would double-tap).
            ui.run_javascript(
                "if (document.activeElement) document.activeElement.blur();"
            )

        cal_tap_btn.on_click(on_tap_click)

        def on_keydown(e) -> None:
            if state.calibration_mode is None:
                return
            if not e.action.keydown:
                return
            # Enter is the primary keyboard alternative; space kept as
            # bonus (may be intercepted by macOS in some webview contexts).
            if e.key.name in (" ", "Enter"):
                register_tap()

        ui.keyboard(on_key=on_keydown)

        # ── SPOTIFY VOLUME ───────────────────────────────────────────
        # No preview buttons — clicking the actual effect (Morning, Night)
        # demonstrates the real volume choreography.
        ui.label("Spotify volume").classes(_hdr_break)
        day_input = (
            ui.number("Day target %", value=config.morning_target_volume,
                      min=0, max=100, step=1)
            .bind_value(config, "morning_target_volume")
            .classes("w-full")
        )
        day_input.on("change", lambda _e: config.save())
        night_input = (
            ui.number("Night target %", value=config.night_target_volume,
                      min=0, max=100, step=1)
            .bind_value(config, "night_target_volume")
            .classes("w-full")
        )
        night_input.on("change", lambda _e: config.save())

        # ── INTERNAL AUDIO ───────────────────────────────────────────
        ui.label("Internal audio").classes(_hdr_break)
        ui.label("Effects volume %").classes(_sub)
        audio_vol_slider = ui.slider(
            min=0, max=100, value=int(config.internal_audio_volume * 100)
        ).props("label-always")

        def _push_audio_vol(persist: bool = True) -> None:
            v_pct = int(audio_vol_slider.value)  # 0..100
            for clip in (*all_audio, click_audio):
                clip_vol = max(0, min(100, v_pct + clip.volume_offset)) / 100
                ui.run_javascript(
                    f'const el = document.getElementById("c{clip.element.id}");'
                    f' if (el) el.volume = {clip_vol};'
                )
            v = v_pct / 100
            if persist:
                config.update(internal_audio_volume=v)
            else:
                config.internal_audio_volume = v

        audio_vol_slider.on("change", lambda _e: _push_audio_vol())

        with ui.row().classes("gap-1 flex-wrap mt-1"):
            ui.button("Thunder", on_click=thunder_audio.element.play).props("size=sm flat")
            ui.button("Gong", on_click=gong_audio.element.play).props("size=sm flat")
            ui.button("Gavel", on_click=gavel_audio.element.play).props("size=sm flat")
            ui.button("Rooster", on_click=rooster_audio.element.play).props("size=sm flat")

      # ╭────────────────────── LOG VIEWER ────────────────────────────╮
      # Visible only when state.expanded_mode is True.
      log_viewer = ui.log(max_lines=80).classes(
        "w-full h-40 font-mono text-xs p-2 border-t border-gray-200 dark:border-gray-700"
      )

    def tail_log() -> None:
        try:
            with open(LOG_FILE) as f:
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
        # Native window only exists when running under pywebview; in
        # browser-only mode app.native.main_window is None.
        window = app.native.main_window
        if state.expanded_mode:
            # Each column takes half the row; their internal max-w + items-center
            # keep content clustered in the middle of each half.
            controls_col.classes(remove="w-full", add="w-1/2")
            toggle_btn.props("icon=unfold_less")
            toggle_btn.tooltip("Hide config + logs")
            if window is not None:
                try:
                    window.resize(960, 720)  # ty: ignore[unresolved-attribute]
                except Exception as exc:
                    logger.info("window resize unavailable: %s", exc)
        else:
            controls_col.classes(remove="w-1/2", add="w-full")
            toggle_btn.props("icon=unfold_more")
            toggle_btn.tooltip("Show config + logs")
            if window is not None:
                try:
                    window.resize(360, 720)  # ty: ignore[unresolved-attribute]
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
