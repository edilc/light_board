# Light Board

A NiceGUI dashboard that drives **Philips Hue Entertainment** lights with audio-synced effects, plus optional Spotify volume choreography. Runs as a native macOS window via `pywebview`.

## Run

```bash
uv run main.py        # launches the native window at port 8080
uv run pytest         # runs the effect trace tests (~1s)
```

App logs go to `logs/light_board.log` (auto-created, gitignored). Useful timing info per effect click — outputLatency, computed light delay, volume choreography starts.

## File map

- **`main.py`** — NiceGUI UI, page-load auto-connect, run_effect orchestration, audio playback, Spotify widget, the `HUE_LATENCY_S = 0.050` constant for AV-sync compensation.
- **`control.py`** — light-control plumbing.
  - `LightController` protocol (`set_color`, `set_colors`, `last_colors`).
  - `HueController` — real implementation; owns `hue_entertainment_pykit.Streaming` + channel ids.
  - `RecordingController` — test double; events are `(t, snapshot)` where snapshot is one color per channel.
  - `Clock` / `VirtualClock` — effects take an optional `clock`; `VirtualClock` advances on `await sleep` so tests run in milliseconds instead of real seconds.
  - `connect()` — discovery → bridge → entertainment area → streaming. Blocking; called via `asyncio.to_thread`.
- **`effects.py`** — audio analysis (cached on first use) plus the effect coroutines:
  - `lightning_onset_effect` / `lightning_hf_effect` / `lightning_amp_effect` — three variants differing only in strike timing source.
  - `gong_effect` — RMS-envelope-driven gold pulse, 1.5s settle to BRIGHT_WHITE.
  - `gavel_effect` — yellow then red flashes synced to onset times in `gavel.wav`, 1.5s settle.
  - `morning_effect` — 1s smoothstep from `ctl.last_colors` to BRIGHT_WHITE.
  - `night_effect` — 4-phase persistent night state (1s dim transition → 2s 60% with candle-flicker emerging → infinite candle sustain). Only ch2 (orange) flickers; ch0/ch1 are steady.
  - Module constant: `BRIGHT_WHITE = (244, 218, 182)` — the settled state every effect except Night ends on.
- **`spotify.py`** — macOS Spotify control via `osascript`. `SpotifyController` (best-effort: silently no-ops when Spotify is closed). Per-effect volume choreographies (`lightning_volume`, `gong_volume`, `gavel_volume`, `morning_volume`, `night_volume`) that run as parallel asyncio tasks alongside the light coroutine.
- **`tests/test_effects.py`** — trace tests using `VirtualClock` + `RecordingController`. `pytest -s` prints a sparse table per effect.
- **`tests/trace.py`** — formatting helpers for the printed traces (auto-detects uniform vs per-channel snapshots).
- **`sounds/*.wav`** — committed audio assets. Renaming or replacing them requires re-running the analysis snippets in this file's history (search for `librosa.onset.onset_strength` in old commits) and updating any hardcoded onset times in `effects.py` (only `gavel_effect` and the three lightning variants have hardcoded times).

## Concepts

- **Effects are async coroutines** with signature `async def X(ctl: LightController, *, clock: ClockLike | None = None)`. They drive lights via `ctl.set_color(r, g, b)` or `ctl.set_colors([(r,g,b), ...])`. Cancellation (Stop button or starting another effect) propagates naturally — effects do **not** zero the lights on cancel, so transitions between effects can read the prior state via `ctl.last_colors` and lerp from there.
- **Volume choreography is independent** of the light coroutine. Each effect button passes a `volume_choreography` to `run_effect`; both tasks start in parallel, both get cancelled by `stop_effect`. Volume changes are stepped (typically 5–10 osascript calls over the fade window) because each `osascript` invocation is ~50–100ms.
- **AV sync** — the dashboard reads `AudioContext.outputLatency` from the browser and delays the light task start by `max(0, audio_latency - HUE_LATENCY_S)` so audio reaching the user's ears is roughly synchronized with bulbs reaching the user's eyes. Built-in speakers ≈ 10ms (no light delay); Bluetooth/AirPods ≈ 200ms (~150ms light delay). On WKWebView (native mode), `outputLatency` may report 0 — fall back to a hardcoded value if needed.

## Setup notes

- **Python 3.12** required (`hue-entertainment-pykit` doesn't ship wheels for 3.13 because `python-mbedtls` doesn't).
- **Hue bridge credentials** live in `data/auth.json` (gitignored). On first connect, the library writes them after a button-press pairing. Keep this file local — it contains the bridge `username` and `clientkey`.
- **`hue-entertainment-pykit` is path-installed** from `../hue-entertainment-pykit` (see `pyproject.toml [tool.uv.sources]`). Editable install — local changes are picked up immediately.
- **Spotify control is optional and macOS-only**. Without Spotify running, the widget shows "Spotify: not detected" and volume choreographies silently no-op. AppleScript guards (`if application "Spotify" is running then ...`) prevent auto-launching Spotify.

## When making changes

- **Effects**: keep them pure (no globals, no UI references). Audio analysis lives in module-level cached functions (`get_thunder`, `get_gong`). Tests should pass after every effect edit — `uv run pytest -s` shows the trace.
- **Hue command rate**: the streaming library queues every `set_input` call without coalescing. 60 Hz updates from effects are fine. The library logs every `set_input` at INFO level — `main.py` silences this logger to keep `logs/light_board.log` readable.
- **Adding a new audio-synced effect**: drop the file in `sounds/`, run `librosa.onset.onset_strength` + `peak_pick` to find hit times, hardcode them in the effect (don't re-analyze at runtime — slow and risks subtle changes between runs). Pattern: see `gavel_effect` and the lightning variants.
- **Adding a button**: handler → `run_effect(effect, audio_el?, volume_choreography?)` → add to the row in the page body → add to `effect_buttons` tuple so it gets enable/disable handling at startup.
