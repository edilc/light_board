# Light Board

A NiceGUI dashboard that drives **Philips Hue Entertainment** lights with
audio-synced effects, plus optional Spotify volume choreography. Runs as
a native macOS window via `pywebview`.

## Run / test / lint

```bash
uv run main.py        # native window at port 8080
uv run pytest -q      # ~1s; 68 tests across the primitive + effect modules
uv run ruff check .   # lint
uv run ty check .     # type-check
```

App logs go to `logs/light_board.log` (auto-created, gitignored). Each
effect click logs `outputLatency`, computed light delay, volume choreography
starts.

## File map

The repo splits responsibilities into small modules so each one fits in
your head. Effects are pure choreography on top of these primitives.

- **`config.py`** — `Config` dataclass; one source of truth for every
  persistent preference (bright_white, target volumes, internal audio
  volume, dark mode, audio/Hue latency overrides). `Config.load(path)` →
  `config.update(field=value)` mutates and atomically writes JSON. Effects
  read `config.bright_white` per-frame, so a mid-effect "Save as Day target"
  takes hold immediately. **No prefs.py — Config absorbs it.**
- **`colors.py`** — RGB type alias + `lerp`, `scale`, `clamp`. Pure functions.
- **`signals.py`** — Easing functions (`smoothstep`, `cosine`, `quadratic_in`),
  the `Envelope` class wrapping per-frame RMS arrays, and audio shapers
  (`quiver`, `shaped`, `candle`).
- **`analysis.py`** — Owns the librosa dependency. `ThunderAnalysis`,
  `GongAnalysis`, cached accessors `get_thunder()` / `get_gong()`, and
  `warm_cache()` that runs both off-thread at startup.
- **`runner.py`** — Effect-runtime primitives: `frame_loop(ctl, clock,
  duration, frame_fn)`, `settle_to(ctl, clock, target, duration, ease)`,
  `hold(ctl, clock, color, duration)`, plus the `FRAME = 1/60` constant.
  Replaces every effect's hand-rolled `start = clock.now(); while True:
  ...` loop.
- **`control.py`** — Light-control plumbing.
  - `LightController` protocol (`set_color`, `set_colors`, `last_colors`)
  - `VolumeController` protocol (`async set_volume`)
  - `HueController` — real implementation; owns `Streaming` + channel ids
  - `RecordingController` / `RecordingVolumeController` — test doubles
    that capture every call as `(t, ...)` events
  - `Clock` / `VirtualClock` — effects take an optional `clock`;
    `VirtualClock` advances on `await sleep` so tests run in milliseconds
  - `connect()` — discovery → bridge → entertainment area → streaming.
    Blocking; called via `asyncio.to_thread`.
- **`effects.py`** — Hue Entertainment effect coroutines:
  - `lightning_effect` — strikes flash white over per-channel quivering
    blue/purple base, then settles to BRIGHT_WHITE
  - `gong_effect` — RMS-envelope-shaped gold pulse, 1.5s settle
  - `gavel_effect` — yellow then red flashes synced to onset times in
    `gavel.wav`, 1.5s settle
  - `morning_effect` is legacy/tested; `night_effect` is used by the UI for
    Hektor's flickery night state.
- **`spotify.py`** — macOS Spotify control via `osascript`.
  `SpotifyController` (best-effort: silently no-ops when Spotify is
  closed). Per-effect volume choreographies for lightning/gong/gavel/
  morning/night that run as parallel asyncio tasks from `main.py`.
- **`main.py`** — NiceGUI UI, page-load auto-connect, `run_effect`
  orchestration, audio playback, Spotify widget, the `HUE_LATENCY_S = 0.050`
  constant for AV-sync compensation, calibration overlay. Rooster/Morning/
  Night use Home Assistant only for `light.dining_room_brighter`; Hektor is
  still controlled through Hue Entertainment. Daytime plays `sounds/rooster.wav`
  and fades Brighter up over 5s with an exponential curve; Night turns Brighter
  off immediately while starting Hektor's flickery night state.
- **`tests/`** — `test_colors`, `test_signals`, `test_runner`,
  `test_config` for primitive unit tests; `test_effects` for trace-based
  effect tests using `VirtualClock` + `RecordingController`.
- **`sounds/*.wav`** — committed audio assets.

## Concepts

- **Effects are async coroutines** with signature
  `async def X(ctl, config, *, clock=None)` (night also takes
  `spotify=None, prev_volume=None`). They drive lights via `ctl.set_color`
  / `ctl.set_colors` and read `config.bright_white` etc. live every frame.
  Cancellation propagates naturally; effects do **not** zero the lights on
  cancel, so transitions between effects can lerp from `ctl.last_colors`.
- **Volume choreography**: most effects (lightning/gong/gavel/morning) run
  a parallel volume task wired by `main.run_effect`. **Night is the
  exception** — it drives its own internal volume fade so the audio is
  synced with the lights' settle into the candle state. New effects that
  want internal volume control just declare a `spotify` kwarg in their
  signature and `run_effect` will inject it (introspected via `inspect`).
- **AV sync** — the dashboard reads `AudioContext.outputLatency` from the
  browser and delays the light task start by
  `max(0, audio_latency - HUE_LATENCY_S)` so audio reaching the user's ears
  is roughly synchronized with bulbs reaching the user's eyes. Built-in
  speakers ≈ 10ms (no light delay); Bluetooth/AirPods ≈ 200ms (~150ms
  light delay). On WKWebView (native mode), `outputLatency` may report 0;
  use the calibration overlay to set per-side overrides.
- **Calibration overlay** — Tap-to-the-beat metronome (100 BPM via
  `ui.timer`, not a fragile asyncio task). Audio mode plays a click each
  beat; light mode flashes the bulbs. After 5 taps with stddev < 30ms the
  mean offset is locked in as `audio_latency_override_ms` or
  `hue_latency_override_ms` on the Config. Signed offsets allow
  anticipatory taps; the saved value is floored at 0.
- **Live config**: `config.update(bright_white=(r,g,b))` (or any other
  field) mutates the dataclass and writes the JSON in one call. Effects
  reading `config.bright_white` on the next frame pick up the new value.

## Setup notes

- **Python 3.12** required (`hue-entertainment-pykit` doesn't ship wheels
  for 3.13 because `python-mbedtls` doesn't).
- **Hue bridge credentials** live in `data/auth.json` (gitignored). On
  first connect, the library writes them after a button-press pairing.
- **`hue-entertainment-pykit` is installed from PyPI** and pinned in
  `pyproject.toml`; no sibling checkout is required.
- **Home Assistant light control** uses `HA_TOKEN` from the environment and
  defaults to `http://t460s:8123` with `light.dining_room_brighter`.
- **Spotify control is optional and macOS-only**. Without Spotify running,
  the widget shows "Spotify: not detected" and volume choreographies
  silently no-op. AppleScript guards (`if application "Spotify" is running
  then ...`) prevent auto-launching Spotify.
- **Persisted prefs** live at `data/preferences.json`. The `Config` class
  loads/saves it atomically; the old `prefs.py` is gone.

## When making changes

- **New effect**: write `async def my_effect(ctl, config, *, clock=None)`.
  Use `frame_loop(ctl, clock, duration, frame_fn)` + the colors/signals
  primitives. Return a single `(r, g, b)` from `frame_fn` for uniform, or
  a sequence for per-channel. Settle into `config.bright_white` with
  `_settle_to_bright_white(ctl, config, clock, duration)` if the effect
  is "active then return to day".
- **New persistent pref**: add a field to `Config` in `config.py`. The
  load/save round-trip handles JSON automatically. Use `config.update(field=v)`
  from UI handlers.
- **New audio-synced effect**: drop the wav in `sounds/`, run
  `librosa.onset.onset_strength` + `peak_pick` to find hit times,
  hardcode them in the effect. Pattern: see `gavel_effect` and the
  lightning STRIKES list.
- **New button**: add a handler that calls `run_effect(my_effect,
  audio_el?, volume_choreography?)`, wire into the buttons grid, and add
  to the `effect_buttons` tuple so it gets enable/disable handling.
- **Tests**: trace tests in `tests/test_effects.py` use `VirtualClock` +
  `RecordingController`. Each effect needs at minimum a final-color
  assertion. For volume-driving effects (currently just night), pass a
  `RecordingVolumeController` and assert on `vol.events`.
- **Hue command rate**: the streaming library queues every `set_input`
  call without coalescing. 60 Hz updates from effects are fine. The
  library logs every `set_input` at INFO; `main.py` silences this logger.

## Lint / type-check expectations

`ruff check .` and `ty check .` should both be clean before committing.
The lint config (in `pyproject.toml`) enables `E`, `F`, `I`, `UP` —
pycodestyle, pyflakes, import sorting, and pyupgrade modernizations.
ty is a structural type checker; it doesn't honor mypy/pyright ignore
syntax, only `# ty: ignore[<rule>]`. The pywebview `WindowProxy` is a
dynamic forwarder so its real attributes aren't typed; the two `resize`
call sites in `main.py` carry inline `ty: ignore` comments.
