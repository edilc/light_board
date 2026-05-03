"""macOS Spotify control via AppleScript + per-effect volume choreography.

Talks to the Spotify desktop app on the local machine through `osascript`.
All calls are best-effort — if Spotify isn't running, methods return None
or silently no-op rather than raising.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Optional

import effects

logger = logging.getLogger("light_board.spotify")

_OSASCRIPT_TIMEOUT = 2.0


class SpotifyController:
    def is_running(self) -> bool:
        try:
            r = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "System Events" to (name of processes) contains "Spotify"',
                ],
                capture_output=True,
                text=True,
                timeout=_OSASCRIPT_TIMEOUT,
            )
            return r.stdout.strip() == "true"
        except Exception:
            return False

    def _get_volume_sync(self) -> Optional[int]:
        if not self.is_running():
            return None
        try:
            r = subprocess.run(
                ["osascript", "-e", 'tell application "Spotify" to sound volume'],
                capture_output=True,
                text=True,
                timeout=_OSASCRIPT_TIMEOUT,
            )
            return int(r.stdout.strip())
        except Exception:
            return None

    def _set_volume_sync(self, percent: int) -> None:
        percent = max(0, min(100, int(percent)))
        # Guard the `tell` inside the AppleScript so it doesn't auto-launch
        # Spotify when the app is closed.
        script = (
            f'if application "Spotify" is running then '
            f'tell application "Spotify" to set sound volume to {percent}'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=_OSASCRIPT_TIMEOUT,
            )
        except Exception as exc:
            logger.warning("set_volume failed: %s", exc)

    def _get_track_sync(self) -> Optional[str]:
        if not self.is_running():
            return None
        try:
            r = subprocess.run(
                [
                    "osascript",
                    "-e",
                    'tell application "Spotify" to (name of current track) & " — " & (artist of current track)',
                ],
                capture_output=True,
                text=True,
                timeout=_OSASCRIPT_TIMEOUT,
            )
            t = r.stdout.strip()
            return t if t and t != " — " else None
        except Exception:
            return None

    async def get_volume(self) -> Optional[int]:
        return await asyncio.to_thread(self._get_volume_sync)

    async def set_volume(self, percent: int) -> None:
        await asyncio.to_thread(self._set_volume_sync, percent)

    async def get_track(self) -> Optional[str]:
        return await asyncio.to_thread(self._get_track_sync)


async def _stepped_fade(
    spotify: SpotifyController, start: int, end: int, duration: float, steps: int = 10
) -> None:
    """osascript is slow (~50–100ms per call), so volume fades happen in
    discrete steps rather than at frame rate."""
    if steps <= 0 or duration <= 0:
        await spotify.set_volume(end)
        return
    step_dur = duration / steps
    for i in range(1, steps + 1):
        f = i / steps
        target = int(start + (end - start) * f)
        await spotify.set_volume(target)
        await asyncio.sleep(step_dur)


async def lightning_volume(spotify: SpotifyController, prev_vol: int) -> None:
    """Duck to 20% during the impact + tail; restore when audio ends."""
    await spotify.set_volume(20)
    await asyncio.sleep(effects.get_thunder().duration)
    await spotify.set_volume(prev_vol)


async def gong_volume(spotify: SpotifyController, prev_vol: int) -> None:
    """Fade prev_vol → 0 over the gong's audio, then 0 → prev_vol over the settle."""
    audio_duration = effects.get_gong().duration  # ~5s
    settle_duration = 1.5
    await _stepped_fade(spotify, prev_vol, 0, audio_duration, steps=10)
    await _stepped_fade(spotify, 0, prev_vol, settle_duration, steps=5)


async def gavel_volume(spotify: SpotifyController, prev_vol: int) -> None:
    """Duck to 0 briefly on each gavel hit, restore between."""
    hits = [0.070, 0.557]
    duck_dur = 0.20
    elapsed = 0.0
    for hit_t in hits:
        if hit_t > elapsed:
            await asyncio.sleep(hit_t - elapsed)
            elapsed = hit_t
        await spotify.set_volume(0)
        await asyncio.sleep(duck_dur)
        elapsed += duck_dur
        await spotify.set_volume(prev_vol)


async def morning_volume(
    spotify: SpotifyController, prev_vol: int, *, target: int = 50
) -> None:
    """Day setpoint: fade to `target`% over 1s. No auto-restore."""
    await _stepped_fade(spotify, prev_vol, target, duration=1.0, steps=10)


async def night_volume(
    spotify: SpotifyController, prev_vol: int, *, target: int = 100
) -> None:
    """Night setpoint: fade to `target`% over 1s. No auto-restore."""
    await _stepped_fade(spotify, prev_vol, target, duration=1.0, steps=10)
