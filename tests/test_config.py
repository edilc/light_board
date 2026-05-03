"""Unit tests for the Config dataclass."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import Config


class TestLatencyProperties:
    """audio_latency_override_s and hue_latency_override_s share the same
    ms→s code path; one set + one None case covers both."""

    def test_set_converts_ms_to_seconds(self):
        c = Config(audio_latency_override_ms=200)
        assert c.audio_latency_override_s == pytest.approx(0.200)

    def test_none_passes_through(self):
        assert Config().audio_latency_override_s is None
        assert Config().hue_latency_override_s is None


class TestLoad:
    def test_missing_file_returns_defaults(self, tmp_path: Path):
        path = tmp_path / "missing.json"
        c = Config.load(path)
        assert c.bright_white == (244, 218, 182)
        assert c.path == path

    def test_existing_file_overrides_defaults(self, tmp_path: Path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({
            "bright_white": [100, 110, 120],
            "morning_target_volume": 75,
            "dark_mode": True,
        }))
        c = Config.load(path)
        assert c.bright_white == (100, 110, 120)
        assert c.morning_target_volume == 75
        assert c.dark_mode is True
        # Unset field falls back to default.
        assert c.night_target_volume == 100

    def test_unknown_keys_are_filtered(self, tmp_path: Path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({
            "bright_white": [50, 60, 70],
            "this_field_does_not_exist": 42,
            "another_junk_key": "lol",
        }))
        # Should not raise — unknown keys silently dropped.
        c = Config.load(path)
        assert c.bright_white == (50, 60, 70)

    def test_malformed_json_falls_back_to_defaults(self, tmp_path: Path):
        path = tmp_path / "prefs.json"
        path.write_text("not valid json {{{")
        c = Config.load(path)
        # Defaults preserved, no exception.
        assert c.bright_white == (244, 218, 182)
        assert c.path == path

    def test_bright_white_loaded_as_tuple(self, tmp_path: Path):
        # JSON arrays come back as lists; we want a tuple internally so
        # comparisons with `(r, g, b)` literals work.
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"bright_white": [10, 20, 30]}))
        c = Config.load(path)
        assert isinstance(c.bright_white, tuple)
        assert c.bright_white == (10, 20, 30)


class TestSave:
    def test_save_round_trips(self, tmp_path: Path):
        path = tmp_path / "prefs.json"
        original = Config(
            path=path,
            bright_white=(10, 20, 30),
            morning_target_volume=33,
            night_target_volume=66,
            internal_audio_volume=0.42,
            dark_mode=True,
            audio_latency_override_ms=150,
            hue_latency_override_ms=80,
        )
        original.save()

        reloaded = Config.load(path)
        assert reloaded.bright_white == (10, 20, 30)
        assert reloaded.morning_target_volume == 33
        assert reloaded.night_target_volume == 66
        assert reloaded.internal_audio_volume == 0.42
        assert reloaded.dark_mode is True
        assert reloaded.audio_latency_override_ms == 150
        assert reloaded.hue_latency_override_ms == 80

    def test_save_atomic_via_tmp(self, tmp_path: Path):
        # Save writes through a `.tmp` sibling first to avoid a torn file
        # on crash. After save completes, no .tmp file should be left.
        path = tmp_path / "prefs.json"
        c = Config(path=path)
        c.save()
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


class TestUpdate:
    def test_sets_field(self, tmp_path: Path):
        path = tmp_path / "prefs.json"
        c = Config(path=path)
        c.update(bright_white=(5, 6, 7))
        assert c.bright_white == (5, 6, 7)

    def test_unknown_field_raises(self, tmp_path: Path):
        c = Config(path=tmp_path / "prefs.json")
        with pytest.raises(AttributeError, match="no field 'wat'"):
            c.update(wat=42)

    def test_unknown_field_does_not_persist_partial(self, tmp_path: Path):
        # If one of the kwargs is invalid, NO writes should occur — even
        # for the valid ones. Otherwise typos cause partial-write surprises.
        path = tmp_path / "prefs.json"
        c = Config(path=path, bright_white=(1, 1, 1))
        with pytest.raises(AttributeError):
            c.update(bright_white=(9, 9, 9), bogus_field=42)
        assert c.bright_white == (1, 1, 1)
        assert not path.exists()
