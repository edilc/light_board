from __future__ import annotations

import pytest

from config import Config
from home_assistant import HomeAssistantClient


class FakeResponse:
    def __init__(self) -> None:
        self.raise_called = False

    def raise_for_status(self) -> None:
        self.raise_called = True


def test_from_config_reads_url_and_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HA_TOKEN", "secret")
    config = Config(home_assistant_url="http://t460s:8123")

    client = HomeAssistantClient.from_config(config)

    assert client.url == "http://t460s:8123"
    assert client.token == "secret"


def test_turn_on_light_posts_to_home_assistant(monkeypatch: pytest.MonkeyPatch):
    calls = []
    response = FakeResponse()

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr("home_assistant.requests.post", fake_post)
    client = HomeAssistantClient(
        url="http://t460s:8123/",
        token="secret",
        timeout_s=3.0,
    )

    client.turn_on_light("light.dining_room_brighter", brightness_pct=30)

    assert calls == [(
        ("http://t460s:8123/api/services/light/turn_on",),
        {
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "json": {
                "entity_id": "light.dining_room_brighter",
                "brightness_pct": 30,
            },
            "timeout": 3.0,
        },
    )]
    assert response.raise_called is True


def test_turn_off_light_posts_to_home_assistant(monkeypatch: pytest.MonkeyPatch):
    calls = []
    response = FakeResponse()

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr("home_assistant.requests.post", fake_post)

    HomeAssistantClient(url="http://t460s:8123", token="secret").turn_off_light(
        "light.dining_room_brighter"
    )

    assert calls == [(
        ("http://t460s:8123/api/services/light/turn_off",),
        {
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            "json": {"entity_id": "light.dining_room_brighter"},
            "timeout": 10.0,
        },
    )]
    assert response.raise_called is True


def test_turn_on_light_clamps_brightness(monkeypatch: pytest.MonkeyPatch):
    calls = []
    response = FakeResponse()

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr("home_assistant.requests.post", fake_post)

    HomeAssistantClient(url="http://t460s:8123", token="secret").turn_on_light(
        "light.dining_room_brighter",
        brightness_pct=130,
    )

    assert calls[0][1]["json"]["brightness_pct"] == 100


def test_turn_on_light_skips_without_token(monkeypatch: pytest.MonkeyPatch):
    def fake_post(*_args, **_kwargs):
        raise AssertionError("requests.post should not be called")

    monkeypatch.setattr("home_assistant.requests.post", fake_post)

    HomeAssistantClient(url="http://t460s:8123", token=None).turn_on_light(
        "light.dining_room_brighter"
    )


def test_turn_on_light_skips_without_entity(monkeypatch: pytest.MonkeyPatch):
    def fake_post(*_args, **_kwargs):
        raise AssertionError("requests.post should not be called")

    monkeypatch.setattr("home_assistant.requests.post", fake_post)

    HomeAssistantClient(url="http://t460s:8123", token="secret").turn_on_light("")
