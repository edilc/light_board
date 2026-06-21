"""Home Assistant light integration."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

import requests

from config import Config

logger = logging.getLogger("light_board.home_assistant")


@dataclass(frozen=True)
class HomeAssistantClient:
    """Minimal Home Assistant REST client for light service calls."""

    url: str
    token: str | None
    timeout_s: float = 10.0

    @classmethod
    def from_config(cls, config: Config) -> HomeAssistantClient:
        return cls(
            url=config.home_assistant_url,
            token=os.environ.get("HA_TOKEN"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.token)

    def _call_light_service(self, service: str, payload: dict) -> None:
        entity_id = str(payload.get("entity_id", ""))
        if not entity_id:
            logger.info("home assistant light not configured; skipping")
            return
        if not self.enabled:
            logger.info("HA_TOKEN not set; skipping home assistant light %s", entity_id)
            return

        response = requests.post(
            f"{self.url.rstrip('/')}/api/services/light/{service}",
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        logger.info("home assistant light %s: %s", service, entity_id)

    def turn_on_light(self, entity_id: str, *, brightness_pct: int | None = None) -> None:
        payload: dict = {"entity_id": entity_id}
        if brightness_pct is not None:
            payload["brightness_pct"] = max(0, min(100, int(brightness_pct)))
        self._call_light_service("turn_on", payload)

    def turn_off_light(self, entity_id: str) -> None:
        self._call_light_service("turn_off", {"entity_id": entity_id})

    async def turn_on_light_async(
        self, entity_id: str, *, brightness_pct: int | None = None
    ) -> None:
        await asyncio.to_thread(
            self.turn_on_light, entity_id, brightness_pct=brightness_pct
        )

    async def turn_off_light_async(self, entity_id: str) -> None:
        await asyncio.to_thread(self.turn_off_light, entity_id)
