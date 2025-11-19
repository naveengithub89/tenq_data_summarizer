from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import monotonic
from typing import Any, Dict, Optional

import httpx

from app.config.settings import get_settings


@dataclass
class RateLimiter:
    max_rps: int
    _lock: asyncio.Lock = asyncio.Lock()
    _last_request_ts: float = 0.0

    async def wait(self) -> None:
        async with self._lock:
            min_interval = 1.0 / float(self.max_rps)
            now = monotonic()
            delta = now - self._last_request_ts
            if delta < min_interval:
                await asyncio.sleep(min_interval - delta)
            self._last_request_ts = monotonic()


class EdgarHttpClient:
    """
    Thin wrapper over httpx.AsyncClient that:
    * Adds SEC User-Agent and headers
    * Applies global rate limiting
    * Centralizes error handling
    """

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        max_rps: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self._client = client or httpx.AsyncClient(
            headers={
                "User-Agent": settings.sec_user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json, text/html, */*",
                "Connection": "keep-alive",
            },
            timeout=30.0,
        )
        self._rate_limiter = RateLimiter(max_rps or settings.sec_max_rps)
        self._sec_base = str(settings.sec_base_url)
        self._data_base = str(settings.sec_data_base_url)

    async def _get(self, url: str) -> httpx.Response:
        await self._rate_limiter.wait()
        resp = await self._client.get(url)
        if resp.status_code == 429:
            # naive backoff; you can refine later
            await asyncio.sleep(2.0)
        resp.raise_for_status()
        return resp

    async def get_json(self, path: str, data_host: bool = False) -> Dict[str, Any]:
        base = self._data_base if data_host else self._sec_base
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        resp = await self._get(url)
        return resp.json()

    async def get_text(self, url: str) -> str:
        resp = await self._get(url)
        return resp.text

    async def aclose(self) -> None:
        await self._client.aclose()
