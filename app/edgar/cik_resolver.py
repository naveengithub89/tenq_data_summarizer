from __future__ import annotations

import asyncio
from datetime import timedelta, datetime
from typing import Dict, Optional

from app.edgar.client import EdgarHttpClient
from app.edgar.models import CikResolutionResult


class CikResolver:
    """
    Resolve a ticker symbol to CIK + company name via company_tickers.json.
    Uses in-memory cache with a simple TTL.
    """

    _TTL = timedelta(hours=6)

    def __init__(self, client: EdgarHttpClient) -> None:
        self._client = client
        self._cache_by_ticker: Dict[str, CikResolutionResult] = {}
        self._last_refresh: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        async with self._lock:
            if self._last_refresh and datetime.utcnow() - self._last_refresh < self._TTL:
                return

            raw = await self._client.get_json("files/company_tickers.json", data_host=False)
            # SEC format: { "0": { "cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc." }, ... }
            cache: Dict[str, CikResolutionResult] = {}
            for _, entry in raw.items():
                ticker = str(entry["ticker"]).upper()
                cik_int = int(entry["cik_str"])
                cik_str = f"{cik_int:010d}"
                result = CikResolutionResult(
                    ticker=ticker,
                    cik_int=cik_int,
                    cik_str=cik_str,
                    company_name=entry["title"],
                )
                cache[ticker] = result

            self._cache_by_ticker = cache
            self._last_refresh = datetime.utcnow()

    async def resolve(self, ticker: str) -> CikResolutionResult:
        await self._ensure_loaded()
        norm = ticker.strip().upper()
        try:
            return self._cache_by_ticker[norm]
        except KeyError:
            raise KeyError(f"Unknown ticker: {ticker!r}")
