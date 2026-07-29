"""Async GeckoTerminal v2 client.

Two load-bearing details:

1. One global rate limiter. The keyless API is documented at both 10 and 30
   calls/min, so every request in the process shares one bucket and adding
   concurrency can never blow the budget.
2. Real 429 handling: honours Retry-After, otherwise exponential backoff with
   jitter, and logs the body so a quota change doesn't look like a network flake.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone

import aiohttp
import pandas as pd

log = logging.getLogger("gecko")

API_BASE = "https://api.geckoterminal.com/api/v2"
API_VERSION = "application/json;version=20230302"


def _num(value) -> float | None:
    """None for absent, 0.0 for a real zero.

    The distinction matters: the live API returns pools doing $7M of daily
    volume with no reserve_in_usd field at all. Coercing that to 0 would fail
    the liquidity floor for a reason that has nothing to do with liquidity.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_pools(payload: dict | None) -> list[dict]:
    """Flatten JSON:API pool objects, resolving included token/dex records."""
    if not payload:
        return []
    included = {(i["type"], i["id"]): i for i in payload.get("included", [])}

    raw = payload.get("data", [])
    if isinstance(raw, dict):
        raw = [raw]

    out: list[dict] = []
    for item in raw:
        attrs = item.get("attributes") or {}
        rels = item.get("relationships") or {}

        def _rel(name: str) -> dict:
            ref = (rels.get(name) or {}).get("data") or {}
            node = included.get((ref.get("type"), ref.get("id")))
            return (node or {}).get("attributes") or {}

        base = _rel("base_token")
        base_id = ((rels.get("base_token") or {}).get("data") or {}).get("id", "")
        # Fall back to the relationship id ("solana_MINT") when include= was
        # omitted or the token record didn't come back.
        base_address = base.get("address") or (
            base_id.split("_", 1)[1] if "_" in base_id else None)

        vol = attrs.get("volume_usd") or {}
        t24 = (attrs.get("transactions") or {}).get("h24") or {}

        out.append({
            "address": attrs.get("address") or item.get("id", "").split("_", 1)[-1],
            "name": attrs.get("name"),
            "dex": _rel("dex").get("name"),
            "base_token_address": base_address,
            "base_token_symbol": base.get("symbol"),
            "pool_created_at": attrs.get("pool_created_at"),
            "price_usd": _num(attrs.get("base_token_price_usd")),
            # market_cap_usd is null unless CoinGecko verified the token, which
            # does not happen at this size. FDV is the usable field.
            "market_cap_usd": _num(attrs.get("market_cap_usd")),
            "fdv_usd": _num(attrs.get("fdv_usd")),
            "reserve_usd": _num(attrs.get("reserve_in_usd")),
            "volume_h1": _num(vol.get("h1")),
            "volume_h24": _num(vol.get("h24")),
            "txns_h24": int((t24.get("buys") or 0) + (t24.get("sells") or 0)),
        })
    return out


def age_days(pool_created_at: str | None) -> float | None:
    if not pool_created_at:
        return None
    try:
        created = datetime.fromisoformat(pool_created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - created).total_seconds() / 86400.0


def effective_mcap(pool: dict) -> float | None:
    """Verified market cap when present, else FDV."""
    mc = pool.get("market_cap_usd")
    return mc if mc is not None else pool.get("fdv_usd")


class RateLimiter:
    """Reserves a slot under the lock, sleeps outside it so latency overlaps."""

    def __init__(self, calls_per_minute: int) -> None:
        self.interval = 60.0 / max(1, calls_per_minute)
        self._lock = asyncio.Lock()
        self._next = 0.0
        self.calls = 0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
            self.calls += 1
        delay = slot - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    def pause(self, seconds: float) -> None:
        self._next = max(self._next, time.monotonic() + seconds)


class GeckoClient:
    def __init__(self, network: str = "solana", calls_per_minute: int = 20,
                 timeout_s: int = 25, max_retries: int = 4) -> None:
        self.network = network
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.limiter = RateLimiter(calls_per_minute)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "GeckoClient":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
            headers={"Accept": API_VERSION, "User-Agent": "scouter/0.1"})
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session:
            await self._session.close()

    async def _get(self, path: str, params: dict | None = None) -> dict | None:
        assert self._session is not None, "use GeckoClient as an async context manager"
        url = f"{API_BASE}{path}"
        for attempt in range(self.max_retries):
            await self.limiter.acquire()
            try:
                async with self._session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status == 429:
                        body = (await resp.text())[:300]
                        ra = resp.headers.get("Retry-After")
                        wait = (float(ra) if ra and ra.isdigit()
                                else min(60.0, 5.0 * 2 ** attempt + random.uniform(0, 3)))
                        log.warning("429 %s attempt %d, sleeping %.1fs. body=%s",
                                    path, attempt + 1, wait, body)
                        self.limiter.pause(wait)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status == 404:
                        return None
                    if 500 <= resp.status < 600:
                        await asyncio.sleep(2.0 * 2 ** attempt + random.uniform(0, 1))
                        continue
                    log.error("HTTP %d %s: %s", resp.status, path,
                              (await resp.text())[:200])
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("network error %s (%s)", path, type(exc).__name__)
                await asyncio.sleep(2.0 * 2 ** attempt + random.uniform(0, 1))
        log.error("gave up on %s after %d attempts", path, self.max_retries)
        return None

    async def new_pools(self, page: int = 1) -> list[dict]:
        """20 per page, pages 1-10 on the free tier."""
        return parse_pools(await self._get(
            f"/networks/{self.network}/new_pools",
            {"page": page, "include": "base_token,quote_token,dex"}))

    async def top_pools(self, page: int = 1,
                        sort: str = "h24_volume_usd_desc") -> list[dict]:
        return parse_pools(await self._get(
            f"/networks/{self.network}/pools",
            {"page": page, "sort": sort, "include": "base_token,quote_token,dex"}))

    async def pools_multi(self, addresses: list[str]) -> list[dict]:
        """Up to 30 addresses per call: the cheapest way to refresh."""
        if not addresses:
            return []
        return parse_pools(await self._get(
            f"/networks/{self.network}/pools/multi/{','.join(addresses[:30])}",
            {"include": "base_token,quote_token,dex"}))

    async def ohlcv_hourly(self, pool_address: str,
                           limit: int = 300) -> pd.DataFrame | None:
        """1H candles, oldest -> newest, USD on the base token."""
        data = await self._get(
            f"/networks/{self.network}/pools/{pool_address}/ohlcv/hour",
            {"aggregate": 1, "limit": limit, "currency": "usd", "token": "base"})
        rows = ((data or {}).get("data", {}).get("attributes", {})
                .get("ohlcv_list", []))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
        return df.set_index("ts").sort_index().astype(float)
