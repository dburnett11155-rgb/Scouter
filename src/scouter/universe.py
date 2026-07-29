"""Pool universe: screening and deduplication.

Built from two findings in the live feed rather than from the spec.

1. Liquidity tracks FDV at roughly 0.79x across launch-mechanic pools, so the
   FDV band and the liquidity floor are not independent filters. A $5,000
   floor implies FDV >= ~$6,300, which silently truncates the bottom of a
   stated $3k-$10k band. Kept deliberately, documented here.
2. The same token appears on multiple pools (LUCY/SOL and LUCY/USDC; PIPEDOG
   on three pools at three different FDVs). Screening per pool would open one
   position per pool and multiply the intended per-trade risk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenParams:
    min_fdv_usd: float = 3_000.0
    max_fdv_usd: float = 10_000.0
    min_liquidity_usd: float = 5_000.0
    min_age_days: float = 3.0
    # Liveness guards: no point spending an OHLCV call on a pool with no
    # trades, since MACD/CMF/EFI need actual bars.
    min_volume_h24_usd: float = 1_000.0
    min_txns_h24: int = 20
    # A missing reserve_in_usd means unknown, not zero. Default is to reject,
    # because an unknown exit depth is not tradeable under a -50% stop.
    allow_unknown_reserve: bool = False


def effective_mcap(pool: dict) -> float | None:
    mc = pool.get("market_cap_usd")
    return mc if mc is not None else pool.get("fdv_usd")


def dedupe_by_mint(pools: list[dict]) -> list[dict]:
    """One pool per base token, keeping the deepest.

    Deepest wins because exit depth is the binding constraint: the stop needs
    a bid on the other side. Pools with no mint address are passed through
    untouched rather than collapsed together.
    """
    best: dict[str, dict] = {}
    passthrough: list[dict] = []
    for p in pools:
        mint = p.get("base_token_address")
        if not mint:
            passthrough.append(p)
            continue
        incumbent = best.get(mint)
        if incumbent is None or (p.get("reserve_usd") or 0) > (
                incumbent.get("reserve_usd") or 0):
            best[mint] = p
    return list(best.values()) + passthrough


def passes_screen(pool: dict, age: float | None, p: ScreenParams) -> bool:
    """Every filter except the indicator confluence."""
    if age is None or age < p.min_age_days:
        return False

    mcap = effective_mcap(pool)
    if mcap is None or not (p.min_fdv_usd <= mcap <= p.max_fdv_usd):
        return False

    reserve = pool.get("reserve_usd")
    if reserve is None:
        if not p.allow_unknown_reserve:
            return False
    elif reserve < p.min_liquidity_usd:
        return False

    if (pool.get("volume_h24") or 0) < p.min_volume_h24_usd:
        return False
    if (pool.get("txns_h24") or 0) < p.min_txns_h24:
        return False
    return True
