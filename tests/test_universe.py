import pytest
from scouter.universe import dedupe_by_mint, passes_screen, ScreenParams

P = ScreenParams()


def pool(addr, mint, fdv, res, vol=3000.0, txns=40, sym="X"):
    return {"address": addr, "base_token_address": mint, "fdv_usd": fdv,
            "market_cap_usd": None, "reserve_usd": res, "volume_h24": vol,
            "txns_h24": txns, "base_token_symbol": sym}


def test_dedupe_keeps_deepest_pool_per_mint():
    """LUCY appeared as both LUCY/SOL and LUCY/USDC in the live feed. One
    token on two pools would open two positions and double the intended risk."""
    pools = [pool("A", "MINT1", 5000, 6000), pool("B", "MINT1", 5200, 9000),
             pool("C", "MINT2", 4000, 7000)]
    kept = dedupe_by_mint(pools)
    assert len(kept) == 2
    assert {p["address"] for p in kept} == {"B", "C"}


def test_dedupe_keeps_pools_with_no_mint():
    pools = [pool("A", None, 5000, 6000), pool("B", None, 5000, 6000)]
    assert len(dedupe_by_mint(pools)) == 2


def test_screen_accepts_in_band_pool():
    assert passes_screen(pool("A", "M", 6000, 7000), age=5.0, p=P)


def test_screen_rejects_out_of_band():
    assert not passes_screen(pool("A", "M", 2000, 7000), age=5.0, p=P)
    assert not passes_screen(pool("A", "M", 50000, 7000), age=5.0, p=P)


def test_screen_rejects_young_pool():
    assert not passes_screen(pool("A", "M", 6000, 7000), age=1.0, p=P)


def test_null_reserve_is_rejected_not_treated_as_zero():
    p = pool("A", "M", 6000, None)
    assert not passes_screen(p, age=5.0, p=P)
    assert passes_screen(p, age=5.0, p=ScreenParams(allow_unknown_reserve=True))


def test_screen_rejects_illiquid():
    assert not passes_screen(pool("A", "M", 6000, 900), age=5.0, p=P)
