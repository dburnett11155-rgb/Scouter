import sqlite3
import pytest
from scouter import store


@pytest.fixture
def conn(tmp_path):
    path = tmp_path / "t.sqlite"
    store.init(str(path))
    c = store.connect(str(path))
    yield c
    c.close()


def pool(addr="POOL1", mint="MINT1", fdv=6000.0, res=7000.0, created="2026-07-01T00:00:00Z"):
    return {"address": addr, "name": "X / SOL", "dex": "Raydium",
            "base_token_address": mint, "base_token_symbol": "X",
            "pool_created_at": created, "price_usd": 1e-5,
            "market_cap_usd": None, "fdv_usd": fdv, "reserve_usd": res,
            "volume_h1": 300.0, "volume_h24": 4000.0, "txns_h24": 60}


def test_upsert_inserts(conn):
    assert store.upsert_pools(conn, [pool()]) == 1
    assert store.universe_size(conn) == 1


def test_upsert_preserves_first_seen_but_updates_stats(conn):
    store.upsert_pools(conn, [pool()])
    before = conn.execute("SELECT first_seen_at FROM pools").fetchone()[0]
    store.upsert_pools(conn, [pool(fdv=6500.0, res=7500.0)])
    row = conn.execute(
        "SELECT first_seen_at, fdv_usd, reserve_usd FROM pools").fetchone()
    assert row["first_seen_at"] == before
    assert row["fdv_usd"] == pytest.approx(6500.0)
    assert store.universe_size(conn) == 1


def test_null_reserve_round_trips_as_null(conn):
    """Must survive storage as NULL, not collapse to 0."""
    store.upsert_pools(conn, [pool(res=None)])
    assert conn.execute("SELECT reserve_usd FROM pools").fetchone()[0] is None


def test_upsert_does_not_null_out_known_fields(conn):
    """pools/multi may return a sparser record than new_pools did."""
    store.upsert_pools(conn, [pool()])
    store.upsert_pools(conn, [{"address": "POOL1", "fdv_usd": 6100.0}])
    row = conn.execute(
        "SELECT base_token_symbol, pool_created_at FROM pools").fetchone()
    assert row["base_token_symbol"] == "X"
    assert row["pool_created_at"] == "2026-07-01T00:00:00Z"


def test_candidates_applies_age_and_band(conn):
    from scouter.universe import ScreenParams
    store.upsert_pools(conn, [
        pool("OLD", "M1", created="2020-01-01T00:00:00Z"),
        pool("NEW", "M2", created="2099-01-01T00:00:00Z"),
        pool("BIG", "M3", fdv=900000.0, created="2020-01-01T00:00:00Z"),
    ])
    got = {p["address"] for p in store.candidates(conn, ScreenParams())}
    assert got == {"OLD"}


def test_candidates_dedupes_by_mint(conn):
    from scouter.universe import ScreenParams
    store.upsert_pools(conn, [
        pool("SHALLOW", "SAME", res=6000.0, created="2020-01-01T00:00:00Z"),
        pool("DEEP", "SAME", res=9000.0, created="2020-01-01T00:00:00Z"),
    ])
    got = store.candidates(conn, ScreenParams())
    assert [p["address"] for p in got] == ["DEEP"]


def test_stale_addresses_orders_by_staleness(conn):
    store.upsert_pools(conn, [pool("A", "M1")])
    conn.execute("UPDATE pools SET last_refreshed_at='2020-01-01' WHERE address='A'")
    store.upsert_pools(conn, [pool("B", "M2")])
    assert store.stale_addresses(conn, 10)[0] == "A"


def test_run_log(conn):
    rid = store.start_run(conn, "test")
    store.finish_run(conn, rid, universe_size=5, discovered=3, note="ok")
    row = store.last_runs(conn, 1)[0]
    assert row["universe_size"] == 5 and row["finished_at"] is not None
