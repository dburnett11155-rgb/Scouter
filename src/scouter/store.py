"""SQLite persistence for the pool universe.

WAL mode with a busy timeout so the discovery daemon can write while a
dashboard process reads, without "database is locked".
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from scouter.universe import ScreenParams, dedupe_by_mint, passes_screen

SCHEMA = """
CREATE TABLE IF NOT EXISTS pools (
    address            TEXT PRIMARY KEY,
    name               TEXT,
    dex                TEXT,
    base_token_address TEXT,
    base_token_symbol  TEXT,
    pool_created_at    TEXT,
    first_seen_at      TEXT NOT NULL,
    last_refreshed_at  TEXT,
    price_usd          REAL,
    market_cap_usd     REAL,
    fdv_usd            REAL,
    reserve_usd        REAL,
    volume_h1          REAL,
    volume_h24         REAL,
    txns_h24           INTEGER,
    status             TEXT NOT NULL DEFAULT 'active'
);
CREATE INDEX IF NOT EXISTS idx_pools_screen ON pools (status, fdv_usd);
CREATE INDEX IF NOT EXISTS idx_pools_mint ON pools (base_token_address);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    trigger     TEXT,
    universe_size INTEGER,
    discovered  INTEGER,
    refreshed   INTEGER,
    candidates  INTEGER,
    api_calls   INTEGER,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS control (key TEXT PRIMARY KEY, value TEXT);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init(path: str) -> None:
    with closing(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def closing(path: str):
    conn = connect(path)
    try:
        yield conn
    finally:
        conn.close()


def upsert_pools(conn: sqlite3.Connection, pools) -> int:
    """Insert new pools, refresh stats on known ones.

    COALESCE(excluded.x, pools.x) on identity fields is load-bearing:
    pools/multi returns a leaner record than new_pools, so a plain overwrite
    would wipe pool_created_at on refresh and reset every pool's age to
    unknown, silently breaking the age filter.

    first_seen_at is never overwritten; it is the fallback age signal for
    pools whose pool_created_at comes back null.
    """
    now = utcnow()
    rows = [(p["address"], p.get("name"), p.get("dex"),
             p.get("base_token_address"), p.get("base_token_symbol"),
             p.get("pool_created_at"), now, now, p.get("price_usd"),
             p.get("market_cap_usd"), p.get("fdv_usd"), p.get("reserve_usd"),
             p.get("volume_h1"), p.get("volume_h24"), p.get("txns_h24"))
            for p in pools if p.get("address")]
    if not rows:
        return 0
    conn.executemany("""
        INSERT INTO pools (address, name, dex, base_token_address,
            base_token_symbol, pool_created_at, first_seen_at,
            last_refreshed_at, price_usd, market_cap_usd, fdv_usd,
            reserve_usd, volume_h1, volume_h24, txns_h24)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(address) DO UPDATE SET
            name = COALESCE(excluded.name, pools.name),
            dex = COALESCE(excluded.dex, pools.dex),
            base_token_address = COALESCE(excluded.base_token_address, pools.base_token_address),
            base_token_symbol = COALESCE(excluded.base_token_symbol, pools.base_token_symbol),
            pool_created_at = COALESCE(excluded.pool_created_at, pools.pool_created_at),
            last_refreshed_at = excluded.last_refreshed_at,
            price_usd = excluded.price_usd,
            market_cap_usd = excluded.market_cap_usd,
            fdv_usd = excluded.fdv_usd,
            reserve_usd = excluded.reserve_usd,
            volume_h1 = excluded.volume_h1,
            volume_h24 = excluded.volume_h24,
            txns_h24 = excluded.txns_h24
    """, rows)
    return len(rows)


def candidates(conn: sqlite3.Connection, p: ScreenParams) -> list[dict]:
    """Screened, deduped pools, deepest first."""
    rows = conn.execute("""
        SELECT *, (julianday('now') -
                   julianday(COALESCE(pool_created_at, first_seen_at))) AS age_days
        FROM pools WHERE status = 'active'
    """).fetchall()
    kept = [dict(r) for r in rows if passes_screen(dict(r), r["age_days"], p)]
    kept = dedupe_by_mint(kept)
    return sorted(kept, key=lambda x: x.get("reserve_usd") or 0, reverse=True)


def stale_addresses(conn: sqlite3.Connection, limit: int) -> list[str]:
    return [r["address"] for r in conn.execute("""
        SELECT address FROM pools WHERE status='active'
        ORDER BY COALESCE(last_refreshed_at, '') ASC LIMIT ?
    """, (limit,)).fetchall()]


def universe_size(conn: sqlite3.Connection) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) AS n FROM pools WHERE status='active'").fetchone()["n"])


def set_control(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO control (key,value) VALUES (?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def get_control(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM control WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def start_run(conn: sqlite3.Connection, trigger: str) -> int:
    return int(conn.execute("INSERT INTO runs (started_at, trigger) VALUES (?,?)",
                            (utcnow(), trigger)).lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, **kw) -> None:
    kw["finished_at"] = utcnow()
    conn.execute(f"UPDATE runs SET {', '.join(f'{k}=?' for k in kw)} WHERE id=?",
                 (*kw.values(), run_id))


def last_runs(conn: sqlite3.Connection, n: int = 10) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (n,)).fetchall()
