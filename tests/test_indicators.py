import numpy as np
import pandas as pd
import pytest
from scouter import indicators as ind


def frame(close, volume=None, spread=0.01):
    close = np.asarray(close, dtype=float)
    vol = np.full(len(close), 100.0) if volume is None else np.asarray(volume, float)
    o = np.roll(close, 1)
    o[0] = close[0]
    idx = pd.date_range("2025-01-01", periods=len(close), freq="h", tz="UTC")
    return pd.DataFrame({
        "open": o,
        "high": np.maximum(o, close) * (1 + spread),
        "low": np.minimum(o, close) * (1 - spread),
        "close": close,
        "volume": vol,
    }, index=idx)


def test_macd_matches_independent_emas():
    c = pd.Series(np.linspace(1.0, 2.0, 60))
    m = ind.macd(c, 12, 26, 9)
    expected = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    assert np.allclose(m["macd"], expected)
    assert np.allclose(m["hist"], m["macd"] - m["signal"])


def test_cmf_hits_bounds():
    n = 30
    at_high = pd.DataFrame({"high": [10.0] * n, "low": [8.0] * n,
                            "close": [10.0] * n, "volume": [100.0] * n})
    at_low = at_high.assign(close=8.0)
    assert ind.cmf(at_high, 20).iloc[-1] == pytest.approx(1.0)
    assert ind.cmf(at_low, 20).iloc[-1] == pytest.approx(-1.0)


def test_cmf_survives_zero_range_and_zero_volume():
    n = 30
    doji = pd.DataFrame({"high": [10.0] * n, "low": [10.0] * n,
                         "close": [10.0] * n, "volume": [100.0] * n})
    novol = doji.assign(low=8.0, volume=0.0)
    assert np.isfinite(ind.cmf(doji, 20).iloc[-1])
    assert ind.cmf(novol, 20).iloc[-1] == 0.0


def test_efi_is_delta_times_volume():
    df = frame(np.linspace(1.0, 1.5, 40), volume=np.linspace(100, 500, 40))
    e = ind.efi(df, 13)
    expected = (df["close"].diff() * df["volume"]).iloc[1:]
    assert np.allclose(e["raw"].iloc[1:], expected)


def test_rolling_z_has_no_lookahead():
    s = pd.Series(np.random.default_rng(0).normal(size=100))
    assert np.isnan(ind.rolling_z(s, 50).iloc[0])


def test_pivot_lows_confirms_both_sides():
    v = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=float)
    assert ind.pivot_lows(v, k=3) == [4, 12]
