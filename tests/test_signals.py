import numpy as np
import pandas as pd
from scouter import indicators as ind
from scouter.signals import Params, evaluate


def series_frame(close, volume, spread=0.012):
    close = np.asarray(close, dtype=float)
    vol = np.asarray(volume, dtype=float)
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


def accumulation_chart(seed=1):
    """Downtrend, choppy base, then a volume-backed markup."""
    r = np.random.default_rng(seed)
    down = np.linspace(1.00, 0.72, 100) * (1 + r.normal(0, 0.006, 100))
    base = 0.72 * (1 + r.normal(0, 0.012, 45)).cumprod()
    up = base[-1] * np.cumprod(
        1 + np.array([.02, .03, .05, .07, .08, .06, .04, .03, .03, .02]))
    close = np.concatenate([down, base, up])
    vol = np.concatenate([
        r.uniform(300, 700, 100), r.uniform(150, 450, 45),
        np.array([900, 1500, 2600, 4200, 6000, 7000, 6500, 5800, 5200, 5600.])])
    return series_frame(close, vol)


def bleed_chart(seed=3):
    r = np.random.default_rng(seed)
    close = np.linspace(1.0, 0.4, 160) * (1 + r.normal(0, 0.01, 160))
    return series_frame(close, r.uniform(100, 400, 160))


def test_divergence_fires_on_lower_price_low_higher_cmf_low():
    n = 80
    lows = np.linspace(1.0, 0.9, n)
    lows[20], lows[60] = 0.80, 0.79
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": lows, "high": lows * 1.02, "low": lows,
                       "close": lows * 1.01, "volume": np.full(n, 100.0)}, index=idx)
    c = pd.Series(np.zeros(n), index=idx)
    c.iloc[20], c.iloc[60] = -0.40, -0.05
    assert ind.cmf_bullish_divergence(df, c, lookback=80, k=3)["divergence"]


def test_no_divergence_when_cmf_also_makes_lower_low():
    n = 80
    lows = np.linspace(1.0, 0.9, n)
    lows[20], lows[60] = 0.80, 0.79
    idx = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": lows, "high": lows * 1.02, "low": lows,
                       "close": lows * 1.01, "volume": np.full(n, 100.0)}, index=idx)
    c = pd.Series(np.zeros(n), index=idx)
    c.iloc[20], c.iloc[60] = -0.40, -0.60
    assert not ind.cmf_bullish_divergence(df, c, lookback=80, k=3)["divergence"]


def test_returns_none_on_insufficient_history():
    assert evaluate(accumulation_chart().iloc[:20], Params()) is None


def test_accumulation_setup_fires():
    df = accumulation_chart()
    fired = [cut for cut in range(45, len(df) + 1)
             if (s := evaluate(df.iloc[:cut], Params())) and s.matched]
    assert fired, "textbook accumulation-to-markup produced no signal"


def test_bleed_chart_never_fires():
    df = bleed_chart()
    assert not any((s := evaluate(df.iloc[:cut], Params())) and s.matched
                   for cut in range(45, len(df) + 1))


def test_strict_same_bar_window_is_far_rarer():
    """The spec's literal reading. Documents why the default window is 6."""
    df = accumulation_chart()
    loose = sum(1 for cut in range(45, len(df) + 1)
                if (s := evaluate(df.iloc[:cut], Params(macd_cross_window=6))) and s.matched)
    strict = sum(1 for cut in range(45, len(df) + 1)
                 if (s := evaluate(df.iloc[:cut], Params(macd_cross_window=1))) and s.matched)
    assert loose > strict
