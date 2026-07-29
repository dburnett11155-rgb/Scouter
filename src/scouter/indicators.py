"""Indicator math on pandas. No pandas_ta: its published build calls
`from numpy import NaN`, removed in NumPy 2.0, so it cannot import here.

Frames are indexed oldest -> newest with open/high/low/close/volume columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def macd(close: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """Standard MACD. adjust=False matches charting platforms."""
    line = (close.ewm(span=fast, adjust=False).mean()
            - close.ewm(span=slow, adjust=False).mean())
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Chaikin Money Flow.

        MFM = ((C - L) - (H - C)) / (H - L)
        CMF = sum(MFM * V, n) / sum(V, n)

    Doji bars (H == L) would divide by zero and get a multiplier of 0, the
    conventional treatment. Zero-volume windows return 0 rather than NaN so a
    quiet stretch doesn't blank the series.
    """
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    span = (high - low).replace(0, np.nan)
    mfm = (((close - low) - (high - close)) / span).fillna(0.0)
    vol_sum = vol.rolling(length).sum()
    out = (mfm * vol).rolling(length).sum() / vol_sum.replace(0, np.nan)
    return out.where(vol_sum.isna() | (vol_sum != 0), 0.0)


def efi(df: pd.DataFrame, length: int = 13) -> pd.DataFrame:
    """Elder's Force Index: raw single-bar force and its EMA."""
    raw = df["close"].diff() * df["volume"]
    return pd.DataFrame({"raw": raw, "efi": raw.ewm(span=length, adjust=False).mean()})


def rolling_z(series: pd.Series, lookback: int) -> pd.Series:
    """Z-score against the PRIOR `lookback` bars, excluding the current bar.

    The shift is what makes an EFI spike testable without look-ahead: the
    current bar is scored against history it could not have influenced.
    """
    prior = series.shift(1)
    floor = max(5, lookback // 3)
    mean = prior.rolling(lookback, min_periods=floor).mean()
    std = prior.rolling(lookback, min_periods=floor).std()
    return (series - mean) / std.replace(0, np.nan)


def pivot_lows(low: pd.Series, k: int = 3) -> list[int]:
    """Positional indices of confirmed swing lows.

    Bar i qualifies when it is the minimum of [i-k, i+k]. The trailing k bars
    can never qualify, which is correct: an unconfirmed low is not a swing low.
    """
    vals = low.to_numpy(dtype=float)
    out: list[int] = []
    for i in range(k, len(vals) - k):
        window = vals[i - k: i + k + 1]
        if np.isnan(window).any():
            continue
        if vals[i] == window.min() and (window > vals[i]).sum() >= k:
            out.append(i)
    return out
