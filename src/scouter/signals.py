"""Confluence evaluation on the most recently closed 1H candle."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from scouter.indicators import cmf, cmf_bullish_divergence, efi, macd, rolling_z


@dataclass(frozen=True)
class Params:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_require_above_zero: bool = True
    # Bars back the bullish cross may have occurred and still count. MACD
    # crosses during the base, several bars before the volume impulse EFI
    # measures; demanding all three on one bar effectively never fires.
    macd_cross_window: int = 6

    cmf_length: int = 20
    efi_length: int = 13
    efi_spike_z: float = 1.5
    efi_z_lookback: int = 50

    divergence_lookback: int = 60
    divergence_pivot_k: int = 3
    divergence_price_tol: float = 0.02
    divergence_cmf_min_delta: float = 0.02

    min_candles: int = 45
    min_data_quality: float = 0.55


@dataclass
class Signal:
    ts: int
    price: float
    macd: float
    macd_signal: float
    bars_since_cross: int
    cmf: float
    efi: float
    efi_z: float
    macd_ok: bool
    cmf_ok: bool
    efi_ok: bool
    divergence: bool
    bars: int
    data_quality: float
    matched: bool
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def _f(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate(df: pd.DataFrame, p: Params,
             data_quality: float = 1.0) -> Signal | None:
    """Run the confluence check on closed candles, oldest -> newest.

    Returns None when history is too short to trust the indicators — a
    distinct outcome from "evaluated and did not match".
    """
    if df is None or len(df) < p.min_candles:
        return None

    m = macd(df["close"], p.macd_fast, p.macd_slow, p.macd_signal)
    c = cmf(df, p.cmf_length)
    e = efi(df, p.efi_length)
    ez = rolling_z(e["raw"], p.efi_z_lookback)

    macd_now, sig_now = _f(m["macd"].iloc[-1]), _f(m["signal"].iloc[-1])
    cmf_now = _f(c.iloc[-1])
    efi_now, efi_raw_now = _f(e["efi"].iloc[-1]), _f(e["raw"].iloc[-1])
    efi_z_now = _f(ez.iloc[-1])

    # 1. A recent bullish cross that is still in effect.
    cross_up = (m["macd"].shift(1) <= m["signal"].shift(1)) & (m["macd"] > m["signal"])
    window = max(1, p.macd_cross_window)
    recent = bool(cross_up.iloc[-window:].any())
    bars_since = -1
    if recent:
        hits = np.flatnonzero(cross_up.to_numpy())
        bars_since = int(len(cross_up) - 1 - hits[-1])
    macd_ok = bool(recent and macd_now > sig_now
                   and (macd_now > 0 if p.macd_require_above_zero else True))

    # 2. CMF positive, or bullish divergence against price.
    div = cmf_bullish_divergence(df, c, p.divergence_lookback,
                                 p.divergence_pivot_k,
                                 p.divergence_price_tol,
                                 p.divergence_cmf_min_delta)
    cmf_ok = bool((np.isfinite(cmf_now) and cmf_now > 0) or div["divergence"])

    # 3. EFI above zero and spiking against its own recent behaviour.
    efi_ok = bool(np.isfinite(efi_now) and efi_now > 0 and efi_raw_now > 0
                  and np.isfinite(efi_z_now) and efi_z_now >= p.efi_spike_z)

    matched = bool(macd_ok and cmf_ok and efi_ok
                   and data_quality >= p.min_data_quality)

    failed = [n for n, ok in (("macd", macd_ok), ("cmf", cmf_ok),
                              ("efi", efi_ok)) if not ok]
    if data_quality < p.min_data_quality:
        failed.append("sparse-data")

    return Signal(
        ts=int(df.index[-1].timestamp()), price=float(df["close"].iloc[-1]),
        macd=macd_now, macd_signal=sig_now, bars_since_cross=bars_since,
        cmf=cmf_now, efi=efi_now,
        efi_z=efi_z_now if np.isfinite(efi_z_now) else 0.0,
        macd_ok=macd_ok, cmf_ok=cmf_ok, efi_ok=efi_ok,
        divergence=bool(div["divergence"]), bars=len(df),
        data_quality=float(data_quality), matched=matched,
        reason="all confirmed" if matched else "failed: " + ",".join(failed))
