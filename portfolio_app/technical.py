"""Technical analysis helpers using TA-Lib."""

from __future__ import annotations

import pandas as pd


def calculate_indicators(prices: pd.Series, indicators: list[str] | None = None) -> pd.DataFrame:
    """Calculate technical indicators using :mod:`TA-Lib`.

    Parameters
    ----------
    prices : pandas.Series
        Price series to analyse.
    indicators : list of str, optional
        Indicators to compute. Supported values: ``rsi``, ``macd``, ``bollinger``,
        ``sma`` and ``ema``. If ``None`` all are calculated.
    """
    try:
        import talib
    except ImportError as exc:  # pragma: no cover - library optional
        raise ImportError("TA-Lib is required for technical analysis") from exc

    if indicators is None:
        indicators = ['rsi', 'macd', 'bollinger', 'sma', 'ema']

    out = pd.DataFrame(index=prices.index)
    if 'rsi' in indicators:
        out['rsi'] = talib.RSI(prices)
    if 'macd' in indicators:
        macd, macdsignal, macdhist = talib.MACD(prices)
        out['macd'] = macd
        out['macdsignal'] = macdsignal
    if 'bollinger' in indicators:
        upper, middle, lower = talib.BBANDS(prices)
        out['bb_upper'] = upper
        out['bb_middle'] = middle
        out['bb_lower'] = lower
    if 'sma' in indicators:
        out['sma'] = talib.SMA(prices)
    if 'ema' in indicators:
        out['ema'] = talib.EMA(prices)
    return out


def generate_signals(indicators: pd.DataFrame) -> pd.Series:
    """Generate simplistic trading signals from indicators."""
    signal = pd.Series(0, index=indicators.index)
    if 'rsi' in indicators:
        signal[indicators['rsi'] < 30] = 1
        signal[indicators['rsi'] > 70] = -1
    return signal
