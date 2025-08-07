"""Data ingestion utilities for portfolio analytics."""

from __future__ import annotations

import pandas as pd


def load_portfolio_from_excel(path: str) -> pd.DataFrame:
    """Load portfolio holdings from an Excel file.

    The Excel file should contain at least two columns: ``Ticker`` and
    ``Shares``. Additional columns are ignored but preserved in the returned
    :class:`~pandas.DataFrame` for potential use (e.g. sector classification).
    """
    df = pd.read_excel(path)
    return df


def fetch_price_history(tickers, start, end):
    """Fetch historical price data using Bloomberg via :mod:`xbbg`.

    Parameters
    ----------
    tickers : sequence of str
        Bloomberg tickers to request.
    start : str or datetime-like
        Start date for data retrieval.
    end : str or datetime-like
        End date for data retrieval.

    Returns
    -------
    pandas.DataFrame
        Historical close prices indexed by date with tickers in columns.
    """
    try:
        from xbbg import blp
    except ImportError as exc:  # pragma: no cover - library optional
        raise ImportError("xbbg is required for Bloomberg data retrieval") from exc

    data = blp.bdh(tickers, "PX_LAST", start_date=start, end_date=end)
    # xbbg returns a multi-index; we only keep the price level
    if isinstance(data.columns, pd.MultiIndex):
        data = data.xs("PX_LAST", level=1, axis=1)
    return data
