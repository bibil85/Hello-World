"""Portfolio analytics and backtesting utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class PerformanceMetrics:
    """Container for common performance metrics."""

    sharpe: float
    alpha: float
    beta: float


@dataclass
class RiskMetrics:
    """Container for risk related metrics."""

    var: float
    cvar: float
    volatility: float


def compute_performance_metrics(portfolio_returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                risk_free_rate: float = 0.0,
                                periods_per_year: int = 252) -> PerformanceMetrics:
    """Compute Sharpe ratio, alpha and beta of a return series."""
    excess_returns = portfolio_returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

    if benchmark_returns is None:
        alpha = np.nan
        beta = np.nan
    else:
        cov = np.cov(portfolio_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1]
        alpha = (portfolio_returns.mean() - beta * benchmark_returns.mean()) * periods_per_year
    return PerformanceMetrics(sharpe=sharpe, alpha=alpha, beta=beta)


def compute_risk_metrics(portfolio_returns: pd.Series,
                         confidence: float = 0.95) -> RiskMetrics:
    """Compute VaR, CVaR and annualized volatility."""
    sorted_returns = np.sort(portfolio_returns)
    index = int((1 - confidence) * len(sorted_returns))
    var = -sorted_returns[index]
    cvar = -sorted_returns[:index].mean()
    volatility = np.sqrt(252) * portfolio_returns.std()
    return RiskMetrics(var=var, cvar=cvar, volatility=volatility)


def asset_allocation_breakdown(portfolio: pd.DataFrame) -> pd.Series:
    """Return weights of each asset based on market value."""
    value = portfolio['Shares'] * portfolio.get('Price', 1)
    weights = value / value.sum()
    return weights


def backtest_portfolio(prices: pd.DataFrame,
                       signals: Optional[pd.DataFrame] = None,
                       engine: str = 'bt'):
    """Backtest a portfolio using the specified engine.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price history of assets.
    signals : pandas.DataFrame, optional
        Trading signals indexed like ``prices``.
    engine : {'bt', 'vectorbt', 'backtrader'}
        Backend to use for the backtest. Only basic examples are provided.
    """
    engine = engine.lower()
    if engine == 'bt':
        try:
            import bt
        except ImportError as exc:  # pragma: no cover - library optional
            raise ImportError("bt library is required for this backtest") from exc
        strategy = bt.Strategy('portfolio', [bt.algos.RunOnDate(prices.index[0]),
                                             bt.algos.SelectAll(),
                                             bt.algos.WeighEqually(),
                                             bt.algos.Rebalance()])
        test = bt.Backtest(strategy, prices)
        return bt.run(test)
    elif engine == 'vectorbt':
        try:
            import vectorbt as vbt
        except ImportError as exc:  # pragma: no cover - library optional
            raise ImportError("vectorbt is required for this backtest") from exc
        pf = vbt.Portfolio.from_signals(prices, entries=signals > 0, exits=signals < 0)
        return pf.stats()
    elif engine == 'backtrader':
        try:
            import backtrader as bttr
        except ImportError as exc:  # pragma: no cover - library optional
            raise ImportError("backtrader is required for this backtest") from exc
        # Minimal backtrader example
        cerebro = bttr.Cerebro()
        data_feed = bttr.feeds.PandasData(dataname=prices)
        cerebro.adddata(data_feed)
        cerebro.addstrategy(bttr.Strategy)
        result = cerebro.run()
        return result
    else:
        raise ValueError(f"Unknown engine: {engine}")
