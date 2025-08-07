"""Plotting helpers for portfolio analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_performance(portfolio: pd.Series, benchmark: pd.Series | None = None):
    """Plot cumulative performance of the portfolio and optionally a benchmark."""
    cum = (1 + portfolio).cumprod()
    plt.figure(figsize=(10, 4))
    plt.plot(cum, label='Portfolio')
    if benchmark is not None:
        plt.plot((1 + benchmark).cumprod(), label='Benchmark')
    plt.legend()
    plt.title('Cumulative Performance')
    plt.tight_layout()


def plot_risk_return(portfolio_metrics: pd.DataFrame):
    """Scatter plot of risk vs return for a set of portfolios."""
    plt.figure(figsize=(6, 4))
    plt.scatter(portfolio_metrics['volatility'], portfolio_metrics['return'])
    for i, txt in enumerate(portfolio_metrics.index):
        plt.annotate(txt, (portfolio_metrics['volatility'][i], portfolio_metrics['return'][i]))
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Risk-Return Profile')
    plt.tight_layout()


def plot_allocation(weights: pd.Series):
    """Pie chart of portfolio allocation."""
    plt.figure(figsize=(6, 6))
    weights.plot.pie(autopct='%1.1f%%')
    plt.title('Asset Allocation')
    plt.ylabel('')
    plt.tight_layout()


def plot_technical(prices: pd.Series, indicators: pd.DataFrame):
    """Overlay technical indicators on price."""
    plt.figure(figsize=(10, 4))
    plt.plot(prices, label='Price')
    if 'sma' in indicators:
        plt.plot(indicators['sma'], label='SMA')
    if 'ema' in indicators:
        plt.plot(indicators['ema'], label='EMA')
    if 'bb_upper' in indicators:
        plt.fill_between(prices.index, indicators['bb_lower'], indicators['bb_upper'], color='grey', alpha=0.3)
    plt.legend()
    plt.title('Technical Indicators')
    plt.tight_layout()
