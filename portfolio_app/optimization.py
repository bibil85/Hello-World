"""Portfolio optimisation utilities using cvxpy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def optimise_portfolio(expected_returns: pd.Series,
                        cov: pd.DataFrame,
                        objective: str = 'sharpe',
                        max_weight: float = 1.0,
                        sector_limits: dict | None = None) -> pd.Series:
    """Optimize portfolio weights using :mod:`cvxpy`.

    Parameters
    ----------
    expected_returns : pandas.Series
        Expected annual returns for each asset.
    cov : pandas.DataFrame
        Covariance matrix of asset returns.
    objective : {'return', 'risk', 'sharpe'}
        Optimisation objective.
    max_weight : float, optional
        Maximum allocation per asset.
    sector_limits : dict, optional
        Mapping of sector name to maximum total weight for that sector.
    """
    import cvxpy as cp

    n = len(expected_returns)
    w = cp.Variable(n)
    ret = expected_returns.values @ w
    risk = cp.quad_form(w, cov.values)

    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]

    if sector_limits:
        for sector, limit in sector_limits.items():
            if sector in expected_returns.index.names:
                mask = expected_returns.index.get_level_values('sector') == sector
                constraints.append(cp.sum(w[mask]) <= limit)

    if objective == 'return':
        prob = cp.Problem(cp.Maximize(ret), constraints)
    elif objective == 'risk':
        prob = cp.Problem(cp.Minimize(risk), constraints)
    elif objective == 'sharpe':
        gamma = cp.Parameter(nonneg=True)
        prob = cp.Problem(cp.Maximize(ret - gamma * risk), constraints)
        gamma.value = 1
    else:
        raise ValueError("Unknown objective")

    prob.solve()
    return pd.Series(w.value, index=expected_returns.index, name='weight')
