"""Streamlit dashboard for portfolio analytics."""

from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from portfolio_app import data_ingestion, analytics, technical, optimization, visualization


st.title("Portfolio Analytics Dashboard")

uploaded_file = st.file_uploader("Upload portfolio Excel", type=["xls", "xlsx"])
if uploaded_file:
    portfolio = data_ingestion.load_portfolio_from_excel(io.BytesIO(uploaded_file.read()))
    st.write("Holdings", portfolio)

    tickers = portfolio['Ticker'].tolist()
    start = st.date_input("Start Date")
    end = st.date_input("End Date")
    if start and end:
        prices = data_ingestion.fetch_price_history(tickers, start, end)
        st.line_chart(prices)

        returns = prices.pct_change().dropna()
        metrics = analytics.compute_performance_metrics(returns.mean(axis=1))
        st.write("Performance", metrics)
        risks = analytics.compute_risk_metrics(returns.mean(axis=1))
        st.write("Risk", risks)

        indicators = technical.calculate_indicators(prices.iloc[:, 0])
        signals = technical.generate_signals(indicators)
        st.write("Signals", signals.tail())

        objective = st.selectbox("Optimization objective", ['return', 'risk', 'sharpe'])
        exp_ret = returns.mean() * 252
        cov = returns.cov() * 252
        weights = optimization.optimise_portfolio(exp_ret, cov, objective)
        st.write("Optimized Weights", weights)

        if st.button("Plot Allocation"):
            visualization.plot_allocation(weights)
            st.pyplot()
