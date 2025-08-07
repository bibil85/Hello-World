# Portfolio Analytics Application

This repository provides a reference implementation of a professional-grade
Python toolkit for financial portfolio analytics, optimisation and technical
analysis. Key features include:

* **Data ingestion** – load holdings from an Excel file and retrieve
  historical market data via Bloomberg using the `xbbg` library.
* **Analytics** – calculate performance (Sharpe, alpha, beta), risk metrics
  (VaR, CVaR, volatility) and basic asset allocation breakdowns.
* **Backtesting** – run simple backtests using `bt`, `vectorbt` or
  `backtrader`.
* **Technical analysis** – compute RSI, MACD, Bollinger Bands and moving
  averages using `TA-Lib` and generate naïve trading signals.
* **Optimisation** – optimise portfolio weights with multiple objectives
  (maximise return, minimise risk, maximise Sharpe ratio) using `cvxpy`.
* **Visualisation** – produce performance charts, risk/return scatter plots,
  allocation pie charts and technical indicator overlays with `matplotlib`.
* **Interactive dashboard** – a `Streamlit` application (`dashboard.py`) allows
  uploading of Excel portfolios, selection of indicators and optimisation
  objectives, viewing of analytics and charts and exporting of results.

## Getting started

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dashboard guides you through uploading a portfolio, fetching data,
performing analytics and viewing optimisation results.
