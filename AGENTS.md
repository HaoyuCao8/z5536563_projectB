# AGENTS.md — AI Agent Instructions for FINS3645 Project B

> Student: z5536563 | Part B: Systematic Multi-Asset Funds with News-Sentiment Analytics

## 1. Project Overview

This repository is my FINS3645 Part B submission. It builds on Part A's data foundation to create systematic multi-asset funds, a news-sentiment index, and a deployed Streamlit investment app named **AtlasFlow**.

- **Station 3A**: Out-of-sample portfolio backtests (Equal Weight, Min-Variance, Max-Sharpe, Risk Parity, HRP) across Equity-only, Crypto-only, and Combined fund families.
- **Station 3B**: Sentiment model using FinVADER (VADER + finance lexicon extension) and equal-weight sector sentiment index.
- **Station 3C**: Sentiment fusion — tilting equity internal weights based on lagged sector sentiment while preserving the equity-vs-crypto allocation.
- **Station 4**: Streamlit app deployed from a public GitHub repo.

Data source: hosted ZIP of Parquet files (equity prices, crypto prices, news headlines). Loaded via `src/data_access.py`.

## 2. Folder Layout

```
src/
  data_access.py   — PROVIDED, do not edit
  etl.py           — Part A ETL (reused)
  features.py      — Part A features (reused)
  portfolios.py    — NEW: optimisers + walk-forward backtest
  sentiment.py     — NEW: VADER/FinVADER + sector index + Fear & Greed
  fusion.py        — NEW: sentiment tilt (preserves equity/crypto share)
scripts/
  run_part_b.py    — Reproduces ALL results
  check_handin.py  — Pre-submission validation
streamlit_app.py   — App entrypoint (reads results/ only)
results/
  data/            — Precomputed CSVs for the app
  tables/          — Metrics, fusion comparison, VADER diagnostic
  figures/         — Growth, drawdown, sentiment, risk-return charts
```

## 3. Coding Conventions

- Python 3.12+, type hints, `from __future__ import annotations`
- pandas for tabular data, numpy for numerics, matplotlib for figures
- All optimisers return `pd.Series` indexed by ticker
- Backtest: walk-forward, **no look-ahead**, weights from past data only
- Annualisation: 252 trading days for Equity and Combined funds; 365 days for Crypto funds
- Rebalance: every 21 trading days; estimation window: 504 days

## 4. Critical Rules

- **NEVER** use look-ahead in backtests. Weights must be formed from estimation window only.
- **NEVER** strip casing/punctuation before VADER scoring.
- Sentiment signal must be **lagged >=1 trading day** before use in portfolio decisions.
- The deployed app **must NOT import nltk**. All sentiment scoring happens in `run_part_b.py`.
- Sentiment fusion must **preserve the original equity-vs-crypto allocation**; only equity internal weights are tilted.

## 5. Innovation Areas Implemented

1. **Hierarchical Risk Parity (HRP)**: `scipy.cluster.hierarchy` + recursive bisection. No covariance inversion.
2. **FinVADER Lexicon**: Extended VADER with ~50 finance terms (surge, plunge, bullish, bearish, etc.).
3. **Equal-weight tickers**: Official sector index uses equal-weight tickers per rubric.
4. **Fear & Greed Index**: Market-wide sentiment standardised with 21-day rolling z-score, aligned with known stress events.
5. **Risk-Return Scatter & VADER Diagnostic**: Additional exhibits beyond the required minimum.

## 6. How I Check AI Output

- Run `python scripts/run_part_b.py` after any code change to verify end-to-end build.
- Run `python scripts/check_handin.py` before submission.
- Verify no `nltk` import in `streamlit_app.py`.
- Visually inspect figures in `results/figures/`.
- Confirm all required output filenames exist:
  - `results/data/fund_returns.csv`
  - `results/data/fund_weights.csv`
  - `results/data/sector_sentiment_index.csv`
  - `results/data/fear_greed_index.csv`
  - `results/tables/performance_metrics.csv`
  - `results/tables/vader_comparison.csv`
