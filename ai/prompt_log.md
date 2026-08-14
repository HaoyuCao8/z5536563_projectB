# AI Prompt and Verification Log - z5536563

## Entry 1 - Part B scope and transition from Part A

## What I wanted

I wanted to understand the remaining Part B requirements for Stations 3 and 4 and map them onto the Part A data foundation I had already built.

## Prompt(s)

"Review the Part B starter folder and explain the remaining work for Stations 3A, 3B, 3C, and 4. Identify which Part A outputs are reused and which new files must be created."

## What the assistant produced

Codex summarised the required optimisers, backtest rules, sentiment model, fusion approach, app pages, and deployment steps. It listed the new files (`src/portfolios.py`, `src/sentiment.py`, `src/fusion.py`, `streamlit_app.py`) and confirmed that `src/etl.py`, `src/features.py`, and `src/data_access.py` are reused unchanged.

## What was wrong or risky

The suggested order began with sentiment scoring before the backtest engine. Building the sentiment model first risked creating an index that was not aligned with the same trading calendar used by the portfolio weights, and it could have encouraged look-ahead if the lag logic was not checked against the rebalance dates.

## What I changed and why

I reordered the build: backtests first, then sentiment, then fusion, then the app. I also confirmed that the estimation window (504 days), rebalance frequency (21 days), and annualisation factor (252 for equity, 365 for crypto) were stated explicitly in the code constants so they could not drift. I checked the plan against `PROJECT_BRIEF.md` and `context/DATA_GUIDE.md` before writing any new module.

---

## Entry 2 - Portfolio optimisers and walk-forward backtest

## What I wanted

I wanted reusable optimiser functions and a walk-forward backtest engine that respected the no-look-ahead rule.

## Prompt(s)

"Write optimiser functions for Equal Weight, Minimum Variance, Maximum Sharpe, and Risk Parity, plus a walk-forward backtest that forms weights only from past data inside a 504-day window and rebalances every 21 trading days."

## What the assistant produced

Codex provided `optimize_equal_weight`, `optimize_min_variance`, `optimize_max_sharpe`, and `optimize_risk_parity` using `scipy.optimize.minimize`, plus a `walk_forward_backtest` loop. The loop extracted `returns_wide.iloc[start:end]` for estimation and `returns_wide.iloc[i:i+rebalance]` for the out-of-sample period.

## What was wrong or risky

The backtest loop used `range(window, len(returns_wide), rebalance)`, which is correct, but the initial draft did not renormalise weights on days where some assets had missing returns. Missing crypto prices on equity holidays would have produced zero portfolio returns instead of rescaling to available assets. There was also a risk that failed optimiser runs could return negative weights or weights that did not sum to one.

## What I changed and why

I added `_renormalise_weights_for_available` to rescale weights to the subset of assets that trade on each day. I added fallback logic so that failed optimiser runs default back to equal weight instead of returning invalid weights. I ran the backtest on the official data, checked that the first live date was after the 504-day window, and confirmed that weights changed across methods rather than silently stalling at the initial guess.

---

## Entry 3 - Hierarchical Risk Parity (HRP) innovation

## What I wanted

I wanted a fifth optimisation method that did not invert the covariance matrix, as an innovation beyond the standard mean-variance and risk-parity baseline.

## Prompt(s)

"Implement Hierarchical Risk Parity using `scipy.cluster.hierarchy` on a correlation distance matrix, then apply recursive bisection to allocate weights. The function should accept the same `pd.DataFrame` of returns as the other optimisers and return a `pd.Series` of weights."

## What the assistant produced

Codex provided `optimize_hrp` with `hierarchy.linkage` on `squareform(dist)`, `_get_quasi_diag` for sorting items according to the linkage tree, and `_get_rec_bisection` for splitting clusters and allocating variance-based weights. The code followed Lopez de Prado (2016) closely.

## What was wrong or risky

The initial implementation triggered a `ClusterWarning` because `squareform` expects a symmetric distance matrix but the condensed form produced by `1 - corr` can have tiny numerical asymmetries. More importantly, the quasi-diagonal sort returned numeric indices that were not always valid column positions, which caused an `IndexError` when the linkage tree produced more leaves than tickers.

## What I changed and why

I set `checks=False` in `squareform` to suppress the numerical warning, since the distance matrix is derived from correlation and the asymmetry is at machine-epsilon level. I filtered the sorted indices to ensure only valid ticker positions were kept, then reindexed the final weights back to the original ticker columns. I compared HRP weights against equal-weight and risk-parity weights for the same estimation window and confirmed they were different and positively constrained. I documented HRP as an innovation in the report.

---

## Entry 4 - FinVADER sentiment model and sector index

## What I wanted

I wanted to extend VADER with finance-specific terms to reduce false-neutrals on financial headlines, then build a sector sentiment index.

## Prompt(s)

"Extend VADER with a finance lexicon of about 50 terms (e.g., surge, plunge, bullish, bearish, guidance raise, guidance cut) with appropriate sentiment scores. Then score the assembled headline panel, weight each ticker-day score by headline count and a sector coverage-reliability score, and build a daily sector-level index with a 1-day lag and a 5-day carry-forward for no-news days."

## What the assistant produced

Codex provided `build_fin_vader_lexicon` with ~50 terms, `score_headlines` that applied `SentimentIntensityAnalyzer` with the updated lexicon, `credibility_weighted_sentiment` that merged coverage scores and computed weighted compounds, and `sector_sentiment_index` that aggregated to sector-day level, forward-filled up to 5 days, and lagged by 1 trading day.

## What was wrong or risky

The initial sentiment scoring iterated over `panel.iterrows()`, which is slow but acceptable for the sample size. A more serious risk was that the NLTK `vader_lexicon` download would fail silently on a fresh machine, and the `SentimentIntensityAnalyzer` constructor would raise a `LookupError`. There was also a risk that the coverage-reliability merge could introduce look-ahead if the coverage scores were computed over the full sample rather than the estimation window; however, coverage is a static sector property in this design, so it is safe.

## What I changed and why

I added `ensure_vader_lexicon` with a one-time `nltk.download('vader_lexicon', quiet=True)` wrapped in a try/except, so the build script fails gracefully with a clear message if the network is unavailable. I verified that the coverage-reliability score is computed from Part A's static sector-level metrics and does not leak future information. I ran a diagnostic comparison of VADER vs FinVADER neutral rates and confirmed that FinVADER reduced the neutral rate, meaning fewer false-neutrals. I also confirmed that the lag is applied after the forward-fill, so day t's decision uses only sentiment from day t-1 or earlier.

---

## Entry 5 - Sentiment fusion into equity weights

## What I wanted

I wanted to tilt the equity weights of a Combined Maximum-Sharpe fund based on the lagged sector sentiment signal, and measure the before-vs-after effect honestly.

## Prompt(s)

"Write a function that takes the base weights of a Combined Max-Sharpe fund, maps each equity ticker to its sector, looks up the lagged sector sentiment z-score for each rebalance date, and applies a multiplicative tilt capped between 0.2x and 2.0x. Then compute the daily returns of the tilted fund and compare its Sharpe ratio to the base fund."

## What the assistant produced

Codex provided `apply_sentiment_tilt` with a daily loop over rebalance dates, sector-to-sentiment lookup, multiplicative tilt `1 + tilt_strength * z_score`, clipping, and renormalisation. It also provided `sentiment_augmented_returns` to compute the daily portfolio returns from the tilted weights.

## What was wrong or risky

The initial tilt function renormalised equity weights within the equity subset but did not preserve the original equity-vs-crypto allocation of the Combined fund. If the sentiment tilt concentrated equity weights, the total equity share of the Combined fund would still sum to its original level only if the renormalisation was applied correctly. A more subtle risk was that the sentiment signal could be missing for some dates or sectors, and the code needed to default to neutral (z = 0) without breaking.

## What I changed and why

I confirmed that the tilt is applied only to equity tickers; crypto weights are left untouched, and the equity subset is renormalised to sum to its original share within each row. I added a `KeyError` catch in the sentiment lookup so missing sectors default to a neutral tilt of 1.0. I computed the before-vs-after metrics table and confirmed that the sentiment-augmented fund did not dramatically underperform or outperform; the result was modest and mixed, which I reported honestly in the report rather than overstating the fusion value.

---

## Entry 6 - Streamlit app and investor journey

## What I wanted

I wanted a multi-page Streamlit app that reads precomputed results, supports the full investor journey (compare funds, read fact sheets, set an allocation, explore sentiment), and does not import `nltk`.

## Prompt(s)

"Build a Streamlit app with five pages: Home, Fund Comparison, Fact Sheets, Allocation, and Sentiment. The app must load precomputed CSVs from `results/data/` and `results/tables/`, use `st.cache_data`, and must not import `nltk`. Include a disclaimer and a coherent colour palette."

## What the assistant produced

Codex provided `streamlit_app.py` with a sidebar radio navigator, cached data loaders, metric cards, growth-of-$1 charts, drawdown charts, a holdings table, an allocation slider page, and a sentiment time-series with a heatmap. The app used the same colour palette as the figures produced by `run_part_b.py`.

## What was wrong or risky

An early draft accidentally included `from src import sentiment` in the app, which would have triggered an `nltk` import on the deployed app and caused a cold-start failure on Streamlit Community Cloud. Another risk was that the app would crash if the precomputed CSVs were missing, so graceful error handling was needed.

## What I changed and why

I removed every import from `src` except `data_access` (which loads the sector map and does not import `nltk`). I added explicit `st.error` messages with instructions to run `python scripts/run_part_b.py` when the precomputed files are missing. I tested the app locally with `streamlit run streamlit_app.py`, confirmed that all five pages load without errors, and checked that the `nltk` string does not appear in the app file using `grep -n "nltk" streamlit_app.py`.

---

## Entry 7 - Week 10 revision: missing exhibits and features

## What I wanted

After reviewing the Week 10 lecture notes, I wanted to compare my codebase against the rubric and identify missing pieces before submission.

## Prompt(s)

"Read the Week 10 lecture notes and compare them to my current codebase. Identify any missing features, figures, or outputs that the rubric expects but I have not built yet. Then implement them."

## What the assistant produced

AI identified three gaps: (1) no Fear & Greed Index (market-wide standardised sentiment), (2) no VADER vs FinVADER diagnostic comparison, and (3) no Risk-Return scatter plot. It also flagged that the crypto annualisation should use 365 days per the rubric, not 252. AI then implemented `fear_greed_index()` in `src/sentiment.py`, `compare_vader_vs_fin_vader()` for the diagnostic, and new chart functions `save_fear_greed`, `save_vader_comparison`, and `save_risk_return_scatter` in `scripts/run_part_b.py`.

## What was wrong or risky

The initial Fear & Greed implementation used full-sample standardisation, which would leak future mean and variance into each day's z-score. That is look-ahead bias. Also, the VADER comparison function ran scoring twice (once with FinVADER, once without), doubling the NLTK load time.

## What I changed and why

I changed the Fear & Greed Index to use a 21-day **rolling** standardisation (subtract rolling mean, divide by rolling std), with `min_periods=5` to handle the start of the sample. This ensures each day's z-score uses only past information. I accepted the VADER comparison as a one-off diagnostic run during `run_part_b.py`, not during app runtime, so the NLTK load is acceptable. I verified all new CSVs and figures were saved to the correct paths and that `check_handin.py` still passes.

---

## Entry 8 - Report cross-check and numerical corrections

## What I wanted

My report was drafted before the Week 10 code improvements. I needed to verify every number in the report matched the actual code output.

## Prompt(s)

"Cross-check every table and statistic in my report against the CSV outputs: `performance_metrics.csv`, `fusion_metrics.csv`, and `vader_comparison.csv`. Flag any mismatches, outdated descriptions, or missing exhibits."

## What the assistant produced

AI found five discrepancies: (1) Equity Max-Sharpe Sharpe ratio missing negative sign (0.138 instead of -0.138); (2) Crypto annualisation not mentioned in methodology section; (3) Missing Fear & Greed Index section entirely.

## What was wrong or risky

Some of the AI's suggested replacements were too aggressive — it proposed rewriting entire paragraphs rather than updating only the numbers. I needed to preserve my own economic interpretation while fixing the data.

## What I changed and why

I accepted the data corrections but rejected the prose rewrites. I updated Table 2, Table 3, and Section 1.3 myself, keeping my own wording. I added the new Fear & Greed Index section (3.4) and updated the Appendix figure list.

---

## Entry 9 - Fusion equity-share preservation bug fix

## What I wanted

During peer review, I was questioned whether the sentiment fusion correctly preserves the Combined fund's equity-vs-crypto allocation. I needed to verify and fix the logic.

## Prompt(s)

"Inspect `src/fusion.py` `apply_sentiment_tilt`. Does the renormalisation preserve the original equity share of the Combined fund, or does it renormalise equity weights to 100% of the fund? If the latter, fix it so equity internal weights are tilted but the total equity share stays constant."

## What the assistant produced

AI confirmed the bug: the code computed `new_equity_weights = new_equity_weights / new_equity_sum`, which renormalises the equity subset to 1.0 regardless of its original share. AI proposed saving `original_equity_sum = row[equity_cols].sum()` before tilting, then scaling the renormalised weights back by `* original_equity_sum`.

## What was wrong or risky

The fix was correct in principle, but I needed to verify it did not break the case where `original_equity_sum` is zero (for example, a fund with no equity allocation at all).

## What I changed and why

I added the `original_equity_sum` preservation with a guard: `if new_equity_sum > 1e-12` before scaling. I then ran a validation script that compared base Combined Max-Sharpe weights against sentiment-augmented weights for the same date, confirming equity sum was identical (0.7546) and crypto sum was identical (0.2454). I updated the report text to document this technical choice.

---

## Entry 10 - Sector sentiment index: equal-weight vs credibility-weighted

## What I wanted

I needed to align the sector sentiment index with the rubric requirement to "equal-weight the tickers".

## Prompt(s)

"The rubric says 'equal-weight the tickers' for the sector sentiment index, but my code uses `headline_count * coverage_reliability_score` weighting. Refactor so the official `sector_sentiment_index.csv` is equal-weight, and delete the credibility-weighted version."

## What the assistant produced

AI refactored `run_part_b.py` to call `sector_sentiment_index(scores, trading_dates, use_weights=False)` for the official output.

## What was wrong or risky

The figure caption in `save_sector_sentiment` still said "Credibility-weighted, equal-weight within sector", which was now misleading. Also, the Streamlit app's Sentiment page description mentioned credibility-weighting as the primary method.

## What I changed and why

I updated the figure caption to "Equal-weight tickers within sector; lagged 1 trading day." I updated the app description to clarify that the displayed sector index is equal-weight.

