# AI Workflow and Critical Reflection – z5536563

## Scope of AI assistance

I used Claude (Web + Claude Code) for coding, debugging, report drafting, and submission validation in Part B. It helped interpret the project brief, implement the portfolio optimisers and walk-forward backtest, design the FinVADER lexicon, build the sentiment fusion logic, develop the Streamlit app, and run the final validation checks. I remained responsible for choosing the fund families and methods, setting the estimation window and rebalance frequency, designing the coverage-reliability weighting scheme, deciding the tilt strength and caps, reviewing the economic interpretation, and confirming that the final app did not import `nltk`.

During the Week 10 revision, I used AI to review the lecture notes against my existing codebase, identify gaps (missing Fear & Greed Index, VADER diagnostic, Risk-Return scatter), implement the missing features, cross-check report numbers against code output, and fix bugs discovered during peer review (fusion equity-share preservation, equal-weight sector index alignment with rubric).

## Main prompts and outcomes

### 1. Part B scope and implementation order

I asked Claude to map the Part B requirements onto the Part A foundation. AI listed the new modules and a suggested build order. I rejected the sentiment-first approach because the backtest calendar must be fixed before the sentiment lag can be verified, and I reordered the build to backtests → sentiment → fusion → app.

### 2. Portfolio optimisers and backtest engine

I used Claude to draft the four baseline optimisers and the walk-forward loop in `src/portfolios.py`. I reviewed the code, added weight renormalisation for missing-asset days, added fallback to equal weight on solver failure, and confirmed the first live date was after the 504-day window. I also asked for the HRP innovation and checked the linkage tree against the original Lopez de Prado (2016) algorithm.

### 3. FinVADER and sector sentiment index

I used AI to propose finance-specific terms and sentiment scores for the VADER lexicon extension. I reviewed the list, added terms that were missing (for example "margin compression", "debt crisis"), removed terms that were too generic, and assigned scores based on financial context rather than accepting the AI's defaults blindly. I verified the diagnostic neutral-rate comparison and confirmed the 1-day lag and 5-day carry-forward were applied correctly.

### 4. Sentiment fusion and app development

I used Claude to draft the multiplicative tilt function and the Streamlit app. I set the tilt strength to 0.30 and the caps to 0.2x–2.0x after testing several values and observing that extreme tilts produced unstable weights. I designed the five-page app structure, confirmed that only `data_access` was imported from `src`, and added the disclaimer and colour palette myself.

### 5. Week 10 revision and feature completeness

After reviewing the Week 10 lecture notes, I asked AI to compare my outputs against the rubric and identify missing pieces. AI flagged that I lacked a Fear & Greed Index, a VADER vs FinVADER diagnostic, and a Risk-Return scatter plot. I directed AI to implement these in `src/sentiment.py` and `scripts/run_part_b.py`, then verified the new figures and tables were saved correctly. I also asked AI to cross-check every number in my pre-written report against the actual code output (`performance_metrics.csv`, `fusion_metrics.csv`, `vader_comparison.csv`), which uncovered several mismatches I then corrected.

### 6. Validation and submission checks

I asked AI to review the final folder structure and identify temporary files, cache folders, or leftover starter code. I removed the residual placeholder code from `streamlit_app.py`. I also asked AI to verify that `AGENTS.md` and `CLAUDE.md` were no longer identical starter stubs and that the prompt logs were complete.

## Errors and corrections

I did not accept Claude suggestions without checking them.

First, the initial backtest did not handle missing asset prices on partial trading days. I identified this by inspecting a Combined fund rebalance date where one crypto had no price; the portfolio return was zero instead of being computed from available assets. I added `_renormalise_weights_for_available` to fix this.

Second, the HRP implementation triggered a `ClusterWarning` and an `IndexError` due to numeric index asymmetry and invalid leaf positions. I set `checks=False` in `squareform` and filtered the quasi-diagonal sort to valid indices only.

Third, an early Streamlit draft imported `src/sentiment`, which would have caused an `nltk` import on deployment. I caught this during a manual code review of `streamlit_app.py` and removed the import before testing locally.

Fourth, the initial sentiment fusion produced weights that were not always positive after aggressive tilting. I added `np.clip(tilt, min_tilt, max_tilt)` and `np.maximum(new_equity_weights, 0)` to enforce positivity, then renormalised.

Fifth, during the Week 10 review I discovered that the fusion renormalised the equity subset to sum to 100% of the fund, diluting the crypto allocation in Combined funds. I added `original_equity_sum` preservation so the tilt only changes equity internal weights, not the equity-vs-crypto mix.

Sixth, the rubric explicitly requires "equal-weight the tickers" for the sector sentiment index, but my code had defaulted to credibility-weighted aggregation. I switched the official `sector_sentiment_index.csv` to equal-weight and saved the credibility-weighted version separately as an extended innovation.

Seventh, AI initially estimated FinVADER's neutral-rate improvement as "50% to 35%" in the report, which did not match the actual computed rate of 24.1% to 23.0%. I caught this during a report-vs-code cross-check and updated the prose and figures.

These corrections show why running the full build, inspecting intermediate outputs, manually reviewing generated code, and cross-checking report claims against data were necessary rather than accepting AI output directly.
