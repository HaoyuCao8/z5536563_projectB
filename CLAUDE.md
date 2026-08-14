# CLAUDE.md — Claude-Specific Workflow for FINS3645 Project B

> Student: z5536563 | Tool: Claude (Web + Claude Code) | Part B

## 1. How I Use Claude

I use Claude as a **coding partner and technical reviewer**, not as a ghost-writer. I direct the agent with specific, bounded prompts; I read every output; I test before accepting; and I keep a log of what was wrong and how I fixed it.

- **Primary mode**: Web Claude for architecture decisions, report drafting, and debugging strategy.
- **Secondary mode**: Claude Code (IDE integration) for iterative file edits, multi-file refactors, and running tests.
- **What I do NOT delegate**: economic interpretation, choosing the app name (AtlasFlow), deciding tilt strength, or writing the final report prose.

## 2. Prompt Strategy

My prompts follow a **context-first, bounded-scope** pattern:

1. **State the goal** — what I want, not how to do it.
2. **Provide constraints** — no look-ahead, 504-day window, 21-day rebalance, etc.
3. **Reference existing code** — "reuse `src/features.py` from Part A", "follow the pattern in `src/portfolios.py`".
4. **Ask for testable output** — functions with docstrings, type hints, and a small smoke test.

Example prompt pattern:

```
"Write a function [X] that accepts [input types] and returns [output types].
Constraints: [list]. Reuse [existing module]. Include a docstring and type hints.
After writing, identify one likely bug and how to test for it."
```

## 3. Review Protocol

For every AI-generated file, I run this checklist before accepting:

- [ ] **No look-ahead** — dates and lags are directionally safe?
- [ ] **No silent failures** — solver failures, missing data, or empty DataFrames have explicit fallbacks?
- [ ] **No heavy imports in app** — `streamlit_app.py` does not import `nltk`, `scipy.optimize`, or backtest code?
- [ ] **Matches rubric** — crypto annualised on 365, equity on 252; equal-weight tickers in sector index; lag >= 1 day?
- [ ] **Reproducible** — `python scripts/run_part_b.py` runs end-to-end on a clean checkout?

## 4. Common AI Mistakes I Caught

| # | AI Mistake | How I Caught It | My Fix |
|---|-----------|----------------|--------|
| 1 | Backtest did not renormalise weights on partial-trading days | Inspected a rebalance date where one crypto had no price; return was zero | Added `_renormalise_weights_for_available` |
| 2 | HRP triggered `ClusterWarning` and `IndexError` | Ran backtest on full data; warning appeared, then crash | Set `checks=False` in `squareform`; filtered valid indices |
| 3 | `streamlit_app.py` imported `src/sentiment` | Manual `grep` for `"nltk"` in app file | Removed import; app only loads precomputed CSVs |
| 4 | Fusion renormalised equity weights to 100% of fund | Compared base vs tilted weights; crypto share dropped unexpectedly | Preserved `original_equity_sum` before renormalising equity subset |
| 5 | Sector index used credibility-weighting as default | Re-read rubric: "equal-weight the tickers" | Switched official index to equal-weight |

## 5. Agent Files I Maintained

- `AGENTS.md` (this folder) — general project instructions, shared across tools.
- `CLAUDE.md` (this file) — Claude-specific workflow, prompt patterns, and review protocol.
- `ai/AI_NOTES.md` — high-level narrative of how AI was directed and checked.
- `ai/prompt_log.md` — per-task log: prompt, AI output, what was wrong, what I changed.

## 6. Version History

- **v1.0** (Week 9): Initial agent instructions for baseline Part B build.
- **v1.1** (Week 10): Added week10 revision requirements — Fear & Greed Index, VADER diagnostic, Risk-Return scatter, crypto 365-day annualisation, equal-weight sector index, fusion equity-share preservation.
