"""Station 3B — Sentiment model and sector index from news headlines.

Innovations:
    * FinVADER lexicon extension — finance-specific terms reduce false-neutrals
    * Credibility-weighted sentiment — leverages Part A coverage-reliability scores
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensure_vader_lexicon() -> None:
    """One-time NLTK download (run during build, never in the deployed app)."""
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)


def build_fin_vader_lexicon() -> dict[str, float]:
    """Finance-specific lexicon extension to reduce VADER false-neutrals.

    Note on tokenisation: VADER matches lexicon entries by single token after
    splitting on whitespace and punctuation. Multi-word phrases such as
    "strong buy" or "bear market" are stored here for completeness, but they
    will only register if the text happens to preserve them as a single token.
    The bulk of the FinVADER improvement comes from the ~30 single-word
    finance terms (e.g. surge, plunge, bullish, bearish) which are matched
    reliably.
    """
    return {
        "surge": 2.5,
        "soar": 2.5,
        "rally": 2.0,
        "bullish": 2.5,
        "upgrade": 2.0,
        "upgrades": 2.0,
        "beat estimates": 2.0,
        "beats estimates": 2.0,
        "earnings surprise": 2.0,
        "revenue beat": 2.0,
        "guidance raise": 2.5,
        "outperform": 2.0,
        "overweight": 1.5,
        "strong buy": 3.0,
        "dividend raise": 2.0,
        "margin expansion": 2.0,
        "expansion": 1.5,
        "growth acceleration": 2.0,
        "record high": 2.0,
        "all-time high": 2.0,
        "breakout": 1.5,
        "plunge": -2.5,
        "tumble": -2.5,
        "crash": -3.5,
        "sell-off": -2.0,
        "bearish": -2.5,
        "downgrade": -2.0,
        "downgrades": -2.0,
        "miss expectations": -2.0,
        "misses expectations": -2.0,
        "revenue miss": -2.0,
        "guidance cut": -2.5,
        "profit warning": -2.5,
        "underperform": -2.0,
        "underweight": -1.5,
        "strong sell": -3.0,
        "dividend cut": -2.0,
        "margin compression": -2.0,
        "recession": -2.5,
        "contraction": -1.5,
        "bankruptcy": -3.5,
        "default": -3.0,
        "debt crisis": -3.0,
        "liquidity crisis": -3.0,
        "layoff": -1.5,
        "layoffs": -1.5,
        "job cuts": -1.5,
        "correction": -1.5,
        "correction territory": -1.5,
        "52-week low": -1.5,
        "short squeeze": 1.5,
        "merger": 1.0,
        "acquisition": 1.0,
        "takeover": 1.0,
        "buyback": 1.0,
        "share buyback": 1.0,
        "write-down": -1.5,
        "impairment": -1.5,
        "sec investigation": -2.0,
        "lawsuit": -1.5,
        "litigation": -1.5,
        "settlement": 0.5,
    }


def score_headlines(
    panel: pd.DataFrame,
    *,
    use_fin_vader: bool = True,
    text_col: str = "combined_headlines",
) -> pd.DataFrame:
    """Apply VADER (optionally FinVADER) to each ticker-day headline bundle."""
    from nltk.sentiment import SentimentIntensityAnalyzer

    ensure_vader_lexicon()
    analyzer = SentimentIntensityAnalyzer()

    if use_fin_vader:
        analyzer.lexicon.update(build_fin_vader_lexicon())

    records: list[dict] = []
    for _, row in panel.iterrows():
        text = str(row.get(text_col, ""))
        vs = analyzer.polarity_scores(text)
        records.append(
            {
                "trading_date": row["trading_date"],
                "ticker": row["ticker"],
                "sector": row["sector"],
                "headline_count": row.get("headline_count", 1),
                "compound": vs["compound"],
                "pos": vs["pos"],
                "neg": vs["neg"],
                "neu": vs["neu"],
            }
        )
    return pd.DataFrame(records)


def sector_sentiment_index(
    scores: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    use_weights: bool = False,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build a daily equal-weight sector sentiment index.

    Treatment of no-news days:
        * Carry-forward the last observed sentiment up to 5 days
        * Beyond 5 days, revert to neutral (0.0)
    """
    if use_weights and "weight" in scores.columns and "weighted_compound" in scores.columns:
        grouped = (
            scores.groupby(["trading_date", "sector"])
            .agg(total_weight=("weight", "sum"), weighted_sum=("weighted_compound", "sum"))
            .reset_index()
        )
        grouped["sentiment"] = grouped["weighted_sum"] / grouped["total_weight"]
        sector_daily = grouped[["trading_date", "sector", "sentiment"]]
    else:
        sector_daily = (
            scores.groupby(["trading_date", "sector"])["compound"]
            .mean()
            .reset_index(name="sentiment")
        )

    all_sectors = sector_daily["sector"].dropna().unique()
    full_index = pd.MultiIndex.from_product(
        [trading_dates, all_sectors], names=["trading_date", "sector"]
    )
    panel = (
        sector_daily.set_index(["trading_date", "sector"])
        .reindex(full_index)
        .reset_index()
    )

    panel["sentiment"] = panel.groupby("sector")["sentiment"].transform(
        lambda s: s.ffill(limit=5).fillna(0.0)
    )
    panel[f"sentiment_lag{lag_days}"] = panel.groupby("sector")["sentiment"].shift(lag_days)

    return panel


def fear_greed_index(
    scores: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    *,
    window: int = 21,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Build a market-wide fear & greed index from ticker-day sentiment.

    Steps (per lecture):
        1. Equal-weight average across all tickers per day.
        2. Rescale compound (-1, +1) onto 0-100.
        3. Standardise with rolling mean/std over ``window`` days.
    """
    daily = (
        scores.groupby("trading_date")["compound"]
        .mean()
        .reindex(trading_dates)
        .fillna(0.0)
    )
    # Rescale to 0-100
    fg = (daily + 1.0) * 50.0
    # Rolling standardisation (expanding until window, then rolling)
    roll_mean = fg.rolling(window=window, min_periods=5).mean()
    roll_std = fg.rolling(window=window, min_periods=5).std()
    z = (fg - roll_mean) / roll_std.replace(0, np.nan)
    z = z.fillna(0.0)

    df = pd.DataFrame(
        {
            "trading_date": trading_dates,
            "fear_greed_raw": fg.values,
            "fear_greed_z": z.values,
        }
    )
    df["fear_greed_z_lag"] = df["fear_greed_z"].shift(lag_days)
    return df


def compare_vader_vs_fin_vader(headline_panel: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic: compare native VADER neutral rate against FinVADER."""
    scores_vader = score_headlines(headline_panel, use_fin_vader=False)
    scores_fin = score_headlines(headline_panel, use_fin_vader=True)

    def _neutral_rate(df):
        return (df["compound"].abs() < 0.05).mean()

    def _mean_compound(df):
        return df["compound"].mean()

    return pd.DataFrame(
        [
            {"model": "VADER", "neutral_rate": _neutral_rate(scores_vader), "mean_compound": _mean_compound(scores_vader)},
            {"model": "FinVADER", "neutral_rate": _neutral_rate(scores_fin), "mean_compound": _mean_compound(scores_fin)},
        ]
    )
