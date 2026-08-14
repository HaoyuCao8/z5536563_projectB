"""Station 3C — Fuse sentiment into the equity funds.

Look-ahead safe: all sentiment used is lagged by at least one trading day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def apply_sentiment_tilt(
    base_weights: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
    ticker_sector_map: pd.DataFrame,
    *,
    sentiment_col: str = "sentiment_lag1",
    tilt_strength: float = 0.30,
    max_tilt: float = 2.0,
    min_tilt: float = 0.2,
) -> pd.DataFrame:
    """Tilt equity weights toward high-sentiment sectors and away from low.

    Parameters
    ----------
    base_weights : DataFrame
        Rows indexed by date, columns are asset weights (only equity tickers
        will be modified).
    sector_sentiment : DataFrame
        Must contain ``trading_date``, ``sector``, and the lagged sentiment column.
    ticker_sector_map : DataFrame
        Columns ``ticker`` and ``sector``.
    tilt_strength : float
        Scaling factor for the sentiment signal (0 = no tilt, 1 = full tilt).
    max_tilt : float
        Maximum multiplicative tilt relative to base weight.
    min_tilt : float
        Minimum multiplicative tilt relative to base weight.

    Returns
    -------
    tilted_weights : DataFrame
        Same shape as ``base_weights``, re-normalised to sum to 1 per row.
    """
    weights = base_weights.copy()
    equity_tickers = set(ticker_sector_map["ticker"].unique())

    # Prepare sentiment lookup: date -> sector -> score
    sent = sector_sentiment[["trading_date", "sector", sentiment_col]].copy()
    sent = sent.dropna(subset=[sentiment_col])
    sent_lookup = sent.set_index(["trading_date", "sector"])[sentiment_col]

    # Normalise sentiment to [-1, 1] already by VADER, but rescale per day
    # to reduce cross-sectional concentration
    sent["sentiment_z"] = sent.groupby("trading_date")[sentiment_col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 1e-6 else 0.0
    )
    z_lookup = sent.set_index(["trading_date", "sector"])["sentiment_z"]

    for date in weights.index:
        row = weights.loc[date]
        equity_mask = [c in equity_tickers for c in weights.columns]
        equity_cols = weights.columns[equity_mask]

        if len(equity_cols) == 0:
            continue

        tilts = []
        for ticker in equity_cols:
            sector = ticker_sector_map.loc[
                ticker_sector_map["ticker"] == ticker, "sector"
            ]
            if sector.empty:
                tilts.append(1.0)
                continue
            sec = sector.iloc[0]
            try:
                z_score = z_lookup.loc[(date, sec)]
            except KeyError:
                z_score = 0.0

            # Multiplicative tilt: positive sentiment -> overweight
            tilt = 1.0 + tilt_strength * z_score
            tilt = np.clip(tilt, min_tilt, max_tilt)
            tilts.append(tilt)

        tilts = np.array(tilts)
        original_equity_sum = row[equity_cols].sum()
        new_equity_weights = row[equity_cols].values * tilts
        new_equity_weights = np.maximum(new_equity_weights, 0)
        new_equity_sum = new_equity_weights.sum()

        if new_equity_sum > 1e-12:
            new_equity_weights = new_equity_weights / new_equity_sum * original_equity_sum

        weights.loc[date, equity_cols] = new_equity_weights

    return weights


def sentiment_augmented_returns(
    base_weights: pd.DataFrame,
    returns_wide: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
    ticker_sector_map: pd.DataFrame,
    *,
    tilt_strength: float = 0.30,
) -> pd.Series:
    """Return the daily returns of a sentiment-tilted version of a base fund.

    This is a convenience wrapper that tilts the *same* base weights each
    rebalance date and then applies them day-by-day to the realised returns.
    """
    tilted = apply_sentiment_tilt(
        base_weights, sector_sentiment, ticker_sector_map, tilt_strength=tilt_strength
    )

    # Align dates
    common_dates = tilted.index.intersection(returns_wide.index)
    tilted = tilted.loc[common_dates]
    ret = returns_wide.loc[common_dates]

    port_rets = []
    for date in common_dates:
        w = tilted.loc[date]
        r = ret.loc[date]
        avail = r.dropna()
        if avail.empty:
            port_rets.append(0.0)
            continue
        w_day = w[avail.index]
        w_day = w_day / w_day.sum() if w_day.sum() > 1e-12 else w_day
        port_rets.append(float((avail * w_day).sum()))

    return pd.Series(port_rets, index=common_dates)
