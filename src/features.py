"""Station 2 return features and calendar-safe panel construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


RETURN_COLUMNS = ["ticker", "date", "return"]


def daily_returns(
    prices: pd.DataFrame,
    price_col: str = "adjClose",
    *,
    drop_initial_missing: bool = True,
) -> pd.DataFrame:
    """Calculate simple daily returns within each ticker's own calendar."""
    required = {"ticker", "date", price_col}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {sorted(missing)}")

    frame = prices[["ticker", "date", price_col]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("returns require unique ticker-date price observations")
    if (frame[price_col] <= 0).any():
        raise ValueError(f"{price_col} must be positive before computing returns")

    frame["return"] = frame.groupby("ticker", sort=False)[price_col].pct_change(
        fill_method=None
    )
    if drop_initial_missing:
        frame = frame.dropna(subset=["return"])
    return frame[RETURN_COLUMNS].reset_index(drop=True)


def returns_to_wide(returns: pd.DataFrame) -> pd.DataFrame:
    """Pivot a unique tidy return panel to one date row and one ticker column."""
    required = set(RETURN_COLUMNS)
    missing = required.difference(returns.columns)
    if missing:
        raise ValueError(f"returns is missing required columns: {sorted(missing)}")
    if returns.duplicated(["ticker", "date"]).any():
        raise ValueError("cannot pivot duplicate ticker-date returns")
    return (
        returns.pivot(index="date", columns="ticker", values="return")
        .sort_index()
        .rename_axis(columns=None)
    )


def combined_returns_panel(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Left-join already-computed crypto returns to the equity calendar."""
    equity_wide = returns_to_wide(equity_returns)
    crypto_wide = returns_to_wide(crypto_returns)
    overlap = set(equity_wide.columns).intersection(crypto_wide.columns)
    if overlap:
        raise ValueError(f"equity and crypto ticker names overlap: {sorted(overlap)}")

    combined = equity_wide.join(crypto_wide, how="left")
    combined.index.name = "date"
    return combined.reset_index()


def descriptive_stats_by_asset_class(
    equity_returns: pd.DataFrame, crypto_returns: pd.DataFrame
) -> pd.DataFrame:
    """Summarise daily returns separately for equities and crypto."""
    rows: list[dict[str, float | int | str]] = []
    for asset_class, frame in (
        ("Equity", equity_returns),
        ("Crypto", crypto_returns),
    ):
        if "return" not in frame.columns:
            raise ValueError(f"{asset_class} return frame has no 'return' column")
        values = pd.to_numeric(frame["return"], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"{asset_class} return frame has no valid observations")
        rows.append(
            {
                "asset_class": asset_class,
                "observations": int(values.size),
                "mean_daily_return": float(values.mean()),
                "daily_volatility": float(values.std(ddof=1)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "skewness": float(values.skew()),
                "excess_kurtosis": float(values.kurt()),
            }
        )
    return pd.DataFrame(rows)


def annualised_asset_summary(
    descriptive_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Add clearly labelled native-calendar annualised mean and volatility."""
    required = {"asset_class", "mean_daily_return", "daily_volatility"}
    missing = required.difference(descriptive_stats.columns)
    if missing:
        raise ValueError(f"descriptive_stats is missing columns: {sorted(missing)}")
    result = descriptive_stats.copy()
    periods = result["asset_class"].map({"Equity": 252, "Crypto": 365})
    if periods.isna().any():
        raise ValueError("annualisation supports only Equity and Crypto rows")
    result["native_calendar_days_per_year"] = periods.astype(int)
    result["annualised_mean_return"] = result["mean_daily_return"] * periods
    result["annualised_volatility"] = result["daily_volatility"] * np.sqrt(periods)
    return result


def align_headlines_to_trading_days(
    headlines: pd.DataFrame, equity_trading_dates: pd.Series | pd.Index
) -> pd.DataFrame:
    """Map each headline to the same or next observed equity trading day."""
    required = {"date", "ticker", "sector", "title"}
    missing = required.difference(headlines.columns)
    if missing:
        raise ValueError(f"headlines is missing required columns: {sorted(missing)}")

    frame = headlines.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    calendar = pd.DatetimeIndex(pd.to_datetime(equity_trading_dates)).normalize()
    calendar = calendar.drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("equity trading calendar is empty")

    positions = calendar.searchsorted(frame["date"], side="left")
    valid = positions < len(calendar)
    aligned = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
    aligned[valid] = calendar.to_numpy()[positions[valid]]
    frame["trading_date"] = pd.to_datetime(aligned)
    frame["alignment_days"] = (frame["trading_date"] - frame["date"]).dt.days
    return frame


def assemble_headline_panel(
    headlines: pd.DataFrame, equity_trading_dates: pd.Series | pd.Index
) -> pd.DataFrame:
    """Build a ticker-sector daily text panel without sentiment scoring."""
    aligned = align_headlines_to_trading_days(headlines, equity_trading_dates)
    aligned = aligned.dropna(subset=["trading_date"]).copy()
    panel = (
        aligned.groupby(["trading_date", "ticker", "sector"], as_index=False)
        .agg(
            headline_count=("title", "size"),
            combined_headlines=("title", lambda values: " || ".join(values.astype(str))),
            mean_alignment_days=("alignment_days", "mean"),
        )
        .sort_values(["trading_date", "ticker"])
        .reset_index(drop=True)
    )
    return panel
