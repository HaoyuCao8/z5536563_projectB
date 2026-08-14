"""Station 1 ETL and reproducible data-integrity checks.

All source data is loaded through :mod:`src.data_access`. Cleaning functions can
optionally return a one-row audit dictionary suitable for later report tables.
Statistical return outliers are reported but retained.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src import data_access


SAMPLE_START = pd.Timestamp("2020-01-01")
SAMPLE_END = pd.Timestamp("2023-12-31")
PRICE_KEY = ["ticker", "date"]
NEWS_KEY = ["ticker", "date", "title"]
PRICE_COLUMNS = ["open", "high", "low", "close", "adjClose", "volume"]


def _normalise_date(series: pd.Series) -> pd.Series:
    """Convert tz-aware or tz-naive timestamps to naive midnight dates."""
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _clean_price_panel(
    raw: pd.DataFrame,
    *,
    name: str,
    expected_calendar: str,
    require_sector: bool,
    outlier_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = set(PRICE_KEY + PRICE_COLUMNS)
    if require_sector:
        required.add("sector")
    _require_columns(raw, required, name)

    df = raw.copy()
    rows_raw = len(df)
    df["date"] = _normalise_date(df["date"])
    invalid_dates = int(df["date"].isna().sum())
    outside_sample = int((~df["date"].between(SAMPLE_START, SAMPLE_END)).sum())
    df = df.loc[df["date"].between(SAMPLE_START, SAMPLE_END)].copy()

    exact_duplicates = int(df.duplicated(keep="first").sum())
    df = df.drop_duplicates().copy()
    conflicting_keys = int(df.duplicated(PRICE_KEY, keep=False).sum())
    if conflicting_keys:
        raise ValueError(
            f"{name} has {conflicting_keys} rows with conflicting ticker-date keys"
        )

    numeric_missing = int(df[PRICE_COLUMNS].isna().sum().sum())
    nonpositive_prices = int((df[["open", "high", "low", "close", "adjClose"]] <= 0).sum().sum())
    negative_volume = int((df["volume"] < 0).sum())
    invalid_ohlc = int(
        (
            (df["high"] < df[["open", "low", "close"]].max(axis=1))
            | (df["low"] > df[["open", "high", "close"]].min(axis=1))
        ).sum()
    )

    if require_sector:
        sector_mappings = df[["ticker", "sector"]].drop_duplicates()
        ambiguous_sector_tickers = int(
            (sector_mappings.groupby("ticker")["sector"].nunique() > 1).sum()
        )
    else:
        ambiguous_sector_tickers = 0

    observed_dates = pd.DatetimeIndex(df["date"].dropna().unique()).sort_values()
    if expected_calendar == "daily":
        expected_dates = pd.date_range(SAMPLE_START, SAMPLE_END, freq="D")
    elif expected_calendar == "observed_market":
        expected_dates = observed_dates
    else:
        raise ValueError("expected_calendar must be 'daily' or 'observed_market'")

    observed_pairs = pd.MultiIndex.from_frame(df[PRICE_KEY])
    expected_pairs = pd.MultiIndex.from_product(
        [df["ticker"].dropna().unique(), expected_dates], names=PRICE_KEY
    )
    missing_ticker_dates = int(len(expected_pairs.difference(observed_pairs)))

    ordered = df.sort_values(PRICE_KEY).copy()
    screening_returns = ordered.groupby("ticker", sort=False)["adjClose"].pct_change(fill_method=None)
    extreme_mask = screening_returns.abs().ge(outlier_threshold)
    extreme_returns = int(extreme_mask.sum())
    max_return = float(screening_returns.max())
    min_return = float(screening_returns.min())

    ordered = ordered.reset_index(drop=True)
    audit: dict[str, Any] = {
        "dataset": name,
        "rows_raw": rows_raw,
        "rows_clean": len(ordered),
        "start_date": ordered["date"].min().date().isoformat(),
        "end_date": ordered["date"].max().date().isoformat(),
        "tickers": int(ordered["ticker"].nunique()),
        "invalid_dates_removed": invalid_dates,
        "outside_sample_rows_removed": outside_sample,
        "exact_duplicate_rows_removed": exact_duplicates,
        "conflicting_key_rows": conflicting_keys,
        "missing_ticker_dates": missing_ticker_dates,
        "missing_numeric_values": numeric_missing,
        "nonpositive_price_values": nonpositive_prices,
        "negative_volume_rows": negative_volume,
        "invalid_ohlc_rows": invalid_ohlc,
        "ambiguous_sector_tickers": ambiguous_sector_tickers,
        "extreme_return_threshold": outlier_threshold,
        "extreme_return_rows_retained": extreme_returns,
        "minimum_daily_return": min_return,
        "maximum_daily_return": max_return,
    }
    return ordered, audit


def clean_equities(
    raw: pd.DataFrame, *, outlier_threshold: float = 0.20
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean equities and audit them against the shared market calendar."""
    return _clean_price_panel(
        raw,
        name="equity_prices",
        expected_calendar="observed_market",
        require_sector=True,
        outlier_threshold=outlier_threshold,
    )


def clean_crypto(
    raw: pd.DataFrame, *, outlier_threshold: float = 0.20
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean crypto, remove the 2024 rows, and audit its daily calendar."""
    return _clean_price_panel(
        raw,
        name="crypto_prices",
        expected_calendar="daily",
        require_sector=False,
        outlier_threshold=outlier_threshold,
    )


def clean_news(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalise and deduplicate headlines using ticker, date, and title."""
    required = {"date", "ticker", "sector", "title", "url", "publisher"}
    _require_columns(raw, required, "news_headlines")

    df = raw.copy()
    rows_raw = len(df)
    df["date"] = _normalise_date(df["date"])
    invalid_dates = int(df["date"].isna().sum())
    df = df.loc[df["date"].between(SAMPLE_START, SAMPLE_END)].copy()
    df["title"] = df["title"].astype("string").str.strip()
    blank_titles = int(df["title"].isna().sum() + df["title"].eq("").sum())

    duplicate_rows = int(df.duplicated(NEWS_KEY, keep="first").sum())
    df = df.drop_duplicates(NEWS_KEY, keep="first")
    df = df.sort_values(["date", "ticker", "title"]).reset_index(drop=True)

    sector_mappings = df[["ticker", "sector"]].drop_duplicates()
    ambiguous_sector_tickers = int(
        (sector_mappings.groupby("ticker")["sector"].nunique() > 1).sum()
    )
    publisher_missing = int(df["publisher"].isna().sum())
    publisher_blank = int(df["publisher"].astype("string").str.strip().eq("").sum())

    audit: dict[str, Any] = {
        "dataset": "news_headlines",
        "rows_raw": rows_raw,
        "rows_clean": len(df),
        "start_date": df["date"].min().date().isoformat(),
        "end_date": df["date"].max().date().isoformat(),
        "tickers": int(df["ticker"].nunique()),
        "sectors": int(df["sector"].nunique()),
        "invalid_dates_removed": invalid_dates,
        "duplicate_headlines_removed": duplicate_rows,
        "blank_titles_retained_for_audit": blank_titles,
        "missing_or_blank_publishers": publisher_missing + publisher_blank,
        "ambiguous_sector_tickers": ambiguous_sector_tickers,
    }
    return df, audit


def load_clean_equities(
    *, return_audit: bool = False, outlier_threshold: float = 0.20
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    clean, audit = clean_equities(
        data_access.load_equity_prices(), outlier_threshold=outlier_threshold
    )
    return (clean, audit) if return_audit else clean


def load_clean_crypto(
    *, return_audit: bool = False, outlier_threshold: float = 0.20
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    clean, audit = clean_crypto(
        data_access.load_crypto_prices(), outlier_threshold=outlier_threshold
    )
    return (clean, audit) if return_audit else clean


def load_clean_news(
    *, return_audit: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    clean, audit = clean_news(data_access.load_news_headlines())
    return (clean, audit) if return_audit else clean
