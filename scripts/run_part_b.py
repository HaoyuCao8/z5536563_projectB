"""Reproduce all Part B results. Run from the project root:

    python scripts/run_part_b.py
"""
from __future__ import annotations

import pathlib
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import data_access, etl, features, portfolios, sentiment, fusion  # noqa: E402

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
DATA_DIR = ROOT / "results" / "data"
SAMPLE_LABEL = "2020-01-01 to 2023-12-31"

COLORS = {
    "navy": "#17324D",
    "blue": "#3D7EA6",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C8553D",
    "gray": "#5F6B73",
    "purple": "#7B68EE",
    "green": "#4CAF50",
}

METHOD_COLORS = {
    "equal_weight": COLORS["gray"],
    "min_variance": COLORS["blue"],
    "max_sharpe": COLORS["gold"],
    "risk_parity": COLORS["teal"],
    "hrp": COLORS["purple"],
    "max_sharpe_sentiment": COLORS["red"],
}

METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "hrp": "HRP",
    "max_sharpe_sentiment": "Max Sharpe + Sentiment",
}


def configure_output() -> None:
    for p in (TABLE_DIR, FIGURE_DIR, DATA_DIR):
        p.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.titlecolor": COLORS["navy"],
            "axes.labelcolor": COLORS["gray"],
            "xtick.color": COLORS["gray"],
            "ytick.color": COLORS["gray"],
        }
    )


def run_backtests(
    families: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run walk-forward backtests for every family x method pair."""
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity", "hrp"]
    all_returns: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []

    for family_name, returns_wide in families.items():
        for method in methods:
            print(f"  Backtesting {family_name} / {method} ...")
            rets, wts = portfolios.walk_forward_backtest(
                returns_wide, method, fund_name=family_name
            )
            if not rets.empty:
                all_returns.append(rets)
                all_weights.append(wts)

    if not all_returns:
        raise RuntimeError("No backtest results were produced.")

    return pd.concat(all_returns, ignore_index=True), pd.concat(
        all_weights, ignore_index=True
    )


def run_sentiment_fusion(
    fund_returns: pd.DataFrame,
    fund_weights: pd.DataFrame,
    combined_wide: pd.DataFrame,
    sector_sentiment: pd.DataFrame,
    ticker_sector_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a sentiment-augmented Combined Maximum-Sharpe fund."""
    # Extract base Combined Max-Sharpe weights
    mask = (fund_weights["fund"] == "Combined") & (fund_weights["method"] == "max_sharpe")
    base_w = fund_weights.loc[mask].copy()
    if base_w.empty:
        print("  Warning: no Combined Max-Sharpe weights found — skipping fusion.")
        return fund_returns, fund_weights

    base_w = base_w.drop(columns=["method", "fund"], errors="ignore")
    base_w = base_w.set_index("date").sort_index()

    # Compute tilted returns
    tilted_rets = fusion.sentiment_augmented_returns(
        base_w, combined_wide, sector_sentiment, ticker_sector_map, tilt_strength=0.30
    )

    if tilted_rets.empty:
        return fund_returns, fund_weights

    fusion_ret = pd.DataFrame(
        {
            "date": tilted_rets.index,
            "return": tilted_rets.values,
            "method": "max_sharpe_sentiment",
            "fund": "Combined",
        }
    )

    # Build tilted weights for saving (approximate: same shape as base)
    tilted_w = fusion.apply_sentiment_tilt(
        base_w, sector_sentiment, ticker_sector_map, tilt_strength=0.30
    )
    tilted_w = tilted_w.reset_index()
    tilted_w["method"] = "max_sharpe_sentiment"
    tilted_w["fund"] = "Combined"

    fund_returns = pd.concat([fund_returns, fusion_ret], ignore_index=True)
    fund_weights = pd.concat([fund_weights, tilted_w], ignore_index=True)
    return fund_returns, fund_weights


def compute_metrics(fund_returns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, method), group in fund_returns.groupby(["fund", "method"]):
        group = group.sort_values("date")
        # Crypto annualises on 365 days; equity and combined on 252 (equity calendar)
        periods = 365 if family == "Crypto" else 252
        m = portfolios.performance_metrics(group["return"].values, periods_per_year=periods)
        m["fund"] = family
        m["method"] = method
        rows.append(m)
    return pd.DataFrame(rows)


def save_growth_of_one(
    fund_returns: pd.DataFrame, filename: str = "growth_of_one_comparison.png"
) -> None:
    """Cumulative-return comparison across all funds and methods."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for (family, method), group in fund_returns.groupby(["fund", "method"]):
        group = group.sort_values("date")
        growth = portfolios.growth_of_one(group["return"].values)
        label = f"{family} — {METHOD_LABELS.get(method, method)}"
        color = METHOD_COLORS.get(method, COLORS["gray"])
        ax.plot(group["date"], growth, label=label, color=color, linewidth=1.3, alpha=0.85)

    ax.set(title="Growth of $1 — Out-of-Sample Backtest", xlabel="Date", ylabel="Portfolio value ($)")
    ax.legend(
        frameon=False, ncol=4, fontsize=7.5,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(1, color=COLORS["navy"], linewidth=0.6, linestyle="--")
    fig.text(
        0.01, 0.01,
        f"Walk-forward out-of-sample performance; estimation window {portfolios.ESTIMATION_WINDOW} observations.",
        fontsize=8, color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_drawdown(
    fund_returns: pd.DataFrame,
    family: str = "Combined",
    method: str = "max_sharpe",
    filename: str = "drawdown_example.png",
) -> None:
    mask = (fund_returns["fund"] == family) & (fund_returns["method"] == method)
    group = fund_returns.loc[mask].sort_values("date")
    if group.empty:
        return

    dd = portfolios.drawdown_series(group["return"].values)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(group["date"], dd * 100, 0, color=COLORS["red"], alpha=0.35)
    ax.plot(group["date"], dd * 100, color=COLORS["red"], linewidth=1.0)
    ax.set(title=f"Drawdown — {family} {METHOD_LABELS.get(method, method)}", xlabel="Date", ylabel="Drawdown (%)")
    ax.grid(axis="y", alpha=0.2)
    fig.text(
        0.01, 0.01,
        "Peak-to-trough decline over the out-of-sample period.",
        fontsize=8,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_weights_over_time(
    fund_weights: pd.DataFrame,
    family: str = "Combined",
    method: str = "max_sharpe",
    top_n: int = 8,
    filename: str = "weights_over_time.png",
) -> None:
    mask = (fund_weights["fund"] == family) & (fund_weights["method"] == method)
    w = fund_weights.loc[mask].copy()
    if w.empty:
        return

    w = w.drop(columns=["fund", "method"], errors="ignore")
    w = w.set_index("date").sort_index()
    mean_w = w.mean().sort_values(ascending=False)
    top_tickers = mean_w.head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6.0))
    palette = plt.cm.tab10(np.linspace(0, 1, len(top_tickers)))
    for ticker, color in zip(top_tickers, palette):
        ax.plot(w.index, w[ticker] * 100, label=ticker, color=color, linewidth=1.1)
    ax.set(title=f"Portfolio Weights Over Time — {family} {METHOD_LABELS.get(method, method)}", xlabel="Date", ylabel="Weight (%)")
    ax.legend(
        frameon=False, ncol=4, fontsize=8,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.01, "Combined-family out-of-sample portfolio weights by method.", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_sharpe_barplot(
    metrics: pd.DataFrame, filename: str = "sharpe_comparison.png"
) -> None:
    m = metrics.copy()
    m["label"] = m["fund"] + "\n" + m["method"].map(METHOD_LABELS)
    m = m.sort_values("sharpe_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [METHOD_COLORS.get(method, COLORS["gray"]) for method in m["method"]]
    ax.barh(m["label"], m["sharpe_ratio"], color=colors, alpha=0.85)
    ax.set(title="Sharpe Ratio by Fund and Method", xlabel="Sharpe ratio (rf = 0)")
    ax.axvline(0, color=COLORS["navy"], linewidth=0.8)
    ax.grid(axis="x", alpha=0.2)
    fig.text(
        0.01, 0.01,
        "Out-of-sample results; annualised using 252 days for Equity/Combined and 365 days for Crypto.",
        fontsize=8,
        color=COLORS["gray"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_sector_sentiment(
    sector_index: pd.DataFrame, filename: str = "sector_sentiment_timeseries.png"
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    sectors = sector_index["sector"].dropna().unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    for sector, color in zip(sorted(sectors), palette):
        sub = sector_index.loc[sector_index["sector"] == sector].sort_values("trading_date")
        ax.plot(sub["trading_date"], sub["sentiment"], label=sector, color=color, linewidth=1.0, alpha=0.8)

    ax.axhline(0, color=COLORS["navy"], linewidth=0.8, linestyle="--")
    ax.set(title="Sector Sentiment Index Over Time", xlabel="Date", ylabel="Sentiment score (FinVADER)")
    ax.legend(
        frameon=False, ncol=5, fontsize=7.5,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
    )
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.01, "Equal-weight tickers within sector; lagged 1 trading day.", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_fusion_comparison(
    fund_returns: pd.DataFrame, filename: str = "fusion_comparison.png"
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method, label, color in [
        ("max_sharpe", "Base Max-Sharpe", METHOD_COLORS["max_sharpe"]),
        ("max_sharpe_sentiment", "Sentiment-Augmented", METHOD_COLORS["max_sharpe_sentiment"]),
    ]:
        mask = (fund_returns["fund"] == "Combined") & (fund_returns["method"] == method)
        group = fund_returns.loc[mask].sort_values("date")
        if group.empty:
            continue
        growth = portfolios.growth_of_one(group["return"].values)
        ax.plot(group["date"], growth, label=label, color=color, linewidth=2.0)

    ax.set(title="Fusion Before-vs-After — Combined Maximum-Sharpe", xlabel="Date", ylabel="Portfolio value ($)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(1, color=COLORS["navy"], linewidth=0.6, linestyle="--")
    fig.text(0.01, 0.01, "Sentiment tilt applied with 1-day lag; no look-ahead.", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_risk_return_scatter(
    metrics: pd.DataFrame, filename: str = "risk_return_scatter.png"
) -> None:
    """Risk-return scatter: each dot is a fund."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    family_markers = {"Equity": "o", "Crypto": "s", "Combined": "D"}
    for family in metrics["fund"].unique():
        sub = metrics.loc[metrics["fund"] == family]
        ax.scatter(
            sub["annualised_volatility"] * 100,
            sub["annualised_return"] * 100,
            label=family,
            marker=family_markers.get(family, "o"),
            s=120,
            alpha=0.75,
            edgecolors=COLORS["navy"],
            linewidths=0.6,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                METHOD_LABELS.get(row["method"], row["method"]),
                (row["annualised_volatility"] * 100, row["annualised_return"] * 100),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=6.5,
                color=COLORS["gray"],
            )
    ax.axhline(0, color=COLORS["navy"], linewidth=0.6, linestyle="--")
    ax.set(title="Risk vs Return — Out-of-Sample Funds", xlabel="Annualised volatility (%)", ylabel="Annualised return (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)
    fig.text(0.01, 0.01, "Each point represents one fund using out-of-sample performance metrics.", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_weights_stacked_area(
    fund_weights: pd.DataFrame,
    ticker_sector_map: pd.DataFrame,
    family: str = "Combined",
    method: str = "max_sharpe",
    filename: str = "weights_over_time.png",
) -> None:
    """Sector-level stacked area chart of portfolio weights over time."""
    mask = (fund_weights["fund"] == family) & (fund_weights["method"] == method)
    w = fund_weights.loc[mask].copy()
    if w.empty:
        return
    w = w.drop(columns=["fund", "method"], errors="ignore")
    w = w.set_index("date").sort_index()

    # Map tickers to sectors; crypto shown as 'Crypto'
    sector_map = dict(zip(ticker_sector_map["ticker"], ticker_sector_map["sector"]))
    sector_weights = pd.DataFrame(index=w.index)
    for sector in sorted(set(sector_map.values())):
        cols = [c for c in w.columns if c in sector_map and sector_map[c] == sector]
        sector_weights[sector] = w[cols].sum(axis=1)
    # Any column not mapped (e.g. crypto tickers)
    unmapped = [c for c in w.columns if c not in sector_map]
    if unmapped:
        sector_weights["Crypto"] = w[unmapped].sum(axis=1)

    sector_weights = sector_weights.clip(lower=0)
    sector_weights = sector_weights.div(sector_weights.sum(axis=1), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(11, 6.0))
    palette = plt.cm.tab10(np.linspace(0, 1, len(sector_weights.columns)))
    ax.stackplot(
        sector_weights.index,
        *[sector_weights[col] * 100 for col in sector_weights.columns],
        labels=sector_weights.columns,
        colors=palette,
        alpha=0.85,
    )
    ax.set(title=f"Portfolio Weights Over Time — {family} {METHOD_LABELS.get(method, method)}", xlabel="Date", ylabel="Weight (%)")
    ax.legend(
        frameon=False, ncol=5, fontsize=7.5,
        loc="upper center", bbox_to_anchor=(0.5, -0.14),
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.01, f"Sector-level allocation; {SAMPLE_LABEL}.", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_weights_methods_comparison(
    fund_weights: pd.DataFrame,
    ticker_sector_map: pd.DataFrame,
    family: str = "Combined",
    filename: str = "weights_over_time.png",
) -> None:
    """Compare sector-level weights across all methods for one fund family."""
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity", "hrp"]
    n_methods = len(methods)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    sector_map = dict(zip(ticker_sector_map["ticker"], ticker_sector_map["sector"]))

    for idx, method in enumerate(methods):
        ax = axes[idx // ncols, idx % ncols]
        mask = (fund_weights["fund"] == family) & (fund_weights["method"] == method)
        w = fund_weights.loc[mask].copy()
        if w.empty:
            ax.set_visible(False)
            continue
        w = w.drop(columns=["fund", "method"], errors="ignore")
        w = w.set_index("date").sort_index()

        sector_weights = pd.DataFrame(index=w.index)
        for sector in sorted(set(sector_map.values())):
            cols = [c for c in w.columns if c in sector_map and sector_map[c] == sector]
            if cols:
                sector_weights[sector] = w[cols].sum(axis=1)
        unmapped = [c for c in w.columns if c not in sector_map]
        if unmapped:
            sector_weights["Crypto"] = w[unmapped].sum(axis=1)

        sector_weights = sector_weights.clip(lower=0)
        sector_weights = sector_weights.div(sector_weights.sum(axis=1), axis=0).fillna(0)

        palette = plt.cm.tab10(np.linspace(0, 1, len(sector_weights.columns)))
        ax.stackplot(
            sector_weights.index,
            *[sector_weights[col] * 100 for col in sector_weights.columns],
            labels=sector_weights.columns,
            colors=palette,
            alpha=0.85,
        )
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        if idx % ncols == 0:
            ax.set_ylabel("Weight (%)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Date")

    # Hide unused subplot(s)
    for idx in range(n_methods, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"Portfolio Weights Over Time — {family} Family (by Method)", fontsize=12, fontweight="bold", y=1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=6, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.text(
        0.01, 0.01,
        "Combined-family out-of-sample portfolio weights by method.",
        fontsize=8,
        color=COLORS["gray"],
    )
    for ax in axes.flat:
        if ax.get_visible():
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_fear_greed(
    fg_index: pd.DataFrame, filename: str = "fear_greed_index.png"
) -> None:
    """Plot the fear & greed index (standardised z-score)."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sub = fg_index.dropna(subset=["fear_greed_z"]).sort_values("trading_date")
    ax.fill_between(sub["trading_date"], sub["fear_greed_z"], 0, where=(sub["fear_greed_z"] >= 0), color=COLORS["green"], alpha=0.35, interpolate=True)
    ax.fill_between(sub["trading_date"], sub["fear_greed_z"], 0, where=(sub["fear_greed_z"] < 0), color=COLORS["red"], alpha=0.35, interpolate=True)
    ax.plot(sub["trading_date"], sub["fear_greed_z"], color=COLORS["navy"], linewidth=0.9)
    ax.axhline(0, color=COLORS["navy"], linewidth=0.6, linestyle="--")
    ax.set(title="Fear & Greed Index (Standardised)", xlabel="Date", ylabel="Z-score")
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.01, "21-day rolling standardisation of market-wide sentiment (0-100 scale).", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_vader_comparison(
    comparison: pd.DataFrame, filename: str = "vader_vs_finvader.png"
) -> None:
    """Bar chart comparing VADER vs FinVADER neutral rates."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLORS["gray"], COLORS["teal"]]
    bars = ax.bar(comparison["model"], comparison["neutral_rate"] * 100, color=colors, alpha=0.85, edgecolor=COLORS["navy"], linewidth=0.6)
    ax.set(title="Neutral Rate: VADER vs FinVADER", xlabel="Model", ylabel="Neutral rate (%)")
    ax.set_ylim(0, max(comparison["neutral_rate"] * 100) * 1.2)
    for bar, val in zip(bars, comparison["neutral_rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.01, "Neutral = |compound| < 0.05; lower is better for financial headlines.", fontsize=8, color=COLORS["gray"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_output()
    print("=" * 60)
    print("Part B — Full Build")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load and clean data (Part A foundation)
    # ------------------------------------------------------------------ #
    print("\n1. Loading data...")
    eq = etl.load_clean_equities()
    cr = etl.load_clean_crypto()
    news = etl.load_clean_news()

    print("2. Computing returns...")
    eq_ret = features.daily_returns(eq)
    cr_ret = features.daily_returns(cr)
    combined_panel = features.combined_returns_panel(eq_ret, cr_ret)

    equity_tickers = sorted(eq["ticker"].unique())
    crypto_tickers = sorted(cr["ticker"].unique())

    combined_wide = combined_panel.set_index("date").sort_index()
    equity_wide = combined_wide[equity_tickers]
    # Crypto-only uses its native 365-day calendar; Combined uses equity-calendar left-join
    crypto_wide_full = features.returns_to_wide(cr_ret)

    families = {
        "Equity": equity_wide,
        "Crypto": crypto_wide_full,
        "Combined": combined_wide,
    }

    # ------------------------------------------------------------------ #
    # 3. Backtests
    # ------------------------------------------------------------------ #
    print("\n3. Running backtests...")
    fund_returns, fund_weights = run_backtests(families)

    # ------------------------------------------------------------------ #
    # 4. Sentiment model
    # ------------------------------------------------------------------ #
    print("\n4. Building sentiment model...")
    trading_dates = pd.DatetimeIndex(eq["date"].unique()).sort_values()
    aligned_news = features.align_headlines_to_trading_days(news, trading_dates)
    headline_panel = features.assemble_headline_panel(news, trading_dates)

    scores = sentiment.score_headlines(headline_panel, use_fin_vader=True)
    # Official sector sentiment index (equal-weight tickers, per rubric requirement)
    sector_index = sentiment.sector_sentiment_index(
        scores, trading_dates, use_weights=False, lag_days=1
    )

    # Fear & Greed Index (market-wide, standardised)
    fg_index = sentiment.fear_greed_index(scores, trading_dates, window=21, lag_days=1)

    # VADER vs FinVADER diagnostic
    vader_comparison = sentiment.compare_vader_vs_fin_vader(headline_panel)

    # ------------------------------------------------------------------ #
    # 5. Fusion
    # ------------------------------------------------------------------ #
    print("\n5. Applying sentiment fusion...")
    ticker_sector_map = eq[["ticker", "sector"]].drop_duplicates()
    fund_returns, fund_weights = run_sentiment_fusion(
        fund_returns, fund_weights, combined_wide, sector_index, ticker_sector_map
    )

    # ------------------------------------------------------------------ #
    # 6. Metrics
    # ------------------------------------------------------------------ #
    print("\n6. Computing performance metrics...")
    metrics = compute_metrics(fund_returns)

    # ------------------------------------------------------------------ #
    # 7. Save artefacts
    # ------------------------------------------------------------------ #
    print("\n7. Saving artefacts...")
    fund_returns.to_csv(DATA_DIR / "fund_returns.csv", index=False)
    fund_weights.to_csv(DATA_DIR / "fund_weights.csv", index=False)
    sector_index.to_csv(DATA_DIR / "sector_sentiment_index.csv", index=False)
    fg_index.to_csv(DATA_DIR / "fear_greed_index.csv", index=False)
    metrics.to_csv(TABLE_DIR / "performance_metrics.csv", index=False)
    vader_comparison.to_csv(TABLE_DIR / "vader_comparison.csv", index=False)

    # Fusion before-vs-after table
    fusion_metrics = metrics.loc[
        (metrics["fund"] == "Combined")
        & (metrics["method"].isin(["max_sharpe", "max_sharpe_sentiment"]))
    ].copy()
    if not fusion_metrics.empty:
        fusion_metrics.to_csv(TABLE_DIR / "fusion_metrics.csv", index=False)

    # ------------------------------------------------------------------ #
    # 8. Figures
    # ------------------------------------------------------------------ #
    print("\n8. Generating figures...")
    save_growth_of_one(fund_returns)
    save_drawdown(fund_returns, family="Combined", method="max_sharpe")
    save_weights_methods_comparison(fund_weights, ticker_sector_map, family="Combined")
    save_sharpe_barplot(metrics)
    save_sector_sentiment(sector_index)
    save_fusion_comparison(fund_returns)
    save_risk_return_scatter(metrics)
    save_fear_greed(fg_index)
    save_vader_comparison(vader_comparison)

    # ------------------------------------------------------------------ #
    # 9. Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("Build complete.")
    print(f"  Funds backtested : {metrics[['fund', 'method']].drop_duplicates().shape[0]}")
    print(f"  Trading days     : {fund_returns['date'].nunique()}")
    print(f"  Tables saved to  : {TABLE_DIR}")
    print(f"  Figures saved to : {FIGURE_DIR}")
    print(f"  Data saved to    : {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
