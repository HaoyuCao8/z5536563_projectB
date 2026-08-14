"""AtlasFlow — Systematic Multi-Asset Funds with News-Sentiment Analytics.

Deployed app: reads precomputed results from results/ only.
No sentiment model at runtime, no backtests, no heavy computation.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import data_access  # noqa: E402

st.set_page_config(page_title="AtlasFlow", layout="wide", page_icon="📈")

# --------------------------------------------------------------------------- #
# Design system — inherits Part A palette
# --------------------------------------------------------------------------- #
COLORS = {
    "navy": "#17324D",
    "blue": "#3D7EA6",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#C8553D",
    "gray": "#5F6B73",
    "purple": "#7B68EE",
}

METHOD_LABELS = {
    "equal_weight": "Equal Weight",
    "min_variance": "Minimum Variance",
    "max_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "hrp": "HRP",
    "max_sharpe_sentiment": "Max Sharpe + Sentiment",
}

METHOD_COLORS = {
    "equal_weight": COLORS["gray"],
    "min_variance": COLORS["blue"],
    "max_sharpe": COLORS["gold"],
    "risk_parity": COLORS["teal"],
    "hrp": COLORS["purple"],
    "max_sharpe_sentiment": COLORS["red"],
}

plt.rcParams.update(
    {
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


# --------------------------------------------------------------------------- #
# Data loaders (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=86_400, show_spinner=False)
def _load_fund_returns() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "data" / "fund_returns.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def _load_fund_weights() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "data" / "fund_weights.csv", parse_dates=["date"])


@st.cache_data(ttl=86_400, show_spinner=False)
def _load_metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / "performance_metrics.csv")


@st.cache_data(ttl=86_400, show_spinner=False)
def _load_sector_sentiment() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "results" / "data" / "sector_sentiment_index.csv",
        parse_dates=["trading_date"],
    )


@st.cache_data(ttl=86_400, show_spinner=False)
def _load_fear_greed() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "results" / "data" / "fear_greed_index.csv",
        parse_dates=["trading_date"],
    )


@st.cache_data(ttl=86_400, show_spinner=False)
def _load_vader_comparison() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "tables" / "vader_comparison.csv")


@st.cache_data(ttl=86_400, show_spinner=False)
def _load_sector_map() -> pd.DataFrame:
    return data_access.load_sector_universe()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _growth_of_one(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod()


def _drawdown(daily_returns: pd.Series) -> pd.Series:
    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak


def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def _fmt_sharpe(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:.3f}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("📈 AtlasFlow")
st.sidebar.caption("Systematic Multi-Asset Funds")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Fund Comparison", "📋 Fact Sheets", "⚖️ Allocation", "📰 Sentiment"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Disclaimer**: Past performance is not a guarantee of future results. "
    "Crypto assets carry material risk. The sentiment index is a research analytic, "
    "not a trading instruction."
)

# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #
if page == "🏠 Home":
    st.title("Welcome to AtlasFlow")
    st.markdown(
        """
        AtlasFlow is a prototype investment app offering systematically managed
        multi-asset funds backed by out-of-sample backtests and news-sentiment
        analytics.

        **What you can do here:**
        - **Fund Comparison** — Compare all funds across return, risk, and Sharpe ratio.
        - **Fact Sheets** — Dive into any fund: growth of $1, drawdowns, current holdings.
        - **Allocation** — Set your own allocation across funds and see combined performance.
        - **Sentiment** — Explore the sector sentiment index built from FinVADER-scored headlines.
        """
    )

    metrics = _load_metrics()
    if not metrics.empty:
        st.subheader("Quick Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Funds offered", metrics[["fund", "method"]].drop_duplicates().shape[0])
        best = metrics.loc[metrics["sharpe_ratio"].idxmax()]
        c2.metric("Best Sharpe", _fmt_sharpe(best["sharpe_ratio"]), f"{best['fund']} {METHOD_LABELS.get(best['method'], best['method'])}")
        c3.metric("OOS trading days", int(metrics["count"].max()))
        c4.metric("Sectors covered", _load_sector_sentiment()["sector"].nunique())

# --------------------------------------------------------------------------- #
# Fund Comparison
# --------------------------------------------------------------------------- #
elif page == "📊 Fund Comparison":
    st.title("Fund Comparison")

    returns = _load_fund_returns()
    metrics = _load_metrics()

    if metrics.empty:
        st.error("No metrics found — run `python scripts/run_part_b.py` first.")
        st.stop()

    # Filters
    families = st.multiselect("Fund family", metrics["fund"].unique(), default=["Combined"])
    methods = st.multiselect("Method", metrics["method"].unique(), default=metrics["method"].unique())

    filtered = metrics.loc[metrics["fund"].isin(families) & metrics["method"].isin(methods)].copy()
    filtered["display"] = filtered["fund"] + " — " + filtered["method"].map(METHOD_LABELS)

    st.dataframe(
        filtered[["display", "annualised_return", "annualised_volatility", "sharpe_ratio", "max_drawdown", "count"]]
        .sort_values("sharpe_ratio", ascending=False)
        .rename(
            columns={
                "display": "Fund",
                "annualised_return": "Ann. Return",
                "annualised_volatility": "Ann. Vol",
                "sharpe_ratio": "Sharpe",
                "max_drawdown": "Max DD",
                "count": "Days",
            }
        )
        .style.format(
            {
                "Ann. Return": "{:.2%}",
                "Ann. Vol": "{:.2%}",
                "Sharpe": "{:.3f}",
                "Max DD": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Growth of $1 chart
    st.subheader("Growth of $1")
    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in filtered.iterrows():
        mask = (returns["fund"] == row["fund"]) & (returns["method"] == row["method"])
        group = returns.loc[mask].sort_values("date")
        if group.empty:
            continue
        growth = _growth_of_one(group["return"])
        color = METHOD_COLORS.get(row["method"], COLORS["gray"])
        ax.plot(group["date"], growth, label=row["display"], color=color, linewidth=1.5, alpha=0.85)
    ax.axhline(1, color=COLORS["navy"], linewidth=0.6, linestyle="--")
    ax.set(xlabel="Date", ylabel="Portfolio value ($)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    # Sharpe bar
    st.subheader("Sharpe Ratio")
    fig2, ax2 = plt.subplots(figsize=(8, max(3, len(filtered) * 0.35)))
    plot_df = filtered.sort_values("sharpe_ratio", ascending=True)
    colors = [METHOD_COLORS.get(m, COLORS["gray"]) for m in plot_df["method"]]
    ax2.barh(plot_df["display"], plot_df["sharpe_ratio"], color=colors, alpha=0.85)
    ax2.axvline(0, color=COLORS["navy"], linewidth=0.8)
    ax2.set(xlabel="Sharpe ratio (rf = 0)")
    ax2.grid(axis="x", alpha=0.2)
    fig2.tight_layout()
    st.pyplot(fig2)

# --------------------------------------------------------------------------- #
# Fact Sheets
# --------------------------------------------------------------------------- #
elif page == "📋 Fact Sheets":
    st.title("Fund Fact Sheets")

    returns = _load_fund_returns()
    weights = _load_fund_weights()
    metrics = _load_metrics()

    if metrics.empty:
        st.error("No data found — run `python scripts/run_part_b.py` first.")
        st.stop()

    # Selector
    metrics["display"] = metrics["fund"] + " — " + metrics["method"].map(METHOD_LABELS)
    selected = st.selectbox("Select a fund", metrics["display"].sort_values())
    sel = metrics.loc[metrics["display"] == selected].iloc[0]
    family, method = sel["fund"], sel["method"]

    mask = (returns["fund"] == family) & (returns["method"] == method)
    ret_series = returns.loc[mask].sort_values("date")
    w_mask = (weights["fund"] == family) & (weights["method"] == method)
    w_series = weights.loc[w_mask].sort_values("date")

    if ret_series.empty:
        st.error("No return data for this fund.")
        st.stop()

    # Metrics cards
    st.subheader("Key Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualised Return", _fmt_pct(sel["annualised_return"]))
    c2.metric("Annualised Volatility", _fmt_pct(sel["annualised_volatility"]))
    c3.metric("Sharpe Ratio", _fmt_sharpe(sel["sharpe_ratio"]))
    c4.metric("Max Drawdown", _fmt_pct(sel["max_drawdown"]))

    # Growth & Drawdown
    col_g, col_d = st.columns(2)
    with col_g:
        st.markdown("**Growth of $1**")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        growth = _growth_of_one(ret_series["return"])
        ax.plot(ret_series["date"], growth, color=METHOD_COLORS.get(method, COLORS["blue"]), linewidth=1.5)
        ax.axhline(1, color=COLORS["navy"], linewidth=0.6, linestyle="--")
        ax.set(xlabel="Date", ylabel="Value ($)")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)

    with col_d:
        st.markdown("**Drawdown**")
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        dd = _drawdown(ret_series["return"])
        ax.fill_between(ret_series["date"], dd * 100, 0, color=COLORS["red"], alpha=0.35)
        ax.plot(ret_series["date"], dd * 100, color=COLORS["red"], linewidth=1.0)
        ax.set(xlabel="Date", ylabel="Drawdown (%)")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)

    # Current holdings
    st.subheader("Current Holdings (latest rebalance)")
    if not w_series.empty:
        latest = w_series.iloc[-1].drop(["date", "fund", "method"])
        latest = latest[latest > 0.001].sort_values(ascending=False)
        holdings = pd.DataFrame({"Ticker": latest.index, "Weight": latest.values})
        holdings["Weight"] = holdings["Weight"].apply(lambda x: f"{x * 100:.2f}%")
        st.dataframe(holdings, use_container_width=True, hide_index=True)
    else:
        st.info("No weight data available.")

# --------------------------------------------------------------------------- #
# Allocation
# --------------------------------------------------------------------------- #
elif page == "⚖️ Allocation":
    st.title("Build Your Portfolio")

    returns = _load_fund_returns()
    metrics = _load_metrics()

    if metrics.empty:
        st.error("No data found — run `python scripts/run_part_b.py` first.")
        st.stop()

    metrics["display"] = metrics["fund"] + " — " + metrics["method"].map(METHOD_LABELS)
    funds = metrics["display"].sort_values().tolist()

    st.markdown("Set your allocation across funds (must sum to 100%).")
    sliders = {}
    cols = st.columns(2)
    for i, fund in enumerate(funds):
        with cols[i % 2]:
            sliders[fund] = st.slider(fund, 0, 100, 0, 5, key=f"slider_{i}")

    total = sum(sliders.values())
    st.markdown(f"**Total allocation: {total}%**")

    if total == 0:
        st.info("Move at least one slider to see the combined portfolio.")
        st.stop()

    if total != 100:
        st.warning("Allocation does not sum to 100% — weights will be normalised.")

    # Normalise
    alloc = {k: v / total for k, v in sliders.items() if v > 0}

    # Compute blended returns
    blended = []
    for fund_display, weight in alloc.items():
        sel = metrics.loc[metrics["display"] == fund_display].iloc[0]
        mask = (returns["fund"] == sel["fund"]) & (returns["method"] == sel["method"])
        sub = returns.loc[mask][["date", "return"]].copy()
        sub["weighted_return"] = sub["return"] * weight
        blended.append(sub[["date", "weighted_return"]])

    blended_df = pd.concat(blended)
    blended_daily = blended_df.groupby("date")["weighted_return"].sum().reset_index()
    blended_daily = blended_daily.sort_values("date")

    # Metrics
    ann_ret = blended_daily["weighted_return"].mean() * 252
    ann_vol = blended_daily["weighted_return"].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    cum = (1 + blended_daily["weighted_return"]).cumprod()
    peak = cum.cummax()
    max_dd = ((cum - peak) / peak).min()

    st.subheader("Your Combined Portfolio")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ann. Return", _fmt_pct(ann_ret))
    c2.metric("Ann. Volatility", _fmt_pct(ann_vol))
    c3.metric("Sharpe Ratio", _fmt_sharpe(sharpe))
    c4.metric("Max Drawdown", _fmt_pct(max_dd))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(blended_daily["date"], cum, color=COLORS["navy"], linewidth=2.0)
    ax.axhline(1, color=COLORS["gray"], linewidth=0.6, linestyle="--")
    ax.set(xlabel="Date", ylabel="Portfolio value ($)", title="Growth of $1 — Your Allocation")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

# --------------------------------------------------------------------------- #
# Sentiment
# --------------------------------------------------------------------------- #
elif page == "📰 Sentiment":
    st.title("Sector Sentiment Analytics")

    sector_index = _load_sector_sentiment()
    fg_index = _load_fear_greed()
    vader_comp = _load_vader_comparison()

    if sector_index.empty:
        st.error("No sentiment data found — run `python scripts/run_part_b.py` first.")
        st.stop()

    st.markdown(
        """
        The sector sentiment index is built from FinVADER-scored headlines,
        aggregated at sector level with equal-weighted tickers.
        Values range from -1 (very negative) to +1 (very positive).
        The signal is lagged by at least one trading day to avoid look-ahead bias.
        """
    )

    # Fear & Greed Index
    st.subheader("Fear & Greed Index")
    if not fg_index.empty:
        c1, c2, c3 = st.columns(3)
        latest_fg = fg_index.dropna(subset=["fear_greed_z"]).iloc[-1]
        c1.metric("Latest Z-Score", f"{latest_fg['fear_greed_z']:.2f}")
        c2.metric("Raw Level (0-100)", f"{latest_fg['fear_greed_raw']:.1f}")
        extreme_days = (fg_index["fear_greed_z"].abs() > 2).sum()
        c3.metric("Extreme Days (|z|>2)", int(extreme_days))

        fig, ax = plt.subplots(figsize=(11, 4))
        sub = fg_index.dropna(subset=["fear_greed_z"]).sort_values("trading_date")
        ax.fill_between(sub["trading_date"], sub["fear_greed_z"], 0, where=(sub["fear_greed_z"] >= 0), color=COLORS["green"], alpha=0.35, interpolate=True)
        ax.fill_between(sub["trading_date"], sub["fear_greed_z"], 0, where=(sub["fear_greed_z"] < 0), color=COLORS["red"], alpha=0.35, interpolate=True)
        ax.plot(sub["trading_date"], sub["fear_greed_z"], color=COLORS["navy"], linewidth=0.9)
        ax.axhline(0, color=COLORS["navy"], linewidth=0.6, linestyle="--")
        ax.axhline(2, color=COLORS["green"], linewidth=0.5, linestyle=":", alpha=0.6)
        ax.axhline(-2, color=COLORS["red"], linewidth=0.5, linestyle=":", alpha=0.6)
        ax.set(xlabel="Date", ylabel="Z-score", title="Market-Wide Fear & Greed (Standardised)")
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Fear & Greed data not available.")

    st.markdown("---")

    # VADER vs FinVADER
    st.subheader("VADER vs FinVADER")
    if not vader_comp.empty:
        c1, c2 = st.columns(2)
        v_row = vader_comp.loc[vader_comp["model"] == "VADER"].iloc[0]
        f_row = vader_comp.loc[vader_comp["model"] == "FinVADER"].iloc[0]
        c1.metric("VADER Neutral Rate", f"{v_row['neutral_rate']*100:.1f}%")
        c2.metric("FinVADER Neutral Rate", f"{f_row['neutral_rate']*100:.1f}%", delta=f"{(f_row['neutral_rate']-v_row['neutral_rate'])*100:.1f}pp")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = [COLORS["gray"], COLORS["teal"]]
        bars = ax.bar(vader_comp["model"], vader_comp["neutral_rate"] * 100, color=colors, alpha=0.85, edgecolor=COLORS["navy"], linewidth=0.6)
        ax.set(xlabel="Model", ylabel="Neutral rate (%)", title="Neutral Headline Rate")
        ax.set_ylim(0, max(vader_comp["neutral_rate"] * 100) * 1.25)
        for bar, val in zip(bars, vader_comp["neutral_rate"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Lower neutral rate means FinVADER captures more financial sentiment. Neutral = |compound| < 0.05.")
    else:
        st.info("VADER comparison data not available.")

    st.markdown("---")

    # Sector time series
    st.subheader("Sentiment Index Over Time")
    sectors = sorted(sector_index["sector"].dropna().unique())
    selected_sectors = st.multiselect("Sectors to display", sectors, default=sectors[:5])

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, len(selected_sectors)))
    for sector, color in zip(selected_sectors, palette):
        sub = sector_index.loc[sector_index["sector"] == sector].sort_values("trading_date")
        ax.plot(sub["trading_date"], sub["sentiment"], label=sector, color=color, linewidth=1.2, alpha=0.85)
    ax.axhline(0, color=COLORS["navy"], linewidth=0.8, linestyle="--")
    ax.set(xlabel="Date", ylabel="Sentiment score")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig)

    # Latest snapshot
    st.subheader("Latest Sentiment Snapshot")
    latest_date = sector_index["trading_date"].max()
    latest = sector_index.loc[sector_index["trading_date"] == latest_date][["sector", "sentiment"]].sort_values("sentiment", ascending=False)
    latest["sentiment"] = latest["sentiment"].apply(lambda x: f"{x:+.3f}")
    st.dataframe(latest, use_container_width=True, hide_index=True)

    # Heatmap by sector x month
    st.subheader("Sentiment Heatmap (monthly average)")
    heat = sector_index.copy()
    heat["month"] = heat["trading_date"].dt.to_period("M").dt.to_timestamp()
    heatmap_data = heat.groupby(["sector", "month"])["sentiment"].mean().unstack("month")
    # Show last 12 months
    if heatmap_data.shape[1] > 12:
        heatmap_data = heatmap_data.iloc[:, -12:]
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels([d.strftime("%Y-%m") for d in heatmap_data.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set(title="Monthly Average Sentiment by Sector")
    fig.colorbar(im, ax=ax, label="Sentiment")
    fig.tight_layout()
    st.pyplot(fig)
