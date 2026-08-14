"""Station 3A — Funds: optimal portfolios + walk-forward out-of-sample backtest.

Methods implemented:
    equal_weight   — 1/n benchmark
    min_variance   — minimise portfolio variance
    max_sharpe     — maximise (mu - rf) / sigma  (tangency portfolio)
    risk_parity    — equal risk contribution
    hrp            — Hierarchical Risk Parity (innovation)

Backtest rules:
    * Walk-forward, no look-ahead
    * Weights formed only from past data inside an estimation window
    * Rebalanced monthly (21 trading days)
    * All funds run on the equity trading calendar (252-day annualisation)
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

PERIODS_PER_YEAR = 252
REBALANCE_DAYS = 21
ESTIMATION_WINDOW = 504  # 2 years of trading days


def _clean_estimation_window(
    returns_wide: pd.DataFrame, start: int, end: int
) -> pd.DataFrame | None:
    """Extract estimation data, drop all-NaN columns, then drop any rows with NaN."""
    est = returns_wide.iloc[start:end]
    est = est.dropna(axis=1, how="all")
    if est.shape[1] < 2:
        return None
    est = est.dropna()
    if len(est) < ESTIMATION_WINDOW * 0.4:
        return None
    return est


def _renormalise_weights_for_available(
    weights: pd.Series, available_returns: pd.Series
) -> pd.Series:
    """On days where some assets do not trade, rescale weights to the subset."""
    w = weights[available_returns.index].copy()
    s = w.sum()
    return w / s if s > 1e-12 else w


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #

def optimize_equal_weight(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    return pd.Series(np.ones(n) / n, index=returns.columns)


def optimize_min_variance(
    returns: pd.DataFrame, allow_short: bool = False
) -> pd.Series:
    n = returns.shape[1]
    cov = returns.cov().values
    w0 = np.ones(n) / n

    def objective(w):
        return float(w @ cov @ w)

    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
    w = result.x if result.success else w0
    w = np.maximum(w, 0) if not allow_short else w
    w = w / w.sum() if w.sum() > 1e-12 else np.ones(n) / n
    return pd.Series(w, index=returns.columns)


def optimize_max_sharpe(
    returns: pd.DataFrame, rf: float = 0.0, allow_short: bool = False
) -> pd.Series:
    n = returns.shape[1]
    mu = returns.mean().values
    cov = returns.cov().values
    w0 = np.ones(n) / n

    def objective(w):
        port_ret = w @ mu - rf
        port_var = w @ cov @ w
        if port_var < 1e-14:
            return 0.0
        return -port_ret / np.sqrt(port_var)

    bounds = [(-1.0, 1.0) if allow_short else (0.0, 1.0) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
    w = result.x if result.success else w0
    w = np.maximum(w, 0) if not allow_short else w
    w = w / w.sum() if w.sum() > 1e-12 else np.ones(n) / n
    return pd.Series(w, index=returns.columns)


def optimize_risk_parity(returns: pd.DataFrame) -> pd.Series:
    n = returns.shape[1]
    cov = returns.cov().values
    w0 = np.ones(n) / n

    def objective(w):
        port_var = w @ cov @ w
        if port_var < 1e-14:
            return 1e6
        rc = w * (cov @ w) / port_var
        return float(np.sum((rc - 1.0 / n) ** 2))

    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 1000},
        )
    w = result.x if result.success else w0
    w = np.maximum(w, 0)
    w = w / w.sum() if w.sum() > 1e-12 else np.ones(n) / n
    return pd.Series(w, index=returns.columns)


# --------------------------------------------------------------------------- #
# Hierarchical Risk Parity (innovation)
# --------------------------------------------------------------------------- #

def _get_quasi_diag(link: np.ndarray) -> list:
    """Sort items according to hierarchical linkage (Lopez de Prado 2016)."""
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j.astype(int), 0]
        df0 = pd.Series(link[j.astype(int), 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _get_cluster_var(cov: pd.DataFrame, cluster_items: list) -> float:
    cov_slice = cov.loc[cluster_items, cluster_items]
    w_ = 1.0 / np.diag(cov_slice.values)
    w_ /= w_.sum()
    return float(w_.T @ cov_slice.values @ w_)


def _get_rec_bisection(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    w = pd.Series(1.0, index=sort_ix)
    cl_items = [sort_ix]
    while len(cl_items) > 0:
        new_cl = []
        for item in cl_items:
            if len(item) > 1:
                mid = len(item) // 2
                new_cl.append(item[:mid])
                new_cl.append(item[mid:])
        for i in range(0, len(new_cl), 2):
            c0 = new_cl[i]
            c1 = new_cl[i + 1]
            v0 = _get_cluster_var(cov, c0)
            v1 = _get_cluster_var(cov, c1)
            alpha = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 1e-14 else 0.5
            w[c0] *= alpha
            w[c1] *= 1.0 - alpha
        cl_items = new_cl
    return w


def optimize_hrp(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity — no covariance inversion required."""
    corr = returns.corr()
    dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1)))
    dist_condensed = squareform(dist.values, checks=False)
    link = hierarchy.linkage(dist_condensed, method="single")
    sort_ix = _get_quasi_diag(link)
    sort_ix = [returns.columns[int(i)] for i in sort_ix if int(i) < len(returns.columns)]
    cov = returns.cov()
    w = _get_rec_bisection(cov, sort_ix)
    w = w.reindex(returns.columns).fillna(0)
    w = w / w.sum() if w.sum() > 1e-12 else pd.Series(1.0 / len(w), index=w.index)
    return w


_OPTIMISERS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "equal_weight": optimize_equal_weight,
    "min_variance": optimize_min_variance,
    "max_sharpe": optimize_max_sharpe,
    "risk_parity": optimize_risk_parity,
    "hrp": optimize_hrp,
}


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #

def walk_forward_backtest(
    returns_wide: pd.DataFrame,
    method: str,
    *,
    window: int = ESTIMATION_WINDOW,
    rebalance: int = REBALANCE_DAYS,
    fund_name: str = "fund",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a walk-forward OOS backtest with no look-ahead bias."""
    if method not in _OPTIMISERS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_OPTIMISERS)}")

    optimiser = _OPTIMISERS[method]
    all_tickers = returns_wide.columns.tolist()

    port_returns: list[dict] = []
    weights: list[dict] = []

    first_live = window
    if first_live >= len(returns_wide):
        raise ValueError("Not enough data for the estimation window.")

    for i in range(first_live, len(returns_wide), rebalance):
        est = _clean_estimation_window(returns_wide, i - window, i)
        if est is None:
            continue

        w = optimiser(est)
        w = w.reindex(all_tickers).fillna(0)

        oos_end = min(i + rebalance, len(returns_wide))
        oos = returns_wide.iloc[i:oos_end]

        for _, row in oos.iterrows():
            available = row.dropna()
            if available.empty:
                continue
            w_day = _renormalise_weights_for_available(w, available)
            port_ret = float((available * w_day).sum())
            port_returns.append(
                {"date": row.name, "return": port_ret, "method": method, "fund": fund_name}
            )
            rec: dict = {"date": row.name, "method": method, "fund": fund_name}
            rec.update(w.to_dict())
            weights.append(rec)

    if not port_returns:
        empty_ret = pd.DataFrame(columns=["date", "return", "method", "fund"])
        empty_w = pd.DataFrame(columns=["date", "method", "fund"] + all_tickers)
        return empty_ret, empty_w

    return pd.DataFrame(port_returns), pd.DataFrame(weights)


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #

def performance_metrics(
    daily_returns: pd.Series | np.ndarray,
    *,
    periods_per_year: int = PERIODS_PER_YEAR,
    rf_annual: float = 0.0,
) -> dict[str, float]:
    """Annualised return, volatility, Sharpe, and maximum drawdown."""
    r = pd.Series(daily_returns).dropna()
    if r.empty:
        return {
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "count": 0,
        }

    mean_daily = r.mean()
    std_daily = r.std(ddof=1)
    ann_ret = mean_daily * periods_per_year
    ann_vol = std_daily * np.sqrt(periods_per_year)
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-12 else np.nan

    cum = (1 + r).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "annualised_return": ann_ret,
        "annualised_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "count": int(len(r)),
    }


def growth_of_one(daily_returns: pd.Series | np.ndarray) -> pd.Series:
    r = pd.Series(daily_returns).dropna()
    return (1 + r).cumprod()


def drawdown_series(daily_returns: pd.Series | np.ndarray) -> pd.Series:
    r = pd.Series(daily_returns).dropna()
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return (cum - peak) / peak
