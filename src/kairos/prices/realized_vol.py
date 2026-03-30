from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def arithmetic_return(prices: pd.Series) -> pd.Series:
    return prices.astype(float).pct_change()


def log_return(prices: pd.Series) -> pd.Series:
    return np.log(prices.astype(float)).diff()


def cumulative_return(prices: pd.Series) -> pd.Series:
    returns = arithmetic_return(prices).fillna(0.0)
    return (1.0 + returns).cumprod() - 1.0


def close_to_close_volatility(returns: pd.Series, annualization: int = TRADING_DAYS) -> float:
    return float(np.sqrt(annualization * returns.dropna().var(ddof=1)))


def ewma_volatility(
    returns: pd.Series,
    lam: float = 0.94,
    annualization: int = TRADING_DAYS,
) -> pd.Series:
    squared = returns.astype(float).pow(2)
    ewma_var = squared.ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(annualization * ewma_var)


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int | None = None,
    annualization: int = TRADING_DAYS,
) -> pd.Series | float:
    rs = np.log(high.astype(float) / low.astype(float)).pow(2) / (4.0 * np.log(2.0))
    if window is None:
        return float(np.sqrt(annualization * rs.mean()))
    return np.sqrt(annualization * rs.rolling(window).mean())


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int | None = None,
    annualization: int = TRADING_DAYS,
) -> pd.Series | float:
    log_hl = np.log(high.astype(float) / low.astype(float))
    log_co = np.log(close.astype(float) / open_.astype(float))
    rs = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    if window is None:
        return float(np.sqrt(annualization * rs.mean()))
    return np.sqrt(annualization * rs.rolling(window).mean())


def downside_semivolatility(
    returns: pd.Series,
    target: float = 0.0,
    annualization: int = TRADING_DAYS,
) -> float:
    downside = np.minimum(returns.astype(float) - target, 0.0)
    return float(np.sqrt(annualization * np.mean(np.square(downside.dropna()))))


def rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    return returns.astype(float).rolling(window).skew()


def rolling_kurtosis(returns: pd.Series, window: int) -> pd.Series:
    return returns.astype(float).rolling(window).kurt()


def max_drawdown(prices: pd.Series) -> float:
    levels = prices.astype(float)
    rolling_peak = levels.cummax()
    drawdown = levels / rolling_peak - 1.0
    return float(drawdown.min())


def z_scored_return(returns: pd.Series, window: int) -> pd.Series:
    rolling_mean = returns.astype(float).rolling(window).mean()
    rolling_std = returns.astype(float).rolling(window).std(ddof=1)
    return (returns - rolling_mean) / rolling_std
