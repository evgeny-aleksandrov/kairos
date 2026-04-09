from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.prices.realized_vol import TRADING_DAYS, arithmetic_return


def realized_vol_frame(
    prices: pd.DataFrame,
    price_column: str = "adj_close",
    windows: tuple[int, ...] = (5, 10, 21, 63),
    annualization: int = TRADING_DAYS,
) -> pd.DataFrame:
    frame = prices.copy().sort_values("date").reset_index(drop=True)
    frame["arith_return"] = arithmetic_return(frame[price_column])
    for window in windows:
        frame[f"realized_vol_{window}d"] = (
            frame["arith_return"].rolling(window).std(ddof=1) * annualization**0.5
        )
    return frame


def _business_day_window(
    quote_date: pd.Timestamp,
    expiry: pd.Timestamp,
) -> int:
    quote = np.datetime64(pd.Timestamp(quote_date).normalize().date())
    exp = np.datetime64(pd.Timestamp(expiry).normalize().date())
    return max(int(np.busday_count(quote, exp)), 1)


def _trailing_realized_vol(
    returns_by_date: pd.Series,
    quote_date: pd.Timestamp,
    window: int,
    annualization: int,
) -> float:
    history = returns_by_date.loc[returns_by_date.index <= quote_date].dropna()
    if len(history) < window:
        return np.nan
    return float(history.iloc[-window:].std(ddof=1) * annualization**0.5)


def compare_implied_vs_realized(
    chain: pd.DataFrame,
    prices: pd.DataFrame,
    iv_column: str = "implied_vol",
    moneyness_tolerance: float = 0.02,
    annualization: int = TRADING_DAYS,
) -> pd.DataFrame:
    chain_frame = chain.copy()
    chain_frame["quote_date"] = pd.to_datetime(chain_frame["quote_date"]).dt.normalize()
    chain_frame["expiry"] = pd.to_datetime(chain_frame["expiry"]).dt.normalize()
    if "moneyness" not in chain_frame.columns:
        chain_frame["moneyness"] = chain_frame["strike"] / chain_frame["underlying_price"]

    atm_slice = chain_frame.loc[
        chain_frame["moneyness"].sub(1.0).abs() <= moneyness_tolerance
    ].copy()
    implied_summary = (
        atm_slice.groupby(["quote_date", "expiry"], as_index=False)
        .agg(
            atm_implied_vol=(iv_column, "mean"),
            atm_bid_ask_width=("bid_ask_width", "mean"),
            contracts=(iv_column, "size"),
        )
        .sort_values(["quote_date", "expiry"])
    )
    implied_summary["tenor_days"] = (
        implied_summary["expiry"] - implied_summary["quote_date"]
    ).dt.days
    implied_summary["tenor_trading_days"] = implied_summary.apply(
        lambda row: _business_day_window(row["quote_date"], row["expiry"]),
        axis=1,
    )

    realized = prices.copy().sort_values("date").reset_index(drop=True)
    realized["date"] = pd.to_datetime(realized["date"]).dt.normalize()
    realized["arith_return"] = arithmetic_return(realized["adj_close"])
    returns_by_date = realized.set_index("date")["arith_return"]

    implied_summary["trailing_realized_vol"] = implied_summary.apply(
        lambda row: _trailing_realized_vol(
            returns_by_date,
            row["quote_date"],
            int(row["tenor_trading_days"]),
            annualization,
        ),
        axis=1,
    )
    
    implied_summary["vol_risk_premium_trailing"] = (
        implied_summary["atm_implied_vol"] - implied_summary["trailing_realized_vol"]
    )

    return implied_summary
