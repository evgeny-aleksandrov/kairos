from __future__ import annotations

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


def compare_implied_vs_realized(
    chain: pd.DataFrame,
    prices: pd.DataFrame,
    iv_column: str = "implied_vol",
    realized_window: int = 21,
    moneyness_tolerance: float = 0.02,
) -> pd.DataFrame:
    chain_frame = chain.copy()
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

    realized = realized_vol_frame(prices, windows=(realized_window,))
    realized_column = f"realized_vol_{realized_window}d"
    merged = implied_summary.merge(
        realized[["date", realized_column]],
        left_on="quote_date",
        right_on="date",
        how="left",
    ).drop(columns=["date"])
    merged["vol_risk_premium"] = merged["atm_implied_vol"] - merged[realized_column]
    merged["tenor_days"] = (merged["expiry"] - merged["quote_date"]).dt.days
    return merged
