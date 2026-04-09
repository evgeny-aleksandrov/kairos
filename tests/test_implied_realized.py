from __future__ import annotations

import pandas as pd

from kairos.prices.implied_realized import compare_implied_vs_realized


def test_compare_implied_vs_realized_builds_summary() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=80, freq="B"),
            "adj_close": [100 + 0.5 * idx + (-1) ** idx * 0.3 for idx in range(80)],
        }
    )
    chain = pd.DataFrame(
        {
            "quote_date": [prices.loc[25, "date"], prices.loc[25, "date"], prices.loc[40, "date"]],
            "expiry": pd.to_datetime([prices.loc[40, "date"], prices.loc[40, "date"], prices.loc[60, "date"]]),
            "strike": [112.5, 113.0, 113.0],
            "underlying_price": [112.5, 112.5, 113.0],
            "bid_ask_width": [0.2, 0.25, 0.2],
            "implied_vol": [0.24, 0.25, 0.23],
        }
    )

    summary = compare_implied_vs_realized(chain, prices)

    assert set(summary.columns) >= {
        "quote_date",
        "expiry",
        "atm_implied_vol",
        "tenor_days",
        "tenor_trading_days",
        "trailing_realized_vol",
        "forward_realized_vol",
        "vol_risk_premium_trailing",
        "vol_risk_premium_forward",
        "tenor_days",
    }
    assert len(summary) == 2
    assert summary["tenor_trading_days"].ge(1).all()
