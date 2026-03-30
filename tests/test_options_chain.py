from __future__ import annotations

import pandas as pd

from kairos.data.options_chain import clean_option_chain


def test_clean_option_chain_computes_mid_and_logs_drops() -> None:
    raw = pd.DataFrame(
        {
            "quote_date": ["2026-03-01"] * 4,
            "expiry": ["2026-04-01", "2026-04-01", "2026-02-28", "2026-04-01"],
            "strike": [500, 510, 520, 530],
            "option_type": ["C", "P", "C", "bad"],
            "bid": [5.0, 6.0, 3.0, 2.0],
            "ask": [5.4, 5.5, 3.2, 2.3],
            "last": [5.2, 5.7, 3.1, 2.1],
            "volume": [100, 80, 50, 10],
            "open_interest": [1000, 900, 700, 20],
            "underlying_price": [505, 505, 505, 505],
            "risk_free_rate": [0.04, 0.04, 0.04, 0.04],
            "dividend_yield": [0.01, 0.01, 0.01, 0.01],
        }
    )

    processed = clean_option_chain(raw)

    assert len(processed.data) == 1
    assert processed.data.loc[0, "option_type"] == "call"
    assert processed.data.loc[0, "mid"] == 5.2
    assert processed.data.loc[0, "time_to_expiry"] > 0
    assert set(processed.drop_log["reason"]) == {"crossed_market", "expired_contract"}


def test_clean_option_chain_drops_non_positive_quotes() -> None:
    raw = pd.DataFrame(
        {
            "quote_date": ["2026-03-01"],
            "expiry": ["2026-04-01"],
            "strike": [500],
            "option_type": ["put"],
            "bid": [0.0],
            "ask": [1.0],
            "last": [0.5],
            "volume": [100],
            "open_interest": [1000],
            "underlying_price": [505],
            "risk_free_rate": [0.04],
            "dividend_yield": [0.01],
        }
    )

    processed = clean_option_chain(raw)

    assert processed.data.empty
    assert processed.drop_log.loc[0, "reason"] == "non_positive_quote"
