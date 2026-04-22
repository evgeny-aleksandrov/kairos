from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.options.surface import (
    fit_smile,
    fit_surface,
    interpolate_surface,
    select_surface_quotes,
    volatility_surface_plot,
)


def test_fit_smile_and_surface() -> None:
    chain = pd.DataFrame(
        {
            "expiry": pd.to_datetime(
                [
                    "2026-04-15",
                    "2026-04-15",
                    "2026-04-15",
                    "2026-05-15",
                    "2026-05-15",
                    "2026-05-15",
                ]
            ),
            "time_to_expiry": [0.05, 0.05, 0.05, 0.12, 0.12, 0.12],
            "strike": [95, 100, 105, 95, 100, 105],
            "option_type": ["put", "call", "call", "put", "put", "call"],
            "symbol": ["QQQ"] * 6,
            "quote_date": pd.to_datetime(["2026-04-01"] * 6),
            "underlying_price": [100, 100, 100, 100, 100, 100],
            "risk_free_rate": [0.03] * 6,
            "dividend_yield": [0.0] * 6,
            "implied_vol": [0.24, 0.20, 0.23, 0.25, 0.21, 0.24],
            "bid_ask_width": [0.2] * 6,
        }
    )

    one_expiry = fit_smile(chain.iloc[:3].copy())
    assert one_expiry.fitted["smile_fitted_vol"].notna().all()

    fitted_chain, params = fit_surface(chain)
    surface_slice = interpolate_surface(params, 0.08, np.array([-0.05, 0.0, 0.05]))
    surface_fig = volatility_surface_plot(params, fitted_chain)

    assert len(fitted_chain) == 6
    assert list(params.columns) == ["expiry", "time_to_expiry", "a", "b", "c"]
    assert surface_slice.shape == (3,)
    assert surface_fig.axes
    assert surface_fig.axes[0].get_title() == "QQQ - Quote 2026-04-01 Volatility Surface"
    surface_fig.clear()


def test_select_surface_quotes_uses_otm_and_best_atm_mid() -> None:
    chain = pd.DataFrame(
        {
            "expiry": pd.to_datetime(["2026-04-15"] * 5),
            "time_to_expiry": [0.05] * 5,
            "strike": [95, 100, 100, 105, 105],
            "option_type": ["put", "put", "call", "call", "put"],
            "underlying_price": [100] * 5,
            "risk_free_rate": [0.0] * 5,
            "dividend_yield": [0.0] * 5,
            "bid": [3.9, 4.5, 4.9, 5.1, 5.0],
            "ask": [4.1, 5.5, 5.1, 5.3, 5.2],
            "mid": [4.0, 5.0, 5.0, 5.2, 5.1],
            "volume": [10, 10, 10, 10, 10],
            "open_interest": [100] * 5,
            "implied_vol": [0.24, 0.21, 0.20, 0.23, 0.25],
        }
    )

    selected = select_surface_quotes(chain)

    assert selected["surface_quote_role"].tolist() == [
        "otm_put",
        "atm_best_mid",
        "otm_call",
    ]
    assert selected["option_type"].tolist() == ["put", "call", "call"]
    assert selected.loc[selected["strike"] == 100, "relative_bid_ask_width"].iloc[0] < 0.05
