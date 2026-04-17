from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.options.surface import (
    fit_smile,
    fit_surface,
    interpolate_surface,
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
