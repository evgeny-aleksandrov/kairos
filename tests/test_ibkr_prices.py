from __future__ import annotations

import pandas as pd

from kairos.data.ibkr import _normalize_historical_points


def test_normalize_historical_points_to_price_schema() -> None:
    payload = {
        "symbol": "QQQ",
        "volumeFactor": 100,
        "data": [
            {"o": 500.0, "h": 505.0, "l": 498.0, "c": 504.0, "v": 12, "t": 1711843200000},
            {"o": 504.0, "h": 506.0, "l": 503.0, "c": 505.5, "v": 15, "t": 1711929600000},
        ],
    }

    frame = _normalize_historical_points(payload)

    assert list(frame.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert frame.loc[0, "adj_close"] == frame.loc[0, "close"]
    assert frame.loc[0, "volume"] == 1200
    assert len(frame) == 2
