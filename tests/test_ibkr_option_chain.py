from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.data import ibkr
from kairos.data.ibkr import (
    _extract_option_months,
    _safe_float,
    _select_strikes_around_spot,
    write_option_chain_snapshot,
)


def test_extract_option_months_from_secdef_search() -> None:
    payload = [
        {
            "conid": "320227571",
            "symbol": "QQQ",
            "sections": [
                {"secType": "STK"},
                {"secType": "OPT", "months": "APR26;MAY26;JUN26", "exchange": "SMART;IBUSOPT"},
            ],
        }
    ]

    conid, months, exchange = _extract_option_months(payload)

    assert conid == 320227571
    assert months == ["APR26", "MAY26", "JUN26"]
    assert exchange == "SMART"


def test_safe_float_handles_blank_and_numeric_strings() -> None:
    assert np.isnan(_safe_float("--"))
    assert _safe_float("123.45") == 123.45
    assert _safe_float("1,234") == 1234.0


def test_select_strikes_around_spot_limits_count() -> None:
    strikes = [450, 460, 470, 480, 490, 500, 510, 520]
    selected = _select_strikes_around_spot(strikes, spot=497, strike_limit=4)

    assert selected == [480.0, 490.0, 500.0, 510.0]


def test_select_strikes_around_spot_samples_moneyness_band() -> None:
    strikes = list(range(60, 141, 5))
    selected = _select_strikes_around_spot(
        strikes,
        spot=100,
        strike_limit=5,
        min_moneyness=0.80,
        max_moneyness=1.20,
    )

    assert selected == [80.0, 90.0, 100.0, 110.0, 120.0]


def test_write_option_chain_snapshot_uses_symbol_filename(tmp_path, monkeypatch) -> None:
    def fake_fetch_option_chain_snapshot(**kwargs) -> pd.DataFrame:
        return pd.DataFrame([{"symbol": kwargs["symbol"].upper(), "strike": 500.0}])

    monkeypatch.setattr(ibkr, "fetch_option_chain_snapshot", fake_fetch_option_chain_snapshot)

    path = write_option_chain_snapshot(
        output_dir=tmp_path,
        symbol="AAPL",
    )

    assert path == tmp_path / "aapl_option_chain_ibkr.parquet"
    assert path.exists()
