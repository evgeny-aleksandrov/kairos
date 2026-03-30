from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.prices.realized_vol import (
    arithmetic_return,
    close_to_close_volatility,
    cumulative_return,
    ewma_volatility,
    garman_klass_volatility,
    log_return,
    max_drawdown,
    parkinson_volatility,
    rolling_kurtosis,
    rolling_skewness,
    downside_semivolatility,
    z_scored_return,
)


def test_return_calculations() -> None:
    prices = pd.Series([100.0, 102.0, 101.0, 105.0])
    arith = arithmetic_return(prices)
    logs = log_return(prices)
    cum = cumulative_return(prices)

    assert np.isclose(arith.iloc[1], 0.02)
    assert np.isclose(logs.iloc[1], np.log(102.0 / 100.0))
    assert np.isclose(cum.iloc[-1], 0.05)


def test_realized_vol_estimators_and_diagnostics() -> None:
    close = pd.Series([100.0, 101.0, 102.0, 100.0, 103.0, 104.0])
    open_ = pd.Series([99.0, 100.0, 101.0, 101.0, 102.0, 103.0])
    high = pd.Series([101.0, 102.0, 103.0, 101.5, 104.0, 105.0])
    low = pd.Series([98.0, 99.5, 100.5, 99.5, 101.0, 102.0])
    returns = arithmetic_return(close)

    assert close_to_close_volatility(returns) > 0
    assert float(ewma_volatility(returns).iloc[-1]) > 0
    assert parkinson_volatility(high, low) > 0
    assert garman_klass_volatility(open_, high, low, close) > 0
    assert downside_semivolatility(returns) >= 0
    assert max_drawdown(close) <= 0
    assert not rolling_skewness(returns, 3).dropna().empty
    assert not rolling_kurtosis(returns, 4).dropna().empty
    assert not z_scored_return(returns, 3).dropna().empty
