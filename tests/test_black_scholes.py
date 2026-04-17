from __future__ import annotations

import numpy as np

from kairos.options.black_scholes import call_price, discount_factor, put_price
from kairos.options.greeks import delta, gamma, rho, theta, vega
from kairos.options.implied_vol import (
    arbitrage_bounds,
    implied_volatility,
    implied_volatility_vectorized,
)


def test_black_scholes_put_call_parity() -> None:
    spot = 500.0
    strike = 510.0
    rate = 0.03
    dividend = 0.01
    sigma = 0.2
    maturity = 0.5

    call = float(call_price(spot, strike, rate, dividend, sigma, maturity))
    put = float(put_price(spot, strike, rate, dividend, sigma, maturity))
    parity_rhs = spot * discount_factor(dividend, maturity) - strike * discount_factor(rate, maturity)

    assert np.isclose(call - put, parity_rhs, atol=1.0e-8)


def test_greeks_match_benchmark_values() -> None:
    spot = 100.0
    strike = 100.0
    rate = 0.05
    dividend = 0.02
    sigma = 0.2
    maturity = 1.0

    assert np.isclose(float(delta("call", spot, strike, rate, dividend, sigma, maturity)), 0.586851, atol=1.0e-6)
    assert np.isclose(float(delta("put", spot, strike, rate, dividend, sigma, maturity)), -0.393348, atol=1.0e-6)
    assert np.isclose(float(gamma(spot, strike, rate, dividend, sigma, maturity)), 0.018951, atol=1.0e-6)
    assert np.isclose(float(vega(spot, strike, rate, dividend, sigma, maturity)), 37.901158, atol=1.0e-6)
    assert np.isclose(float(theta("call", spot, strike, rate, dividend, sigma, maturity)), -5.089319, atol=1.0e-6)
    assert np.isclose(float(rho("call", spot, strike, rate, dividend, sigma, maturity)), 49.458109, atol=1.0e-6)


def test_implied_volatility_recovers_input_sigma() -> None:
    market_price = float(call_price(100.0, 105.0, 0.03, 0.01, 0.25, 0.75))
    solved = implied_volatility("call", market_price, 100.0, 105.0, 0.03, 0.01, 0.75)
    assert np.isclose(solved, 0.25, atol=1.0e-6)


def test_implied_volatility_clips_price_just_below_lower_bound() -> None:
    lower, _ = arbitrage_bounds("call", 100.0, 80.0, 0.03, 0.01, 0.5)
    solved = implied_volatility("call", lower * 0.99, 100.0, 80.0, 0.03, 0.01, 0.5)

    assert np.isfinite(solved)


def test_implied_volatility_clips_price_just_above_upper_bound() -> None:
    _, upper = arbitrage_bounds("call", 100.0, 80.0, 0.03, 0.01, 0.5)
    solved = implied_volatility("call", upper * 1.01, 100.0, 80.0, 0.03, 0.01, 0.5)

    assert np.isfinite(solved)


def test_vectorized_implied_volatility() -> None:
    prices = np.array(
        [
            float(call_price(100.0, 95.0, 0.01, 0.0, 0.2, 0.5)),
            float(put_price(100.0, 105.0, 0.01, 0.0, 0.3, 0.5)),
        ]
    )
    vols = implied_volatility_vectorized(
        np.array(["call", "put"]),
        prices,
        np.array([100.0, 100.0]),
        np.array([95.0, 105.0]),
        np.array([0.01, 0.01]),
        np.array([0.0, 0.0]),
        np.array([0.5, 0.5]),
    )
    assert np.allclose(vols, np.array([0.2, 0.3]), atol=1.0e-6)
