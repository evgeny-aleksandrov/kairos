from __future__ import annotations

import numpy as np
from scipy.stats import norm

SQRT_EPS = 1.0e-12


def _as_array(value: float | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float)


def discount_factor(rate: float | np.ndarray, maturity: float | np.ndarray) -> np.ndarray:
    rate_arr = _as_array(rate)
    maturity_arr = np.maximum(_as_array(maturity), 0.0)
    return np.exp(-rate_arr * maturity_arr)


def forward_price(
    spot: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    carry = _as_array(rate) - _as_array(dividend_yield)
    maturity_arr = np.maximum(_as_array(maturity), 0.0)
    return spot_arr * np.exp(carry * maturity_arr)


def d1(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    strike_arr = _as_array(strike)
    sigma_arr = np.maximum(_as_array(sigma), SQRT_EPS)
    maturity_arr = np.maximum(_as_array(maturity), SQRT_EPS)
    numerator = np.log(spot_arr / strike_arr) + (
        _as_array(rate) - _as_array(dividend_yield) + 0.5 * sigma_arr**2
    ) * maturity_arr
    denominator = sigma_arr * np.sqrt(maturity_arr)
    return numerator / denominator


def d2(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    sigma_arr = np.maximum(_as_array(sigma), SQRT_EPS)
    maturity_arr = np.maximum(_as_array(maturity), SQRT_EPS)
    return d1(spot, strike, rate, dividend_yield, sigma_arr, maturity_arr) - sigma_arr * np.sqrt(
        maturity_arr
    )


def call_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    strike_arr = _as_array(strike)
    sigma_arr = _as_array(sigma)
    maturity_arr = np.maximum(_as_array(maturity), 0.0)
    small = (maturity_arr <= SQRT_EPS) | (sigma_arr <= SQRT_EPS)
    d1_arr = d1(spot_arr, strike_arr, rate, dividend_yield, sigma_arr, maturity_arr)
    d2_arr = d2(spot_arr, strike_arr, rate, dividend_yield, sigma_arr, maturity_arr)
    discounted_spot = spot_arr * discount_factor(dividend_yield, maturity_arr)
    discounted_strike = strike_arr * discount_factor(rate, maturity_arr)
    price = discounted_spot * norm.cdf(d1_arr) - discounted_strike * norm.cdf(d2_arr)
    fallback = np.maximum(discounted_spot - discounted_strike, 0.0)
    return np.where(small, fallback, price)


def put_price(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    strike_arr = _as_array(strike)
    sigma_arr = _as_array(sigma)
    maturity_arr = np.maximum(_as_array(maturity), 0.0)
    small = (maturity_arr <= SQRT_EPS) | (sigma_arr <= SQRT_EPS)
    d1_arr = d1(spot_arr, strike_arr, rate, dividend_yield, sigma_arr, maturity_arr)
    d2_arr = d2(spot_arr, strike_arr, rate, dividend_yield, sigma_arr, maturity_arr)
    discounted_spot = spot_arr * discount_factor(dividend_yield, maturity_arr)
    discounted_strike = strike_arr * discount_factor(rate, maturity_arr)
    price = discounted_strike * norm.cdf(-d2_arr) - discounted_spot * norm.cdf(-d1_arr)
    fallback = np.maximum(discounted_strike - discounted_spot, 0.0)
    return np.where(small, fallback, price)


def option_price(
    option_type: str | np.ndarray,
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    if np.isscalar(option_type):
        return (
            call_price(spot, strike, rate, dividend_yield, sigma, maturity)
            if str(option_type).lower() == "call"
            else put_price(spot, strike, rate, dividend_yield, sigma, maturity)
        )

    option_arr = np.char.lower(np.asarray(option_type, dtype=str))
    calls = call_price(spot, strike, rate, dividend_yield, sigma, maturity)
    puts = put_price(spot, strike, rate, dividend_yield, sigma, maturity)
    return np.where(option_arr == "call", calls, puts)
