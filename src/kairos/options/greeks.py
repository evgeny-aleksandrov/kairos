from __future__ import annotations

import numpy as np
from scipy.stats import norm

from kairos.options.black_scholes import SQRT_EPS, _as_array, d1, d2, discount_factor


def delta(
    option_type: str | np.ndarray,
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    d1_arr = d1(spot, strike, rate, dividend_yield, sigma, maturity)
    disc_q = discount_factor(dividend_yield, maturity)
    if np.isscalar(option_type):
        return disc_q * norm.cdf(d1_arr) if str(option_type).lower() == "call" else disc_q * (
            norm.cdf(d1_arr) - 1.0
        )
    option_arr = np.char.lower(np.asarray(option_type, dtype=str))
    return np.where(option_arr == "call", disc_q * norm.cdf(d1_arr), disc_q * (norm.cdf(d1_arr) - 1.0))


def gamma(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    sigma_arr = np.maximum(_as_array(sigma), SQRT_EPS)
    maturity_arr = np.maximum(_as_array(maturity), SQRT_EPS)
    d1_arr = d1(spot_arr, strike, rate, dividend_yield, sigma_arr, maturity_arr)
    return discount_factor(dividend_yield, maturity_arr) * norm.pdf(d1_arr) / (
        spot_arr * sigma_arr * np.sqrt(maturity_arr)
    )


def vega(
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    maturity_arr = np.maximum(_as_array(maturity), SQRT_EPS)
    d1_arr = d1(spot_arr, strike, rate, dividend_yield, sigma, maturity_arr)
    return spot_arr * discount_factor(dividend_yield, maturity_arr) * norm.pdf(d1_arr) * np.sqrt(
        maturity_arr
    )


def theta(
    option_type: str | np.ndarray,
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    spot_arr = _as_array(spot)
    strike_arr = _as_array(strike)
    maturity_arr = np.maximum(_as_array(maturity), SQRT_EPS)
    sigma_arr = np.maximum(_as_array(sigma), SQRT_EPS)
    d1_arr = d1(spot_arr, strike_arr, rate, dividend_yield, sigma_arr, maturity_arr)
    d2_arr = d2(spot_arr, strike_arr, rate, dividend_yield, sigma_arr, maturity_arr)
    disc_q = discount_factor(dividend_yield, maturity_arr)
    disc_r = discount_factor(rate, maturity_arr)
    front = -(spot_arr * disc_q * norm.pdf(d1_arr) * sigma_arr) / (2.0 * np.sqrt(maturity_arr))

    if np.isscalar(option_type):
        if str(option_type).lower() == "call":
            return front - rate * strike_arr * disc_r * norm.cdf(d2_arr) + dividend_yield * spot_arr * disc_q * norm.cdf(d1_arr)
        return front + rate * strike_arr * disc_r * norm.cdf(-d2_arr) - dividend_yield * spot_arr * disc_q * norm.cdf(-d1_arr)

    option_arr = np.char.lower(np.asarray(option_type, dtype=str))
    calls = front - rate * strike_arr * disc_r * norm.cdf(d2_arr) + dividend_yield * spot_arr * disc_q * norm.cdf(d1_arr)
    puts = front + rate * strike_arr * disc_r * norm.cdf(-d2_arr) - dividend_yield * spot_arr * disc_q * norm.cdf(-d1_arr)
    return np.where(option_arr == "call", calls, puts)


def rho(
    option_type: str | np.ndarray,
    spot: float | np.ndarray,
    strike: float | np.ndarray,
    rate: float | np.ndarray,
    dividend_yield: float | np.ndarray,
    sigma: float | np.ndarray,
    maturity: float | np.ndarray,
) -> np.ndarray:
    strike_arr = _as_array(strike)
    maturity_arr = np.maximum(_as_array(maturity), 0.0)
    d2_arr = d2(spot, strike, rate, dividend_yield, sigma, np.maximum(maturity_arr, SQRT_EPS))
    disc_r = discount_factor(rate, maturity_arr)
    if np.isscalar(option_type):
        return (
            strike_arr * maturity_arr * disc_r * norm.cdf(d2_arr)
            if str(option_type).lower() == "call"
            else -strike_arr * maturity_arr * disc_r * norm.cdf(-d2_arr)
        )
    option_arr = np.char.lower(np.asarray(option_type, dtype=str))
    calls = strike_arr * maturity_arr * disc_r * norm.cdf(d2_arr)
    puts = -strike_arr * maturity_arr * disc_r * norm.cdf(-d2_arr)
    return np.where(option_arr == "call", calls, puts)
