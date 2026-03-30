from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from kairos.options.black_scholes import discount_factor, option_price
from kairos.options.greeks import vega


@dataclass(slots=True)
class ImpliedVolResult:
    implied_vol: float
    converged: bool
    method: str
    error: str | None = None


def arbitrage_bounds(
    option_type: Literal["call", "put"],
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
) -> tuple[float, float]:
    discounted_spot = spot * discount_factor(dividend_yield, maturity)
    discounted_strike = strike * discount_factor(rate, maturity)
    if option_type == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    else:
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike
    return float(lower), float(upper)


def implied_volatility(
    option_type: Literal["call", "put"],
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    lower_vol: float = 1.0e-6,
    upper_vol: float = 5.0,
    method: Literal["brent", "newton"] = "brent",
    max_iter: int = 100,
    tol: float = 1.0e-8,
    return_error: bool = False,
) -> float | ImpliedVolResult:
    opt_type = option_type.lower()
    lower_bound, upper_bound = arbitrage_bounds(
        opt_type, spot, strike, rate, dividend_yield, maturity
    )
    if (
        not np.isfinite(market_price)
        or market_price < lower_bound - 1.0e-10
        or market_price > upper_bound + 1.0e-10
    ):
        result = ImpliedVolResult(np.nan, False, method, "price_outside_arbitrage_bounds")
        return result if return_error else np.nan

    if maturity <= 0:
        result = ImpliedVolResult(np.nan, False, method, "non_positive_maturity")
        return result if return_error else np.nan

    def objective(vol: float) -> float:
        return float(
            option_price(opt_type, spot, strike, rate, dividend_yield, vol, maturity)
            - market_price
        )

    try:
        if method == "newton":
            sigma = min(max(0.2, lower_vol), upper_vol)
            for _ in range(max_iter):
                price_error = objective(sigma)
                if abs(price_error) < tol:
                    result = ImpliedVolResult(float(sigma), True, "newton")
                    return result if return_error else result.implied_vol
                local_vega = float(vega(spot, strike, rate, dividend_yield, sigma, maturity))
                if local_vega <= 1.0e-10:
                    break
                sigma = np.clip(sigma - price_error / local_vega, lower_vol, upper_vol)

        f_low = objective(lower_vol)
        f_high = objective(upper_vol)
        if f_low * f_high > 0:
            result = ImpliedVolResult(np.nan, False, "brent", "root_not_bracketed")
            return result if return_error else np.nan

        vol = brentq(objective, lower_vol, upper_vol, xtol=tol, maxiter=max_iter)
        result = ImpliedVolResult(float(vol), True, "brent")
        return result if return_error else result.implied_vol
    except ValueError:
        result = ImpliedVolResult(np.nan, False, method, "solver_failure")
        return result if return_error else np.nan


def implied_volatility_vectorized(
    option_type: np.ndarray,
    market_price: np.ndarray,
    spot: np.ndarray,
    strike: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    maturity: np.ndarray,
    **kwargs: float | int | str | bool,
) -> np.ndarray:
    iv_solver = np.vectorize(implied_volatility, otypes=[float])
    return iv_solver(
        option_type,
        market_price,
        spot,
        strike,
        rate,
        dividend_yield,
        maturity,
        **kwargs,
    )


def benchmark_iv_runtime(chain_df, price_column: str = "mid") -> dict[str, float]:
    import time

    start = time.perf_counter()
    implied_volatility_vectorized(
        chain_df["option_type"].to_numpy(),
        chain_df[price_column].to_numpy(),
        chain_df["underlying_price"].to_numpy(),
        chain_df["strike"].to_numpy(),
        chain_df["risk_free_rate"].to_numpy(),
        chain_df["dividend_yield"].to_numpy(),
        chain_df["time_to_expiry"].to_numpy(),
    )
    elapsed = time.perf_counter() - start
    rows = len(chain_df)
    return {
        "rows": float(rows),
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed if elapsed > 0 else np.inf,
    }
