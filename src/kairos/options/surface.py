from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from kairos.options.black_scholes import forward_price
from kairos.options.greeks import vega


@dataclass(slots=True)
class SmileFitResult:
    expiry: pd.Timestamp
    coefficients: np.ndarray
    fitted: pd.DataFrame


def prepare_smile_frame(chain: pd.DataFrame, iv_column: str = "implied_vol") -> pd.DataFrame:
    frame = chain.copy()
    frame["forward"] = forward_price(
        frame["underlying_price"].to_numpy(),
        frame["risk_free_rate"].to_numpy(),
        frame["dividend_yield"].to_numpy(),
        frame["time_to_expiry"].to_numpy(),
    )
    frame["log_moneyness"] = np.log(frame["strike"] / frame["forward"])
    if "vega" not in frame.columns:
        frame["vega"] = vega(
            frame["underlying_price"].to_numpy(),
            frame["strike"].to_numpy(),
            frame["risk_free_rate"].to_numpy(),
            frame["dividend_yield"].to_numpy(),
            frame[iv_column].to_numpy(),
            frame["time_to_expiry"].to_numpy(),
        )
    return frame


def fit_smile(
    expiry_frame: pd.DataFrame,
    iv_column: str = "implied_vol",
    weighting: str = "vega",
) -> SmileFitResult:
    if expiry_frame.empty:
        raise ValueError("Cannot fit smile on empty expiry frame.")

    frame = prepare_smile_frame(expiry_frame, iv_column=iv_column)
    x = frame["log_moneyness"].to_numpy()
    y = frame[iv_column].to_numpy()
    design = np.column_stack([np.ones_like(x), x, x**2])

    if weighting == "vega":
        weights = np.clip(frame["vega"].to_numpy(), 1.0e-8, None)
    elif weighting == "inverse_bid_ask":
        if "bid_ask_width" not in frame.columns:
            raise ValueError("bid_ask_width required for inverse_bid_ask weighting.")
        weights = 1.0 / np.clip(frame["bid_ask_width"].to_numpy(), 1.0e-8, None)
    else:
        weights = np.ones_like(y)

    sqrt_w = np.sqrt(weights)
    lhs = design * sqrt_w[:, None]
    rhs = y * sqrt_w
    coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    fitted_iv = design @ coeffs

    fitted = frame.assign(
        smile_fitted_vol=fitted_iv,
        smile_residual=frame[iv_column] - fitted_iv,
        smile_weight=weights,
    )
    expiry = pd.Timestamp(frame["expiry"].iloc[0])
    return SmileFitResult(expiry=expiry, coefficients=coeffs, fitted=fitted)


def fit_surface(
    chain: pd.DataFrame,
    iv_column: str = "implied_vol",
    weighting: str = "vega",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    smile_rows: list[pd.DataFrame] = []
    param_rows: list[dict[str, float | pd.Timestamp]] = []

    for expiry, group in chain.groupby("expiry", sort=True):
        fit = fit_smile(group, iv_column=iv_column, weighting=weighting)
        smile_rows.append(fit.fitted)
        param_rows.append(
            {
                "expiry": pd.Timestamp(expiry),
                "time_to_expiry": float(group["time_to_expiry"].iloc[0]),
                "a": float(fit.coefficients[0]),
                "b": float(fit.coefficients[1]),
                "c": float(fit.coefficients[2]),
            }
        )

    fitted_chain = pd.concat(smile_rows, ignore_index=True) if smile_rows else pd.DataFrame()
    params = pd.DataFrame(param_rows).sort_values("time_to_expiry").reset_index(drop=True)
    return fitted_chain, params


def interpolate_surface(
    params: pd.DataFrame,
    maturity: float,
    log_moneyness: np.ndarray | float,
) -> np.ndarray:
    if params.empty:
        raise ValueError("No smile parameters available for interpolation.")

    maturity_grid = params["time_to_expiry"].to_numpy()
    coeff_matrix = params[["a", "b", "c"]].to_numpy()
    interp = interp1d(
        maturity_grid,
        coeff_matrix,
        axis=0,
        fill_value="extrapolate",
        bounds_error=False,
    )
    coeffs = interp(maturity)
    k = np.asarray(log_moneyness, dtype=float)
    return coeffs[..., 0] + coeffs[..., 1] * k + coeffs[..., 2] * k**2


def smile_plot_by_expiry(
    fitted_chain: pd.DataFrame,
    iv_column: str = "implied_vol",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    for expiry, group in fitted_chain.groupby("expiry", sort=True):
        group = group.sort_values("log_moneyness")
        ax.scatter(
            group["log_moneyness"],
            group[iv_column],
            s=18,
            alpha=0.7,
            label=f"{expiry:%Y-%m-%d} obs",
        )
        ax.plot(
            group["log_moneyness"],
            group["smile_fitted_vol"],
            linewidth=1.8,
            label=f"{expiry:%Y-%m-%d} fit",
        )
    ax.set_title("QQQ Volatility Smile by Expiry")
    ax.set_xlabel("Log-moneyness log(K / F)")
    ax.set_ylabel("Implied volatility")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)
    return fig


def residual_plot(fitted_chain: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(
        fitted_chain["log_moneyness"],
        fitted_chain["smile_residual"],
        alpha=0.7,
        s=18,
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Smile Fit Residuals")
    ax.set_xlabel("Log-moneyness")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.2)
    return fig


def parameter_table_over_time(params: pd.DataFrame) -> pd.DataFrame:
    table = params.copy()
    table["curvature_rank"] = table["c"].rank(ascending=False, method="dense")
    return table
