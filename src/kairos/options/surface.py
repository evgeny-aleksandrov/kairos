from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.interpolate import interp1d

from kairos.options.black_scholes import forward_price
from kairos.options.greeks import vega


@dataclass(slots=True)
class SmileFitResult:
    expiry: pd.Timestamp
    coefficients: np.ndarray
    fitted: pd.DataFrame


def add_quote_quality_metrics(chain: pd.DataFrame) -> pd.DataFrame:
    """Add quote-quality fields used to choose mids for surface fitting."""

    frame = chain.copy()
    if "mid" not in frame.columns:
        frame["mid"] = 0.5 * (frame["bid"] + frame["ask"])
    if "bid_ask_width" not in frame.columns:
        frame["bid_ask_width"] = frame["ask"] - frame["bid"]

    frame["relative_bid_ask_width"] = frame["bid_ask_width"] / frame["mid"]

    frame["quote_quality_score"] = (
        frame["relative_bid_ask_width"].replace([np.inf, -np.inf], np.nan).fillna(np.inf)
    )
    return frame


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


def select_surface_quotes(
    chain: pd.DataFrame,
    iv_column: str = "implied_vol",
    atm_log_moneyness_band: float = 0.01,
) -> pd.DataFrame:

    if chain.empty:
        return chain.copy()

    frame = add_quote_quality_metrics(prepare_smile_frame(chain, iv_column=iv_column))
    finite = (
        np.isfinite(frame[iv_column])
        & np.isfinite(frame["mid"])
        & np.isfinite(frame["relative_bid_ask_width"])
        & (frame[iv_column] > 0.0)
        & (frame["mid"] > 0.0)
        & (frame["relative_bid_ask_width"] >= 0.0)
    )
    frame = frame.loc[finite].copy()
    if frame.empty:
        return frame

    option_type = frame["option_type"].astype(str).str.lower()
    log_moneyness = frame["log_moneyness"]
    band = abs(float(atm_log_moneyness_band))

    wing_puts = frame.loc[(option_type == "put") & (log_moneyness < -band)].copy()
    wing_puts["surface_quote_role"] = "otm_put"
    wing_calls = frame.loc[(option_type == "call") & (log_moneyness > band)].copy()
    wing_calls["surface_quote_role"] = "otm_call"

    atm_candidates = frame.loc[log_moneyness.abs() <= band].copy()
    if not atm_candidates.empty:
        group_columns = ["expiry", "strike"]
        if "quote_date" in atm_candidates.columns:
            group_columns.insert(0, "quote_date")
        atm_candidates = atm_candidates.sort_values(
            [
                *group_columns,
                "quote_quality_score",
                "relative_bid_ask_width"
            ],
            ascending=[True] * (len(group_columns) + 2) + [False], #Th sorting here might be wrong
        )
        atm_candidates = atm_candidates.groupby(group_columns, as_index=False).head(1)
        atm_candidates["surface_quote_role"] = "atm_best_mid"

    selected_parts = [wing_puts, atm_candidates, wing_calls]
    non_empty_parts = [part for part in selected_parts if not part.empty]
    if not non_empty_parts:
        empty = frame.iloc[0:0].copy()
        empty["surface_quote_role"] = pd.Series(dtype=object)
        return empty

    selected = pd.concat(non_empty_parts, ignore_index=True)
    return selected.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def fit_smile(
    expiry_frame: pd.DataFrame,
    iv_column: str = "implied_vol",
) -> SmileFitResult:
    if expiry_frame.empty:
        raise ValueError("Cannot fit smile on empty expiry frame.")

    frame = prepare_smile_frame(expiry_frame, iv_column=iv_column)
    x = frame["log_moneyness"].to_numpy()
    y = frame[iv_column].to_numpy()
    design = np.column_stack([np.ones_like(x), x, x**2])

    weights = np.clip(frame["vega"].to_numpy(), 1.0e-8, None)

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
    select_quotes: bool = True,
    atm_log_moneyness_band: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    smile_rows: list[pd.DataFrame] = []
    param_rows: list[dict[str, float | pd.Timestamp]] = []
    surface_chain = (
        select_surface_quotes(
            chain,
            iv_column=iv_column,
            atm_log_moneyness_band=atm_log_moneyness_band,
        )
        if select_quotes
        else chain
    )

    for expiry, group in surface_chain.groupby("expiry", sort=True):
        fit = fit_smile(group, iv_column=iv_column)
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


def volatility_surface_plot(
    params: pd.DataFrame,
    fitted_chain: pd.DataFrame | None = None,
    iv_column: str = "implied_vol",
    log_moneyness_points: int = 45,
    maturity_points: int = 45,
    azimuth: float = -55.0,
    elevation: float = 24.0,
) -> plt.Figure:
    if params.empty:
        raise ValueError("Cannot render volatility surface without smile parameters.")

    maturity_min = float(params["time_to_expiry"].min())
    maturity_max = float(params["time_to_expiry"].max())
    maturity_grid = np.linspace(maturity_min, maturity_max, maturity_points)

    if fitted_chain is not None and not fitted_chain.empty:
        log_moneyness_min = float(fitted_chain["log_moneyness"].quantile(0.02))
        log_moneyness_max = float(fitted_chain["log_moneyness"].quantile(0.98))
    else:
        log_moneyness_min = -0.25
        log_moneyness_max = 0.25

    log_moneyness_grid = np.linspace(
        log_moneyness_min,
        log_moneyness_max,
        log_moneyness_points,
    )
    x_grid, y_grid = np.meshgrid(log_moneyness_grid, maturity_grid)
    z_grid = np.vstack(
        [
            interpolate_surface(params, maturity, log_moneyness_grid)
            for maturity in maturity_grid
        ]
    )

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        x_grid,
        y_grid * 365.0,
        z_grid,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.86,
    )

    if fitted_chain is not None and not fitted_chain.empty:
        points = fitted_chain[
            np.isfinite(fitted_chain["log_moneyness"])
            & np.isfinite(fitted_chain["time_to_expiry"])
            & np.isfinite(fitted_chain[iv_column])
        ]
        ax.scatter(
            points["log_moneyness"],
            points["time_to_expiry"] * 365.0,
            points[iv_column],
            color="black",
            s=10,
            alpha=0.42,
            depthshade=False,
        )

    title = "Volatility Surface"
    if fitted_chain is not None and not fitted_chain.empty:
        title_parts: list[str] = []
        if "symbol" in fitted_chain.columns:
            symbols = fitted_chain["symbol"].dropna().astype(str).unique()
            if len(symbols) == 1:
                title_parts.append(symbols[0])
        if "quote_date" in fitted_chain.columns:
            quote_dates = pd.to_datetime(
                fitted_chain["quote_date"].dropna(),
                errors="coerce",
            ).dropna()
            if not quote_dates.empty:
                title_parts.append(f"Quote {quote_dates.max():%Y-%m-%d}")
        if title_parts:
            title = f"{' - '.join(title_parts)} Volatility Surface"

    ax.set_title(title)
    ax.set_xlabel("Log-moneyness log(K / F)")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied volatility")
    ax.view_init(elev=elevation, azim=azimuth)
    fig.colorbar(surface, ax=ax, shrink=0.62, pad=0.08, label="Implied volatility")
    fig.tight_layout()
    return fig
