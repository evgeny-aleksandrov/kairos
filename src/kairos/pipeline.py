from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kairos.data.options_chain import (
    clean_option_chain,
    load_option_chain,
    write_processed_option_chain,
)
from kairos.data.prices import clean_prices, load_prices, write_processed_prices
from kairos.options.greeks import delta, gamma, rho, theta, vega
from kairos.options.implied_vol import benchmark_iv_runtime, implied_volatility_vectorized
from kairos.options.surface import fit_surface, volatility_surface_plot
from kairos.prices.implied_realized import compare_implied_vs_realized


@dataclass(slots=True)
class PipelineArtifacts:
    processed_prices: pd.DataFrame
    processed_chain: pd.DataFrame
    fitted_chain: pd.DataFrame
    smile_parameters: pd.DataFrame
    implied_realized_comparison: pd.DataFrame
    iv_runtime: dict[str, float]


def enrich_chain_with_iv_and_greeks(
    chain: pd.DataFrame,
    price_column: str = "mid",
) -> pd.DataFrame:
    enriched = chain.copy()
    enriched["implied_vol"] = implied_volatility_vectorized(
        enriched["option_type"].to_numpy(),
        enriched[price_column].to_numpy(),
        enriched["underlying_price"].to_numpy(),
        enriched["strike"].to_numpy(),
        enriched["risk_free_rate"].to_numpy(),
        enriched["dividend_yield"].to_numpy(),
        enriched["time_to_expiry"].to_numpy(),
    )
    enriched["delta"] = delta(
        enriched["option_type"].to_numpy(),
        enriched["underlying_price"].to_numpy(),
        enriched["strike"].to_numpy(),
        enriched["risk_free_rate"].to_numpy(),
        enriched["dividend_yield"].to_numpy(),
        enriched["implied_vol"].to_numpy(),
        enriched["time_to_expiry"].to_numpy(),
    )
    enriched["gamma"] = gamma(
        enriched["underlying_price"].to_numpy(),
        enriched["strike"].to_numpy(),
        enriched["risk_free_rate"].to_numpy(),
        enriched["dividend_yield"].to_numpy(),
        enriched["implied_vol"].to_numpy(),
        enriched["time_to_expiry"].to_numpy(),
    )
    enriched["vega"] = vega(
        enriched["underlying_price"].to_numpy(),
        enriched["strike"].to_numpy(),
        enriched["risk_free_rate"].to_numpy(),
        enriched["dividend_yield"].to_numpy(),
        enriched["implied_vol"].to_numpy(),
        enriched["time_to_expiry"].to_numpy(),
    )
    enriched["theta"] = theta(
        enriched["option_type"].to_numpy(),
        enriched["underlying_price"].to_numpy(),
        enriched["strike"].to_numpy(),
        enriched["risk_free_rate"].to_numpy(),
        enriched["dividend_yield"].to_numpy(),
        enriched["implied_vol"].to_numpy(),
        enriched["time_to_expiry"].to_numpy(),
    )
    enriched["rho"] = rho(
        enriched["option_type"].to_numpy(),
        enriched["underlying_price"].to_numpy(),
        enriched["strike"].to_numpy(),
        enriched["risk_free_rate"].to_numpy(),
        enriched["dividend_yield"].to_numpy(),
        enriched["implied_vol"].to_numpy(),
        enriched["time_to_expiry"].to_numpy(),
    )
    return enriched


def run_pipeline(
    prices_path: str | Path,
    chain_path: str | Path,
    output_dir: str | Path,
) -> PipelineArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_prices = load_prices(prices_path)
    raw_chain = load_option_chain(chain_path)

    processed_prices = clean_prices(raw_prices)
    processed_chain = clean_option_chain(raw_chain)

    write_processed_prices(
        processed_prices,
        output_dir / "processed_prices.parquet",
        output_dir / "processed_prices_drop_log.parquet",
    )
    write_processed_option_chain(
        processed_chain,
        output_dir / "processed_option_chain.parquet",
        output_dir / "processed_option_chain_drop_log.parquet",
    )

    enriched_chain = enrich_chain_with_iv_and_greeks(processed_chain.data)
    enriched_chain.to_parquet(output_dir / "option_chain_with_iv_greeks.parquet", index=False)

    fitted_chain, smile_params = fit_surface(enriched_chain)
    fitted_chain.to_parquet(output_dir / "fitted_smiles.parquet", index=False)
    surface_fig = volatility_surface_plot(smile_params, fitted_chain)
    surface_fig.savefig(output_dir / "volatility_surface.png", dpi=180)

    comparison = compare_implied_vs_realized(enriched_chain, processed_prices.data)
    comparison.to_parquet(output_dir / "implied_vs_realized.parquet", index=False)

    return PipelineArtifacts(
        processed_prices=processed_prices.data,
        processed_chain=processed_chain.data,
        fitted_chain=fitted_chain,
        smile_parameters=smile_params,
        implied_realized_comparison=comparison,
        iv_runtime=benchmark_iv_runtime(enriched_chain),
    )
