from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kairos.options.black_scholes import option_price


def generate_synthetic_option_chain(
    quote_date: str = "2026-03-31",
    spot: float = 500.0,
    risk_free_rate: float = 0.04,
    dividend_yield: float = 0.0,
    expiries_days: tuple[int, ...] = (14, 30, 60, 90),
    moneyness_grid: tuple[float, ...] = (0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15),
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quote_ts = pd.Timestamp(quote_date)
    rows: list[dict[str, float | int | str]] = []

    for expiry_days in expiries_days:
        expiry = quote_ts + pd.Timedelta(days=expiry_days)
        maturity = expiry_days / 365.0
        for strike in [round(spot * m, 2) for m in moneyness_grid]:
            log_moneyness = np.log(strike / spot)
            base_vol = 0.18 + 0.04 * np.sqrt(maturity)
            smile_bump = 0.18 * log_moneyness**2 - 0.03 * log_moneyness
            sigma = max(base_vol + smile_bump, 0.05)

            for option_type in ("call", "put"):
                theo = float(
                    option_price(
                        option_type,
                        spot,
                        strike,
                        risk_free_rate,
                        dividend_yield,
                        sigma,
                        maturity,
                    )
                )
                width = max(0.02, 0.01 * theo + 0.03 + 0.15 * abs(log_moneyness))
                micro_noise = rng.normal(0.0, 0.05 * width)
                mid = max(theo + micro_noise, 0.01)
                bid = max(mid - 0.5 * width, 0.01)
                ask = max(mid + 0.5 * width, bid + 0.01)
                last = max(mid + rng.normal(0.0, 0.25 * width), 0.01)

                rows.append(
                    {
                        "quote_date": quote_ts.date().isoformat(),
                        "expiry": expiry.date().isoformat(),
                        "strike": strike,
                        "option_type": option_type,
                        "bid": round(bid, 4),
                        "ask": round(ask, 4),
                        "last": round(last, 4),
                        "volume": int(rng.integers(10, 500)),
                        "open_interest": int(rng.integers(100, 5000)),
                        "underlying_price": spot,
                        "risk_free_rate": risk_free_rate,
                        "dividend_yield": dividend_yield,
                        "synthetic_sigma": round(sigma, 6),
                    }
                )

    frame = pd.DataFrame(rows).sort_values(
        ["expiry", "option_type", "strike"]
    ).reset_index(drop=True)
    return frame


def write_synthetic_option_chain(
    output_path: str | Path,
    **kwargs: object,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_option_chain(**kwargs)
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path
