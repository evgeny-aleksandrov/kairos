from __future__ import annotations

import argparse

from kairos.data.ibkr import write_option_chain_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch option-chain snapshots from the IBKR Web API."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for the fetched option chain.",
    )
    parser.add_argument("--symbol", default="QQQ", help="Underlying ticker symbol.")
    parser.add_argument(
        "--month-limit",
        type=int,
        default=6,
        help="Number of available expiry months to fetch when --month is not provided.",
    )
    parser.add_argument(
        "--strike-limit-per-month",
        type=int,
        default=25,
        help="Limit selected strikes per month, sampled across the moneyness band.",
    )
    parser.add_argument("--moneyness-spread", type=float, default=0.20)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = write_option_chain_snapshot(
        output_dir=args.output_dir,
        symbol=args.symbol,
        month_limit=args.month_limit,
        strike_limit_per_month=args.strike_limit_per_month,
        moneyness_spread=args.moneyness_spread,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
    )
    print(f"Wrote IBKR option chain snapshot to {path}")


if __name__ == "__main__":
    main()
