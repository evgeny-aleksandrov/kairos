from __future__ import annotations

import argparse

from kairos.data.ibkr import write_option_chain_snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch option-chain snapshots from the IBKR Web API."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output csv or parquet path, e.g. data/option_chain_ibkr.parquet",
    )
    parser.add_argument("--symbol", default="QQQ", help="Underlying ticker symbol.")
    parser.add_argument(
        "--month",
        action="append",
        dest="months",
        help="Expiry month in IBKR format, e.g. APR26. Repeat for multiple months.",
    )
    parser.add_argument(
        "--month-limit",
        type=int,
        default=6,
        help="Number of available expiry months to fetch when --month is not provided.",
    )
    parser.add_argument("--exchange", help="Optional explicit exchange. Defaults to SMART or the secdef value.")
    parser.add_argument(
        "--strike-limit-per-month",
        type=int,
        default=25,
        help="Limit selected strikes per month, sampled across the moneyness band.",
    )
    parser.add_argument("--min-moneyness", type=float, default=0.80)
    parser.add_argument("--max-moneyness", type=float, default=1.20)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = write_option_chain_snapshot(
        output_path=args.output,
        symbol=args.symbol,
        months=args.months,
        exchange=args.exchange,
        month_limit=args.month_limit,
        strike_limit_per_month=args.strike_limit_per_month,
        min_moneyness=args.min_moneyness,
        max_moneyness=args.max_moneyness,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
    )
    print(f"Wrote IBKR option chain snapshot to {path}")


if __name__ == "__main__":
    main()
