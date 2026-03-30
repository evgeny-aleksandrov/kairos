from __future__ import annotations

import argparse

from kairos.data.synthetic import write_synthetic_option_chain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QQQ option-chain snapshot."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output csv or parquet path, e.g. data/qqq_chain.parquet",
    )
    parser.add_argument("--quote-date", default="2026-03-31", help="Snapshot date YYYY-MM-DD.")
    parser.add_argument("--spot", type=float, default=500.0, help="Underlying spot price.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = write_synthetic_option_chain(
        output_path=args.output,
        quote_date=args.quote_date,
        spot=args.spot,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        seed=args.seed,
    )
    print(f"Wrote synthetic option chain to {path}")


if __name__ == "__main__":
    main()
