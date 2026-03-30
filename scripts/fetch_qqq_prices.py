from __future__ import annotations

import argparse

from kairos.data.twelve_data import write_qqq_prices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch QQQ daily prices from Twelve Data.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output csv or parquet path, e.g. data/qqq_prices.parquet",
    )
    parser.add_argument("--start-date", help="Optional YYYY-MM-DD start date.")
    parser.add_argument("--end-date", help="Optional YYYY-MM-DD end date.")
    parser.add_argument(
        "--outputsize",
        type=int,
        default=5000,
        help="Maximum number of rows to request.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = write_qqq_prices(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
        outputsize=args.outputsize,
    )
    print(f"Wrote QQQ prices to {path}")


if __name__ == "__main__":
    main()
