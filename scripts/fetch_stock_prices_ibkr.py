from __future__ import annotations

import argparse

from kairos.data.ibkr import write_stock_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch historical stock prices from the IBKR Web API."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output csv or parquet path, e.g. data/prices_ibkr.parquet",
    )
    parser.add_argument("--symbol", default="QQQ", help="Underlying ticker symbol.")
    parser.add_argument("--period", default="1y", help="IBKR history period, e.g. 30d, 1y, 2y.")
    parser.add_argument("--bar", default="1d", help="IBKR bar size, e.g. 1d, 1h.")
    parser.add_argument(
        "--exchange",
        default="SMART",
        help="Exchange routing value for historical data.",
    )
    parser.add_argument(
        "--outside-rth",
        action="store_true",
        help="Include bars outside regular trading hours.",
    )
    parser.add_argument(
        "--source",
        default="trades",
        choices=["trades", "midpoint", "bid_ask", "bid", "ask"],
        help="IBKR historical source type.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = write_stock_history(
        output_path=args.output,
        symbol=args.symbol,
        period=args.period,
        bar=args.bar,
        exchange=args.exchange,
        outside_rth=args.outside_rth,
        source=args.source,
    )
    print(f"Wrote IBKR historical prices to {path}")


if __name__ == "__main__":
    main()
