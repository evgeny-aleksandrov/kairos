from __future__ import annotations

import argparse

from kairos.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the volatility research pipeline.")
    parser.add_argument("--symbol", default="QQQ", help="Underlying ticker symbol.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing IBKR Parquet inputs.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for Parquet outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_pipeline(
        output_dir=args.output_dir,
        symbol=args.symbol,
        data_dir=args.data_dir,
    )
    print("Pipeline completed.")
    print(f"Processed price rows: {len(artifacts.processed_prices)}")
    print(f"Processed option rows: {len(artifacts.processed_chain)}")
    print(f"Smile expiries fit: {artifacts.smile_parameters['expiry'].nunique()}")
    print(f"IV rows/sec: {artifacts.iv_runtime['rows_per_second']:.2f}")


if __name__ == "__main__":
    main()
