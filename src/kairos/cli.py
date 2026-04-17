from __future__ import annotations

import argparse

from kairos.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the QQQ volatility research pipeline.")
    parser.add_argument("--prices", required=True, help="Path to raw QQQ daily prices csv/parquet.")
    parser.add_argument("--chain", required=True, help="Path to raw QQQ option chain csv/parquet.")
    parser.add_argument("--output-dir", required=True, help="Directory for parquet outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_pipeline(
        prices_path=args.prices,
        chain_path=args.chain,
        output_dir=args.output_dir,
    )
    print("Pipeline completed.")
    print(f"Processed price rows: {len(artifacts.processed_prices)}")
    print(f"Processed option rows: {len(artifacts.processed_chain)}")
    print(f"Smile expiries fit: {artifacts.smile_parameters['expiry'].nunique()}")
    print(f"IV rows/sec: {artifacts.iv_runtime['rows_per_second']:.2f}")


if __name__ == "__main__":
    main()
