from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kairos.data.schemas import DroppedRow, ProcessedFrame, ProcessingSummary

PRICE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def load_prices(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load QQQ daily prices from csv or parquet."""

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    return pd.read_csv(path, **kwargs)


def clean_prices(df: pd.DataFrame) -> ProcessedFrame:
    """Validate and normalize OHLCV daily prices."""

    frame = df.copy()
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric_columns = [column for column in PRICE_COLUMNS if column != "date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

    drop_log: list[DroppedRow] = []

    def mark_invalid(mask: pd.Series, reason: str, details: str | None = None) -> None:
        for idx in frame.index[mask]:
            drop_log.append(
                DroppedRow(dataset="prices", row_index=int(idx), reason=reason, details=details)
            )

    null_mask = frame[["date", "open", "high", "low", "close"]].isna().any(axis=1)
    mark_invalid(null_mask, "missing_required_fields")

    non_positive_mask = (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    mark_invalid(non_positive_mask & ~null_mask, "non_positive_ohlc")

    inconsistent_ohlc = (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    )
    mark_invalid(inconsistent_ohlc & ~null_mask & ~non_positive_mask, "inconsistent_ohlc")

    invalid = null_mask | non_positive_mask | inconsistent_ohlc
    cleaned = (
        frame.loc[~invalid]
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    cleaned["adj_close"] = cleaned["adj_close"].fillna(cleaned["close"])
    cleaned["volume"] = cleaned["volume"].fillna(0.0)
    cleaned["simple_return"] = cleaned["adj_close"].pct_change()
    cleaned["log_return"] = np.log(cleaned["adj_close"]).diff()

    drop_log_df = pd.DataFrame([record.model_dump() for record in drop_log])
    summary = ProcessingSummary(
        dataset="prices",
        input_rows=len(df),
        output_rows=len(cleaned),
        dropped_rows=int(invalid.sum()),
    )
    return ProcessedFrame(cleaned, drop_log_df, summary)


def write_processed_prices(
    processed: ProcessedFrame,
    output_path: str | Path,
    drop_log_path: str | Path | None = None,
) -> ProcessingSummary:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.data.to_parquet(output_path, index=False)

    resolved_drop_log_path: Path | None = None
    if drop_log_path is not None:
        resolved_drop_log_path = Path(drop_log_path)
        resolved_drop_log_path.parent.mkdir(parents=True, exist_ok=True)
        processed.drop_log.to_parquet(resolved_drop_log_path, index=False)

    return processed.summary.model_copy(
        update={"output_path": output_path, "drop_log_path": resolved_drop_log_path}
    )
