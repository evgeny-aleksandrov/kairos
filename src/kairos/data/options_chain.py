from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from kairos.data.schemas import DroppedRow, ProcessedFrame, ProcessingSummary

REQUIRED_CHAIN_COLUMNS = [
    "quote_date",
    "expiry",
    "strike",
    "option_type",
    "bid",
    "ask",
    "last",
    "volume",
    "open_interest",
    "underlying_price",
    "risk_free_rate",
    "dividend_yield",
]

OPTION_TYPE_MAP = {
    "c": "call",
    "call": "call",
    "calls": "call",
    "p": "put",
    "put": "put",
    "puts": "put",
}


def load_option_chain(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix != ".parquet":
        raise ValueError(f"Option chain data must be a Parquet file: {path}")
    return pd.read_parquet(path)


def _append_drop_records(
    frame: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    drop_log: list[DroppedRow],
    details: str | None = None,
) -> None:
    for idx in frame.index[mask]:
        drop_log.append(
            DroppedRow(dataset="options", row_index=int(idx), reason=reason, details=details)
        )


def clean_option_chain(df: pd.DataFrame, day_count: int = 365) -> ProcessedFrame:
    """Clean option chain snapshots and compute normalized fields."""

    frame = df.copy()
    missing = [column for column in REQUIRED_CHAIN_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required option chain columns: {missing}")

    frame["quote_date"] = pd.to_datetime(frame["quote_date"], errors="coerce")
    frame["expiry"] = pd.to_datetime(frame["expiry"], errors="coerce")

    numeric_columns = [
        "strike",
        "bid",
        "ask",
        "last",
        "volume",
        "open_interest",
        "underlying_price",
        "risk_free_rate",
        "dividend_yield",
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame["option_type"] = (
        frame["option_type"].astype(str).str.strip().str.lower().map(OPTION_TYPE_MAP)
    )

    drop_log: list[DroppedRow] = []

    missing_required = frame[
        [
            "quote_date",
            "expiry",
            "strike",
            "option_type",
            "underlying_price",
            "risk_free_rate",
            "dividend_yield",
        ]
    ].isna().any(axis=1)
    _append_drop_records(frame, missing_required, "missing_required_fields", drop_log)

    invalid_option_type = frame["option_type"].isna()
    _append_drop_records(
        frame,
        invalid_option_type & ~missing_required,
        "invalid_option_type",
        drop_log,
    )

    invalid_strike_underlying = (frame["strike"] <= 0) | (frame["underlying_price"] <= 0)
    _append_drop_records(
        frame,
        invalid_strike_underlying & ~(missing_required | invalid_option_type),
        "non_positive_strike_or_spot",
        drop_log,
    )

    has_bid_ask = frame["bid"].notna() & frame["ask"].notna()
    crossed_quotes = has_bid_ask & (frame["bid"] > frame["ask"])
    _append_drop_records(
        frame,
        crossed_quotes & ~(missing_required | invalid_option_type | invalid_strike_underlying),
        "crossed_market",
        drop_log,
    )

    frame["mid"] = np.where(has_bid_ask, 0.5 * (frame["bid"] + frame["ask"]), np.nan)
    non_positive_quotes = has_bid_ask & (
        (frame["bid"] <= 0) | (frame["ask"] <= 0) | (frame["mid"] <= 0)
    )
    _append_drop_records(
        frame,
        non_positive_quotes
        & ~(missing_required | invalid_option_type | invalid_strike_underlying | crossed_quotes),
        "non_positive_quote",
        drop_log,
    )

    invalid_last = frame["last"].notna() & (frame["last"] < 0)
    _append_drop_records(
        frame,
        invalid_last
        & ~(
            missing_required
            | invalid_option_type
            | invalid_strike_underlying
            | crossed_quotes
            | non_positive_quotes
        ),
        "negative_last",
        drop_log,
    )

    raw_tte_days = (frame["expiry"] - frame["quote_date"]).dt.total_seconds() / 86400.0
    expired = raw_tte_days <= 0
    _append_drop_records(
        frame,
        expired
        & ~(
            missing_required
            | invalid_option_type
            | invalid_strike_underlying
            | crossed_quotes
            | non_positive_quotes
            | invalid_last
        ),
        "expired_contract",
        drop_log,
    )

    invalid = (
        missing_required
        | invalid_option_type
        | invalid_strike_underlying
        | crossed_quotes
        | non_positive_quotes
        | invalid_last
        | expired
    )

    cleaned = frame.loc[~invalid].copy()
    cleaned["time_to_expiry"] = raw_tte_days.loc[~invalid] / day_count
    cleaned["bid_ask_width"] = cleaned["ask"] - cleaned["bid"]
    cleaned["moneyness"] = cleaned["strike"] / cleaned["underlying_price"] #Maybe adjust this to S/K or log(S/K)
    cleaned["quote_date"] = cleaned["quote_date"].dt.normalize()
    cleaned["expiry"] = cleaned["expiry"].dt.normalize()
    cleaned = cleaned.sort_values(
        ["quote_date", "expiry", "option_type", "strike"]
    ).reset_index(drop=True)

    drop_log_df = pd.DataFrame([record.model_dump() for record in drop_log])
    summary = ProcessingSummary(
        dataset="options",
        input_rows=len(df),
        output_rows=len(cleaned),
        dropped_rows=int(invalid.sum()),
    )
    return ProcessedFrame(cleaned, drop_log_df, summary)


def write_processed_option_chain(
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
