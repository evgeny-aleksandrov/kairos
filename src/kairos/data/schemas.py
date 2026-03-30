from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DroppedRow(BaseModel):
    """Structured record for a row removed during cleaning."""

    model_config = ConfigDict(frozen=True)

    dataset: Literal["prices", "options"]
    row_index: int
    reason: str
    details: str | None = None


class ProcessingSummary(BaseModel):
    """Counts and output locations for a processing run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: Literal["prices", "options"]
    input_rows: int = Field(ge=0)
    output_rows: int = Field(ge=0)
    dropped_rows: int = Field(ge=0)
    output_path: Path | None = None
    drop_log_path: Path | None = None


@dataclass(slots=True)
class ProcessedFrame:
    data: pd.DataFrame
    drop_log: pd.DataFrame
    summary: ProcessingSummary
