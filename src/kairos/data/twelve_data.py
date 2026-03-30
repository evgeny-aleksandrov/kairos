from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://api.twelvedata.com"


class TwelveDataError(RuntimeError):
    pass


def _api_key(api_key: str | None = None) -> str:
    resolved = api_key or os.getenv("TWELVE_DATA_API_KEY")
    if not resolved:
        raise TwelveDataError("TWELVE_DATA_API_KEY is not set.")
    return resolved


def _request_json(
    endpoint: str,
    params: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    resolved_api_key = _api_key(api_key)
    query = urlencode(params)
    url = f"{BASE_URL}/{endpoint}?{query}"
    request = Request(
        url,
        headers={
            "Authorization": f"apikey {resolved_api_key}",
            "User-Agent": "kairos/0.1 (+https://local.dev)",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
            message = payload.get("message") or payload.get("status") or body
        except json.JSONDecodeError:
            message = body or str(exc)
        raise TwelveDataError(f"Twelve Data HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise TwelveDataError(f"Twelve Data network error: {exc.reason}") from exc
    if payload.get("status") == "error":
        raise TwelveDataError(payload.get("message", "Twelve Data request failed."))
    return payload


def fetch_qqq_prices(
    start_date: str | None = None,
    end_date: str | None = None,
    outputsize: int = 5000,
    api_key: str | None = None,
) -> pd.DataFrame:
    payload = _request_json(
        "time_series",
        {
            "symbol": "QQQ",
            "interval": "1day",
            "format": "JSON",
            "outputsize": outputsize,
            **({"start_date": start_date} if start_date else {}),
            **({"end_date": end_date} if end_date else {}),
        },
        api_key=api_key,
    )
    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    frame = pd.DataFrame(values).rename(columns={"datetime": "date"})
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["adj_close"] = frame["close"]
    frame = frame[["date", "open", "high", "low", "close", "adj_close", "volume"]]
    return frame.sort_values("date").reset_index(drop=True)


def write_qqq_prices(
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    outputsize: int = 5000,
    api_key: str | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices = fetch_qqq_prices(
        start_date=start_date,
        end_date=end_date,
        outputsize=outputsize,
        api_key=api_key,
    )
    if output_path.suffix == ".parquet":
        prices.to_parquet(output_path, index=False)
    else:
        prices.to_csv(output_path, index=False)
    return output_path


def fetch_qqq_quote(api_key: str | None = None) -> dict[str, Any]:
    return _request_json("quote", {"symbol": "QQQ"}, api_key=api_key)
