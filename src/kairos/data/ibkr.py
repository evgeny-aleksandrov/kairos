from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_BASE_URL = "https://localhost:5000/v1/api"


class IBKRError(RuntimeError):
    pass


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class IBKRWebApiClient:
    """
    Minimal IBKR Web API client for historical stock bars.

    TODO:
    - Add Interactive Brokers option-chain retrieval via secdef endpoints.
    """

    def __init__(
        self,
        base_url: str | None = None,
        access_token: str | None = None,
        cookie: str | None = None,
        insecure: bool | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("IBKR_WEB_API_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.access_token = access_token or os.getenv("IBKR_WEB_API_ACCESS_TOKEN")
        self.cookie = cookie or os.getenv("IBKR_WEB_API_COOKIE")
        self.insecure = _bool_env("IBKR_WEB_API_INSECURE", True) if insecure is None else insecure

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self.insecure:
            return None
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "User-Agent": "kairos/0.1 (+https://local.dev)",
        }
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        if self.cookie:
            headers["Cookie"] = self.cookie
        return headers

    def _request_json(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        query = urlencode(params or {})
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers=self._headers(), method=method.upper())
        try:
            with urlopen(request, context=self._ssl_context()) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network path
            raise IBKRError(f"IBKR request failed for {endpoint}: {exc}") from exc
        if isinstance(payload, dict) and payload.get("error"):
            raise IBKRError(str(payload["error"]))
        return payload

    def ensure_brokerage_session(self) -> Any:
        return self._request_json("GET", "iserver/accounts")

    def resolve_stock_conid(self, symbol: str = "QQQ") -> int:
        payload = self._request_json("GET", "trsrv/stocks", {"symbols": symbol.upper()})
        candidates = payload.get(symbol.upper(), [])
        for candidate in candidates:
            if candidate.get("assetClass") != "STK":
                continue
            for contract in candidate.get("contracts", []):
                if contract.get("isUS"):
                    return int(contract["conid"])
        raise IBKRError(f"Could not resolve a US stock conid for {symbol}.")

    def fetch_historical_bars(
        self,
        conid: int,
        exchange: str = "SMART",
        period: str = "1y",
        bar: str = "1d",
        outside_rth: bool = False,
        source: str = "trades",
    ) -> dict[str, Any]:
        params = {
            "conid": str(conid),
            "exchange": exchange,
            "period": period,
            "bar": bar,
            "outsideRth": str(outside_rth).lower(),
            "source": source.title(),
        }
        return self._request_json("GET", "iserver/marketdata/history", params)


def _normalize_historical_points(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("data", [])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "adj_close", "volume"]
        )

    rename_map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "t": "date",
    }
    frame = frame.rename(
        columns={key: value for key, value in rename_map.items() if key in frame.columns}
    )

    if "date" in frame.columns:
        frame["date"] = (
            pd.to_datetime(frame["date"], unit="ms", errors="coerce").dt.tz_localize(None)
        )
    elif "time" in frame.columns:
        frame["date"] = (
            pd.to_datetime(frame["time"], unit="ms", errors="coerce").dt.tz_localize(None)
        )
    else:
        raise IBKRError(
            "IBKR historical response did not contain a recognized time field."
        )

    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.NA

    volume_factor = payload.get("volumeFactor")
    if volume_factor not in (None, 0, "0"):
        frame["volume"] = frame["volume"] * float(volume_factor)

    frame["adj_close"] = frame["close"]
    normalized = frame[
        ["date", "open", "high", "low", "close", "adj_close", "volume"]
    ]
    return normalized.sort_values("date").reset_index(drop=True)


def fetch_stock_history(
    symbol: str = "QQQ",
    period: str = "1y",
    bar: str = "1d",
    exchange: str = "SMART",
    outside_rth: bool = False,
    source: str = "trades",
    client: IBKRWebApiClient | None = None,
) -> pd.DataFrame:
    resolved_client = client or IBKRWebApiClient()
    resolved_client.ensure_brokerage_session()
    conid = resolved_client.resolve_stock_conid(symbol=symbol)
    payload = resolved_client.fetch_historical_bars(
        conid=conid,
        exchange=exchange,
        period=period,
        bar=bar,
        outside_rth=outside_rth,
        source=source,
    )
    return _normalize_historical_points(payload)


def write_stock_history(
    output_path: str | Path,
    symbol: str = "QQQ",
    period: str = "1y",
    bar: str = "1d",
    exchange: str = "SMART",
    outside_rth: bool = False,
    source: str = "trades",
    client: IBKRWebApiClient | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_stock_history(
        symbol=symbol,
        period=period,
        bar=bar,
        exchange=exchange,
        outside_rth=outside_rth,
        source=source,
        client=client,
    )
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path


def ibkr_option_chain_todo() -> None:
    raise NotImplementedError(
        "IBKR option-chain integration is not implemented yet."
    )
