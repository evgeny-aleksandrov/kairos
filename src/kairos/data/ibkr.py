from __future__ import annotations

import json
import os
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np


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
    - Add more complete session/bootstrap helpers when the repo needs them.
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

    def resolve_stock_conid(self, symbol: str) -> int:
        payload = self._request_json("GET", "trsrv/stocks", {"symbols": symbol.upper()})
        candidates = payload.get(symbol.upper(), [])
        for candidate in candidates:
            if candidate.get("assetClass") != "STK":
                continue
            for contract in candidate.get("contracts", []):
                if contract.get("isUS"):
                    return int(contract["conid"])
        raise IBKRError(f"Could not resolve a US stock conid for {symbol}.")

    def search_secdef(self, symbol: str, sec_type: str = "STK") -> list[dict[str, Any]]:
        payload = self._request_json(
            "GET",
            "iserver/secdef/search",
            {"symbol": symbol.upper(), "secType": sec_type},
        )
        if not isinstance(payload, list):
            raise IBKRError(f"Unexpected secdef search response for {symbol}: {payload}")
        return payload

    def fetch_option_strikes(
        self,
        underlying_conid: int,
        month: str,
        exchange: str = "SMART",
    ) -> dict[str, Any]:
        return self._request_json(
            "GET",
            "iserver/secdef/strikes",
            {
                "conid": str(underlying_conid),
                "sectype": "OPT",
                "month": month,
                "exchange": exchange,
            },
        )

    def fetch_option_contract_info(
        self,
        underlying_conid: int,
        month: str,
        strike: float,
        right: str,
        exchange: str = "SMART",
    ) -> list[dict[str, Any]]:
        payload = self._request_json(
            "GET",
            "iserver/secdef/info",
            {
                "conid": str(underlying_conid),
                "sectype": "OPT",
                "month": month,
                "exchange": exchange,
                "strike": str(strike),
                "right": right,
            },
        )
        if not isinstance(payload, list):
            raise IBKRError(
                f"Unexpected secdef info response for conid={underlying_conid}, month={month}, strike={strike}, right={right}: {payload}"
            )
        return payload

    def fetch_marketdata_snapshot(
        self,
        conids: list[int],
        fields: list[str],
        hydrate_seconds: float = 0.75, #Increasing to 0.75 seconds to allow IBKR to populate the bid/ask data.
    ) -> list[dict[str, Any]]:
        if not conids:
            return []
        params = {
            "conids": ",".join(str(conid) for conid in conids),
            "fields": ",".join(fields),
        }
        self._request_json("GET", "iserver/marketdata/snapshot", params)
        if hydrate_seconds > 0:
            time.sleep(hydrate_seconds)
        payload = self._request_json("GET", "iserver/marketdata/snapshot", params)
        if not isinstance(payload, list):
            raise IBKRError(f"Unexpected marketdata snapshot response: {payload}")
        return payload

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
    output_dir: str | Path,
    output_format: str = "parquet",
    symbol: str = "QQQ",
    period: str = "1y",
    bar: str = "1d",
    exchange: str = "SMART",
    outside_rth: bool = False,
    source: str = "trades",
    client: IBKRWebApiClient | None = None,
) -> Path:
    output_path = Path(output_dir) / f"{symbol.lower()}_prices_ibkr.{output_format}"
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
        "IBKR option-chain integration is now in fetch_option_chain_snapshot()."
    )


def _extract_option_months(search_results: list[dict[str, Any]]) -> tuple[int, list[str], str]:
    for result in search_results:
        conid = result.get("conid")
        for section in result.get("sections", []):
            if section.get("secType") == "OPT":
                months = str(section.get("months", "")).split(";")
                months = [month for month in months if month]
                exchange = str(section.get("exchange", "SMART")).split(";")[0]
                if conid and months:
                    return int(conid), months, exchange
    raise IBKRError("Could not extract option months from secdef search response.")


def _safe_float(value: Any) -> float:
    if value in (None, "", "--"):
        return np.nan
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return np.nan


def _select_strikes_around_spot(
    strikes: list[float],
    spot: float,
    strike_limit: int | None,
    min_moneyness: float | None = None,
    max_moneyness: float | None = None,
) -> list[float]:
    unique = sorted({float(strike) for strike in strikes})

    if not np.isnan(spot) and min_moneyness is not None:
        unique = [strike for strike in unique if strike >= spot * min_moneyness]
    if not np.isnan(spot) and max_moneyness is not None:
        unique = [strike for strike in unique if strike <= spot * max_moneyness]

    if not unique:
        return []
    if strike_limit is None or strike_limit <= 0 or strike_limit >= len(unique):
        return unique
    if np.isnan(spot):
        indexes = np.linspace(0, len(unique) - 1, strike_limit).round().astype(int)
        return [unique[idx] for idx in sorted(set(indexes))]
    if min_moneyness is None and max_moneyness is None:
        ordered = sorted(unique, key=lambda strike: (abs(strike - spot), strike))
        return sorted(ordered[:strike_limit])

    atm_index = min(range(len(unique)), key=lambda idx: abs(unique[idx] - spot))
    indexes = set(np.linspace(0, len(unique) - 1, strike_limit - 1).round().astype(int))
    indexes.add(atm_index)
    return [unique[idx] for idx in sorted(indexes)]


def _chunked(values: list[int], chunk_size: int) -> list[list[int]]:
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def fetch_option_chain_snapshot(
    symbol: str = "QQQ",
    months: list[str] | None = None,
    exchange: str | None = None,
    month_limit: int | None = 6,
    strike_limit_per_month: int | None = 25,
    min_moneyness: float | None = 0.80,
    max_moneyness: float | None = 1.20,
    rights: tuple[str, ...] = ("C", "P"),
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    client: IBKRWebApiClient | None = None,
) -> pd.DataFrame:
    resolved_client = client or IBKRWebApiClient()
    resolved_client.ensure_brokerage_session()

    search_results = resolved_client.search_secdef(symbol=symbol, sec_type="STK")
    underlying_conid, available_months, discovered_exchange = _extract_option_months(search_results)
    resolved_exchange = exchange or discovered_exchange or "SMART"
    if months is not None:
        target_months = months
    elif month_limit is None or month_limit <= 0:
        target_months = available_months
    else:
        target_months = available_months[:month_limit]

    underlying_snapshot = resolved_client.fetch_marketdata_snapshot(
        [underlying_conid],
        fields=["31"],
    )
    spot = _safe_float(underlying_snapshot[0].get("31")) if underlying_snapshot else np.nan

    contracts: list[dict[str, Any]] = []
    for month in target_months:
        strike_payload = resolved_client.fetch_option_strikes(
            underlying_conid=underlying_conid,
            month=month,
            exchange=resolved_exchange,
        )
        strike_pool: list[float] = []
        for right in rights:
            side_key = "call" if right == "C" else "put"
            strike_pool.extend(strike_payload.get(side_key, []))
        selected_strikes = _select_strikes_around_spot(
            strike_pool,
            spot=spot,
            strike_limit=strike_limit_per_month,
            min_moneyness=min_moneyness,
            max_moneyness=max_moneyness,
        )

        for strike in selected_strikes:
            for right in rights:
                info_rows = resolved_client.fetch_option_contract_info(
                    underlying_conid=underlying_conid,
                    month=month,
                    strike=strike,
                    right=right,
                    exchange=resolved_exchange,
                )
                if info_rows:
                    contracts.append(info_rows[0])

    if not contracts:
        return pd.DataFrame(
            columns=[
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
                "symbol",
                "conid",
                "exchange",
                "has_delayed",
            ]
        )

    snapshot_fields = ["31", "84", "86", "7089", "7638", "6509"]
    snapshot_rows: list[dict[str, Any]] = []
    for conid_batch in _chunked([int(contract["conid"]) for contract in contracts], 100):
        snapshot_rows.extend(
            resolved_client.fetch_marketdata_snapshot(conid_batch, fields=snapshot_fields)
        )
    snapshot_map = {int(row["conid"]): row for row in snapshot_rows if row.get("conid") is not None}

    quote_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    rows: list[dict[str, Any]] = []
    for contract in contracts:
        conid = int(contract["conid"])
        snapshot = snapshot_map.get(conid, {})
        maturity = str(contract.get("maturityDate", ""))
        expiry = pd.to_datetime(maturity, format="%Y%m%d", errors="coerce")
        rows.append(
            {
                "quote_date": quote_date,
                "expiry": expiry,
                "strike": _safe_float(contract.get("strike")),
                "option_type": "call" if contract.get("right") == "C" else "put",
                "bid": _safe_float(snapshot.get("84")),
                "ask": _safe_float(snapshot.get("86")),
                "last": _safe_float(snapshot.get("31")),
                "volume": _safe_float(snapshot.get("7089")),
                "open_interest": _safe_float(snapshot.get("7638")),
                "underlying_price": spot,
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "symbol": contract.get("symbol", symbol.upper()),
                "conid": conid,
                "exchange": contract.get("exchange", resolved_exchange),
                "has_delayed": snapshot.get("6509"),
                "maturity_date": maturity,
                "trading_class": contract.get("tradingClass"),
                "multiplier": contract.get("multiplier"),
            }
        )

    frame = pd.DataFrame(rows).sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)
    return frame


def write_option_chain_snapshot(
    output_path: str | Path,
    symbol: str = "QQQ",
    months: list[str] | None = None,
    exchange: str | None = None,
    month_limit: int | None = 6,
    strike_limit_per_month: int | None = 25,
    min_moneyness: float | None = 0.80,
    max_moneyness: float | None = 1.20,
    rights: tuple[str, ...] = ("C", "P"),
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    client: IBKRWebApiClient | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_option_chain_snapshot(
        symbol=symbol,
        months=months,
        exchange=exchange,
        month_limit=month_limit,
        strike_limit_per_month=strike_limit_per_month,
        min_moneyness=min_moneyness,
        max_moneyness=max_moneyness,
        rights=rights,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        client=client,
    )
    if output_path.suffix == ".parquet":
        frame.to_parquet(output_path, index=False)
    else:
        frame.to_csv(output_path, index=False)
    return output_path
