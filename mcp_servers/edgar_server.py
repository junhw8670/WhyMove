from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()
mcp = FastMCP("EDGAR")

UA = os.getenv("EDGAR_USER_AGENT", "WhyMove research junhw8670@gmail.com")
EDGAR = "https://www.sec.gov"
EDGAR_DATA = "https://data.sec.gov"

MATERIAL_FORMS = {
    "10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A",
    "S-1", "S-3", "S-4", "DEF 14A", "20-F", "6-K", "13F",
}

_ticker_to_cik: dict[str, str] | None = None


def _req(url: str) -> requests.Response:
    """EDGAR HTTP GET with required User-Agent + error guard."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r

def _cik(ticker: str) -> str:
    """Resolve US symbol -> 10-digit CIK string. Lazy-cache the full index on first call."""
    global _ticker_to_cik
    if _ticker_to_cik is None:
        raw = _req(f"{EDGAR}/files/company_tickers.json").json()
        _ticker_to_cik = {
            e["ticker"].upper(): str(e["cik_str"]).zfill(10)
            for e in raw.values()
        }
    cik = _ticker_to_cik.get(ticker.upper())
    if cik is None:
        raise ValueError(f"US ticker not in EDGAR index: {ticker}")
    return cik

@mcp.tool()
def fetch_filings_around(ticker: str, event_date: str, lookback_days: int = 7) -> dict:
    """Find material SEC filings from lookback_days to event_date.

    Args:
        ticker: US symbol
        event_date: 'YYYY-MM-DD' or 'YYYYMMDD'.
        lookback_days: days to look back. Default 7.

    Returns:
        {ticker, filings: [{filing_id, filing_date, form, title, primary_doc}, ...]}
    """
    cik = _cik(ticker)

    d = datetime.strptime(event_date.replace("-", ""), "%Y%m%d")
    bgn = d - timedelta(days=lookback_days)

    recent = _req(f"{EDGAR_DATA}/submissions/CIK{cik}.json").json().get("filings", {}).get("recent", {})

    out = []
    for form, acc, fdate, pdoc, pdesc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("filingDate", []),
        recent.get("primaryDocument", []),
        recent.get("primaryDocDescription", []),
    ):
        if form not in MATERIAL_FORMS:
            continue
        try:
            fd = datetime.strptime(fdate, "%Y-%m-%d")
        except ValueError:
            continue
        if fd < bgn or fd > d:
            continue
        out.append({
            "filing_id": acc,
            "filing_date": fdate,
            "form": form,
            "title": pdesc or form,
            "primary_doc": pdoc,
        })
    return {"ticker": ticker, "filings": out}


@mcp.tool()
def fetch_multi_quarters(ticker: str, n_quarters: int = 5) -> dict:
    """Return quarterly figures of last n_quarters (chronological).

    Returns:
        {ticker, periods: ['2024-09-30', ...], by_period: {<period>: {...}}}
    """
    tk = yf.Ticker(ticker)
    fin, bs = tk.quarterly_financials, tk.quarterly_balance_sheet
    if fin.empty:
        raise ValueError(f"No US quarterly financial for {ticker}")

    cols = list(fin.columns[:n_quarters])[::-1]
    periods = [c.strftime("%Y-%m-%d") for c in cols]

    def _val(df: pd.DataFrame, name: str, col) -> Optional[float]:
        if name not in df.index:
            return None
        v = df.at[name, col]
        return None if pd.isna(v) else float(v)

    by_period = {
        periods[i]: {
            "Revenue":          _val(fin, "Total Revenue", cols[i]),
            "OperatingIncome":  _val(fin, "Operating Income", cols[i]),
            "NetIncome":        _val(fin, "Net Income", cols[i]),
            "TotalAssets":      _val(bs,  "Total Assets", cols[i]),
            "TotalLiabilities": _val(bs,  "Total Liabilities Net Minority Interest", cols[i]),
            "TotalEquity":      _val(bs,  "Stockholders Equity", cols[i]),
        }
        for i in range(len(cols))
    }
    return {"ticker": ticker, "periods": periods, "by_period": by_period}


@mcp.tool()
def fetch_multi_years(ticker: str, n_years: int = 5) -> dict:
    """Return last n_years of annual figures (chronological) + YoY growth %.

    Args:
        ticker: US symbol.
        n_years: number of years being fetched. Default 5.

    Returns:
        {
            "ticker": "AAPL",
            "years": [2021, 2022, 2023, 2024, 2025],
            "by_year": {
                2021: {Revenue, OperatingIncome, ..., TotalEquity},
                2022: {...},
                ...
            },
            "growth_rates": {
                "Revenue": [0, 7.5, 2.0, -2.1, 5.1],
                "OperatingIncome": [...],
                ...
            }
        }
    """
    tk = yf.Ticker(ticker)
    fin, bs = tk.financials, tk.balance_sheet
    if fin.empty:
        raise ValueError(f"No US financial for {ticker}")

    cols = list(fin.columns[:n_years])[::-1]
    years = [c.year for c in cols]

    def _val(df: pd.DataFrame, name: str, col) -> Optional[float]:
        if name not in df.index:
            return None
        v = df.at[name, col]
        return None if pd.isna(v) else float(v)

    by_year = {
        y: {
            "Revenue":          _val(fin, "Total Revenue", c),
            "OperatingIncome":  _val(fin, "Operating Income", c),
            "NetIncome":        _val(fin, "Net Income", c),
            "TotalAssets":      _val(bs, "Total Assets", c),
            "TotalLiabilities": _val(bs, "Total Liabilities Net Minority Interest", c),
            "TotalEquity":      _val(bs, "Stockholders Equity", c),
        }
        for y, c in zip(years, cols)
    }

    def _growth(series: list[Optional[float]]) -> list[Optional[float]]:
        out: list[Optional[float]] = [None]
        for prev, cur in zip(series, series[1:]):
            if prev in (None, 0) or cur is None:
                out.append(None)
            else:
                out.append(round((cur - prev) / abs(prev) * 100, 2))
        return out
    
    growth_rates = {
        key: _growth([by_year[y][key] for y in years])
        for key in by_year[years[0]]
    }

    return {
        "ticker": ticker,
        "years": years,
        "by_year": by_year,
        "growth_rates": growth_rates,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")