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
def fetch_filings_around(ticker: str, event_date: str, window_days: int = 7) -> dict:
    """Find material SEC filings within ±window_days of event_date.

    Args:
        ticker: US symbol
        event_date: 'YYYY-MM-DD' or 'YYYYMMDD'.
        window_days: both sides of event_date. Default 7 -> total 15 days window.

    Returns:
        {ticker, filings: [{filing_id, filing_date, form, title, primary_doc}, ...]}
    """
    cik = _cik(ticker)

    d = datetime.strptime(event_date.relace("-", ""), "%Y%m%d")
    bgn, end = d - timedelta(days=window_days), d + timedelta(days=window_days)

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
            fd = datatime.strptime(fdate, "%Y-%m-%d")
        except ValueError:
            continue
        if fd < bgn or fd > end:
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
def fetch_filing_text(
    ticker: str,
    accession: str,
    primary_doc: str,
    max_chars: int = 8000,
) -> dict:
    """Fetch and extract body text from a US filing's primary document.

    Args:
        ticker: US symbol (For CIK lookup).
        accession: fetch_filings_around -> filing_id.
        primary_doc: fetch_filings_around -> primary_doc.
        max_chars: length for truncate.
    
    Returns:
        {ticker, filing_id, text, truncated: True}
    """
    cik = _cik(ticker)
    url = f"{EDGAR}/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{primary_doc}"

    soup = BeautifulSoup(_req(url).text, "lxml")
    text = re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True))
    return {
        "ticker": ticker,
        "filing_id": accession,
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


@mcp.tool()
def fetch_recent_financial(ticker: str) -> dict:
    """Return most recent annual financial figures via yfinance.

    Args: 
        ticker: US symbol.

    Returns:
        {ticker, year, figures: {Revenue, OperatingIncome, NetIncome, TotalAssets, TotalLiabilities, TotalEquity}}
    """
    tk = yf.Ticker(ticker)
    fin, bs = tk.financials, tk.balance_sheet

    if fin.empty:
        raise ValueError(f"No US financial for {ticker}")

    col = fin.columns[0]
    year = col.year

    def _val(df: pd.DataFrame, name: str) -> Optional[float]:
        if name not in df.index:
            return None
        v = df.at[name, col]
        return None if pd.isna(v) else float(v)

    return {
        "ticker": ticker,
        "year": year,
        "figures": {
            "Revenue":          _val(fin, "Total Revenue"),
            "OperatingIncome":  _val(fin, "Operating Income"),
            "NetIncome":        _val(fin, "Net Income"),
            "TotalAssets":      _val(bs,  "Total Assets"),
            "TotalLiabilities": _val(bs,  "Total Liabilities Net Minority Interest"),
            "TotalEquity":      _val(bs,  "Stockholders Equity"),
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
