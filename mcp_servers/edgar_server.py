from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, date
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

DUR = {
    "Revenue": ["RevenueFromContractWithCustomerExcludingAsssessedTax", "Revenues", "SalesRevenueNet"],
    "OperatingIncome": ["OperatingIncomeLoss"],
    "NetIncome": ["NetIncomeLoss"],
}
INST = {
    "TotalAssets": ["Assets"],
    "TotalLiabilities": ["Liabilities"],
    "TotalEquity": ["StockholdersEquity"],
}


_ticker_to_cik: dict[str, str] | None = None




def _req(url: str) -> requests.Response:
    """EDGAR HTTP GET with required User-Agent + error guard."""
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r
    

def _facts(cik: str) -> dict:
    return _req(f"{EDGAR_DATA}/api/xbrl/companyfacts/CIK{cik}.json").json().get("facts", {}).get("us-gaap", {})


def _flow(facts, concepts, lo, hi):
    for c in concepts:
        out = {}
        for it in facts.get(c, {}).get("units", {}).get("USD", []):
            s, e, v = it.get("start"), it.get("end"), it.get("val")
            if s and e and v is not None and lo <= (date.fromisoformat(e) - date.fromisoformat(s)).days <= hi:
                out[e] = v
        if out:
            return out
    return {}


def _inst(facts, concepts):
    for c in concepts:
        out = {it["end"]: it["val"] for it in facts.get(c, {}).get("units", {}).get("USD", []) if it.get("end") and it.get("val") is not None}
        if out:
            return out
    return {}


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
    """as-reported GAAP 분기 수치 (EDGAR XBRL)."""
    facts = _facts(_cik(ticker))
    flows = {k: _flow(facts, c, 60, 100) for k, c in DUR.items()}
    insts = {k: _inst(facts, c) for k, c in INST.items()}

    ends = sorted(flows["NetIncome"])[-n_quarters:]
    by_period = {
        e: {**{k: flows[k].get(e) for k in DUR},
            **{k: insts[k].get(e) for k in INST}}
        for e in ends
    }
    return {"ticker": ticker, "periods": ends, "by_period": by_period}


@mcp.tool()
def fetch_multi_years(ticker: str, n_years: int = 5) -> dict:
    """as-reported GAAP 연간 수치 (EDGAR XBRL)."""
    facts = _facts(_cik(ticker))
    flows = {k: _flow(facts, c, 330, 400) for k, c in DUR.items()}
    insts = {k: _inst(facts, c) for k, c in INST.items()}
    ends = sorted(flows["NetIncome"])[-n_years:]
    by_year = {int(e[:4]): {**{k: flows[k].get(e) for k in DUR},
                            **{k: insts[k].get(e) for k in INST}} for e in ends}
    return {"ticker": ticker, "years": [int(e[:4]) for e in ends], "by_year": by_year}


if __name__ == "__main__":
    mcp.run(transport="stdio")