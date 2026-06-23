from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv()

mcp = FastMCP("Market")

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"

_p = os.getenv("DART_INDUSTRY_PATH")
if not _p:
    raise RuntimeError("DART_INDUSTRY_PATH를 .env에 설정하세요")
DART_INDUSTRY_PATH = Path(_p)

def _rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.insert(0, "date", pd.to_datetime(out.index).strftime("%Y-%m-%d"))
    return json.loads(out.to_json(orient="records"))

@mcp.tool()
def fetch_ohlcv(ticker:str, market:str, start: str, end: str) -> dict:
    """Fetch US daily OHLCV via yfinance.

    Args:
        ticker: US symbol.
        market: 'US'.
        start, end: 'YYYY-MM-DD' or 'YYYYMMDD'.

    Returns:
        {ticker, market, rows: [{date, Open, High, Low, Close, Volume}, ...]}
    """
    if market == "US":
        end_incl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.Ticker(ticker).history(start=start, end=end_incl, auto_adjust=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
    else:
        raise ValueError(f"Unknown market: {market!r}")
    return {"ticker": ticker, "market": market, "rows": _rows(df)}


@mcp.tool()
def get_sector_map(market:str) -> dict:
    """Return ticker -> sector mapping.

    Args: US or KR

    Returns: {market, sector_map: {<ticker>: <sector>, ...}}
    """
    if market == "KR":
        raw = json.loads(DART_INDUSTRY_PATH.read_text(encoding="utf-8"))
        mapping = {
            e["stock_code"]: e["industry"]
            for e in raw.values()
            if e.get("stock_code") and e.get("industry")
        }
    else:
        path = CACHE_DIR / "us_sector_map.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Sector map missing: {path}. Run scripts/build_us_sector_map.py."
            )
        mapping = json.loads(path.read_text(encoding="utf-8"))
    return {"market": market, "sector_map": mapping}


if __name__ == "__main__":
    mcp.run(transport="stdio")
