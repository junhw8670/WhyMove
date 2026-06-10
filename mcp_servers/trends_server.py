from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Trends")


@mcp.tool()
def search_spike(name: str, market: str, event_date: str,
                 lookback_days: int = 90, retries: int = 2) -> dict:
    """Google Trends search-interest spike around an event.
    Args:
        name: Search keyword (company name), market: 'US'/'KR', event_date: 'YYYY-MM-DD'.
    Returns:
        {found, search_now, search_base, search_mult}
    """
    from pytrends.request import TrendReq

    ev = datetime.strptime(event_date.replace("-", ""), "%Y%m%d").date()
    start = ev - timedelta(days=lookback_days)
    timeframe = f"{start:%Y-%m-%d} {ev:%Y-%m-%d}"
    geo = {"US": "US", "KR": "KR"}.get(market, "")

    for attempt in range(retries):
        try:
            pt = TrendReq(
                hl="ko" if market == "KR" else "en-US",
                tz=540,
                timeout=(4, 8),
                retries=0,
                backoff_factor=0,
            )
            pt.build_payload([name], timeframe=timeframe, geo=geo)
            df = pt.interest_over_time()
            if df.empty or name not in df.columns:
                return {"found": False}
            s = df[name].astype(float)
            s.index = pd.to_datetime(s.index)
            ev_ts = pd.Timestamp(ev)
            now = s[(s.index >= ev_ts - pd.Timedelta(days=2)) & (s.index <= ev_ts)]
            base = s[s.index <= ev_ts - pd.Timedelta(days=7)]
            if now.empty or base.empty:
                return {"found": False}
            n, b = float(now.max()), float(base.mean())
            return {
                "found": True,
                "search_now": round(n, 1),
                "search_base": round(b, 1),
                "search_mult": round(n / b, 1) if b > 0 else None,
            }
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {"found": False}


if __name__ == "__main__":
    mcp.run(transport="stdio")