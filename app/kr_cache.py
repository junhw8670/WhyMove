from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from pykrx import stock
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CACHE = Path(__file__).resolve().parent.parent / "cache" / "kr_ohlcv.parquet"
MARKETS = ["KOSPI", "KOSDAQ"]
REN = {"시가": "Open", "고가": "High", "저가": "Low", "종가": "Close", "거래량": "Volume"}

def _fetch_day(d: date) -> list[pd.DataFrame]:
    frames = []
    ds = d.strftime("%Y%m%d")
    for mkt in MARKETS:
        try:
            df = stock.get_market_ohlcv_by_ticker(ds, market=mkt)
        except Exception:
            df = None
        if df is not None and not df.empty:
            df = df.rename(columns=REN)[["Open", "High", "Low", "Close", "Volume"]]
            df = df[df["Close"] > 0]
            df = df.reset_index().rename(columns={"티커": "ticker"})
            df.insert(0, "date", pd.Timestamp(d))
            frames.append(df)
    return frames


def update(days: int = 400, upto: Optional[date] = None) -> pd.DataFrame:
    end = upto or datetime.today().date()
    if CACHE.exists():
        old = pd.read_parquet(CACHE)
        start = old["date"].max().date() + timedelta(days=1)
    else:
        old, start = None, end - timedelta(days=days)
        
    frames = []
    d = start
    while d <= end:
        frames += _fetch_day(d)
        time.sleep(0.3)
        d += timedelta(days=1)

    new = pd.concat(frames, ignore_index=True) if frames else None
    parts = [x for x in (old, new) if x is not None and not x.empty]
    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out.drop_duplicates(["date", "ticker"], keep="last")
    out = out[out["date"] >= pd.Timestamp(end - timedelta(days=days * 2))]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


if __name__ == "__main__":
    df = update()
    print(f"cache: {len(df)} rows, {df['ticker'].nunique()} tickers, {df['date'].min().date()}~{df['date'].max().date()}")