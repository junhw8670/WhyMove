from __future__ import annotations

import requests
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


CACHE_PATH = Path(__file__).resolve().parent.parent / "cache" / "us_sector_map.json"
CAP_PATH = CACHE_PATH.parent / "us_marketcap.json"


def fetch_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    tables = pd.read_html(r.text)
    df = tables[0]
    return [s.replace(".", "-") for s in df["Symbol"].astype(str).tolist()]

def main() -> None:
    tickers = fetch_sp500_tickers()
    print(f"{len(tickers)} 개")

    mapping: dict[str, str] = {}
    caps: dict[str, int] = {}
    for i, t in enumerate(tickers, 1):
        try:
            info = yf.Ticker(t).info
            sector = info.get("sector") or ""
            if sector:
                mapping[t] = sector
            mc = info.get("marketCap")
            if mc:
                caps[t] = mc
        except Exception as e:
            print(f"{i}/{len(tickers)} {t}: error {e}")
            continue

        time.sleep(0.2)

        if i % 50 == 0:
            print(f" [{i}/{len(tickers)}]  resolved={len(mapping)}")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    CAP_PATH.write_text(
        json.dumps(caps, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(mapping)} entries → {CACHE_PATH}, {CAP_PATH}")


if __name__ == "__main__":
    main()