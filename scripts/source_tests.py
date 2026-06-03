from __future__ import annotations

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()
UA = os.getenv("EDGAR_USER_AGENT", "WhyMove research junhw8670@gmail.com")


def head(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def show_df(df, title: str, n: int = 3) -> None:
    head(title)
    print(f"shape={df.shape}, index.name={df.index.name}, index.dtype={df.index.dtype}")
    print(f"columns={list(df.columns)}")
    print(f"dtypes:\n{df.dtypes}")
    print(f"\nhead({n}):\n{df.head(n)}")


def inspect_pykrx_ohlcv():
    from pykrx import stock
    df = stock.get_market_ohlcv_by_date("20250501", "20250530", "005930")
    show_df(df, "pykrx OHLCV — 삼성전자 20250501~20250530")


def inspect_pykrx_flow():
    from pykrx import stock
    df = stock.get_market_trading_volume_by_date("20250501", "20250530", "005930")
    show_df(df, "pykrx flow — 삼성전자 20250501~20250530")


def inspect_yf_ohlcv():
    import yfinance as yf
    df = yf.Ticker("AAPL").history(start="2025-05-01", end="2025-05-30", auto_adjust=False)
    show_df(df, "yfinance OHLCV — AAPL 2025-05-01~30")


def inspect_yf_financials():
    import yfinance as yf
    tk = yf.Ticker("AAPL")
    head("yfinance financials / balance_sheet — AAPL")
    print(f"[financials] columns(연도): {[str(c) for c in tk.financials.columns]}")
    print(f"index 상위 15:\n{list(tk.financials.index[:15])}")
    print(f"\n[balance_sheet] columns: {[str(c) for c in tk.balance_sheet.columns]}")
    print(f"index 상위 15:\n{list(tk.balance_sheet.index[:15])}")


def inspect_fdr_listings():
    import FinanceDataReader as fdr
    # 'Sector' 컬럼이 실제로 있는지·값 형태가 무엇인지 확인.
    df = fdr.StockListing("KRX")
    show_df(df, "FinanceDataReader StockListing('KRX')")


def inspect_edgar_tickers():
    head("SEC EDGAR company_tickers.json")
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": UA}, timeout=30,
    )
    data = r.json()
    print(f"type={type(data).__name__}, total entries={len(data)}")
    # 첫 3개 entry 구조 확인 — cik_str/ticker/title 키 패턴 검증.
    for k in list(data.keys())[:3]:
        print(f"key={k!r}: {json.dumps(data[k], indent=2)}")


def inspect_edgar_submissions():
    head("SEC EDGAR submissions — AAPL CIK=0000320193")
    r = requests.get(
        "https://data.sec.gov/submissions/CIK0000320193.json",
        headers={"User-Agent": UA}, timeout=30,
    )
    data = r.json()
    print(f"top-level keys: {list(data.keys())}")
    recent = data.get("filings", {}).get("recent", {})
    # column-major 배열들이 같은 i 인덱스에서 한 건을 이룬다는 점 확인.
    print(f"recent keys: {list(recent.keys())}")
    for k in ["form", "accessionNumber", "filingDate", "primaryDocument", "primaryDocDescription"]:
        print(f"recent['{k}'][:5] = {recent.get(k, [])[:5]}")


if __name__ == "__main__":
    inspect_pykrx_ohlcv()
    inspect_pykrx_flow()
    inspect_yf_ohlcv()
    inspect_yf_financials()
    inspect_fdr_listings()
    inspect_edgar_tickers()
    inspect_edgar_submissions()