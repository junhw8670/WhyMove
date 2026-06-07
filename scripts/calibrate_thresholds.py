from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
 
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.signal import detect_events

US_SECTOR_MAP = ROOT / "cache" / "us_sector_map.json"
DART_INDUSTRY = Path("C:/DartCopilot/cache/industry_codes.json")

DEFAULT_HORIZONS = [1, 5, 20]


def load_universe(market: str, top_n: int | None) -> list[str]:
    if market == "US":
        sector_map = json.loads(US_SECTOR_MAP.read_text(encoding="utf-8"))
        tickers = list(sector_map.keys())
    elif market == "KR":
        raw = json.loads(DART_INDUSTRY.read_text(encoding="utf-8"))
        tickers = [e["stock_code"] for e in raw.values() if e.get("stock_code")]
    else:
        raise ValueError(f"Unknown market: {market!r}")
    return tickers[:top_n] if top_n else tickers


def fetch_ohlcv(ticker: str, market: str, start: str, end: str) -> pd.DataFrame:
    if market == "KR":
        from pykrx import stock
        df = stock.get_market_ohlcv_by_date(
            start.replace("-", ""), end.replace("-", ""), ticker
        )
        df = df.rename(columns={
            "시가": "Open", "고가": "High", "저가": "Low",
            "종가": "Close", "거래량": "Volume",
        })[["Open", "High", "Low", "Close", "Volume"]]
    else:
        import yfinance as yf
        df = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index)
    return df


def forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    close = df["Close"]
    fwd = pd.DataFrame(index=df.index)
    for h in horizons:
        fwd[f"fwd_{h}"] = close.shift(-h) / close - 1.0
    return fwd


def backtest_ticker(ticker: str, market: str, start: str, end: str,
                    horizons: list[int]) -> list[dict]:
    df = fetch_ohlcv(ticker, market, start, end)
    if df.empty:
        return []
 
    events = detect_events(df, ticker, market, last_only=False)
    if not events:
        return []
 
    fwd = forward_returns(df, horizons)
 
    records: list[dict] = []
    for ev in events:
        ts = pd.Timestamp(ev.event_date)
        if ts not in fwd.index:
            continue
        rets = fwd.loc[ts]
        for sig in ev.signals:
            rec = {"ticker": ticker, "date": ev.event_date,
                   "signal": sig, "score": ev.score}
            for h in horizons:
                rec[f"fwd_{h}"] = float(rets[f"fwd_{h}"])
            records.append(rec)
    return records


def summarize(records: list[dict], horizons: list[int]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
 
    rows = []
    for sig, g in df.groupby("signal"):
        row = {"signal": sig, "n": len(g)}
        for h in horizons:
            col = f"fwd_{h}"
            s = g[col].dropna()
            row[f"mean_{h}"] = round(s.mean() * 100, 2)
            row[f"median_{h}"] = round(s.median() * 100, 2)
            row[f"win_{h}"] = round((s > 0).mean() * 100, 1)
        rows.append(row)
 
    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def run(market: str, top_n: int | None, horizons: list[int],
        history_days: int = 1825, end: str | None = None) -> pd.DataFrame:
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.today().date()
    start_str = (end_date - timedelta(days=history_days)).strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
 
    tickers = load_universe(market, top_n)
    print(f"[backtest] market={market}  tickers={len(tickers)}  "
          f"window={start_str}~{end_str}  horizons={horizons}")
 
    all_records: list[dict] = []
    for i, t in enumerate(tickers, 1):
        try:
            all_records += backtest_ticker(t, market, start_str, end_str, horizons)
        except Exception as e:
            print(f"  ! {t}: {e}")
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] events_so_far={len(all_records)}")
 
    summary = summarize(all_records, horizons)
    print(f"\n총 신호 레코드: {len(all_records)}\n")
    print("=== 신호별 향후 수익률 (mean/median/win = %) ===")
    print(summary.to_string(index=False) if not summary.empty else "(신호 없음)")
    return summary
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="신호별 향후 수익률 백테스트")
    p.add_argument("--market", choices=["KR", "US"], default="US")
    p.add_argument("--top-n", type=int, default=100, help="유니버스 앞 N개 (0=전체)")
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                   help="향후 수익률 구간(거래일)")
    p.add_argument("--end", type=str, default=None, help="기준 종료일 YYYY-MM-DD (기본=오늘)")
    p.add_argument("--save", type=str, default=None, help="결과 CSV 저장 경로")
    args = p.parse_args()
 
    summary = run(
        market=args.market,
        top_n=args.top_n or None,
        horizons=args.horizons,
        end=args.end,
    )
    if args.save and not summary.empty:
        summary.to_csv(args.save, index=False, encoding="utf-8-sig")
        print(f"\nsaved → {args.save}")