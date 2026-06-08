from __future__ import annotations

import pandas as pd

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from pykrx import stock


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.signal import detect_events

US_SECTOR_MAP = ROOT / "cache" / "us_sector_map.json"
DART_INDUSTRY = Path("C:/DartCopilot/cache/industry_codes.json")

DEFAULT_HORIZONS = [1, 5, 20, 60]


def load_universe(market: str, top_n: int | None) -> list[str]:
    if market == "US":
        sector_map = json.loads(US_SECTOR_MAP.read_text(encoding="utf-8"))
        tickers = list(sector_map.keys())
    elif market == "KR":
        base_day = stock.get_nearest_business_day_in_a_week()
        cap = stock.get_market_cap_by_ticker(base_day, market="KOSPI")
        tickers = cap.sort_values("시가총액", ascending=False).index.tolist()
    else:
        raise ValueError(f"Unknown market: {market!r}")
    return tickers[:top_n] if top_n else tickers


def fetch_ohlcv(ticker: str, market: str, start: str, end: str) -> pd.DataFrame:
    if market == "KR":
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


def fetch_market(market, start, end):
    if market == "US":
        import yfinance as yf
        df = yf.Ticker("SPY").history(start=start, end=end, auto_adjust=False)
        s = df["Close"]
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    else:
        df = stock.get_market_ohlcv_by_date(
            start.replace("-", ""), end.replace("-", ""), "069500"
        )
        s = df["종가"]
    s.index = pd.to_datetime(s.index)
    return s


def forward_returns(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    close = df["Close"]
    fwd = pd.DataFrame(index=df.index)
    for h in horizons:
        fwd[f"fwd_{h}"] = close.shift(-h) / close - 1.0
    return fwd


def backtest_ticker(ticker, market, fetch_start, end, horizons,
                    test_start, mkt_fwd, cooldown=0):
    df = fetch_ohlcv(ticker, market, fetch_start, end)
    if df.empty:
        return []

    events = detect_events(df, ticker, market, last_only=False)
    if not events:
        return []

    fwd = forward_returns(df, horizons)
    day_ret = df["Close"].pct_change()

    pos = {ts: i for i, ts in enumerate(df.index)}
    events = sorted(
        (ev for ev in events if ev.event_date >= test_start),
        key=lambda e: e.event_date,
    )

    last_pos: dict[str, int] = {}
    records = []
    for ev in events:
        ts = pd.Timestamp(ev.event_date)
        if ts not in fwd.index:
            continue
        i = pos[ts]
        rets = fwd.loc[ts]
        dr = day_ret.loc[ts]

        for sig in ev.signals:
            label = sig
            if sig == "volume_spike" and pd.notna(dr):
                label = "volume_spike_up" if dr >= 0 else "volume_spike_down"

            if cooldown and label in last_pos and i - last_pos[label] < cooldown:
                continue
            last_pos[label] = i

            rec = {"ticker": ticker, "date": ev.event_date,
                   "signal": label, "score": ev.score}

            for h in horizons:
                v = float(rets[f"fwd_{h}"])
                m = float(mkt_fwd.loc[ts, f"fwd_{h}"]) if ts in mkt_fwd.index else float("nan")
                rec[f"fwd_{h}"] = v
                rec[f"exc_{h}"] = v - m
            records.append(rec)
    return records


def summarize(records, horizons):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    rows = []
    for sig, g in df.groupby("signal"):
        row = {"signal": sig, "n": len(g)}
        for h in horizons:
            r = g[f"fwd_{h}"].dropna()
            e = g[f"exc_{h}"].dropna()
            row[f"mean_{h}"] = round(r.mean() * 100, 2)
            row[f"exc_{h}"]  = round(e.mean() * 100, 2)
            row[f"win_{h}"]  = round((e > 0).mean() * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def bootstrap_excess(records, h, signal=None, n_boot=2000, seed=2026):
    import numpy as np
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(records)
    if signal:
        df = df[df.signal == signal]
    per_ticker = df.groupby("ticker")[f"exc_{h}"].mean().dropna().to_numpy()
    if len(per_ticker) < 2:
        return None
    boots = np.array([
        rng.choice(per_ticker, len(per_ticker), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return round(per_ticker.mean()*100, 2), round(lo*100, 2), round(hi*100, 2)


def run(market, top_n, horizons, history_days=1825, warmup_days=1825,
        cooldown=0, end=None):
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else datetime.today().date()
    test_start  = end_date - timedelta(days=history_days)
    fetch_start = test_start - timedelta(days=warmup_days)
    fetch_start_str = fetch_start.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    mkt_close = fetch_market(market, fetch_start_str, end_str)
    mkt_fwd = forward_returns(mkt_close.to_frame("Close"), horizons)

    tickers = load_universe(market, top_n)
    print(f"[backtest] market={market} tickers={len(tickers)} "
          f"eval={test_start}~{end_date} warmup_from={fetch_start} "
          f"horizons={horizons} cooldown={cooldown}")

    all_records = []
    for i, t in enumerate(tickers, 1):
        try:
            all_records += backtest_ticker(
                t, market, fetch_start_str, end_str, horizons,
                test_start=test_start, mkt_fwd=mkt_fwd, cooldown=cooldown,
            )
        except Exception as e:
            print(f"  ! {t}: {e}")
        if i % 25 == 0:
            print(f"  [{i}/{len(tickers)}] events_so_far={len(all_records)}")

    summary = summarize(all_records, horizons)
    print(f"\n총 신호 레코드: {len(all_records)}\n")
    print("=== 신호별 향후 수익률 (mean/exc/win = %; exc=baseline 대비 초과) ===")
    print(summary.to_string(index=False) if not summary.empty else "(신호 없음)")
    return summary, all_records
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="신호별 향후 수익률 백테스트")
    p.add_argument("--market", choices=["KR", "US"], default="US")
    p.add_argument("--top-n", type=int, default=100, help="유니버스 앞 N개 (0=전체)")
    p.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS,
                   help="향후 수익률 구간(거래일)")
    p.add_argument("--warmup", type=int, default=1825,
                   help="평가구간 이전 워밍업 일수")
    p.add_argument("--cooldown", type=int, default=0,
                   help="신호별 최소 간격(거래일), 0=끔")
    p.add_argument("--end", type=str, default=None, help="기준 종료일 YYYY-MM-DD (기본=오늘)")
    p.add_argument("--save", type=str, default=None, help="결과 CSV 저장 경로")
    args = p.parse_args()

    summary, records = run(
        market=args.market,
        top_n=args.top_n or None,
        horizons=args.horizons,
        warmup_days=args.warmup,
        cooldown=args.cooldown,
        end=args.end,
    )
    if args.save and not summary.empty:
        summary.to_csv(args.save, index=False, encoding="utf-8-sig")
        print(f"\nsaved → {args.save}")

    for sig in ["5_years_high", "price_jump_up", "gap_up", "volume_spike_up"]:
        for h in [20, 60]:
            print(sig, h, bootstrap_excess(records, h, signal=sig))