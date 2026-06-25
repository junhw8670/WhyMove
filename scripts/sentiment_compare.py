from __future__ import annotations

import argparse, os, sys, time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from signal_backtest import run, summarize, bootstrap_excess

FINN = os.getenv("FINNHUB_API_KEY")


from transformers import pipeline

fin = pipeline("sentiment-analysis", model="ProsusAI/finbert")


def finbert_score(text: str) -> float:
    r = _fin(text[:512])[0]
    return r["score"] if r["label"] == "positive" else -r["score"] if r["label"] == "negative" else 0.0


def finnhub_sentiment(ticker: str, date, lookback: int = 7):
    frm = (pd.Timestamp(date) - timedelta(days=lookback)).strftime("%Y-%m-%d")
    to = pd.Timestamp(date).strftime("%Y-%m-%d")
    try:
        items = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker, "from": frm, "to": to, "token": FINN}, 
            timeout=15,
        ).json()
    except Exception:
        return None
    if not isinstance(items, list):
        if not getattr(finnhub_sentiment, "_warned", False):
            print("not list", items)
            finnhub_sentiment._warned = True
        return None
    heads = [it.get("headline") for it in (items or [])[:10] if it.get("headline")]
    if not heads:
        return None
    return sum(finbert_score(h) for h in heads) / len(heads)


def group(sig: str) -> str:
    return "bull" if sig.endswith(("_up", "_high")) else "bear"


def sent_label(s, thr: float):
    if s is None:
        return None
    return "pos" if s > thr else "neg" if s < -thr else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=500)
    p.add_argument("--horizons", type=int, nargs="+", default=[5, 20, 60])
    p.add_argument("--thr", type=float, default=0.15)
    p.add_argument("--cooldown", type=int, default=5)
    args = p.parse_args()

    _, records = run("US", top_n=args.top_n, horizons=args.horizons,
                     history_days=365, warmup_days=365, cooldown=args.cooldown)
    print(f"records: {len(records)}")

    cache = {}
    for i, rec in enumerate(records, 1):
        key = (rec["ticker"], rec["date"])
        if key not in cache:
            cache[key] = finnhub_sentiment(rec["ticker"], rec["date"])
            time.sleep(1.1)
        rec["sent"] = cache[key]
        if i % 50 == 0:
            print(f" sent: {i}/{len(records)}")
        

    have = [r["sent"] for r in records if r.get("sent") is not None]
    print(f"have sent: {len(have)}/{len(records)}, samp: {[round(x,3) for x in have[:10]]}")

    combined = []
    for rec in records:
        sl = sent_label(rec.get("sent"), args.thr)
        if sl is None:
            continue
        r2 = dict(rec)
        r2["signal"] = f"{group(rec['signal'])}+{sl}"
        combined.append(r2)

    print(f"combined records: {len(combined)}")
    print(summarize(combined, args.horizons).to_string(index=False))

    for cell in ["bull+pos", "bull+neg", "bear+pos", "bear+neg"]:
        for h in args.horizons:
            print(cell, h, bootstrap_excess(combined, h, signal=cell))

    
if __name__ == "__main__":
    main()