from pathlib import Path
import sys

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detect import detect_events


MARKET = "US"     
TICKER = "MDB"
START_DATE = "2026-05-16"
END_DATE = "2026-07-16"

KR_CACHE = ROOT / "cache" / "kr_ohlcv.parquet"


def load_ohlcv():
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    if MARKET == "KR":
        cache = pd.read_parquet(KR_CACHE)
        cache["date"] = pd.to_datetime(cache["date"])

        return (
            cache[
                (cache["ticker"] == TICKER)
                & (cache["date"] <= end)
            ]
            .sort_values("date")
            .set_index("date")[
                ["Open", "High", "Low", "Close", "Volume"]
            ]
        )

    fetch_start = start - pd.Timedelta(days=370)
    fetch_end = end + pd.Timedelta(days=1)

    df = yf.Ticker(TICKER).history(
        start=fetch_start.strftime("%Y-%m-%d"),
        end=fetch_end.strftime("%Y-%m-%d"),
        auto_adjust=True,
    )

    df = df[["Open", "High", "Low", "Close", "Volume"]]

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def main():
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    df = load_ohlcv()

    events = detect_events(
        df=df,
        ticker=TICKER,
        market=MARKET,
        name=TICKER,
        last_only=False,
    )

    events = [
        event
        for event in events
        if start.date() <= event.event_date <= end.date()
    ]

    print(
        f"\n{MARKET} {TICKER} "
        f"{START_DATE} ~ {END_DATE}\n"
    )

    if not events:
        print("탐지된 이벤트 없음")
        return

    for event in events:
        print(
            f"{event.event_date} | "
            f"score={event.score:.2f} | "
            f"{', '.join(event.signals)}"
        )
        print(f"  {event.detail}")


if __name__ == "__main__":
    main()
