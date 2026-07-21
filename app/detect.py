from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .models import Event, Market, SignalType


def build_features(df: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["ret"] = df["Close"].pct_change()

    vmean = df["Volume"].ewm(span=span).mean()
    vstd = df["Volume"].ewm(span=span).std()    
    
    out["vol_z"] = (df["Volume"] - vmean) / (vstd + 1e-9)
    out["vol_mult"] = df["Volume"] / (vmean + 1e-9)

    out["gap"] = df["Open"] / df["Close"].shift(1) - 1

    out["ret_z"] = out["ret"] / (out["ret"].ewm(span=span).std() + 1e-9)
    out["gap_z"] = out["gap"] / (out["gap"].ewm(span=span).std() + 1e-9)
    
    return out.dropna()


def detect_events(
    df: pd.DataFrame,
    ticker: str,
    market: Market,
    name: str = "",
    span: int = 60,
    z_floor: float = 2.5,
    w_52w: float = 0.5,
    score_cutoff: float = 1.0,
    last_only: bool = False
) -> list[Event]:
    feats = build_features(df, span)
    if len(feats) < span:
        return []

    h_52w = df["Close"].rolling(252, min_periods=60).max().shift(1).reindex(feats.index)
    l_52w = df["Close"].rolling(252, min_periods=60).min().shift(1).reindex(feats.index)
    start = df["Open"].reindex(feats.index)
    close = df["Close"].reindex(feats.index)

    events: list[Event] = []

    rows_to_check = feats.iloc[[-1]] if last_only else feats
    for ts, row in rows_to_check.iterrows():
        sigs: list[SignalType] = []
        score = 0.0

        detail: dict = {
            "open": round(float(start.loc[ts]), 2),
            "close": round(float(close.loc[ts]), 2),
            "ret_pct": round(float(row["ret"]) * 100, 2),
            "gap_pct": round(float(row["gap"]) * 100, 2),
            "vol_mult": round(float(row["vol_mult"]), 2),
            "vol_z": round(float(row["vol_z"]), 2),
            "ret_z": round(float(row["ret_z"]), 2),
            "gap_z": round(float(row["gap_z"]), 2),
        }

        c = max(0.0, row["vol_z"] - z_floor)
        if c > 0: 
            sigs.append("volume_spike")
            score += c

        ret_z = float(row["ret_z"])
        c = max(0.0, abs(ret_z) - z_floor)
        if c > 0: 
            sigs.append("price_jump_up" if ret_z > 0 else "price_jump_down")
            score += c

        gap_z = float(row["gap_z"])
        c = max(0.0, abs(gap_z) - z_floor)
        if c > 0: 
            sigs.append("gap_up" if gap_z > 0 else "gap_down")
            score += c

        if pd.notna(h_52w.loc[ts]) and close.loc[ts] > h_52w.loc[ts]:
            sigs.append("52_weeks_high")
            score += w_52w
            detail["h_52w"] = float(h_52w.loc[ts])
        if pd.notna(l_52w.loc[ts]) and close.loc[ts] < l_52w.loc[ts]:
            sigs.append("52_weeks_low")
            score += w_52w
            detail["l_52w"] = float(l_52w.loc[ts])

        if score >= score_cutoff:
            events.append(
                Event(
                    ticker=ticker,
                    name=name or ticker,
                    market=market,
                    event_date=ts.date() if hasattr(ts, "date") else ts,
                    scope="single",
                    signals=sigs,
                    score=round(score, 2),
                    detail=detail,
                )
            )
    return events


def detect_sector_breadth(
    events: Iterable[Event],
    sector_map: dict[str, str],
    market: Market,
    breadth_floor: float = 0.3,
    min_members: int = 5,
    universe: Optional[Iterable[str]] = None,
) -> list[Event]:
    if universe is not None:
        sector_sizes = Counter(
            sector_map[t] for t in universe if t in sector_map
        )
    else:
        sector_sizes = Counter(sector_map.values())

    bucket: dict[tuple, set[str]] = defaultdict(set)
    for e in events:
        sec = sector_map.get(e.ticker)
        if not sec:
            continue
        key = (e.event_date, sec)
        bucket[key].add(e.ticker)
    
    sector_events: list[Event] = []
    for (day, sec), triggered in bucket.items():
        size = sector_sizes.get(sec, 0)
        if size < min_members:
            continue
        
        breadth = len(triggered) / size
        if breadth < breadth_floor:
            continue
        
        sector_events.append(
            Event(
                ticker=sec,
                name=sec,
                market=market,
                event_date=day,
                scope="sector",
                sector=sec,
                signals=["breadth_surge"],
                score=round(breadth, 2),
                detail={
                    "breadth": round(breadth, 2),
                    "n_triggered": len(triggered),
                    "n_members": size,
                    "triggered_tickers": sorted(triggered)[:20]
                },
            )
        )
        
    return sector_events    


def rank_and_cap_daily(events: Iterable[Event], max_per_day: int) -> list[Event]:
    by_day: dict[object, list[Event]] = defaultdict(list)
    for e in events:
        by_day[e.event_date].append(e)

    kept: list[Event] = []
    for day in sorted(by_day):
        day_events = sorted(by_day[day], key=lambda x: x.score, reverse=True)
        kept.extend(day_events[:max_per_day])
    return kept


