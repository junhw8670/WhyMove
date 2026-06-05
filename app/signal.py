from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .models import Event, Market, SignalType

IF_FEATURES = ["ret", "ret_abs", "vol_z", "gap"]


def build_features(df: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["ret"] = df["Close"].pct_change()
    out["ret_abs"] = out["ret"].abs()

    vmean = df["Volume"].ewm(span=span).mean()
    vstd = df["Volume"].ewm(span=span).std()    
    out["vol_z"] = (df["Volume"] - vmean) / (vstd + 1e-9)

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
    z_floor: float = 3.0,
    w_at: float = 1.0,
    w_if: float = 1.5,
    score_cutoff: float = 3.0,
    last_only: bool = False
) -> list[Event]:
    feats = build_features(df, span)
    if len(feats) < span:
        return []

    if len(feats) >= 120:
        X = feats[IF_FEATURES].values

        iso = IsolationForest(n_estimators=200, max_samples=256, random_state=2026).fit(X)

        if_raw = -iso.score_samples(X)
        feats = feats.assign(if_z=(if_raw - if_raw.mean()) / (if_raw.std() + 1e-9))
    else:
        feats = feats.assign(if_z=0.0)

    ath = df["Close"].cummax().shift(1).reindex(feats.index)
    atl = df["Close"].cummin().shift(1).reindex(feats.index)
    close = df["Close"].reindex(feats.index)

    events: list[Event] = []

    rows_to_check = feats.iloc[[-1]] if last_only else feats
    for ts, row in rows_to_check.iterrows():
        sigs: list[SignalType] = []
        rule_score = 0.0

        detail: dict = {
            "vol_z": round(float(row["vol_z"]), 2),
            "ret_z": round(float(row["ret_z"]), 2),
            "gap_z": round(float(row["gap_z"]), 2),
            "if_z": round(float(row["if_z"]), 2),
        }

        c = max(0.0, abs(row["vol_z"]) - z_floor)
        if c > 0: 
            sigs.append("volume_spike")
            rule_score += c

        c = max(0.0, abs(row["ret_z"]) - z_floor)
        if c > 0: 
            sigs.append("price_jump")
            rule_score += c

        c = max(0.0, abs(row["gap_z"]) - z_floor)
        if c > 0: 
            sigs.append("gap")
            rule_score += c

        if pd.notna(ath.loc[ts]) and close.loc[ts] > ath.loc[ts]:
            sigs.append("all_time_high")
            rule_score += w_at
            detail["ath"] = float(ath.loc[ts])
        if pd.notna(atl.loc[ts]) and close.loc[ts] < atl.loc[ts]:
            sigs.append("all_time_low")
            rule_score += w_at
            detail["atl"] = float(atl.loc[ts])


        if_contrib = max(0.0, float(row["if_z"])) * w_if

        score = rule_score + if_contrib

        if score >= score_cutoff:
            if not sigs:
                sigs.append("anomaly_outlier")

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
    breadth_floor: float = 0.5,
    min_members: int = 5,
) -> list[Event]:
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
        
        sector_events.append(Event(
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
        ))
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


