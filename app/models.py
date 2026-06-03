from __future__ import annotations

from datetime import date
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

Market = Literal["KR", "US"]

SignalType = Literal[
    "volume_spike",
    "price_jump",
    "gap",
    "all_time_high",
    "all_time_low",
    "flow_shift",
    "anomaly_outlier",
    "breadth_surge",
]

Scope = Literal["single", "sector"]


class Event(BaseModel):
    ticker: str
    name: str
    market: Market
    event_date: date
    scope: Scope = "single"
    sector: Optional[str] = None
    signals: list[SignalType] = []
    score: float = 0.0
    detail: dict = Field(default_factory=dict)


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    url: str = ""
    source: str = ""
    published: Optional[date] = None
    

class FinancialFigure(BaseModel):
    label: str
    period: str
    value: float
    yoy: Optional[float] = None


class Memo(BaseModel):
    event: Event
    news_context: str = ""
    filing_context: str = ""
    figures: list[FinancialFigure] = Field(default_factory=list)
    discrepancy: str = ""
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    backend_used: str = "cloud"


class GraphState(TypedDict, total=False):
    event: Event
    news: list[NewsItem]
    has_news: bool
    has_filing: bool
    figures: list[FinancialFigure]
    memo: Memo

