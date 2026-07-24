from __future__ import annotations

from datetime import date
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

Market = Literal["KR", "US"]

SignalType = Literal[
    "volume_spike",
    "price_jump_up",
    "price_jump_down",
    "gap_up", 
    "gap_down",
    "52_weeks_high",
    "52_weeks_low",
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
    signals: list[SignalType] = Field(default_factory=list)
    score: float = 0.0
    detail: dict = Field(default_factory=dict)


class NewsItem(BaseModel):
    title: str
    summary: str = ""
    content: str = ""
    llm_text: str = ""
    url: str = ""
    source: str = ""
    published: Optional[date] = None
    sources: list[str] = Field(default_factory=list)

class FinancialFigure(BaseModel):
    label: str
    period: str
    value: float


class Memo(BaseModel):
    event: Event
    figures: list[FinancialFigure] = Field(default_factory=list)
    summary: str = ""
    sources: list[str] = Field(default_factory=list)
    backend_used: str = ""


class NewsAnalysis(BaseModel):
    key_facts: list[str] = Field(default_factory=list)
    likely_catalysts: list[str] = Field(default_factory=list)
    timing_assessment: str = ""

    causal_strength: Literal[
        "none",
        "weak",
        "moderate",
        "strong",
    ] = "none"

    news_findings: Optional[str] = None
    sources: list[str] = Field(default_factory=list)

class FinancialAnalysis(BaseModel):
    filing_form: Optional[str] = None
    filing_date: Optional[str] = None

    key_figures: list[str] = Field(default_factory=list)
    notable_changes: list[str] = Field(default_factory=list)

    event_relevance: Literal[
        "none",
        "weak",
        "moderate",
    ] = "none"

    financial_findings: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    event: Event
    news: list[NewsItem]
    has_news: bool
    has_filing: bool
    filing_info: str
    news_analysis: NewsAnalysis
    financial_analysis: FinancialAnalysis
    figures: list[FinancialFigure]
    news_sources: list[str]
    memo: Memo