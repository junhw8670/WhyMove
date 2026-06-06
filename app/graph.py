from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
import logging

from .llm_utils import get_llm
from .models import Event, FinancialFigure, GraphState, Market, Memo, NewsItem
from .signal import detect_events, detect_sector_breadth, rank_and_cap_daily

logger = logging.getLogger(__name__)


def _find(tools: list[BaseTool], name: str) -> BaseTool:
    for t in tools:
        if t.name == name:
            return t
    raise KeyError(f"Tool not found: {name}")


def _parse_tool_payload(content) -> dict | None:
    """Normalize a ToolMessage.content into a dict."""
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
        
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and "text" in item:
                try:
                    inner = json.loads(item["text"])
                    if isinstance(inner, dict):
                        return inner
                except (json.JSONDecodeError, TypeError):
                    continue
    return None


@lru_cache(maxsize=1)
def _stock_to_corp() -> dict[str, str]:
    raw = json.loads(
        Path("C:/DartCopilot/cache/industry_codes.json").read_text(encoding="utf-8")
    )
    return {e["stock_code"]: cc for cc, e in raw.items() if e.get("stock_code")}


def _parse_kr_report(report_nm: str) -> Optional[tuple[str, int]]:
    if not report_nm:
        return None
    m = re.search(r"(\d{4})\.(\d{2})", report_nm)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if "사업보고서" in report_nm:
        return ("11011", year)
    if "반기보고서" in report_nm:
        return ("11012", year)
    if "분기보고서" in report_nm:
        if month == 3:
            return ("11013", year)
        if month == 9:
            return ("11014", year)
    return None

FLOW_ACCOUNTS = {"매출액", "영업이익", "당기순이익"}
STOCK_ACCOUNTS = {"자산총계", "부채총계", "자본총계"}
KEY_KR = FLOW_ACCOUNTS | STOCK_ACCOUNTS


def _derive_q4(annual: dict[str, float], q1: dict, q2: dict, q3: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in FLOW_ACCOUNTS:
        if k in annual:
            out[k] = annual[k] - q1.get(k, 0.0) - q2.get(k, 0.0) - q3.get(k, 0.0)
    for k in STOCK_ACCOUNTS:
        if k in annual:
            out[k] = annual[k]
    return out


def _sum_flow(*reports: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in FLOW_ACCOUNTS:
        vals = [r[k] for r in reports if k in r]
        if vals:
            out[k] = sum(vals)
    last = next((r for r in reversed(reports) if r), {})
    for k in STOCK_ACCOUNTS:
        if k in last:
            out[k] = last[k]
    return out


async def scan_universe(
    tools: dict[str, list[BaseTool]],
    market: Market,
    date: str,
    top_n: Optional[int] = None,
    history_days: int = 1825,
) -> list[Event]:
    get_sm = _find(tools["market"], "get_sector_map")
    fetch_ohlcv = _find(tools["market"], "fetch_ohlcv")

    sm: dict[str, str] = {}
    raw = await get_sm.ainvoke({"market": market})
    payload = _parse_tool_payload(raw)
    if payload is None:
        raise RuntimeError("get_sector_map: invalid MCP payload")
    sm = payload["sector_map"]

    tickers = list(sm.keys())[: int(top_n)] if top_n else list(sm.keys())

    end_date = datetime.strptime(date.replace("-", ""), "%Y%m%d").date()
    start_str = (end_date - timedelta(days=history_days)).strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    sem = asyncio.Semaphore(8)

    async def _scan_one(ticker: str) -> list[Event]:
        async with sem:
            try:
                raw = await fetch_ohlcv.ainvoke({
                    "ticker": ticker, "market": market,
                    "start": start_str, "end": end_str,
                })
                payload = _parse_tool_payload(raw)
                if payload is None:
                    raise RuntimeError("fetch_ohlcv: invalid MCP payload")
                rows = payload.get("rows", [])
                if not rows:
                    return []
                df = pd.DataFrame(rows).set_index("date")
                df.index = pd.to_datetime(df.index)
                return detect_events(df, ticker, market, last_only=True)
            except Exception as e:
                logger.warning(f"_scan_one({ticker}) failed: {e}")
                return []

    results = await asyncio.gather(*[_scan_one(t) for t in tickers])
    all_singles: list[Event] = [ev for chunk in results for ev in chunk]
    sectors = detect_sector_breadth(all_singles, sm, market)

    return (
        rank_and_cap_daily(all_singles, max_per_day=20)
        + rank_and_cap_daily(sectors, max_per_day=5)
    )


def build_event_graph(tools_by_server: dict[str, list[BaseTool]]):
    news_tool = _find(tools_by_server["news"], "fetch_news")
    dart_list = _find(tools_by_server["dart"], "list_disclosures")
    dart_fin  = _find(tools_by_server["dart"], "fetch_financial")
    edgar_list  = _find(tools_by_server["edgar"], "fetch_filings_around")
    edgar_qmul  = _find(tools_by_server["edgar"], "fetch_multi_quarters")
    edgar_ymul  = _find(tools_by_server["edgar"], "fetch_multi_years")

    import os
    backend = os.getenv("LLM_BACKEND", "cloud")
    llm = get_llm(backend)

    KR_ACCOUNT_ALIASES: dict[str, set[str]] = {
        "매출액": {"매출액", "수익(매출액)", "영업수익", "수익"},
        "영업이익": {"영업이익", "영업손실", "영업이익(손실)"},
        "당기순이익": {
            "당기순이익", "분기순이익", "반기순이익",
            "당기순이익(손실)", "분기순이익(손실)", "반기순이익(손실)",
            "당기순손실",
        },
        "자산총계": {"자산총계"},
        "부채총계": {"부채총계"},
        "자본총계": {"자본총계"},
    }

    _KR_STANDARD_TO_SJ: dict[str, str] = {
        "매출액": "손익",
        "영업이익": "손익",
        "당기순이익": "손익",
        "자산총계": "재무상태표",
        "부채총계": "재무상태표",
        "자본총계": "재무상태표",
    }

    _KR_ALIAS_TO_KEY: dict[str, str] = {
        alias: key for key, aliases in KR_ACCOUNT_ALIASES.items() for alias in aliases
    }

    async def _kr_cum(corp_code: str, year: int, report_code: str) -> dict[str, float]:
        raw = await dart_fin.ainvoke({
            "corp_code": corp_code, "year": year, "report_code": report_code,
        })
        payload = _parse_tool_payload(raw)
        if payload is None:
            raise RuntimeError("dart_financial: invalid MCP payload")
        out: dict[str, float] = {}
        for row in payload.get("accounts", []):
            standard = _KR_ALIAS_TO_KEY.get(row.get("account_nm", ""))
            if not standard:
                continue
            expected_sj = _KR_STANDARD_TO_SJ.get(standard, "")
            if expected_sj not in row.get("sj_nm", ""):
                continue
            if standard in out:
                continue
            amt = row.get("thstrm_amount")
            if amt is not None:
                out[standard] = float(amt)
        return out

    async def fetch_news_node(state: GraphState) -> dict:
        ev = state["event"]
        raw = await news_tool.ainvoke({
            "ticker": ev.ticker, "market": ev.market,
            "event_date": ev.event_date.isoformat(),
            "lookback_days": 7, "name": ev.name,
        })
        payload = _parse_tool_payload(raw)
        if payload is None:
            raise RuntimeError("news_tool: invalid MCP payload")
        items = [NewsItem(**it) for it in payload.get("items", [])][:10]
        return {"news": items, "has_news": bool(items)}

    async def fetch_financial_node(state: GraphState) -> dict:
        ev = state["event"]
        if ev.scope != "single":
            return{"figures": []}

        if ev.market == "KR":
            corp_code = _stock_to_corp().get(ev.ticker)
            if not corp_code:
                return {"figures": [], "has_filing": False}

            bgn = (ev.event_date - timedelta(days=30)).strftime("%Y%m%d")
            end = ev.event_date.strftime("%Y%m%d")
            raw = await dart_list.ainvoke({
                "corp_code": corp_code, "bgn_de": bgn, "end_de": end,
            })
            payload = _parse_tool_payload(raw)
            if payload is None:
                raise RuntimeError("dart_list: invalid MCP payload")
            latest = None
            for f in sorted(payload.get("list", []), key=lambda x: x.get("rcept_dt", ""), reverse=True):
                parsed = _parse_kr_report(f.get("report_nm", ""))
                if parsed:
                    latest = parsed
                    break
            if latest is None:
                return {"figures": [], "has_filing": False}

            report_code, year = latest
            y_prev = year - 1
           
            cache: dict[tuple[int, str], dict[str, float]] = {}

            async def report(y: int, rc: str) -> dict[str, float]:
                key = (y, rc)
                if key not in cache:
                    cache[key] = await _kr_cum(corp_code, y, rc)
                return cache[key]

            async def q4(y: int) -> dict[str, float]:
                return _derive_q4(
                    await report(y, "11011"),
                    await report(y, "11013"),
                    await report(y, "11012"),
                    await report(y, "11014"),
                )

            async def h1(y: int) -> dict[str, float]:
                return _sum_flow(
                    await report(y, "11013"),
                    await report(y, "11012"),
                )

            async def h2(y: int) -> dict[str, float]:
                return _sum_flow(
                    await report(y, "11014"),
                    await q4(y),
                )

            if report_code == "11013":
                periods = [
                    (f"{year} Q1",   await report(year, "11013")),
                    (f"{y_prev} Q4", await q4(y_prev)),
                    (f"{y_prev} Q1", await report(y_prev, "11013")),
                ]
            elif report_code == "11012":
                periods = [
                    (f"{year} H1",   await h1(year)),
                    (f"{y_prev} H2", await h2(y_prev)),
                    (f"{y_prev} H1", await h1(y_prev)),
                ]
            elif report_code == "11014":
                periods = [
                    (f"{year} Q3",   await report(year, "11014")),
                    (f"{year} Q2",   await report(year, "11012")),
                    (f"{y_prev} Q3", await report(y_prev, "11014")),
                ]
            elif report_code == "11011":
                periods = [
                    (f"{year} Annual",   await report(year, "11011")),
                    (f"{y_prev} Annual", await report(y_prev, "11011")),
                ]
            else:
                periods = []
            
            figs = [
                FinancialFigure(label=k, period=label, value=v)
                for label, data in periods for k, v in data.items()
            ]
            return {"figures": figs, "has_filing": True}

        elif ev.market == "US":
            raw = await edgar_list.ainvoke({
                "ticker": ev.ticker,
                "event_date": ev.event_date.isoformat(),
                "lookback_days": 30,
            })
            payload = _parse_tool_payload(raw)
            if payload is None:
                raise RuntimeError("edgar_list: invalid MCP payload")
            periodic = sorted(
                [f for f in payload.get("filings", [])
                 if f.get("form") in ("10-K", "10-K/A", "10-Q", "10-Q/A")],
                key=lambda f: f["filing_date"], reverse=True,
            )
            if not periodic:
                return {"figures": [], "has_filing": False}

            if "10-K" in periodic[0]["form"]:
                raw = await edgar_ymul.ainvoke({"ticker": ev.ticker, "n_years": 2})
                payload = _parse_tool_payload(raw)
                if payload is None:
                    raise RuntimeError("edgar_multi_years: invalid MCP payload")
                figs = [
                    FinancialFigure(label=k, period=f"{year} Annual", value=float(v))
                    for year, metrics in payload.get("by_year", {}).items()
                    for k, v in metrics.items() if v is not None
                ]
            else:
                raw = await edgar_qmul.ainvoke({"ticker": ev.ticker, "n_quarters": 5})
                payload = _parse_tool_payload(raw)
                if payload is None:
                    raise RuntimeError("edgar_multi_quarters: invalid MCP payload")
                figs = [
                    FinancialFigure(label=k, period=period, value=float(v))
                    for period, metrics in payload.get("by_period", {}).items()
                    for k, v in metrics.items() if v is not None
                ]
            return {"figures": figs, "has_filing": True}

        return {"figures": [], "has_filing": False}

    async def compose_memo_node(state: GraphState) -> dict:
        ev = state["event"]
        news = state.get("news", [])
        figures = state.get("figures", [])

        news_block = "\n".join(
            f"- [{n.published}] {n.title} ({n.source}) — {n.summary[:120]}"
            for n in news[:5]
        ) or "(뉴스 없음)"

        from collections import defaultdict
        by_label: dict[str, list[FinancialFigure]] = defaultdict(list)
        for f in figures:
            by_label[f.label].append(f)
        lines = []
        for label, items in by_label.items():
            items.sort(key=lambda x: x.period)
            series = " → ".join(f"{x.period}: {x.value:,.0f}" for x in items)
            lines.append(f"- {label}: {series}")
        figures_block = "\n".join(lines) or "(재무 없음)"

        prompt = (
            "Compose a Korean memo about the remarkable event below.\n\n"
            f"ticker: {ev.name} ({ev.ticker}, {ev.market}, scope={ev.scope})\n"
            f"date: {ev.event_date}  signal: {', '.join(ev.signals)}\n\n"
            f"news:\n{news_block}\n\n"
            f"financial figures:\n{figures_block} -> flow account(Revenue, OI, NI,...) figures reflect the exact time period, NOT the accumulative figure.\n\n "
            "Include:\n"
            "1) Possible trigger - quote news/financial figures.\n"
            "2) Information quality - note any discrepancies between news and financial figures.\n"
            "3) Summary review.\n"
        )

        result = await llm.ainvoke(prompt)
        summary = result.content.strip() if hasattr(result, "content") else str(result)

        return {"memo": Memo(
            event=ev,
            figures=figures,
            summary=summary,
            sources=[n.url for n in news if n.url],
            backend_used=backend,
        )}
    def _should_compose(state: GraphState) -> str:
        if state.get("has_news") or state.get("has_filing"):
            return "compose_memo"
        return END


    g = StateGraph(GraphState)
    g.add_node("fetch_news", fetch_news_node)
    g.add_node("fetch_financial", fetch_financial_node)
    g.add_node("compose_memo", compose_memo_node)
    g.add_edge(START, "fetch_news")
    g.add_edge("fetch_news", "fetch_financial")
    g.add_conditional_edges("fetch_financial", _should_compose, {"compose_memo": "compose_memo", END: END})
    g.add_edge("compose_memo", END)
    return g.compile()