from __future__ import annotations

import asyncio
import json
import re
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os

import pandas as pd
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.graph import END, START, StateGraph
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
import logging

from .llm_utils import get_llm
from .models import Event, FinancialFigure, GraphState, Market, Memo, NewsItem, NewsAnalysis, FinancialAnalysis, OrchestrationPlan
from .detect import detect_events, detect_sector_breadth, rank_and_cap_daily

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
    _p = os.getenv("DART_INDUSTRY_PATH")
    if not _p:
        raise RuntimeError("DART_INDUSTRY_PATH를 .env에 설정하세요")
    path = Path(_p)    
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
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
    history_days: int = 370,
    kr_market: str = "ALL",
) -> list[Event]:
    get_sm = _find(tools["market"], "get_sector_map")
    raw = await get_sm.ainvoke({"market": market})
    payload = _parse_tool_payload(raw)
    if payload is None:
        raise RuntimeError("get_sector_map: invalid MCP payload")
    sm = payload["sector_map"]

    end = datetime.strptime(date.replace("-", ""), "%Y%m%d").date()
    st = pd.Timestamp(end - timedelta(days=history_days))

    frames: list[tuple[str, pd.DataFrame]] = []
    if market == "KR":
        from .kr_cache import update, CACHE
        from pykrx import stock
        if not CACHE.exists():
            raise RuntimeError("Execute `python -m app.kr_cache`.")
        try:
            cache = update()
        except Exception as e:
            logger.warning(f"KR cache update failed, using stale: {e}")
            cache = pd.read_parquet(CACHE)

        cache = cache[cache["date"] <= pd.Timestamp(end)]
        live = set(cache.loc[cache["date"] == cache["date"].max(), "ticker"])
        
        base = cache["date"].max().strftime("%Y%m%d")
        mkts = [kr_market] if kr_market in ("KOSPI", "KOSDAQ") else ["KOSPI", "KOSDAQ"]
        try:
            cap = pd.concat([stock.get_market_cap_by_ticker(base, market=m)["시가총액"] for m in mkts])
            ranked = [t for t in cap.sort_values(ascending=False).index if t in live]
            tickers = ranked[:top_n] if top_n else ranked
        except Exception as e:
            logger.warning(f"market cap fetch failed: {e}")
            tickers = list(live)[:top_n] if top_n else list(live)

        for t, g in cache[cache["ticker"].isin(tickers)].groupby("ticker"):
            df = g[g["date"] >= st].set_index("date").sort_index()[["Open", "High", "Low", "Close", "Volume"]]
            if len(df) < 60 or (df["Close"] * df["Volume"]).median() < 1e8:
                continue
            frames.append((t, df))
    else:
        fetch = _find(tools["market"], "fetch_ohlcv")
        tickers = list(sm)
        if top_n:
            cap_path = Path(__file__).resolve().parent.parent / "cache" / "us_marketcap.json"
            if cap_path.exists():
                caps = json.loads(cap_path.read_text(encoding="utf-8"))
                tickers = sorted((t for t in tickers if t in caps), key=lambda t: caps[t], reverse=True)[:top_n]
            else:
                tickers = tickers[:top_n]

        sem = asyncio.Semaphore(8)

        async def grab(ticker: str) -> None:
            async with sem:
                try:
                    r = await asyncio.wait_for(fetch.ainvoke({
                        "ticker": ticker, "market": market,
                        "start": st.strftime("%Y-%m-%d"), "end": end.strftime("%Y-%m-%d"),
                    }), 15)
                    rows = (_parse_tool_payload(r) or {}).get("rows", [])
                    if rows:
                        df = pd.DataFrame(rows).set_index("date")
                        df.index = pd.to_datetime(df.index)
                        frames.append((ticker, df))
                except Exception as e:
                    logger.warning(f"grab({ticker}): {e}")

        await asyncio.gather(*[grab(t) for t in tickers])
        
    singles = [ev for t, df in frames for ev in detect_events(df, t, market, last_only=True)]
    sectors = detect_sector_breadth(singles, sm, market, universe=[t for t, _ in frames])
    return (
        rank_and_cap_daily(singles, max_per_day=10)
        + rank_and_cap_daily(sectors, max_per_day=3)
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

    news_agent = create_agent(
        model=llm,
        tools=[news_tool],
        response_format=ToolStrategy(NewsAnalysis),
        system_prompt=
        """
        You are a News Analysis Agent that searches and analyzes news articles related to a specific stock event.
        Given the news_tool, you MUST follow these instructions:
        1) Search for news articles published right before or during the market event.
        2) You may call the tool again with a different lookback_days if the initial results are insufficient.
        3) Do not treat post-event news as the original cause of the event; instead, use it to understand the market's reaction and sentiment.
        4) Separate confirmed facts from speculation and inference.
        5) Assess whether the news plausibly explains the direction and magnitude of the price movement.
        6) Never invent facts, article contents, publication times, or sources.
        7) Stop when sufficient evidence is found or further searches are unlikely to help.
        8) If no plausible catalyst is found, state that clearly.
        9) Considering the time gap between the event and the news, evaluate whether the news could have influenced the market's reaction.
        10) If the news handles multiple companies, sectors or markets, just use the information for providing context.
        11) Irrelevent news should be ignored, and you should focus on the most relevant articles.
        12) Write the final answer in Korean.
        """,
        name="news_agent",
    )

    orchestration_llm = llm.with_structured_output(OrchestrationPlan)

    KR_ACCOUNT_ALIASES: dict[str, set[str]] = {
        "매출액": {"매출액", "수익(매출액)", "영업수익", "수익"},
        "영업이익": {"영업이익", "영업손실", "영업이익(손실)"},
        "당기순이익": {"당기순이익", "분기순이익", "반기순이익", "당기순이익(손실)", "분기순이익(손실)", "반기순이익(손실)", "당기순손실"},
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

    async def fetch_financial_node(state: GraphState, lookback_days: int = 30) -> dict:
        ev = state["event"]
        if ev.scope != "single":
            return{"figures": [], "has_filing": False}

        if ev.market == "KR":
            corp_code = _stock_to_corp().get(ev.ticker)
            if not corp_code:
                return {"figures": [], "has_filing": False}

            bgn = (ev.event_date - timedelta(days=lookback_days)).strftime("%Y%m%d")
            end = ev.event_date.strftime("%Y%m%d")
            raw = await dart_list.ainvoke({
                "corp_code": corp_code, "bgn_de": bgn, "end_de": end,
            })
            payload = _parse_tool_payload(raw)
            if payload is None:
                raise RuntimeError("dart_list: invalid MCP payload")
            latest = None
            latest_nm = ""
            for f in sorted(payload.get("list", []), key=lambda x: x.get("rcept_dt", ""), reverse=True):
                parsed = _parse_kr_report(f.get("report_nm", ""))
                if parsed:
                    latest = parsed
                    latest_nm = f.get("report_nm", "")
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
            return {"figures": figs, "has_filing": True, "filing_info": latest_nm}

        elif ev.market == "US":
            raw = await edgar_list.ainvoke({
                "ticker": ev.ticker,
                "event_date": ev.event_date.isoformat(),
                "lookback_days": lookback_days,
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
            return {"figures": figs, "has_filing": True, 
                    "filing_info": f"{periodic[0]['form']} (filed {periodic[0]['filing_date']})"}

        return {"figures": [], "has_filing": False}


    async def fetch_financial_context(
        ticker: str,
        name: str,
        market: str,
        event_date: str,
        lookback_days: int = 30,
    ) -> dict:
        event = Event(
            ticker=ticker,
            name=name,
            market=market,
            event_date=date.fromisoformat(event_date),
            scope="single",
            signals=[],
            score=0,
            detail={},
        )

        result = await fetch_financial_node(
                {"event": event},
                lookback_days= lookback_days,
        )

        return {
            "has_filing": result.get("has_filing", False),
            "filing_info": result.get("filing_info", ""),
            "figures": [
                figure.model_dump()
                for figure in result.get("figures", [])
            ],
        }

    financial_context_tool = StructuredTool.from_function(
        coroutine=fetch_financial_context,
        name="fetch_financial_context",
        description=(
            "Fetch deterministic filing metadata and normalized financial figures for a Korean or US stock around a market event."
        ),
    )

    financial_agent = create_agent(
        model=llm,
        tools=[financial_context_tool],
        response_format=ToolStrategy(FinancialAnalysis),
        system_prompt=
        """
        You are a Financial Analysis Agent that searches and analyzes financial information related to a specific stock event.
        Given the financial_context_tool, you MUST follow these instructions:
        1) Search for news articles published right before or during the market event.
        2) You may call the tool again with a different lookback_days if the initial results are insufficient(for example, 7 -> 30 -> 60 -> 180).
        3) Use `fetch_financial_context` to obtain filing metadata and normalized financial figures.
        4) Use only values returned by the tool.
        5) Identify material changes across periods.
        6) Considering the distance between the filing date and the market-event date, evaluate whether the filing could have influenced the market's reaction.
        7) Do not claim that financial results caused the market event.
        8) State clearly when the available figures are insufficient.
        9) Write the final answer in Korean.
        """,
        name="financial_agent",
    )


    def build_event_prompt(event: Event) -> str:
        return f"""
        Analyze the following market event.

        Company/Target: {event.name}
        Ticker: {event.ticker}
        Market: {event.market}
        Event Date: {event.event_date.isoformat()}
        Scope: {event.scope}
        Detected Signals: {", ".join(event.signals)}
        Score: {event.score}
        Price/Volume Information: {json.dumps(event.detail, ensure_ascii=False)}
        """.strip()


    async def orchestrator_node(state: GraphState) -> dict:
        event = state["event"]

        return {
            "research_plan": OrchestrationPlan(
                run_news=True,
                run_financial=event.scope == "single",
                reason=(
                    "Basically, news research is always performed, but financial analysis is only performed for individual company events."
                ),
            )
        }


    async def news_agent_node(state: GraphState) -> dict:
        plan = state["research_plan"]

        if not plan.run_news:
            return {
                "news_analysis": NewsAnalysis(
                    news_findings="오케스트레이터가 뉴스 조사를 생략했습니다."
                )
            }

        result = await news_agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": build_event_prompt(state["event"]),
                }
            ]
        })

        sources = []

        for message in result["messages"]:
            payload = _parse_tool_payload(message.content)

            if not payload:
                continue
            
            for item in payload.get("items", []):
                url = item.get("url")
                if url and url not in sources:
                    sources.append(url)

        return {
            "news_analysis": result["structured_response"],
            "news_sources": sources,    
        }

    async def financial_agent_node(state: GraphState) -> dict:
        plan = state["research_plan"]

        if not plan.run_financial:
            return {
                "financial_analysis": FinancialAnalysis(
                    financial_findings=[
                        "섹터 이벤트이거나 오케스트레이터가 재무 조사를 생략했습니다."
                    ]
                ),
                "figures": [],
            }

        result = await financial_agent.ainvoke({
            "messages": [
                {
                    "role": "user",
                    "content": build_event_prompt(state["event"]),
                }
            ]
        })

        figures = []

        for message in result["messages"]:
            payload = _parse_tool_payload(message.content)

            if not payload:
                continue
            
            tool_figures = payload.get("figures", [])

            if tool_figures:
                figures = [
                    FinancialFigure(**fig)
                    for fig in tool_figures
                ]

        return {
            "financial_analysis": result["structured_response"],
            "figures": figures,
        }


    async def compose_memo_node(state: GraphState) -> dict:
        event = state["event"]
        plan = state["research_plan"]
        news = state["news_analysis"]
        financial = state["financial_analysis"]

        prompt = f"""
    Compose a Korean market-event memo based on the following agents' findings.
    
    Event:
    {build_event_prompt(event)}

    Orchestration Plan:
    {plan.model_dump_json(indent=2)}

    News Agent results:
    {news.model_dump_json(indent=2)}

    Financial Agent results:
    {financial.model_dump_json(indent=2)}

    작성 규칙:
    1. Distinguish between confirmed facts and speculation. Clearly indicate which statements are based on evidence and which are inferences or opinions.
    2. DO NOT invent any facts by fusing multiple news articles or financial reports. Only use the information provided by the agents.
    3. Clarify the time order of news and financial information in explaining the market event.
    4. Evaluate whether the price movements are sufficiently explained.
    5. If evidence is insufficient, clearly state "cause unclear".
    """

        result = await llm.ainvoke(prompt)
        summary = (
            result.content.strip()
            if isinstance(result.content, str)
            else str(result.content)
        )

        return {
            "memo": Memo(
                event=event,
                figures=state.get("figures", []),
                summary=summary,
                sources=state.get("news_sources", []),
                backend_used=backend,
                model_used=os.getenv(
                    "OLLAMA_MODEL" if backend == "local" else "OPENAI_MODEL",
                    "",
                )
            )
        }


    g = StateGraph(GraphState)

    g.add_node("orchestrator", orchestrator_node)
    g.add_node("news_agent", news_agent_node)
    g.add_node("financial_agent", financial_agent_node)
    g.add_node("compose_memo", compose_memo_node)

    g.add_edge(START, "orchestrator")

    g.add_edge("orchestrator", "news_agent")
    g.add_edge("orchestrator", "financial_agent")

    g.add_edge(
        ["news_agent", "financial_agent"],
        "compose_memo",
    )

    g.add_edge("compose_memo", END)

    return g.compile()