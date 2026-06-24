from __future__ import annotations

import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Optional
import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel

from .graph import build_event_graph, scan_universe

BASE_DIR = Path(__file__).resolve().parent.parent
MCP_DIR = BASE_DIR / "mcp_servers"

_p = os.getenv("DART_MCP_PATH")
if not _p:
    raise RuntimeError("DART_MCP_PATH를 .env에 설정하세요")
DART_MCP_PATH = Path(_p)

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncExitStack() as stack:
        servers = [
            ("market", MCP_DIR / "market_server.py"),
            ("edgar", MCP_DIR / "edgar_server.py"),
            ("news",   MCP_DIR / "news_server.py"),
            ("dart",   DART_MCP_PATH),
        ]
    
        tools_by_server: dict[str, list] = {}
        for name, path in servers:
            params = StdioServerParameters(command=sys.executable, args=[str(path)])
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            tools_by_server[name] = await load_mcp_tools(session)

        app.state.graph = build_event_graph(tools_by_server)
        app.state.tools = tools_by_server
        yield


app = FastAPI(
    title="WhyMove",
    description="시장 신호 탐지 -> 뉴스/공시 자동 추적 -> AI 분석 메모",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict:
    """For health check."""
    return {"status": "ok", "service": "WhyMove"}


class ScanRequest(BaseModel):
    market: str
    date: str
    top_n: Optional[int] = None
    kr_market: Optional[str] = "ALL"


@app.post("/api/whymove/scan")
async def scan(payload: ScanRequest) -> dict:
    """Per-day universe scan + memo generation.

    Stream:
    1. scan_universe:
        - get_sector_map -> select universe
        - each ticker fetch_ohlcv -> signal.detect_events
        - aggregated ticker events -> signal.detect_sector_breadth -> add sector event
        - apply daily cap using rank_and_cap_daily
        Result: Event list('single' | 'sector')
    2. ainvoke compiled LangGraph for each Event: fetch_news / fetch_filings / fetch_financial -> create LLM memo
    3. respond based on memos
    """
    try:
        events = await scan_universe(
            tools=app.state.tools,
            market=payload.market,
            date=payload.date,
            top_n=payload.top_n,
            kr_market=payload.kr_market or "ALL"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scan failed: {e}")

    sem = asyncio.Semaphore(5)

    async def _build_memo(ev):
        async with sem:
            try:
                state = await app.state.graph.ainvoke({"event": ev})
                return state.get("memo")
            except Exception as e:
                logging.warning(f"memo failed for {ev.ticker}: {e}")
                return None
        
    results = await asyncio.gather(*[_build_memo(ev) for ev in events])
    memos = [m for m in results if m]

    return {
        "market": payload.market,
        "date": payload.date,
        "n_events": len(events),
        "memos": memos,
    }