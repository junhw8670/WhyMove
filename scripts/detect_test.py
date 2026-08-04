from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.graph import scan_universe


MARKET = "US"     
KR_MARKET = "ALL"     
START_DATE = "2026-07-27"
END_DATE = "2026-08-02"
TOP_N_UNIVERSE = 500

KR_CACHE = ROOT / "cache" / "kr_ohlcv.parquet"
MARKET_SERVER = ROOT / "mcp_servers" / "market_server.py"


def get_scan_dates() -> list[str]:
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)

    if start > end:
        raise ValueError("START_DATE가 END_DATE보다 늦습니다.")

    if MARKET == "KR":
        cache = pd.read_parquet(KR_CACHE)
        cache["date"] = pd.to_datetime(cache["date"])

        dates = (
            cache.loc[
                cache["date"].between(start, end),
                "date",
            ]
            .drop_duplicates()
            .sort_values()
        )

        return dates.dt.strftime("%Y-%m-%d").tolist()

    return [
        day.strftime("%Y-%m-%d")
        for day in pd.bdate_range(start, end)
    ]


async def main() -> None:
    scan_dates = get_scan_dates()

    async with AsyncExitStack() as stack:
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(MARKET_SERVER)],
        )

        read, write = await stack.enter_async_context(
            stdio_client(params)
        )

        session = await stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()

        tools = {
            "market": await load_mcp_tools(session)
        }

        for scan_date in scan_dates:
            events = await scan_universe(
                tools=tools,
                market=MARKET,
                date=scan_date,
                top_n=TOP_N_UNIVERSE,
                kr_market=KR_MARKET,
            )

            companies = [
                event
                for event in events
                if event.scope == "single"
                and event.event_date.isoformat() == scan_date
            ]

            if not companies:
                continue

            print(f"\n[{scan_date}] {len(companies)}개")

            for rank, event in enumerate(companies, start=1):
                print(
                    f"{rank:>2}. "
                    f"{event.ticker} | "
                    f"score={event.score:.2f} | "
                    f"{', '.join(event.signals)}"
                )


if __name__ == "__main__":
    asyncio.run(main())