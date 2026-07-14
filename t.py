from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.graph import build_news_agent
from app.llm_utils import get_llm


BASE_DIR = Path(__file__).resolve().parent
NEWS_SERVER_PATH = BASE_DIR / "mcp_servers" / "news_server.py"

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--market", choices=["KR", "US"], required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--signals", default="")
    args = parser.parse_args()

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(NEWS_SERVER_PATH)],
        env=os.environ.copy(),
    )

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(
            stdio_client(params)
        )

        session = await stack.enter_async_context(
            ClientSession(read, write)
        )

        await session.initialize()
        tools = await load_mcp_tools(session)

        news_tool = next(
            tool for tool in tools
            if tool.name == "fetch_news"
        )

        llm = get_llm(os.getenv("LLM_BACKEND", "cloud"))
        news_agent = build_news_agent(llm, news_tool)

        request = f"""
Investigate whether news can explain this market event.

Ticker: {args.ticker}
Company: {args.name}
Market: {args.market}
Event date: {args.date}
Detected signals: {args.signals or "not provided"}
"""

        async for state in news_agent.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": request,
                    }
                ]
            },
            stream_mode="values",
        ):
            message = state["messages"][-1]

            tool_calls = getattr(message, "tool_calls", None)

            if tool_calls:
                for call in tool_calls:
                    print("\n[TOOL CALL]")
                    print("name:", call["name"])
                    print("args:", call["args"])

            elif getattr(message, "type", "") == "tool":
                print("\n[TOOL RESULT]")
                print(str(message.content)[:2000])

            elif message.content:
                print("\n[AGENT]")
                print(message.content)


if __name__ == "__main__":
    asyncio.run(main())