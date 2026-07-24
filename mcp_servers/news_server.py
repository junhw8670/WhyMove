from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import trafilatura
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("News")

NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")
FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

TAG_RE = re.compile(r'<[^>]+>')
DOMAIN_RE = re.compile(r'https?://(?:www\.)?([^/]+)')

def _strip_html(s: str) -> str:
    s = TAG_RE.sub("", s)
    for ent, ch in (("&quot;", '"'), ("&amp;", "&"),
                    ("&lt;", "<"), ("&gt;", ">"), ("&#39;", "'")):
        s = s.replace(ent, ch)
    return s

def _domain(url: str) -> str:
    m = DOMAIN_RE.match(url)
    return m.group(1) if m else ""

def _kr_name(ticker: str) -> str:
    from pykrx import stock
    return stock.get_market_ticker_name(ticker) or ticker


def _us_names(ticker: str) -> list[str]:
    import re
    import yfinance as yf

    info = yf.Ticker(ticker).info

    long_name = (
        info.get("longName")
        or info.get("shortName")
        or ticker
    )

    short_name = re.sub(
        r"\s+(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|PLC)$",
        "",
        long_name,
        flags=re.IGNORECASE,
    ).strip()

    return list(dict.fromkeys([long_name, short_name]))


def _fetch_naver(query: str, n: int = 100) -> list[dict]:
    if not (NAVER_ID and NAVER_SECRET):
        raise RuntimeError("NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set in .env")

    r = requests.get(
        "https://openapi.naver.com/v1/search/news.json",
        headers={
            "X-Naver-Client-Id": NAVER_ID,
            "X-Naver-Client-Secret": NAVER_SECRET,
        },
        params={"query": query, "display": min(n, 100), "sort": "date"},
        timeout=15,
    )
    r.raise_for_status()
    raw = r.json().get("items", [])

    out = []
    for it in raw:
        try:
            dt = datetime.strptime(it["pubDate"], "%a, %d %b %Y %H:%M:%S %z")
        except (KeyError, ValueError):
            continue
        url = it.get("originallink") or it.get("link") or ""
        out.append({
            "title": _strip_html(it.get("title", "")),
            "summary": _strip_html(it.get("description", "")),
            "url": url,
            "source": _domain(url),
            "published": dt.strftime("%Y-%m-%d"),
        })
    return out


def _fetch_finnhub(symbol: str, frm: str, to: str) -> list[dict]:
    if not FINNHUB_KEY:
        raise RuntimeError("FINNHUB_API_KEY not set in .env")

    r = requests.get(
        "https://finnhub.io/api/v1/company-news",
        params={"symbol": symbol, "from": frm, "to": to, "token": FINNHUB_KEY},
        timeout=15,
    )
    r.raise_for_status()
    raw = r.json()

    out = []
    for it in raw:
        try:
            dt = datetime.fromtimestamp(it["datetime"], tz=timezone.utc)
        except (KeyError, ValueError, TypeError):
            continue
        out.append({
            "title": it.get("headline", ""),
            "summary": it.get("summary", ""),
            "url": it.get("url", ""),
            "source": it.get("source", ""),
            "published": dt.strftime("%Y-%m-%d"),
        })
    return out


def _is_relevant(item: dict, keywords: list[str]) -> bool:
    title = item.get("title", "").lower()
    summary = item.get("summary", "").lower()

    return any(
        keyword.lower() in title
        and keyword.lower() in summary
        for keyword in keywords
        if keyword
    )


def _fetch_article_text(url: str, max_chars: int = 5000) -> str:
    if not url:
        return ""

    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/124.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()

    text = trafilatura.extract(
        response.content,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )

    if not text:
        return ""

    return text.strip()[:max_chars]


def _filter_and_enrich(
    items: list[dict],
    keywords: list[str],
    limit: int = 10,
) -> list[dict]:
    relevant = [
        item for item in items
        if _is_relevant(item, keywords)
    ]

    result = []

    for item in relevant:
        item = item.copy()

        try:
            content = _fetch_article_text(
                item.get("url", "")
            )
        except Exception:
            content = ""

        summary = item.get("summary", "").strip()
        item["content"] = content

        if len(content.strip()) <= 250:
            continue
        
        item["llm_text"] = content

        result.append(item)

        if len(result) >= limit:
            break

    return result


@mcp.tool()
def fetch_news(
    ticker: str, market: str, event_date: str, lookback_days: int = 7, name: Optional[str] = None, limit: int = 10
) -> dict:
    """Fetch news within lookback_days to event_date.

    Args:
        ticker: KR 6-digit or US symbol.
        market: 'KR' or 'US'.
        event_date: 'YYYY-MM-DD' or 'YYYYMMDD'.
        lookback_days: days to look back. Default 7.
        name: KR search keyword. If not, ticker -> company name auto transform.

    Returns:
        {
            "ticker": "...",
            "market": "KR" | "US",
            "items": [
                {"title": "...", "summary": "...", "content": "...", "llm_text": "...", 
                "url": "...", "source": "...", "published": "YYYY-MM-DD"},
                ...
            ]
        }
    """
    d = datetime.strptime(event_date.replace("-", ""), "%Y%m%d").date()
    bgn = (d - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end = d.strftime("%Y-%m-%d")

    if market == "KR":
        query = name or _kr_name(ticker)
        candidates = [
            it for it in _fetch_naver(query, n=100)
            if bgn <= it["published"] <= end
        ]

        keywords = [query]

    elif market == "US":
        candidates = _fetch_finnhub(
            symbol=ticker,
            frm=bgn,
            to=end,
        )

        keywords = [name] if name else _us_names(ticker)
    else:
        raise ValueError(f"Unknown market: {market!r}")

    items = _filter_and_enrich(
        candidates,
        keywords=keywords,
        limit=limit,
    )

    return {"ticker": ticker, "market": market, "keywords": keywords, "candidate_count": len(candidates), "filtered_count": len(items), "items": items}


if __name__ == "__main__":
    mcp.run(transport="stdio")
        