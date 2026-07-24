import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp_servers.news_server import fetch_news


OUTPUT = ROOT / "results" / "data" / "news_accuracy_after_2.csv"
rows = []


def save():
    OUTPUT.parent.mkdir(exist_ok=True)

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "provider",
                "title",
                "summary",
                "content",
                "url",
                "published",
                "relevant",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(provider, result):
    print(
        f"\n{provider}: 후보 {result['candidate_count']}건 "
        f"→ 필터 후 {result['filtered_count']}건"
    )

    for i, item in enumerate(result["items"], 1):
        print("\n" + "=" * 80)
        print(f"[{provider} {i}] {item['title']}")
        print(f"요약: {item['summary']}")
        print(f"내용: {item['llm_text']}")
        print(f"URL: {item['url']}")

        while True:
            answer = input(
                "관련 뉴스? [y/n]: "
            ).strip().lower()

            if answer in {"y", "n"}:
                break

        rows.append({
            "provider": provider,
            "title": item["title"],
            "summary": item.get("summary", ""),
            "content": item.get("content", ""),
            "url": item["url"],
            "published": item["published"],
            "relevant": answer,
        })

        save()

    relevant = sum(
        row["relevant"] == "y"
        for row in rows
        if row["provider"] == provider
    )
    total = sum(
        row["provider"] == provider
        for row in rows
    )

    accuracy = relevant / total * 100 if total else 0

    print(
        f"\n{provider}: "
        f"{relevant}/{total} = {accuracy:.1f}%"
    )


today = date.today().isoformat()

# evaluate(
#     "naver",
#     fetch_news(
#         ticker="009150",
#         market="KR",
#         event_date=today,
#         lookback_days=7,
#         name="",
#         limit=10,
#     ),
# )
# evaluate(
#     "naver",
#     fetch_news(
#         ticker="012330",
#         market="KR",
#         event_date=today,
#         lookback_days=7,
#         name="",
#         limit=10,
#     ),
# )
# evaluate(
#     "naver",
#     fetch_news(
#         ticker="352820",
#         market="KR",
#         event_date=today,
#         lookback_days=7,
#         name="",
#         limit=10,
#     ),
# )
# evaluate(
#     "naver",
#     fetch_news(
#         ticker="000660",
#         market="KR",
#         event_date=today,
#         lookback_days=7,
#         name="",
#         limit=10,
#     ),
# )
# evaluate(
#     "naver",
#     fetch_news(
#         ticker="035420",
#         market="KR",
#         event_date=today,
#         lookback_days=7,
#         name="",
#         limit=10,
#     ),
# )

evaluate(
    "finnhub",
    fetch_news(
        ticker="GOOG",
        market="US",
        event_date=today,
        lookback_days=7,
        name="",
        limit=10,
    ),
)
evaluate(
    "finnhub",
    fetch_news(
        ticker="JOBY",
        market="US",
        event_date=today,
        lookback_days=7,
        name="",
        limit=10,
    ),
)
evaluate(
    "finnhub",
    fetch_news(
        ticker="ARM",
        market="US",
        event_date=today,
        lookback_days=7,
        name="",
        limit=10,
    ),
)
evaluate(
    "finnhub",
    fetch_news(
        ticker="ASTS",
        market="US",
        event_date=today,
        lookback_days=7,
        name="",
        limit=10,
    ),
)
evaluate(
    "finnhub",
    fetch_news(
        ticker="NFLX",
        market="US",
        event_date=today,
        lookback_days=7,
        name="",
        limit=10,
    ),
)
print(f"\n저장 완료: {OUTPUT}")