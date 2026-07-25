import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\WhyMove\mcp_servers")

from edgar_server import (
    fetch_filings_around,
    fetch_multi_quarters,
    fetch_multi_years,
)


OUTPUT_DIR = Path("results/data/edgar_tool_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


COMPANIES = [
    {
        "name": "Apple",
        "ticker": "AAPL",
    },
    {
        "name": "Microsoft",
        "ticker": "MSFT",
    },
    {
        "name": "NVIDIA",
        "ticker": "NVDA",
    },
]


def run_tool(tool_name, function, arguments):
    try:
        result = function(**arguments)

        return {
            "input": arguments,
            "result": result,
        }

    except Exception as error:
        return {
            "input": arguments,
            "error": str(error),
        }


def save_result(tool_name, results):
    output_path = OUTPUT_DIR / f"{tool_name}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print(f"저장 완료: {output_path}")


# 1. fetch_filings_around

filing_results = []

for company in COMPANIES:
    filing_results.append(
        run_tool(
            "fetch_filings_around",
            fetch_filings_around,
            {
                "ticker": company["ticker"],
                "event_date": "2026-07-25",
                "lookback_days": 90,
            },
        )
    )

save_result("fetch_filings_around", filing_results)


# 2. fetch_multi_quarters

quarter_results = []

for company in COMPANIES:
    quarter_results.append(
        run_tool(
            "fetch_multi_quarters",
            fetch_multi_quarters,
            {
                "ticker": company["ticker"],
                "n_quarters": 5,
            },
        )
    )

save_result("fetch_multi_quarters", quarter_results)


# 3. fetch_multi_years

year_results = []

for company in COMPANIES:
    year_results.append(
        run_tool(
            "fetch_multi_years",
            fetch_multi_years,
            {
                "ticker": company["ticker"],
                "n_years": 5,
            },
        )
    )

save_result("fetch_multi_years", year_results)


print()
print("전체 EDGAR 도구 실행 완료")
print(f"결과 폴더: {OUTPUT_DIR.resolve()}")