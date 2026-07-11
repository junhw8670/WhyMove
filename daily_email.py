from __future__ import annotations

import os
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.main import app

load_dotenv()

def get_report(market: str, report_date: str) -> dict:
    with TestClient(app) as client:
        response = client.post(
            "/api/whymove/scan",
            json={
                "market": market,
                "date": report_date,
                "top_n": None,
                "kr_market": "ALL",
            },
        )
        response.raise_for_status()
        return response.json()


def build_email_body(report: dict) -> str:
    memos = report["memos"]

    lines = [
        "WhyMove Daily Report",
        f"Date: {report['date']}",
        f"Detected: {report['n_events']} items",
        "",
    ]

    if not memos:
        lines.append("No anomalous stocks detected.")
        return "\n".join(lines)

    for index, memo in enumerate(memos, 1):
        event = memo["event"]

        lines.extend(
            [
                f"{index}. {event['name']} ({event['ticker']})",
                f"Anomaly Score: {event['score']}",
                f"Detection Signals: {', '.join(event['signals'])}",
                "",
                memo["summary"],
            ]
        )

        sources = memo.get("sources") or []

        if sources:
            lines.append("")
            lines.append("Sources:")
            lines.extend(f"- {source}" for source in sources)

        lines.extend(["", "-" * 60, ""])

    return "\n".join(lines)


def send_email(body: str, report_date: str) -> None:
    sender = os.environ["EMAIL_SENDER"]
    password = os.environ["EMAIL_APP_PASSWORD"]

    recipients = [
        address.strip()
        for address in os.environ["EMAIL_RECIPIENTS"].split(",")
        if address.strip()
    ]

    if not recipients:
        raise RuntimeError("EMAIL_RECIPIENTS가 비어 있습니다.")

    message = EmailMessage()
    message["Subject"] = f"[WhyMove] {report_date} Daily Report"
    message["From"] = sender
    message["To"] = sender
    message["Bcc"] = ", ".join(recipients)
    message.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)


def main() -> None:
    report_date = (datetime.now(ZoneInfo("Asia/Seoul")).date() - timedelta(days=1)).isoformat()

    kr_report = get_report("KR", report_date)
    us_report = get_report("US", report_date)

    email_body = (
        build_email_body(kr_report)
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + build_email_body(us_report)
    )

    send_email(
        body=email_body,
        report_date=report_date,
    )

if __name__ == "__main__":
    main()