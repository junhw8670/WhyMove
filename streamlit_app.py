from __future__ import annotations

from datetime import date

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/api/whymove/scan"

# ─── 페이지 설정 ─────────────────────────────────────────
st.set_page_config(
    page_title="WhyMove",
    page_icon="📊",
    layout="wide",
)

st.title("WhyMove")
st.caption("시장 신호 탐지 → 뉴스·공시 추적 → AI 분석 메모")

# ─── 사이드바 — 입력 폼 ──────────────────────────────────
with st.sidebar:
    st.header("스캔 설정")

    market = st.selectbox(
        "시장",
        options=["KR", "US"],
        index=0,
        help="KR = pykrx·DART·Naver / US = yfinance·EDGAR·Finnhub",
    )
    kr_market = st.selectbox(
        "KR 세부시장", ["ALL", "KOSPI", "KOSDAQ"], index=0,
    )
    target_date = st.date_input(
        "거래일",
        value=date(2025, 5, 30),
        help="이 거래일의 종가 기준으로 신호 탐지",
    )

    top_n = st.number_input(
        "Universe top N",
        min_value=1,
        max_value=3000,
        value=100,
        step=10,
        help="시총 상위 N개 종목 스캔",
    )

    submit = st.button("스캔 실행", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "메모 1건당 LLM 1회 호출. "
        "n_events 가 많을수록 응답 시간·토큰 비용 증가."
    )


if "result" not in st.session_state:
    st.session_state.result = None

if submit:
    with st.spinner(f"{market} 시장 / {target_date} / top {top_n}종목 스캔 중..."):
        try:
            r = requests.post(
                API_URL,
                json={
                    "market": market,
                    "date": target_date.isoformat(),
                    "top_n": int(top_n),
                    "kr_market": kr_market,
                },
                timeout=1200,
            )
            r.raise_for_status()
            st.session_state.result = r.json()
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except Exception:
                pass
            st.session_state.result = None
            st.error(f"백엔드 오류 (HTTP {r.status_code})\n\n{detail}")
        except requests.exceptions.RequestException as e:
            st.session_state.result = None
            st.error(f"백엔드 통신 실패 — uvicorn 떠있는지 확인:\n{e}")

result = st.session_state.result

if result is None:
    st.info("좌측 사이드바에서 설정 후 [스캔 실행]")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("시장", result.get("market", "—"))
    col2.metric("거래일", result.get("date", "—"))
    col3.metric("이벤트 수", result.get("n_events", 0))

    memos = result.get("memos", [])
    if not memos:
        st.warning("이 조건에서 탐지된 이벤트 없음. top_n 늘리거나 cutoff 낮춰서 재시도.")
    else:
        st.divider()
        for i, memo in enumerate(memos, 1):
            ev = memo.get("event", {})
            ticker = ev.get("ticker", "?")
            name = ev.get("name", "")
            scope = ev.get("scope", "single")
            score = ev.get("score", 0)
            signals = ", ".join(ev.get("signals", []))

            header = f"#{i}  [{scope}]  {ticker}  {name}  · score {score}  · {signals}"
            with st.expander(header, expanded=(i == 1)):
                left, right = st.columns([2, 1])

                left.subheader("AI 분석 메모")
                summary = memo.get("summary", "")
                if summary:
                    left.markdown(summary)
                else:
                    left.caption("(메모 없음)")

                figures = memo.get("figures", [])
                if figures:
                    right.subheader("재무")
                    df = pd.DataFrame(figures)
                    pivot = df.pivot_table(
                        index="label", columns="period", values="value",
                        aggfunc="first",
                    )
                    right.dataframe(
                        pivot.style.format("{:,.0f}", na_rep="—"),
                        use_container_width=True,
                    )
                else:
                    right.caption("재무 데이터 없음")

                sources = memo.get("sources", [])
                if sources:
                    right.subheader(f"출처 ({len(sources)})")
                    for url in sources[:10]:
                        right.markdown(f"- [{url[:60]}…]({url})")

                detail = ev.get("detail", {})
                if detail:
                    with left.expander("이벤트 detail (z-score 등)"):
                        st.json(detail)