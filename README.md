# WhyMove
한국·미국 주식의 주가·거래량·수급 변동 이상치를 탐지하고, 그 발생 원인을 뉴스와 공시·재무로 자동 추적해 분석 메모를 제공하여 투자 의사결정에 도움을 주는 멀티 에이전트 모니터링 시스템입니다. 

---

### 이 프로젝트로 할 수 있는 것
1. 주목할 만한 변화가 있는 종목/섹터 탐지
2. 이상치의 원인을 추적하는 뉴스·공시 자동 추적
3. 뉴스와 공시·재무자료를 대조하여 정보의 품질 분석
4. 종합 분석 메모 제공

---

### 기술 스택
- Orchestration: `LangGraph` (커스텀 StateGraph - 명시적 노드 + 조건부 엣지)
- LLM: `ChatOpenAI`(기본)·`ChatOllama`(로컬)
- MCP Servers:
    - `market_mcp`
    - `news_mcp`
    - `filings_mcp`
- Backend: `FastAPI`
- Frontend: `Streamlit` + `Plotly`

---

### 데이터 소스
- 시세·거래량: `pykrx`, `yfinance`
- 뉴스: 네이버 검색 API, `Finnhub`
- 공시·재무: OpenDART, SEC EDGAR

---

### 프로젝트 구조
```
WhyMove/
    app/
        main.py                 # FastAPI 엔트리포인트, 3개 MCP 서버 로드
        graph.py                # LangGraph 커스텀 StateGraph
        signal.py               # 이상탐지, 단일/섹터 분류
        llm_utils.py            # 하이브리드 LLM 스위치
        models.py               # Pydantic State / Event /Memo
        metrics.py              # 재무비율·수치 계산
    mcp_servers/
        market_server.py        # 주가·거래량·수급
        news_server.py          # 뉴스
        filings_server.py       # 공시·재무
    scripts/
        calibrate_thresholds.py # 기존 비슷한 사례의 수익률 정보 분석
    docs/                       
        devlog/                 # 개발일지
    streamlit_app.py            # Streamlit UI (대시보드)
    requirements.txt
    .env
    