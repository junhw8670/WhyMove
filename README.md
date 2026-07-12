# WhyMove
거래일마다 한국·미국 주식의 주가·거래량·수급 변동 이상징후를 탐지하고, 관련 뉴스와 공시·재무자료를 자동으로 추적하여 근거가 포함된 AI 분석 메모를 생성하고 이메일로 전송하는 멀티에이전트 시장 모니터링 시스템입니다.

---

### 이 프로젝트로 할 수 있는 것
1. 주목할 만한 변화가 있는 종목/섹터 탐지
2. 이상치의 원인을 추적하는 뉴스·공시 자동 수집
3. 수치와 외부 근거를 종합한 AI 분석 메모 생성
4. 출처 링크를 통한 분석 결과의 역추적 및 검증
5. 일정 시각 자동 실행 및 이메일 보고서 전송

---

### 기술 스택
- Orchestration: `LangGraph` (커스텀 StateGraph - 명시적 노드 + 조건부 엣지)
- LLM: `ChatOpenAI`(cloud)·`ChatOllama`(local)
- Backend: `FastAPI`
- Frontend: `Streamlit`
- MCP:
    - 시장 데이터 MCP
    - 뉴스 검색 MCP
    - SEC EDGAR MCP
    - OpenDART MCP (`DartCopilot` 연동)
- Automation:
    - `GitHub Actions` scheduled workflow
- Delivery:
    - Gmail SMTP
    - Python `EmailMessage`

---

### 데이터 소스
- 시세·거래량: `pykrx`, `yfinance`
- 뉴스: 네이버 검색 API, `Finnhub`
- 공시·재무: OpenDART, SEC EDGAR

---

### 프로젝트 구조
```
WhyMove/
    .github/
        workflows/
            daily-email.yml     # 자동 실행 및 이메일 발송
    app/
        main.py                 # FastAPI 엔트리포인트
        graph.py                # LangGraph 분석 워크플로
        detect.py               # 단일/섹터 이상탐지
        llm_utils.py            # 하이브리드 LLM 스위치
        models.py               # Pydantic State / Event / Memo
        kr_cache.py             # 국내 주가 데이터 캐시
    mcp_servers/
        market_server.py        # 주가·거래량·섹터 정보
        news_server.py          # 뉴스
        edgar_server.py         # 공시·재무
    scripts/
        signal_backtest.py      # 시그널 별 수익률 백테스트
        build_us_sector_map.py  # 미국 주식 섹터 매핑 빌드
        sentiment_backtest.py   # 시그널 + 뉴스 감성 분석 백테스트
        make_chart.py           # 시각화 자료 생성
    cache/
        kr_ohlcv.parquet        # KR 주가정보
        us_marketcap.json       # US 시가총액
        us_sector_map.json      # US 섹터맵
    docs/                       
        devlog/                 # 개발일지
    daily_email.py              # 분석 실행 및 이메일 보고서 생성
    streamlit_app.py            # Streamlit UI (대시보드)
    requirements.txt
    .env
```

---

### 동기
관심 종목이 급등락하면 "왜?"를 찾느라 시간을 쓰고, 안 보던 종목의 급등은 뒤늦게 안다.
임계치를 넘는 움직임을 빠르게 **탐지**하고 원인까지 **요약**해주면 시간 절약 + 기회 포착에 유용하겠다는 생각에서 시작.

---

### 동작
1. **신호 탐지** — 전일대비 수익률·갭·거래량을 과거 1년 대비 z-score화, **z>2.5** 초과분을 score 합산 + 52주 고·저가 시 +0.5. score >= 1.0이면 이벤트. 같은 섹터 30% 동반 시 섹터 이벤트.
2. **원인 추적** — 탐지 종목의 최근 뉴스·최신 공시 자동 수집 -> LLM 프롬프트에 결합.
3. **AI 메모** — 신호·수치·재무·뉴스를 종합한 분석 메모 생성.

---

### 검증 (백테스트)
- 신호별 향후 수익률이 시장(baseline) 대비 유의한지 부트스트랩으로 검정.  

  KR - 유의미한 평균 차이 없음.  
  US - **52주 신고·저가** 지표에서 양방향으로 큰 평균 초과수익률 발생. 검정 결과 95% 신뢰구간에서 통계적으로 유의함을 입증.  

    <img src="results/img/exc_by_signal_US.png" width="800">

    <img src="results/img/bootstrap_ci_20d.png" width="450">  <img src="results/img/bootstrap_ci_60d.png" width="450">
    
    > 나머지 지표들도 긍정 신호는 양의 초과수익률, 부정 신호는 음의 초과수익률 평균을 나타냈으나 신뢰구간이 0을 포함. -> 확신 불가.

- 신호 + 뉴스 감성 결합 분석으로 확장.(US 단독)  

  **긍정 신호 + 긍정 뉴스 20일 초과 수익률 평균(+1.56%) > 긍정 신호 + 부정 뉴스 20일 초과 수익률 평균(-1.38%)**. 그러나 95% 신뢰구간이 상당부분 겹치는 결과. 통계적 확신은 얻지 못함.  

- **결론**  

  52주 신고가(20일)·신저가(60일)는 시장 대비 통계적으로 유의한 초과/미달 수익을 보여 추세 전환 포착 신호로서의 유효성을 확인했음. 나머지 신호들은 유의성까지 도달하지는 못했으나 긍정 신호는 양(+), 부정 신호는 음(-)의 평균 초과수익이라는 일관된 방향성을 보임.  
  -> 지표 설계의 방향성은 타당했다고 판단.  

  뉴스 감성을 결합한 결과 '긍정 신호 + 긍정 뉴스'와 '긍정 신호 + 부정 뉴스'의 20일 평균 초과수익이 약 3%p 벌어졌다(+1.56% vs -1.38%). 신뢰구간이 겹쳐 통계적 단정은 어려우나 신호에 맥락을 더하면 결과가 분리되는 경향은 관찰됨.  

  요컨대 본 시스템은 단독으로 초과수익을 보장하는 알파 도구라기보다 시장의 움직임을 추적·해석하는 도구로서 가치가 있음. 이는 원인을 파악하여 투자 판단에 활용하고자 하는 본 프로젝트의 목표와 부합함.  

---

### 최종 산출물 (예시)

#### Streamlit 화면
<img src="results/img/streamlit_sample.png" width="1000">  

#### 수신된 E-mail 화면
<img src="results/img/email_sam1.png" width="800">
<img src="results/img/email_sam2.png" width="800">
<img src="results/img/email_sam3.png" width="950">
<img src="results/img/email_sam4.png" width="800">
