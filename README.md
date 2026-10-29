
# 브랜드 A 재고 코파일럿

**Ajou University AI Mini Project — Inventory Copilot**

광고비 및 할인율 변경에 따른 예상 판매량과 재고 부족 가능성을 분석하는 AI 기반 재고 의사결정 지원 프로젝트입니다.

---

## 프로젝트 소개

브랜드 A의 판매·광고·재고 데이터를 분석하고, 머신러닝 기반 판매량 예측과 LangChain AI Agent를 결합하여 재고 운영 의사결정을 지원합니다.

Streamlit 대시보드에서 판매 및 광고비 추이를 확인할 수 있으며, 자연어 질문을 통해 데이터 조회, 판매량 예측, 광고비·할인율 시뮬레이션, 재고 소진 및 발주 필요성 분석을 수행할 수 있습니다.

**프로젝트 목표**

- 제품별·채널별 판매 및 광고비 현황 분석
- XGBoost 기반 머신러닝 판매량 예측
- 광고비·할인율 변경에 따른 예상 판매량 비교
- 재고 소진 기간과 발주 필요성 검토
- AI Agent와 Tool을 활용한 자연어 기반 분석

---

## 기술 스택

<div align="center">

<img src="https://skillicons.dev/icons?i=python,sklearn,git,github" alt="Python, Scikit-learn, Git, GitHub" />

<br/>

<img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas"/>
<img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy"/>
<img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square" alt="Matplotlib"/>
<img src="https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square" alt="Seaborn"/>
<img src="https://img.shields.io/badge/XGBoost-EC6B23?style=flat-square" alt="XGBoost"/>
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit"/>
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain"/>
<img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white" alt="OpenAI"/>

</div>

---

## 주요 기능

### 1. 판매량 분석

- 주차별 판매량 추이 시각화
- 제품별 판매량 비교
- 판매 채널별 판매량 분석
- 판매 데이터 기반 주요 패턴 확인

### 2. 광고비 분석

- 주차별 광고비 변화 시각화
- 제품별·채널별 광고비 분석
- 광고비와 판매량의 관계 분석

### 3. 재고 분석

- 제품별 현재 보유 재고 확인
- 현재 재고 및 발주 진행 수량 조회
- 재고 데이터 기반 현황 분석

### 4. 머신러닝 기반 판매량 예측

XGBoost의 `XGBRFRegressor` 모델과 전처리 Scaler를 활용하여 제품 및 채널별 예상 판매량을 계산합니다.

- 개별 제품 및 전체 제품 예측
- 쿠팡, 네이버, 자사몰 채널 지원
- 1~12주 누적 예상 판매량 계산
- 광고비 및 할인율 조건을 반영한 예측
- 학습된 머신러닝 모델을 활용한 추론

**참고:** 현재 예측은 동일한 주간 조건을 반복 적용한 누적 시뮬레이션 방식입니다. 미래 각 주의 변화를 개별적으로 예측하는 방식은 아닙니다.

### 5. LangChain 기반 AI Agent

LangChain의 `create_agent`와 OpenAI API를 활용하여 사용자의 자연어 질문을 분석하고 적절한 Tool을 호출합니다.

**일반 AI Agent**

| Tool | 기능 |
|---|---|
| `get_sales_data` | 판매 실적 데이터 조회 |
| `get_ad_spend_data` | 광고비 데이터 조회 |
| `get_inventory_data` | 현재 재고 데이터 조회 |

**판매량 예측 AI Agent**

| Tool | 기능 |
|---|---|
| `predict_sales` | 제품 및 채널별 예상 판매량 계산 |
| `simulate_ad_spend` | 광고비 변경 전후 판매량 비교 |
| `simulate_discount` | 할인율 변경 전후 판매량 비교 |
| `simulate_inventory` | 재고 소진 및 발주 필요성 분석 |

**Middleware 및 대화 처리**

- 업무 범위 외 질문 차단
- Tool 호출 및 토큰 사용량 로깅
- 이전 대화의 제품·채널·기간 조건 활용
- Tool 계산 결과 기반 응답 생성

### 6. 재고 소진 및 발주 시뮬레이션

- 현재 재고 기반 예상 소진 기간 계산
- 예상 판매량을 활용한 품절 위험 분석
- 리드타임 및 목표 재고 기준 발주 필요성 검토
- 진행 중인 발주 수량 반영

재고 분석은 제공된 데이터 기준일의 스냅샷을 활용하며, 실시간 재고를 의미하지 않습니다.

---

## AI 챗봇 질문 예시

프로젝트에서 지원하는 주요 자연어 질문 예시입니다.

### 일반 데이터 조회

**질문 1**

> 최근 4주간 전체 채널의 판매량을 알려줘.

**질문 2**

> 네이버와 쿠팡의 최근 8주 광고비를 비교해줘.

**질문 3**

> 현재 제품별 보유 재고와 발주 중인 수량을 알려줘.

### 판매량 예측 및 시뮬레이션

**질문 4**

> 장벽 크림의 국내 전체 채널 기준 4주 예상 판매량은 얼마야?

**질문 5**

> 장벽 크림의 네이버 광고비를 30% 늘리면 4주 예상 판매량이 얼마나 달라져?

**질문 6**

> 그럼 할인율을 10%p 올리면 어떻게 달라져?

**질문 7**

> 진정 캡슐 앰플의 현재 재고는 몇 주 동안 유지될까?

**질문 8**

> 전체 제품의 국내 3개 채널 기준 4주 예상 판매량을 알려줘.

**참고:** 위 질문은 기능 시연용 예시입니다. 모델 예측 결과는 입력 조건에 따라 달라질 수 있습니다.

---

## 프로젝트 폴더 구조

```text
ajou-mini-project/
│
├── data/                       # 판매·광고·재고 CSV 데이터
│
├── models/                     # 학습된 머신러닝 모델
│   ├── robust_scaler.pkl       # 데이터 전처리 Scaler
│   └── xgbrfr_model.json       # XGBoost 예측 모델
│
├── src/                        # 주요 Python 코드
│   ├── agent.py                # 일반 AI Agent
│   ├── forecast.py             # 예측 모델 로딩 및 추론
│   ├── inventory_tool.py       # 재고 시뮬레이션 Tool
│   ├── middleware.py           # 업무 범위 제한 및 토큰 로깅
│   ├── prediction_chatbot.py   # 판매량 예측 AI Agent
│   ├── prediction_tool.py      # 예측 및 시뮬레이션 Tool
│   └── tool.py                 # 일반 데이터 조회 Tool
│
├── ui/                         # Streamlit 챗봇 UI
│   ├── __init__.py
│   ├── agent_chat.py           # 일반 AI 챗봇 화면
│   └── prediction_chat.py      # 예측 AI 챗봇 화면
│
├── app.py                      # Streamlit 메인 대시보드
├── advcamp_cap.ipynb           # 데이터 분석 노트북
├── requirements.txt            # Python 라이브러리 목록
├── .gitignore                  # Git 추적 제외 설정
└── README.md                   # 프로젝트 설명
```

---

## 실행 방법

### 1. 저장소 복제

```bash
git clone https://github.com/Kim-Ki-yeol/ajou-mini-project.git
cd ajou-mini-project
```

### 2. 가상환경 생성

```powershell
python -m venv .venv
```

### 3. 가상환경 활성화 (Windows PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. 라이브러리 설치

```powershell
python -m pip install -r requirements.txt
```

### 5. 환경변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 OpenAI API 키를 설정합니다.

```env
OPENAI_API_KEY=your_openai_api_key
```

### 6. 데이터 준비

프로젝트에서 참조하는 CSV 파일을 `data/` 폴더에 배치합니다.

- `sales_weekly.csv`
- `sales_export.csv`
- `inventory_now.csv`
- `future_plan_4w.csv`
- `product_master.csv`

기업 제공 데이터는 공개하지 않으므로 별도로 준비해야 합니다.

### 7. Streamlit 실행

```powershell
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

---

## 개발 진행 현황

- [x] 프로젝트 폴더 구조 구성
- [x] Streamlit 대시보드 구현
- [x] 판매량 분석 및 시각화
- [x] 광고비 분석 및 시각화
- [x] 재고 현황 분석
- [x] XGBoost 기반 판매량 예측 모델 연결
- [x] 일반 데이터 조회 AI Agent 구현
- [x] 광고비 변경 시뮬레이션
- [x] 할인율 변경 시뮬레이션
- [x] 재고 소진 및 발주 시뮬레이션
- [x] 판매량 예측 AI Agent 구현
- [x] 멀티턴 대화 처리
- [x] Middleware 기반 업무 외 질문 차단
- [x] 주요 예측 및 Tool 호출 테스트
- [ ] 최종 발표 및 시연 정리

---

## 데이터 및 보안

- API 키와 환경변수 파일(`.env`)은 Git 추적에서 제외합니다.
- Python 가상환경(`.venv/`)과 `__pycache__/`는 Git 추적에서 제외합니다.
- 판매량 예측 및 시뮬레이션 결과는 모델 기반 추정값으로 실제 판매량을 보장하지 않습니다.
- 재고 분석은 데이터 기준일의 스냅샷을 사용하며 실시간 재고와 다를 수 있습니다.
