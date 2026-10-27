# 브랜드 A 재고 코파일럿

**Ajou University AI Mini Project — Inventory Copilot**

광고비를 변경했을 때 예상 판매량과 재고 부족 가능성을 분석하는 AI 기반 재고 의사결정 지원 프로젝트입니다.

##프로젝트 소개

브랜드 A의 판매, 광고비, 재고 데이터를 활용하여 판매량을 분석하고, 머신러닝 기반 수요 예측과 AI Agent를 통해 재고 운영 의사결정을 지원하는 것을 목표로 합니다.

##기술 스택

- **Language:** Python
- **Dashboard:** Streamlit
- **Data Analysis:** Pandas, Matplotlib
- **Machine Learning:** Xgbrfr  
- **AI Agent:** LangChain 
- **Version Control:** Git, GitHub

##프로젝트 폴더 구조

```text
ajou-mini-project/
│
├── .venv/                 # Python 가상환경 (Git 제외)
│
├── data/                  # 판매·광고·재고 CSV 데이터
│
├── models/                # 학습된 머신러닝 모델 저장
│
├── src/                   # 주요 기능별 Python 코드
│   ├── agent.py           # LangChain AI Agent (개발 예정)
│   ├── forecast.py        # 판매량 예측 및 WAPE 평가 (개발 예정)
│   └── simulation.py      # 광고비 변경 시뮬레이션 (개발 예정)
│
├── app.py                 # Streamlit 대시보드
├── requirements.txt       # Python 라이브러리 목록
├── .gitignore             # Git 추적 제외 파일 설정
└── README.md              # 프로젝트 설명
```

##주요 기능

### 1. 판매량 분석
- 주별 판매량 추이 시각화
- 판매 채널 및 제품별 필터링

### 2. 광고비 분석
- 광고비와 판매량 관계 시각화
- 주별 광고비 변화 분석

### 3. 판매량 예측 (개발 예정)
- 최근 4주 평균 Baseline 구현
- 머신러닝 기반 판매량 예측
- WAPE를 활용한 모델 성능 비교

### 4. AI 재고 코파일럿 (개발 예정)
- 자연어 질문을 통한 분석
- 광고비 변경 시나리오 분석
- 예상 재고 부족 위험 확인

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

```bash
python -m pip install -r requirements.txt
```

### 5. Streamlit 실행

```bash
python -m streamlit run app.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

**참고:** 데이터 CSV 파일은 별도로 준비하여 `data/` 폴더에 배치해야 합니다. 제공 데이터는 GitHub에 공개하지 않습니다.

개발 진행 현황

- [x] 프로젝트 폴더 구조 구성
- [x] Streamlit 기본 대시보드 구성
- [x] 판매량 분석 시각화
- [x] 광고비 분석 시각화
- [ ] 판매량 예측 모델 구현
- [ ] WAPE 성능 평가
- [ ] 광고비 변경 시뮬레이션
- [ ] LangChain AI Agent 연결
- [ ] 최종 테스트 및 발표

데이터 및 보안

- 기업 제공 데이터는 공개 저장소에 업로드하지 않습니다.
- API 키와 환경변수 파일(`.env`)은 Git 추적에서 제외합니다.
- Python 가상환경(`.venv/`)은 Git에 업로드하지 않습니다.
