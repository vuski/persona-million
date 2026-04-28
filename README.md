# persona-million

100만 합성 페르소나(NVIDIA Nemotron-Personas-Korea)를 LLM에게 입력해, 한국 정치 상황에서 어떻게 투표할지 시뮬레이션하는 실험.

**라이브 대시보드**: 깃헙 페이지 활성화 시 `https://<username>.github.io/persona-million/`

---

## 무엇을 하나

1. NVIDIA가 공개한 한국 합성 페르소나(100만 명, 26 필드: 인구통계 + 직업·예술·여행·요리·가족 등 6개 영역 서사)를 다운로드
2. 무작위 샘플링한 페르소나 1명을 LLM에 주입
3. "2026년 4월 한국 정치 상황 컨텍스트"와 함께 가상 투표 의향을 묻고 JSON으로 받음
4. 결과를 CSV에 누적, 모델·인구통계별 비교 분석

OpenAI(GPT-5.4 / 5.5 / mini) 와 로컬(EXAONE 4.0 / Qwen3 / Gemma3) 6개 모델로 30명씩 = **180건** 수집된 상태.

---

## 직접 돌려보기

### 0. 환경

- Python 3.10+
- Windows: PowerShell 기준 (코드는 OS 무관, 명령어만 PowerShell)
- (선택) NVIDIA GPU 24GB+ — 로컬 모델 돌릴 때만
- (선택) Ollama — 로컬 모델 돌릴 때만

### 1. 클론 + venv

```powershell
git clone https://github.com/<username>/persona-million.git
cd persona-million
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install langchain langchain-openai langchain-ollama python-dotenv pyarrow pandas streamlit plotly streamlit-autorefresh huggingface_hub
```

### 2. 페르소나 데이터 다운로드 (~3GB, 9 parquet 샤드)

데이터셋은 용량 때문에 git에 포함 안 됨. 직접 받아야 함.

Python에서:
```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/Nemotron-Personas-Korea",
    repo_type="dataset",
    local_dir="backend/nvidia-personas",
    local_dir_use_symlinks=False,
)
```

### 3. API 키 설정 (OpenAI 모델 쓸 때만)

프로젝트 루트에 `.env.local` 파일 생성:
```
OPENAI_API_KEY=sk-...
```

### 4. 로컬 모델 받기 (Ollama 쓸 때만)

```powershell
ollama pull ingu627/exaone4.0:32b   # LG, 한국어 특화 (19GB)
ollama pull qwen3:32b               # Alibaba, 다국어 (20GB)
ollama pull gemma3:27b              # Google, 다국어 (17GB)
```

### 5. 실행

#### 옵션 A — 노트북에서 한 명씩 인터랙티브
```
backend/testPersona.ipynb
```
Cell 1~8 실행 후 Cell 9~12 중 원하는 모델 셀에서 N 조정 → 실행.

#### 옵션 B — 배치 스크립트로 N명 누적

```powershell
# OpenAI
.\.venv\Scripts\python.exe backend\_run_openai_batch.py gpt-5.4-mini 30

# Ollama
.\.venv\Scripts\python.exe backend\_run_batch.py qwen3:32b 30
```

결과: `backend/vote_results_all.csv` 에 누적, 응답 raw JSON은 `backend/response/` 에 1건당 파일로 저장.

### 6. 대시보드 보기

#### 로컬 Streamlit (실시간 자동 새로고침)
```powershell
streamlit run backend\dashboard.py
```
브라우저 자동 열림.

#### 정적 HTML (Stlite — Pyodide 기반, 서버 불필요)
```powershell
python -m http.server 8765
```
브라우저에서 `http://localhost:8765/` 접속. 첫 로딩 5~10초.

GitHub Pages 배포된 페이지도 같은 `index.html` 로 동작.

---

## 디렉토리 구조

```
persona-million/
├── index.html                      # Stlite 정적 대시보드 (GitHub Pages 진입점)
├── backend/
│   ├── _run_batch.py               # Ollama 배치 실행
│   ├── _run_openai_batch.py        # OpenAI 배치 실행
│   ├── testPersona.ipynb           # 인터랙티브 노트북
│   ├── dashboard.py                # Streamlit 대시보드 (실시간)
│   ├── loadData.ipynb              # 페르소나 다운로드 노트북
│   ├── vote_results_all.csv        # 단일 결과 CSV (27 컬럼)
│   ├── context/
│   │   ├── voter_context.md        # LLM 프롬프트 주입용 (중립, 수치·출처 제거)
│   │   └── research_notes.md       # 리서처용 (출처·여론조사 수치 포함)
│   ├── nvidia-personas/            # 페르소나 데이터 (gitignore, 직접 다운로드)
│   ├── response/                   # 응답 1건당 raw JSON
│   └── log/                        # 배치 실행 로그 (gitignore)
├── .env.local                      # API 키 (gitignore)
└── README.md
```

---

## 결과 CSV 스키마 (27 컬럼)

- 식별자: `persona_uuid`
- 인구통계 (10): `sex`, `age`, `marital_status`, `family_type`, `housing_type`, `education_level`, `bachelors_field`, `occupation`, `province`, `district`
- 페르소나 서사 (11): `persona_summary`, `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, `family_persona`, `cultural_background`, `skills_and_expertise`, `hobbies_and_interests`, `career_goals_and_ambitions`
- 결과 (6): `vote`, `reason`, `model`, `elapsed_sec`, `response_file`

---

## 주요 결정 / 한계

- **`voter_context.md` 는 정교하게 구성된 컨텍스트가 아님**: Claude Code가 웹 리서치한 결과를 단순 요약·정리해 만든 초안 수준. 정당 명단·쟁점 선정·표현 톤·중립성 모두 검증 미흡. **정밀한 시뮬레이션을 위해서는 컨텍스트 자체에 대한 섬세한 조율(어떤 사실을 넣고 뺄지, 어떤 톤으로 서술할지)과 주입 방식에 대한 별도 연구가 필요함.** 이 결과는 그 연구의 출발점일 뿐, 결론이 아님.
- **시점 고정**: 컨텍스트의 정치 상황은 2026년 4월 기준. 다른 시점 실험하려면 파일을 갈아끼우면 됨.
- **여론조사 수치는 컨텍스트에서 의도적으로 제외**: 모델이 그 비율로 답을 맞추려는 앵커링 방지. 출처·수치는 `research_notes.md` 에만 보존.
- **temperature=0.7**: 결정적 결과 원하면 코드에서 0으로 변경.
- **샘플 수가 작음**: 모델당 30명은 파일럿 수준. 통계적 결론보다 모델별 응답 패턴 차이 보기 위함.
- **합성 데이터**: 페르소나는 NVIDIA가 한국 통계청·대법원 등 실제 통계로 합성한 가상 인물. 실재 인물과 무관.

---

## 라이선스 / 저작권

- 코드: 자유롭게 사용
- 페르소나 데이터: NVIDIA Nemotron-Personas-Korea, **CC BY 4.0** (상업 OK, 출처 표기 필요)
- EXAONE 4.0: **EXAONE NC** (비상업 연구만, 상업 금지)
- Qwen 3: Apache 2.0
- Gemma 3: Gemma Terms of Use

---

## 참고

- 데이터셋: https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea
- Stlite: https://github.com/whitphx/stlite
- LangChain Ollama: https://python.langchain.com/docs/integrations/chat/ollama/
