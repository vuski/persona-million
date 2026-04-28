"""모델별 N회 페르소나 투표 추론 → CSV + raw 응답 파일 누적."""
import os, glob, time, random, sys, json, re
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "nvidia-personas" / "data"
CONTEXT_PATH = BASE / "context" / "voter_context.md"
RESULTS_PATH = BASE / "vote_results_all.csv"
RESPONSE_DIR = BASE / "response"
RESPONSE_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE / "log"
LOG_DIR.mkdir(exist_ok=True)

for env_path in [BASE / ".env.local", BASE.parent / ".env.local"]:
    if env_path.exists():
        load_dotenv(env_path)
        break

POLITICAL_CONTEXT = CONTEXT_PATH.read_text(encoding="utf-8")
shards = sorted(glob.glob(str(DATA_DIR / "train-*.parquet")))

SYSTEM_TEMPLATE = """당신은 사회조사 시뮬레이션을 위한 가상 응답자다. 주어진 페르소나의 인물 서사·인구통계·가치관·관심사를 바탕으로, 그 인물이 실제로 할 법한 정치 선택을 1인칭 시점에서 추론한다.

다음 규칙을 따른다:
1. 아래 [정치상황 컨텍스트]에 명시된 정당과 쟁점만 사용한다. 외부 정보·최신 추정 금지.
2. 페르소나의 연령·지역·직업·생애 맥락·가치관과 정합적인 선택을 한다. 단순히 인구통계 평균에 의존하지 말고 인물 서사를 우선한다.
3. 무당층·기권도 정당한 선택지다. 페르소나가 정치에 무관심하거나 기존 정당 모두에 거리감이 있으면 그렇게 답한다.
4. 출력은 반드시 다음 JSON 형식만 (코드블록·설명·머리말 일절 금지):
{{"vote": "<정당명 또는 '무당층/기권'>", "reason": "<해당 인물의 1인칭 시점으로 200~400자 이내 한국어 설명>"}}

## [정치상황 컨텍스트]
{political_context}
"""

USER_TEMPLATE = """다음 페르소나가 2026년 6월 3일 지방선거(또는 가까운 미래의 총선·대선)에서 어느 정당을 지지·투표할지 추론하시오.

{persona_card}

JSON만 출력."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE), ("user", USER_TEMPLATE),
])

def persona_card(r) -> str:
    fields = [
        ("성별", r["sex"]), ("연령", f"{r['age']}세"),
        ("혼인상태", r["marital_status"]),
        ("가구형태", r["family_type"]), ("주거", r["housing_type"]),
        ("학력", r["education_level"]), ("전공", r["bachelors_field"]),
        ("직업", r["occupation"]),
        ("지역", f"{r['province']} {r['district']}"),
    ]
    demo = "\n".join(f"- {k}: {v}" for k, v in fields if v)
    narr = (
        f"[요약]\n{r['persona']}\n\n"
        f"[직업적 면모]\n{r['professional_persona']}\n\n"
        f"[가족 면모]\n{r['family_persona']}\n\n"
        f"[문화적 배경]\n{r['cultural_background']}\n\n"
        f"[관심사]\n{r['hobbies_and_interests']}\n\n"
        f"[목표]\n{r['career_goals_and_ambitions']}"
    )
    return f"## 인구통계\n{demo}\n\n## 인물 서사\n{narr}"

def parse_response(raw: str) -> dict:
    """thinking/코드블록/잡소리 제거하고 JSON 추출."""
    s = raw
    # <think>...</think> 블록 제거 (Qwen3 등)
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    # 코드블록 제거
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    # JSON 객체 추출 (첫 { 부터 마지막 } 까지)
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    return json.loads(m.group(0))

def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", s)

def run_loop(model_name: str, n: int):
    # Qwen3는 thinking 토큰 많이 뱉으니 ctx·predict 더 크게 잡음
    is_qwen3 = "qwen3" in model_name.lower()
    llm = ChatOllama(
        model=model_name, temperature=0.7,
        num_ctx=8192 if is_qwen3 else 4096,
        num_predict=4096 if is_qwen3 else 800,
    )
    chain = prompt | llm

    # stdout + 로그파일 동시 출력
    log_path = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_filename(model_name)}.log"
    log_f = log_path.open("w", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"=== ollama/{model_name} × {n} ===")

    success = 0
    for i in range(n):
        shard = random.choice(shards)
        df = pq.read_table(shard).to_pandas()
        r = df.sample(n=1).iloc[0]
        t0 = time.perf_counter()
        raw = ""
        try:
            msg = chain.invoke({"political_context": POLITICAL_CONTEXT,
                                "persona_card": persona_card(r)})
            raw = msg.content if hasattr(msg, "content") else str(msg)
            p = parse_response(raw)
            vote = p.get("vote") or p.get("party") or p.get("정당") or p.get("투표")
            reason = p.get("reason") or p.get("이유") or p.get("rationale")
            if not vote or not reason:
                raise ValueError(f"missing keys: {list(p.keys())}")
        except Exception as e:
            el = time.perf_counter() - t0
            log(f"[{i+1:>3}/{n}] ERROR ({el:.1f}s): {str(e)[:120]}")
            # 실패한 raw도 디버깅용으로 저장
            ts = time.strftime("%Y%m%d-%H%M%S")
            fname = f"FAIL_{ts}_{safe_filename(model_name)}_{r['uuid'][:8]}.txt"
            (RESPONSE_DIR / fname).write_text(
                f"=== ERROR: {e}\n=== MODEL: {model_name}\n=== UUID: {r['uuid']}\n\n{raw}",
                encoding="utf-8")
            continue
        el = time.perf_counter() - t0
        success += 1

        # raw 응답 파일 저장 — 파일명: 타임스탬프_모델_uuid.json
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}_{safe_filename(model_name)}_{r['uuid'][:8]}.json"
        out = {
            "timestamp": ts,
            "model": f"ollama/{model_name}",
            "persona_uuid": r["uuid"],
            "elapsed_sec": round(el, 3),
            "persona": {
                "sex": r["sex"], "age": int(r["age"]),
                "marital_status": r["marital_status"],
                "family_type": r["family_type"],
                "housing_type": r["housing_type"],
                "education_level": r["education_level"],
                "bachelors_field": r["bachelors_field"],
                "occupation": r["occupation"],
                "province": r["province"], "district": r["district"],
                "persona_summary": r["persona"],
            },
            "raw_response": raw,
            "parsed": {"vote": vote, "reason": reason},
        }
        (RESPONSE_DIR / fname).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        # CSV 누적
        rec = {
            "persona_uuid": r["uuid"],
            "sex": r["sex"], "age": int(r["age"]),
            "marital_status": r["marital_status"],
            "family_type": r["family_type"], "housing_type": r["housing_type"],
            "education_level": r["education_level"],
            "bachelors_field": r["bachelors_field"],
            "occupation": r["occupation"],
            "province": r["province"], "district": r["district"],
            "persona_summary": r["persona"],
            "vote": vote, "reason": reason,
            "model": f"ollama/{model_name}",
            "elapsed_sec": round(el, 3),
            "response_file": fname,
            # 풀 페르소나 10개 필드
            "professional_persona": r["professional_persona"],
            "sports_persona": r["sports_persona"],
            "arts_persona": r["arts_persona"],
            "travel_persona": r["travel_persona"],
            "culinary_persona": r["culinary_persona"],
            "family_persona": r["family_persona"],
            "cultural_background": r["cultural_background"],
            "skills_and_expertise": r["skills_and_expertise"],
            "hobbies_and_interests": r["hobbies_and_interests"],
            "career_goals_and_ambitions": r["career_goals_and_ambitions"],
        }
        df_rec = pd.DataFrame([rec])
        if RESULTS_PATH.exists():
            df_rec.to_csv(RESULTS_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            df_rec.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
        log(f"[{i+1:>3}/{n}] {r['sex']} {r['age']}세 {r['province']:<6} "
            f"{r['occupation'][:14]:<14} → {vote[:10]:<10} ({el:.1f}s) → {fname}")

    log(f"\n결과: {success}/{n} 성공")
    log_f.close()

if __name__ == "__main__":
    model = sys.argv[1]
    n = int(sys.argv[2])
    run_loop(model, n)
