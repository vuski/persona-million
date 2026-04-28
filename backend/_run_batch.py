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
CONTEXT_DIR = BASE / "context"
RESULTS_PATH = BASE / "vote_results_all.csv"
RESPONSE_DIR = BASE / "response"
RESPONSE_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE / "log"
LOG_DIR.mkdir(exist_ok=True)

for env_path in [BASE / ".env.local", BASE.parent / ".env.local"]:
    if env_path.exists():
        load_dotenv(env_path)
        break

shards = sorted(glob.glob(str(DATA_DIR / "train-*.parquet")))

# ── 컨텍스트·시스템 프롬프트 버전 자동 감지 ──
def list_versions(prefix: str) -> list[str]:
    """context/<prefix>_v*.md 파일에서 v1, v2, ... 추출."""
    pat = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.md$")
    versions = []
    for p in CONTEXT_DIR.iterdir():
        m = pat.match(p.name)
        if m:
            versions.append(f"v{m.group(1)}")
    return sorted(versions, key=lambda v: int(v[1:]))

def load_context(version: str) -> str:
    return (CONTEXT_DIR / f"voter_context_{version}.md").read_text(encoding="utf-8")

def load_system_prompt(version: str) -> str:
    return (CONTEXT_DIR / f"system_prompt_{version}.md").read_text(encoding="utf-8")

USER_TEMPLATE = """다음 페르소나가 2026년 6월 3일 지방선거(또는 가까운 미래의 총선·대선)에서 어느 정당을 지지·투표할지 추론하시오.

{persona_card}

JSON만 출력."""

def build_prompt(system_template: str) -> ChatPromptTemplate:
    """system_template 의 '{political_context}' 자리에 컨텍스트가 주입됨."""
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", USER_TEMPLATE),
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

VOTE_KEYS = ("vote", "party", "정당", "지지정당", "투표", "선택", "지지")
REASON_KEYS = ("reason", "이유", "rationale", "사유", "근거", "설명")

def _try_complete_json(s: str) -> str:
    last_close = s.rfind("}")
    last_open = s.rfind("{")
    if last_open == -1:
        return s
    if last_close > last_open:
        return s[last_open:last_close + 1]
    candidate = s[last_open:].rstrip().rstrip(",")
    if candidate.count('"') % 2 == 1:
        candidate += '"'
    candidate += "}"
    return candidate

def parse_response(raw: str) -> dict:
    """thinking/코드블록 제거 후 JSON 추출. 잘린 응답은 자동 복구 시도."""
    s = raw
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    repaired = _try_complete_json(s)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        raise ValueError(f"no parseable JSON in response: {raw[:200]!r}")

def pick(d: dict, keys: tuple):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None

def safe_filename(s: str) -> str:
    return re.sub(r"[^\w\-.]", "_", s)

def run_loop(model_name: str, n: int, voter_v: str = "v1", system_v: str = "v1"):
    if voter_v != system_v:
        raise ValueError(
            f"voter_v({voter_v}) != system_v({system_v}). 같은 버전 쌍만 허용."
        )
    political_context = load_context(voter_v)
    system_template = load_system_prompt(system_v)

    # Qwen3는 user 메시지 끝에 '/no_think' 토큰 붙여 thinking 비활성화 (Qwen 공식)
    is_qwen3 = "qwen3" in model_name.lower()
    if is_qwen3:
        # user 템플릿에 /no_think 추가
        from langchain_core.prompts import ChatPromptTemplate
        no_think_user = USER_TEMPLATE + "\n\n/no_think"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", no_think_user),
        ])
    else:
        prompt = build_prompt(system_template)

    llm = ChatOllama(
        model=model_name, temperature=0.7,
        num_ctx=4096,
        num_predict=1200,  # qwen3 잘림 방지
    )
    chain = prompt | llm

    # stdout + 로그파일 동시 출력
    log_path = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe_filename(model_name)}_{voter_v}_{system_v}.log"
    log_f = log_path.open("w", encoding="utf-8")
    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"=== ollama/{model_name} × {n} (voter_context={voter_v}, system_prompt={system_v}) ===")

    success = 0
    for i in range(n):
        shard = random.choice(shards)
        df = pq.read_table(shard).to_pandas()
        r = df.sample(n=1).iloc[0]
        t0 = time.perf_counter()
        raw = ""
        try:
            msg = chain.invoke({"political_context": political_context,
                                "persona_card": persona_card(r)})
            raw = msg.content if hasattr(msg, "content") else str(msg)
            p = parse_response(raw)
            vote = pick(p, VOTE_KEYS)
            reason = pick(p, REASON_KEYS)
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

        # raw 응답 파일 저장 — 파일명: 타임스탬프_모델_uuid_voter_system.json
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = f"{ts}_{safe_filename(model_name)}_{r['uuid'][:8]}_{voter_v}_{system_v}.json"
        out = {
            "timestamp": ts,
            "model": f"ollama/{model_name}",
            "voter_context_version": voter_v,
            "system_prompt_version": system_v,
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
            "voter_context_version": voter_v,
            "system_prompt_version": system_v,
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
    # 사용: python _run_batch.py <model> <n> [version]
    # version 미지정 시 v1. voter/system 항상 동일 버전으로 페어링.
    model = sys.argv[1]
    n = int(sys.argv[2])
    v = sys.argv[3] if len(sys.argv) > 3 else "v1"
    run_loop(model, n, voter_v=v, system_v=v)
