"""v2 / v3 / v4(gpt) / v4(haiku) 페르소나별 응답 비교 — 같은 100명.

실행:
    streamlit run backend/compareV2V3.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
RESULTS_PATH = BASE / "vote_results_all.csv"
INTERESTS_PATH = BASE / "interests" / "personas_gpt-5.4-mini_all_interests.tsv"
CONTEXT_DIR = BASE / "context"
GPT_MODEL = "openai/gpt-5.4-mini"
HAIKU_MODEL = "anthropic/claude-haiku-4-5-20251001"


@st.cache_data
def load_prompt(version: str) -> tuple[str, str]:
    sp = (CONTEXT_DIR / f"system_prompt_{version}.md").read_text(encoding="utf-8")
    vc = (CONTEXT_DIR / f"voter_context_{version}.md").read_text(encoding="utf-8")
    return sp, vc


@st.dialog("프롬프트 보기", width="large")
def show_prompt(version: str) -> None:
    sp, vc = load_prompt(version)
    st.caption(
        f"**{version}** · `system_prompt_{version}.md` 의 `{{political_context}}` 자리에 "
        f"`voter_context_{version}.md` 가 치환되어 LLM에 전달됨"
    )
    tab_sys, tab_ctx = st.tabs(["system_prompt", "voter_context"])
    with tab_sys:
        st.markdown(sp)
    with tab_ctx:
        st.markdown(vc)

PARTY_COLORS = {
    "더불어민주당": "#004EA2",
    "국민의힘": "#E61E2B",
    "개혁신당": "#FF7920",
    "조국혁신당": "#06275E",
    "진보당": "#D6001C",
    "자유통일당": "#8E44AD",
    "무당층/기권": "#9E9E9E",
}

def party_badge(vote: str) -> str:
    color = PARTY_COLORS.get(vote, "#666")
    return (
        f"<span style='background:{color};color:white;padding:3px 10px;"
        f"border-radius:12px;font-weight:600;font-size:0.9em;"
        f"display:inline-block;white-space:nowrap'>{vote}</span>"
    )

st.set_page_config(page_title="v2 / v3 / v4 / v5 비교", layout="wide", initial_sidebar_state="collapsed")

# 메인 컨테이너 가로폭을 화면 거의 전체로
st.markdown(
    """
    <style>
    .block-container {max-width: 100% !important; padding-left: 1.5rem; padding-right: 1.5rem;}
    section[data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("페르소나별 v2 / v3 / v4 / v5 / v6 응답 비교")
st.caption(
    "같은 100명에게 voter_context/system_prompt v2 → v3 (지역·세대 단서 제거) → v4 (정치적 이해관계 투입) → v5 (정당 정책 풍부화 + 매칭 강제·무당층 억제) → v6 (권력 시그널 뒤집기 — 가상). "
    "v4·v5·v6는 `openai/gpt-5.4-mini` 와 `anthropic/claude-haiku-4-5-20251001` 두 모델 응답을 나란히 표시."
)

# ── 버전 on/off 토글 ──
ALL_VERSIONS = [
    ("v2", "v2", "v2 · gpt"),
    ("v3", "v3", "v3 · gpt"),
    ("v4_gpt", "v4", "v4 · gpt"),
    ("v4_haiku", "v4", "v4 · haiku"),
    ("v5_gpt", "v5", "v5 · gpt"),
    ("v5_haiku", "v5", "v5 · haiku"),
    ("v6_gpt", "v6", "v6 · gpt"),
    ("v6_haiku", "v6", "v6 · haiku"),
]
toggle_cols = st.columns(len(ALL_VERSIONS))
SHOW = {}
for (key, _, label), tcol in zip(ALL_VERSIONS, toggle_cols):
    SHOW[key] = tcol.toggle(label, value=True, key=f"toggle_{key}")


@st.cache_data(ttl=10)
def load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")

    base_cols = [
        "persona_uuid", "sex", "age", "marital_status", "family_type",
        "housing_type", "education_level", "bachelors_field", "occupation",
        "province", "district", "persona_summary",
    ]

    def pick(model: str, version: str, suffix: str) -> pd.DataFrame:
        sub = df[(df["model"] == model) & (df["voter_context_version"] == version)]
        sub = sub.drop_duplicates("persona_uuid", keep="last")
        if len(sub) == 0:
            return pd.DataFrame(columns=["persona_uuid", f"vote_{suffix}", f"reason_{suffix}"])
        return sub[base_cols + ["vote", "reason"]].rename(
            columns={"vote": f"vote_{suffix}", "reason": f"reason_{suffix}"}
        )

    v2 = pick(GPT_MODEL, "v2", "v2")
    v3 = pick(GPT_MODEL, "v3", "v3")
    v4_gpt = pick(GPT_MODEL, "v4", "v4_gpt")
    v4_haiku = pick(HAIKU_MODEL, "v4", "v4_haiku")
    v5_gpt = pick(GPT_MODEL, "v5", "v5_gpt")
    v5_haiku = pick(HAIKU_MODEL, "v5", "v5_haiku")
    v6_gpt = pick(GPT_MODEL, "v6", "v6_gpt")
    v6_haiku = pick(HAIKU_MODEL, "v6", "v6_haiku")

    merged = v2.copy()
    for sub, suffix in [
        (v3, "v3"),
        (v4_gpt, "v4_gpt"), (v4_haiku, "v4_haiku"),
        (v5_gpt, "v5_gpt"), (v5_haiku, "v5_haiku"),
        (v6_gpt, "v6_gpt"), (v6_haiku, "v6_haiku"),
    ]:
        if len(sub) > 0:
            merged = merged.merge(
                sub[["persona_uuid", f"vote_{suffix}", f"reason_{suffix}"]],
                on="persona_uuid", how="left",
            )
        else:
            merged[f"vote_{suffix}"] = pd.NA
            merged[f"reason_{suffix}"] = pd.NA

    if INTERESTS_PATH.exists():
        ints = pd.read_csv(INTERESTS_PATH, sep="\t", encoding="utf-8-sig")[
            ["persona_uuid", "political_interest"]
        ]
        merged = merged.merge(ints, on="persona_uuid", how="left")
    else:
        merged["political_interest"] = pd.NA

    return merged.reset_index(drop=True)


df = load()

# ── 메트릭 ──
both23 = df.dropna(subset=["vote_v2", "vote_v3"])
changed23 = (both23["vote_v2"] != both23["vote_v3"]).sum()
c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
c1.metric("페르소나", len(df))
c2.metric("v2·v3 비교", f"{len(both23)} (변경 {changed23})")
c3.metric("v4 (gpt)", df["vote_v4_gpt"].notna().sum())
c4.metric("v4 (haiku)", df["vote_v4_haiku"].notna().sum())
c5.metric("v5 (gpt)", df["vote_v5_gpt"].notna().sum())
c6.metric("v5 (haiku)", df["vote_v5_haiku"].notna().sum())
c7.metric("v6 (gpt)", df["vote_v6_gpt"].notna().sum())
c8.metric("v6 (haiku)", df["vote_v6_haiku"].notna().sum())
c9.metric("political_interest", df["political_interest"].notna().sum())

# ── 필터: 헤더 가로 배치 ──
with st.container():
    f1, f2, f3, f4, f5 = st.columns([1.2, 1.5, 1.5, 1.5, 1.5])
    with f1:
        f_changed = st.selectbox(
            "v2→v3 변경",
            ["전체", "바뀐 사람만", "유지된 사람만"], index=0,
        )
    with f2:
        f_v2 = st.multiselect("v2 정당", sorted(df["vote_v2"].dropna().unique().tolist()))
    with f3:
        f_v3 = st.multiselect("v3 정당", sorted(df["vote_v3"].dropna().unique().tolist()))
    with f4:
        f_prov = st.multiselect("지역", sorted(df["province"].dropna().unique().tolist()))
    with f5:
        f_sex = st.multiselect("성별", sorted(df["sex"].dropna().unique().tolist()))

view = df.copy()
if f_changed == "바뀐 사람만":
    view = view[view["vote_v2"].notna() & view["vote_v3"].notna() & (view["vote_v2"] != view["vote_v3"])]
elif f_changed == "유지된 사람만":
    view = view[view["vote_v2"].notna() & view["vote_v3"].notna() & (view["vote_v2"] == view["vote_v3"])]
if f_v2:
    view = view[view["vote_v2"].isin(f_v2)]
if f_v3:
    view = view[view["vote_v3"].isin(f_v3)]
if f_prov:
    view = view[view["province"].isin(f_prov)]
if f_sex:
    view = view[view["sex"].isin(f_sex)]

def party_bar(col, counts: pd.Series) -> None:
    if len(counts) == 0:
        return
    total = int(counts.sum())
    max_v = int(counts.max())
    rows = []
    for party, n in counts.items():
        color = PARTY_COLORS.get(party, "#666")
        pct = n / total * 100
        bar_w = n / max_v * 100
        rows.append(
            f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:0.85em'>"
            f"<div style='width:88px;color:#000;font-weight:700;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis'>{party}</div>"
            f"<div style='flex:1;background:#fff;border:1px solid #ddd;border-radius:3px;height:18px;position:relative'>"
            f"<div style='background:{color};width:{bar_w:.1f}%;height:100%;border-radius:3px'></div>"
            f"</div>"
            f"<div style='width:64px;color:#333;font-variant-numeric:tabular-nums'>{n} ({pct:.0f}%)</div>"
            f"</div>"
        )
    col.markdown("".join(rows), unsafe_allow_html=True)


st.markdown("**분포**")
VERSION_NOTES = {
    "v2": "프롬프트에 지역·성·연령 편향 단서 포함",
    "v3": "지역·성·연령 편향 단서 제거",
    "v4": "편향 단서 제거 + 정치적 이해관계 투입",
    "v5": "v4 + 정당 정책 풍부화 + 정당↔이해관계 매칭 강제, 무당층 억제",
    "v6": "v5 + 권력 시그널 뒤집기 (가상 대통령 장재현·국민의힘 여당 160석) — major 착시",
}
DIST_DATA = {
    "v2": ("openai/gpt-5.4-mini", both23["vote_v2"].value_counts()),
    "v3": ("openai/gpt-5.4-mini", both23["vote_v3"].value_counts()),
    "v4_gpt": ("openai/gpt-5.4-mini", df["vote_v4_gpt"].dropna().value_counts()),
    "v4_haiku": ("anthropic/claude-haiku-4-5-20251001", df["vote_v4_haiku"].dropna().value_counts()),
    "v5_gpt": ("openai/gpt-5.4-mini", df["vote_v5_gpt"].dropna().value_counts()),
    "v5_haiku": ("anthropic/claude-haiku-4-5-20251001", df["vote_v5_haiku"].dropna().value_counts()),
    "v6_gpt": ("openai/gpt-5.4-mini", df["vote_v6_gpt"].dropna().value_counts()),
    "v6_haiku": ("anthropic/claude-haiku-4-5-20251001", df["vote_v6_haiku"].dropna().value_counts()),
}
active_versions = [(k, v) for k, v, _ in ALL_VERSIONS if SHOW[k]]
if not active_versions:
    st.warning("적어도 하나의 버전을 켜주세요.")
    st.stop()
dist_cols = st.columns(len(active_versions))
series_list = [
    (dist_cols[i], ver, DIST_DATA[key][0], DIST_DATA[key][1])
    for i, (key, ver) in enumerate(active_versions)
]
for col, ver, model, s in series_list:
    tag = "<span class='v6-fictional-tag'>가상 시나리오</span>  \n" if ver == "v6" else ""
    col.markdown(
        f"{tag}**{ver}** · `{model}`  \n"
        f"<span style='font-size:0.78em;color:#888'>{VERSION_NOTES[ver]}</span>",
        unsafe_allow_html=True,
    )
    col.dataframe(s, use_container_width=True)
    party_bar(col, s)

st.write(f"**표시: {len(view)}명**")
st.divider()

# ── 페르소나별 6열 비교 ──
def persona_md(r) -> str:
    return (
        f"<code style='color:#999;font-size:0.78em;background:transparent;padding:0'>"
        f"{r['persona_uuid']}</code>  \n"
        f"**{r['sex']} {int(r['age'])}세** · {r['province']} {r['district']}  \n"
        f"{r['marital_status']} · {r['family_type']} · {r['housing_type']}  \n"
        f"{r['education_level']}"
        f"{(' (' + str(r['bachelors_field']) + ')') if pd.notna(r['bachelors_field']) and r['bachelors_field'] != '해당없음' else ''}"
        f" · {r['occupation']}  \n\n"
        f"{r['persona_summary']}"
    )

def vote_block(vote, reason, label, fictional: bool = False) -> str:
    wrapper_open = "<div class='v6-cell'>" if fictional else ""
    wrapper_close = "</div>" if fictional else ""
    tag = "<span class='v6-fictional-tag'>가상 시나리오</span>" if fictional else ""
    if pd.isna(vote):
        return (
            f"{wrapper_open}{tag}"
            f"<div style='font-size:0.85em;color:#888;margin-bottom:6px'>{label}</div>"
            f"<div style='color:#aaa'><em>(없음)</em></div>"
            f"{wrapper_close}"
        )
    badge = party_badge(vote)
    return (
        f"{wrapper_open}{tag}"
        f"<div style='font-size:0.85em;color:#888;margin-bottom:6px'>{label}</div>"
        f"<div style='margin-bottom:8px'>{badge}</div>"
        f"<div style='font-size:0.92em;line-height:1.55'>{reason}</div>"
        f"{wrapper_close}"
    )

# 컬럼 비율: 페르소나·이해관계 좀 더 넓게, 응답 컬럼은 활성 버전 수만큼
COL_RATIO = [2.8, 2.8] + [2] * len(active_versions)
VERSION_HEADERS = {
    "v2": ("v2", "openai/gpt-5.4-mini", VERSION_NOTES["v2"]),
    "v3": ("v3", "openai/gpt-5.4-mini", VERSION_NOTES["v3"]),
    "v4_gpt": ("v4", "openai/gpt-5.4-mini", VERSION_NOTES["v4"]),
    "v4_haiku": ("v4", "anthropic/claude-haiku-4-5-20251001", VERSION_NOTES["v4"]),
    "v5_gpt": ("v5", "openai/gpt-5.4-mini", VERSION_NOTES["v5"]),
    "v5_haiku": ("v5", "anthropic/claude-haiku-4-5-20251001", VERSION_NOTES["v5"]),
    "v6_gpt": ("v6", "openai/gpt-5.4-mini", VERSION_NOTES["v6"]),
    "v6_haiku": ("v6", "anthropic/claude-haiku-4-5-20251001", VERSION_NOTES["v6"]),
}
PROMPT_VERSION = {
    "v2": "v2", "v3": "v3",
    "v4_gpt": "v4", "v4_haiku": "v4",
    "v5_gpt": "v5", "v5_haiku": "v5",
    "v6_gpt": "v6", "v6_haiku": "v6",
}

# 헤더 행 (sticky) — :has() 로 마커가 든 horizontal block 만 잡고,
# main view 의 overflow 를 visible 로 풀어줘서 viewport 기준 sticky 가 작동하게
st.markdown(
    """
    <style>
    div[class*="st-key-sticky_header_box"] {
      background: #f7f7f9;
      border-top: 2px solid #888;
      border-bottom: 2px solid #888;
      padding: 8px 0;
    }
    .v6-cell {
      border: 2px dashed #d4a017;
      background: #fffbe6;
      padding: 8px 10px;
      border-radius: 6px;
    }
    .v6-fictional-tag {
      display: inline-block;
      background: #d4a017;
      color: white;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 0.72em;
      margin-bottom: 4px;
      font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
def render_header(suffix: str = "") -> None:
    box = st.container(key=f"sticky_header_box{suffix}")
    hdr = box.columns(COL_RATIO)
    hdr[0].markdown("**페르소나**")
    hdr[1].markdown(
        "**정치적 이해관계**  \n"
        "<span style='font-size:0.82em;color:#888'>v4·v5·v6 컨텍스트에 투입 · 최대 10회 프로파일 관심사 위주의 에이전트 자율 검색 기반 1인칭 정리</span>",
        unsafe_allow_html=True,
    )
    for i, (key, _) in enumerate(active_versions):
        ver, model, note = VERSION_HEADERS[key]
        tag = "<span class='v6-fictional-tag'>가상 시나리오</span>  \n" if ver == "v6" else ""
        hdr[2 + i].markdown(
            f"{tag}**{ver}** · `{model}`  \n"
            f"<span style='font-size:0.82em;color:#888'>{note}</span>",
            unsafe_allow_html=True,
        )
        if hdr[2 + i].button("📄 프롬프트", key=f"hdr_{key}{suffix}"):
            show_prompt(PROMPT_VERSION[key])


render_header()
st.divider()

@st.fragment
def render_row(r: pd.Series) -> None:
    cols = st.columns(COL_RATIO)
    with cols[0]:
        st.markdown(persona_md(r), unsafe_allow_html=True)
    with cols[1]:
        if pd.notna(r["political_interest"]):
            st.markdown(f"<div style='font-size:0.92em;line-height:1.55'>{r['political_interest']}</div>", unsafe_allow_html=True)
        else:
            st.markdown("_(없음)_")
    for i, (key, _) in enumerate(active_versions):
        ver, model, _note = VERSION_HEADERS[key]
        with cols[2 + i]:
            st.markdown(
                vote_block(
                    r[f"vote_{key}"], r[f"reason_{key}"],
                    f"{ver} · {model}",
                    fictional=(ver == "v6"),
                ),
                unsafe_allow_html=True,
            )
    st.divider()


REPEAT_HEADER_EVERY = 10
for i, (_, r) in enumerate(view.iterrows()):
    if i > 0 and i % REPEAT_HEADER_EVERY == 0:
        render_header(suffix=f"_rep{i}")
        st.divider()
    render_row(r)
