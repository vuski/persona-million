"""페르소나 가상 투표 실험 대시보드.

실행:
    streamlit run backend/dashboard.py
"""
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh

BASE = Path(__file__).parent
RESULTS_PATH = BASE / "vote_results_all.csv"
VOTER_CONTEXT_PATH = BASE / "context" / "voter_context.md"

# 정당별 색상 — 한국 통상 인식 기준 (대략적)
PARTY_COLORS = {
    "더불어민주당": "#004EA2",
    "국민의힘": "#E61E2B",
    "개혁신당": "#FF7920",
    "조국혁신당": "#06275E",
    "진보당": "#D6001C",
    "자유통일당": "#8E44AD",
    "무당층/기권": "#9E9E9E",
}

st.set_page_config(page_title="페르소나 투표 실험", page_icon="🗳️", layout="wide")

# 자동 새로고침 (3초)
st_autorefresh(interval=3000, key="auto_refresh")

@st.cache_data(ttl=2)
def load_data(mtime: float) -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(RESULTS_PATH, encoding="utf-8-sig")
    # 연령대 버킷
    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 29, 39, 49, 59, 69, 200],
        labels=["20대", "30대", "40대", "50대", "60대", "70+"],
    )
    return df

mtime = RESULTS_PATH.stat().st_mtime if RESULTS_PATH.exists() else 0
df = load_data(mtime)

# ───────────────────────────── HEADER
st.title("🗳️ 페르소나 가상 투표 실험")
st.markdown(
    "<div style='color:#888;font-size:0.9em;margin-bottom:8px'>"
    "합성 페르소나(Nemotron-Personas-Korea) 기반 LLM 투표 시뮬레이션 · "
    "<a href='https://github.com/vuski/persona-million' target='_blank' "
    "style='color:#888;text-decoration:underline'>github.com/vuski/persona-million</a>"
    "</div>",
    unsafe_allow_html=True,
)
st.caption(
    f"데이터 출처: `{RESULTS_PATH.name}` · "
    f"마지막 갱신: {datetime.fromtimestamp(mtime).strftime('%H:%M:%S') if mtime else '—'} · "
    f"3초마다 자동 새로고침"
)

# voter context — LLM 프롬프트에 주입한 정치 상황 컨텍스트
if VOTER_CONTEXT_PATH.exists():
    with st.expander("📋 LLM에 주입한 정치 상황 컨텍스트 (voter_context.md)"):
        st.markdown(VOTER_CONTEXT_PATH.read_text(encoding="utf-8"))

if df.empty:
    st.warning("아직 결과가 없습니다. 노트북에서 셀을 실행해주세요.")
    st.stop()

# ───────────────────────────── KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("총 응답 수", f"{len(df):,}")
c2.metric("모델 수", df["model"].nunique())
c3.metric("평균 응답시간", f"{df['elapsed_sec'].mean():.1f}s")
c4.metric("고유 페르소나", df["persona_uuid"].nunique())

st.divider()

# ───────────────────────────── 섹션 1: 모델별 비교
st.subheader("모델별 비교")

col_a, col_b = st.columns([2, 1])

with col_a:
    # 모델 × 정당 그룹 막대그래프 (비율)
    pivot = (
        df.groupby(["model", "vote"]).size()
        .reset_index(name="count")
    )
    pivot["pct"] = pivot.groupby("model")["count"].transform(lambda x: x / x.sum() * 100)
    fig = px.bar(
        pivot, x="model", y="pct", color="vote",
        color_discrete_map=PARTY_COLORS,
        labels={"pct": "득표율 (%)", "model": "모델", "vote": "정당"},
        title="모델별 정당 득표율",
        barmode="stack",
        text=pivot["pct"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="inside")
    fig.update_layout(height=400, legend_title="")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    # 모델별 평균 응답시간
    timing = df.groupby("model")["elapsed_sec"].agg(["mean", "count"]).reset_index()
    timing.columns = ["model", "평균시간(s)", "응답수"]
    fig2 = px.bar(
        timing, x="model", y="평균시간(s)",
        labels={"model": "모델"},
        title="모델별 평균 응답시간",
        text=timing["평균시간(s)"].round(1),
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(timing, hide_index=True, use_container_width=True)

st.divider()

# ───────────────────────────── 섹션 2: 인구통계 교차분석
st.subheader("인구통계 교차분석")

model_filter = st.multiselect(
    "모델 필터 (다중 선택)",
    options=sorted(df["model"].unique()),
    default=sorted(df["model"].unique()),
)
df_filt = df[df["model"].isin(model_filter)]

if not df_filt.empty:
    col1, col2, col3 = st.columns(3)

    for col, dim, title in [
        (col1, "age_bucket", "연령대 × 정당"),
        (col2, "province",   "지역(시도) × 정당"),
        (col3, "sex",        "성별 × 정당"),
    ]:
        agg = df_filt.groupby([dim, "vote"]).size().reset_index(name="count")
        fig = px.bar(
            agg, x=dim, y="count", color="vote",
            color_discrete_map=PARTY_COLORS,
            labels={"count": "응답 수", dim: ""},
            title=title, barmode="stack",
        )
        fig.update_layout(height=350, legend_title="", showlegend=(col is col3))
        col.plotly_chart(fig, use_container_width=True)
else:
    st.info("선택된 모델 없음")

st.divider()

# ───────────────────────────── 섹션 3: 최근 응답 피드
st.subheader("최근 응답")

# 필터 5종 — 모두 다중 선택, 비어 있으면 = 전체 허용
fcols = st.columns(5)
sex_opts = sorted(df["sex"].dropna().unique())
age_opts = ["20대", "30대", "40대", "50대", "60대", "70+"]
prov_opts = sorted(df["province"].dropna().unique())
occ_opts = sorted(df["occupation"].dropna().unique())
edu_opts = sorted(df["education_level"].dropna().unique())

with fcols[0]:
    f_sex = st.multiselect("성별", sex_opts, default=[], placeholder="전체")
with fcols[1]:
    f_age = st.multiselect("연령대", age_opts, default=[], placeholder="전체")
with fcols[2]:
    f_prov = st.multiselect("시도", prov_opts, default=[], placeholder="전체")
with fcols[3]:
    f_occ = st.multiselect("직업", occ_opts, default=[], placeholder="전체")
with fcols[4]:
    f_edu = st.multiselect("학력", edu_opts, default=[], placeholder="전체")

dfr = df.copy()
if f_sex:  dfr = dfr[dfr["sex"].isin(f_sex)]
if f_age:  dfr = dfr[dfr["age_bucket"].isin(f_age)]
if f_prov: dfr = dfr[dfr["province"].isin(f_prov)]
if f_occ:  dfr = dfr[dfr["occupation"].isin(f_occ)]
if f_edu:  dfr = dfr[dfr["education_level"].isin(f_edu)]

st.caption(f"필터 결과: {len(dfr):,} / {len(df):,}건")

n_recent = st.slider("표시 개수", 5, 50, 10)
recent = dfr.tail(n_recent).iloc[::-1]

if recent.empty:
    st.info("필터 조건에 맞는 응답이 없습니다.")

for _, r in recent.iterrows():
    party_color = PARTY_COLORS.get(r["vote"], "#666")
    with st.container(border=True):
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(
                f"**`{r['model']}`** "
                f"<span style='color:#999;font-weight:300;font-size:0.85em'>persona_uuid</span> "
                f"<code style='color:#999;font-size:0.85em;background:transparent;padding:0'>"
                f"{r['persona_uuid']}</code>",
                unsafe_allow_html=True,
            )
            # 풀 프로파일 — 한 줄당 라벨 + 값
            profile_pairs = [
                ("성별", r["sex"]),
                ("연령", f"{r['age']}세"),
                ("혼인", r["marital_status"]),
                ("가구", r["family_type"]),
                ("주거", r["housing_type"]),
                ("학력", r["education_level"]),
                ("전공", r["bachelors_field"]),
                ("직업", r["occupation"]),
                ("지역", f"{r['province']} {r['district']}"),
            ]
            profile_html = "<div style='font-size:0.85em;line-height:1.6;margin:2px 0'>"
            profile_html += " · ".join(
                f"<span style='color:#999;font-weight:300'>{k}</span> "
                f"<b style='color:inherit'>{v}</b>"
                for k, v in profile_pairs
            )
            profile_html += "</div>"
            st.markdown(profile_html, unsafe_allow_html=True)

            st.markdown(
                f"<div style='color:#888;font-size:0.9em;margin:4px 0 8px 0'>"
                f"{r['persona_summary']}</div>",
                unsafe_allow_html=True,
            )

            # 풀 페르소나 (펼침)
            full_pairs = [
                ("직업적 면모", "professional_persona"),
                ("가족 면모", "family_persona"),
                ("문화적 배경", "cultural_background"),
                ("스포츠", "sports_persona"),
                ("예술", "arts_persona"),
                ("여행", "travel_persona"),
                ("음식", "culinary_persona"),
                ("관심사", "hobbies_and_interests"),
                ("숙련·전문성", "skills_and_expertise"),
                ("목표·포부", "career_goals_and_ambitions"),
            ]
            full_pairs = [(k, r[c]) for k, c in full_pairs if c in r and pd.notna(r[c])]
            if full_pairs:
                with st.expander("페르소나 자세히 보기"):
                    for k, v in full_pairs:
                        st.markdown(
                            f"<div style='margin:6px 0'>"
                            f"<span style='color:#999;font-weight:300;font-size:0.85em'>{k}</span><br/>"
                            f"<span style='font-size:0.9em'>{v}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown(
                f"<span style='background:{party_color};color:white;padding:2px 8px;"
                f"border-radius:4px;font-weight:bold'>{r['vote']}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"_{r['reason']}_")
        with cols[1]:
            st.metric("응답시간", f"{r['elapsed_sec']:.1f}s")
            st.caption(f"`{r['persona_uuid'][:8]}…`")

st.divider()

# ───────────────────────────── 섹션 4: 원본 데이터
with st.expander("원본 데이터 (전체 행 보기)"):
    st.dataframe(
        df[["persona_uuid", "model", "vote", "sex", "age", "province",
            "district", "occupation", "education_level", "reason", "elapsed_sec"]],
        hide_index=True, use_container_width=True, height=400,
    )
