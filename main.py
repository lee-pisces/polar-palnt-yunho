import io
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# -----------------------------
# Streamlit page + Korean font
# -----------------------------
st.set_page_config(page_title="극지식물 최적 EC 농도 연구", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_FONT_FAMILY = "Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif"


# -----------------------------
# Constants
# -----------------------------
SCHOOLS: List[str] = ["송도고", "하늘고", "아라고", "동산고"]
SCHOOL_EC_TARGET: Dict[str, float] = {"송도고": 1.0, "하늘고": 2.0, "아라고": 4.0, "동산고": 8.0}
SCHOOL_N_EXPECTED: Dict[str, int] = {"동산고": 58, "송도고": 29, "아라고": 106, "하늘고": 45}

SCHOOL_COLOR: Dict[str, str] = {
    "송도고": "#1f77b4",
    "하늘고": "#2ca02c",
    "아라고": "#ff7f0e",
    "동산고": "#d62728",
}

ENV_REQUIRED_COLS = ["time", "temperature", "humidity", "ph", "ec"]
GROW_REQUIRED_COLS = ["개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]


# -----------------------------
# Filename normalization helpers
# -----------------------------
def _norm_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _norm_nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _is_same_text(a: str, b: str) -> bool:
    a_variants = {_norm_nfc(a), _norm_nfd(a)}
    b_variants = {_norm_nfc(b), _norm_nfd(b)}
    return len(a_variants.intersection(b_variants)) > 0


def _norm_for_match(s: str) -> str:
    """
    파일명 변형(공백/언더스코어/하이픈/괄호 등)에도 견고하도록
    비교용 문자열을 단순화.
    """
    x = _norm_nfc(s).lower()
    for ch in [" ", "_", "-", "(", ")", "[", "]", "{", "}", ".", ","]:
        x = x.replace(ch, "")
    return x


def find_file_by_exact_names(folder: Path, exact_names: List[str]) -> Optional[Path]:
    """
    iterdir + NFC/NFD 양방향 비교로 '정확 이름' 매칭
    """
    if not folder.exists() or not folder.is_dir():
        return None

    for p in folder.iterdir():
        if not p.is_file():
            continue
        for name in exact_names:
            if _is_same_text(p.name, name):
                return p
    return None


def find_best_env_csv(folder: Path, school: str) -> Optional[Path]:
    """
    정확 매칭 실패 시 fallback:
    - iterdir()로 data 폴더 내 .csv 파일을 모두 스캔
    - 파일명에 학교명/환경 관련 키워드가 포함된 후보를 점수화하여 최적 1개 선택
    - glob-only 탐색 금지 준수 (glob 사용 안 함)
    """
    if not folder.exists() or not folder.is_dir():
        return None

    school_key = _norm_for_match(school)
    # 환경데이터 파일에서 흔히 등장하는 토큰(변형 포함)
    env_tokens = ["환경", "환경데이터", "환경data", "env", "environment"]
    env_tokens_norm = [_norm_for_match(t) for t in env_tokens]

    candidates: List[Tuple[int, Path]] = []

    for p in folder.iterdir():
        if not p.is_file():
            continue
        # 확장자 검사도 정규화로 처리(대문자 CSV 등)
        if _norm_for_match(p.suffix) != "csv":
            continue

        name_norm = _norm_for_match(p.name)

        score = 0
        # 학교명 포함 여부(가장 중요)
        if school_key and (school_key in name_norm):
            score += 100

        # 환경 토큰 포함 여부
        for tok in env_tokens_norm:
            if tok and (tok in name_norm):
                score += 30
                break

        # 너무 일반적인 csv(학교명도 환경도 없는 경우)는 탈락
        if score == 0:
            continue

        # 파일명이 짧고 명확할수록 가산(과대선정 방지용)
        score += max(0, 50 - len(name_norm))
        candidates.append((score, p))

    if not candidates:
        return None

    # 최고 점수 1개 선택
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def best_match_sheet_name(sheet_names: List[str], school: str) -> Optional[str]:
    """
    시트명 하드코딩 금지:
    - 실제 sheet_names에서 학교명 포함 시트를 정규화 포함 매칭
    """
    school_norm = _norm_for_match(school)

    scored: List[Tuple[int, str]] = []
    for s in sheet_names:
        s_norm = _norm_for_match(s)
        score = 0
        if school_norm and (school_norm in s_norm or s_norm in school_norm):
            score += 100
        # 짧고 명확한 시트명 선호
        score += max(0, 80 - len(s_norm))
        if score > 0:
            scored.append((score, s))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# -----------------------------
# Data loading with caching
# -----------------------------
@st.cache_data(show_spinner=False)
def load_env_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in ENV_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"환경 CSV에 필수 컬럼이 없습니다: {missing}")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")

    for col in ["temperature", "humidity", "ph", "ec"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["temperature", "humidity", "ph", "ec"])
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx_all_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    data = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    cleaned: Dict[str, pd.DataFrame] = {}

    for sheet, df in data.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        if not all(c in df.columns for c in GROW_REQUIRED_COLS):
            continue

        for col in ["잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["생중량(g)"])
        cleaned[sheet] = df

    if not cleaned:
        raise ValueError("엑셀에서 유효한 생육 데이터 시트를 찾지 못했습니다(필수 컬럼 불일치).")

    return cleaned


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "data") -> io.BytesIO:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer


# -----------------------------
# Build file paths robustly
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# "정확 이름" 후보는 유지하되, 실패하면 자동 탐색으로 대체
CSV_NAME_CANDIDATES: Dict[str, List[str]] = {
    "송도고": ["송도고_환경데이터.csv"],
    "하늘고": ["하늘고_환경데이터.csv"],
    "아라고": ["아라고_환경데이터.csv"],
    "동산고": ["동산고_환경데이터.csv"],
}
XLSX_NAME_CANDIDATES = ["4개교_생육결과데이터.xlsx"]


def load_all_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Path, Dict[str, Path]]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"data 폴더를 찾지 못했습니다: {DATA_DIR}")

    env_by_school: Dict[str, pd.DataFrame] = {}
    csv_paths: Dict[str, Path] = {}

    for school in SCHOOLS:
        # 1) 정확 이름 매칭 시도
        p = find_file_by_exact_names(DATA_DIR, CSV_NAME_CANDIDATES[school])
        # 2) 실패 시 fallback: 폴더 내 csv 스캔 후 최적 후보 선택
        if p is None:
            p = find_best_env_csv(DATA_DIR, school)

        if p is None:
            # 디버깅용: data 폴더의 실제 파일명 목록 제공(Cloud에서도 확인 가능)
            existing = [q.name for q in DATA_DIR.iterdir() if q.is_file()]
            raise FileNotFoundError(
                f"환경 데이터 파일을 찾지 못했습니다: {CSV_NAME_CANDIDATES[school]}\n"
                f"- data 폴더 파일 목록: {existing}"
            )

        csv_paths[school] = p
        env_by_school[school] = load_env_csv(p)

    xlsx_path = find_file_by_exact_names(DATA_DIR, XLSX_NAME_CANDIDATES)
    if xlsx_path is None:
        existing = [q.name for q in DATA_DIR.iterdir() if q.is_file()]
        raise FileNotFoundError(
            f"생육 결과 엑셀 파일을 찾지 못했습니다: {XLSX_NAME_CANDIDATES}\n"
            f"- data 폴더 파일 목록: {existing}"
        )

    sheets = load_growth_xlsx_all_sheets(xlsx_path)
    sheet_names = list(sheets.keys())

    growth_by_school: Dict[str, pd.DataFrame] = {}
    for school in SCHOOLS:
        matched = best_match_sheet_name(sheet_names, school)
        if matched is None:
            raise FileNotFoundError(
                f"엑셀 시트 중 '{school}'에 해당하는 시트를 찾지 못했습니다.\n"
                f"- 현재 유효 시트: {sheet_names}"
            )
        growth_by_school[school] = sheets[matched].copy()

    return env_by_school, growth_by_school, xlsx_path, csv_paths


# -----------------------------
# Analysis helpers
# -----------------------------
def env_summary_table(env_by_school: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in env_by_school.items():
        rows.append(
            {
                "학교": s,
                "평균 온도(°C)": df["temperature"].mean(),
                "평균 습도(%)": df["humidity"].mean(),
                "평균 pH": df["ph"].mean(),
                "실측 평균 EC": df["ec"].mean(),
                "목표 EC": SCHOOL_EC_TARGET[s],
            }
        )
    out = pd.DataFrame(rows)
    return out


def growth_summary_table(growth_by_school: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in growth_by_school.items():
        rows.append(
            {
                "학교": s,
                "EC 목표": SCHOOL_EC_TARGET[s],
                "개체수(n)": int(df.shape[0]),
                "평균 생중량(g)": df["생중량(g)"].mean(),
                "평균 잎 수(장)": df["잎 수(장)"].mean(),
                "평균 지상부 길이(mm)": df["지상부 길이(mm)"].mean(),
            }
        )
    return pd.DataFrame(rows)


def best_ec_from_growth(growth_by_school: Dict[str, pd.DataFrame]) -> Tuple[float, str, float]:
    best_school = None
    best_mean = None
    best_ec = None

    for s, df in growth_by_school.items():
        m = float(df["생중량(g)"].mean())
        if best_mean is None or m > best_mean:
            best_mean = m
            best_school = s
            best_ec = SCHOOL_EC_TARGET[s]

    return float(best_ec), str(best_school), float(best_mean)


# -----------------------------
# UI
# -----------------------------
st.title("🌱 극지식물 최적 EC 농도 연구")

with st.sidebar:
    st.header("설정")
    school_option = st.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        env_by_school, growth_by_school, xlsx_path, csv_paths = load_all_data()
except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다.\n\n- 원인: {e}")
    st.stop()

env_summary = env_summary_table(env_by_school)
growth_summary = growth_summary_table(growth_by_school)
best_ec, best_school, best_mean_w = best_ec_from_growth(growth_by_school)

selected_schools = SCHOOLS if school_option == "전체" else [school_option]

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# -----------------------------
# Tab 1
# -----------------------------
with tab1:
    st.subheader("연구 배경 및 목적")
    st.write(
        """
본 연구는 4개 학교에서 서로 다른 목표 EC 조건(1.0, 2.0, 4.0, 8.0)을 적용하여 극지식물의 생육 차이를 비교하고,
환경 데이터(온도/습도/pH/EC)와 생육 결과(생중량/잎 수/길이)를 종합해 최적 EC 농도를 도출하는 것을 목적으로 합니다.
"""
    )

    st.subheader("학교별 EC 조건")
    cond_df = pd.DataFrame(
        [
            {"학교명": s, "EC 목표": SCHOOL_EC_TARGET[s], "개체수": SCHOOL_N_EXPECTED.get(s, None), "색상": SCHOOL_COLOR[s]}
            for s in SCHOOLS
        ]
    )
    st.dataframe(cond_df, use_container_width=True)

    total_n = int(sum(int(growth_by_school[s].shape[0]) for s in SCHOOLS))
    avg_temp_all = float(pd.concat([env_by_school[s][["temperature"]] for s in SCHOOLS])["temperature"].mean())
    avg_hum_all = float(pd.concat([env_by_school[s][["humidity"]] for s in SCHOOLS])["humidity"].mean())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총 개체수", f"{total_n} 개")
    c2.metric("평균 온도", f"{avg_temp_all:.2f} °C")
    c3.metric("평균 습도", f"{avg_hum_all:.2f} %")
    c4.metric("최적 EC(평균 생중량 기준)", f"{best_ec:.1f}  (학교: {best_school})")


# -----------------------------
# Tab 2
# -----------------------------
with tab2:
    st.subheader("학교별 환경 평균 비교")

    env_view = env_summary.copy()
    env_view["학교"] = pd.Categorical(env_view["학교"], categories=SCHOOLS, ordered=True)
    env_view = env_view.sort_values("학교")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC 비교(평균)"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(x=env_view["학교"], y=env_view["평균 온도(°C)"], name="평균 온도",
               marker_color=[SCHOOL_COLOR[str(s)] for s in env_view["학교"]]),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=env_view["학교"], y=env_view["평균 습도(%)"], name="평균 습도",
               marker_color=[SCHOOL_COLOR[str(s)] for s in env_view["학교"]]),
        row=1, col=2
    )

    fig.add_trace(
        go.Bar(x=env_view["학교"], y=env_view["평균 pH"], name="평균 pH",
               marker_color=[SCHOOL_COLOR[str(s)] for s in env_view["학교"]]),
        row=2, col=1
    )

    fig.add_trace(go.Bar(x=env_view["학교"], y=env_view["목표 EC"], name="목표 EC", opacity=0.7), row=2, col=2)
    fig.add_trace(go.Bar(x=env_view["학교"], y=env_view["실측 평균 EC"], name="실측 평균 EC", opacity=0.7), row=2, col=2)

    fig.update_layout(
        barmode="group",
        height=720,
        font=dict(family=PLOTLY_FONT_FAMILY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("선택한 학교 시계열")

    if school_option == "전체":
        st.info("전체 선택 시, 학교별 측정 주기가 다르므로 학교별 시계열을 각각 표시합니다.")
        for s in selected_schools:
            df = env_by_school[s]
            st.markdown(f"**{s}**")
            t1, t2, t3 = st.columns(3)

            fig_t = px.line(df, x="time", y="temperature", title="온도 변화")
            fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=320)
            t1.plotly_chart(fig_t, use_container_width=True)

            fig_h = px.line(df, x="time", y="humidity", title="습도 변화")
            fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=320)
            t2.plotly_chart(fig_h, use_container_width=True)

            fig_ec = px.line(df, x="time", y="ec", title="EC 변화(목표선 포함)")
            fig_ec.add_hline(y=SCHOOL_EC_TARGET[s], line_dash="dash")
            fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=320)
            t3.plotly_chart(fig_ec, use_container_width=True)
    else:
        s = school_option
        df = env_by_school[s]
        c1, c2, c3 = st.columns(3)

        fig_t = px.line(df, x="time", y="temperature", title="온도 변화")
        fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=360)
        c1.plotly_chart(fig_t, use_container_width=True)

        fig_h = px.line(df, x="time", y="humidity", title="습도 변화")
        fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=360)
        c2.plotly_chart(fig_h, use_container_width=True)

        fig_ec = px.line(df, x="time", y="ec", title="EC 변화(목표선 포함)")
        fig_ec.add_hline(y=SCHOOL_EC_TARGET[s], line_dash="dash")
        fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=360)
        c3.plotly_chart(fig_ec, use_container_width=True)

    with st.expander("환경 데이터 원본 테이블 + CSV 다운로드"):
        if school_option == "전체":
            for s in SCHOOLS:
                st.markdown(f"**{s}**  (파일: {csv_paths[s].name})")
                st.dataframe(env_by_school[s], use_container_width=True)
                st.download_button(
                    label=f"{s} CSV 다운로드",
                    data=df_to_csv_bytes(env_by_school[s]),
                    file_name=f"{s}_환경데이터.csv",
                    mime="text/csv",
                )
        else:
            s = school_option
            st.markdown(f"**{s}**  (파일: {csv_paths[s].name})")
            st.dataframe(env_by_school[s], use_container_width=True)
            st.download_button(
                label=f"{s} CSV 다운로드",
                data=df_to_csv_bytes(env_by_school[s]),
                file_name=f"{s}_환경데이터.csv",
                mime="text/csv",
            )


# -----------------------------
# Tab 3
# -----------------------------
with tab3:
    st.subheader("🥇 핵심 결과: EC별 평균 생중량")

    growth_sum = growth_summary.copy()
    growth_sum["학교"] = pd.Categorical(growth_sum["학교"], categories=SCHOOLS, ordered=True)
    growth_sum = growth_sum.sort_values("학교")

    max_idx = growth_sum["평균 생중량(g)"].idxmax()
    max_row = growth_sum.loc[max_idx]
    max_ec = float(max_row["EC 목표"])
    max_school = str(max_row["학교"])
    max_mean = float(max_row["평균 생중량(g)"])

    colA, colB = st.columns([2, 3])
    colA.metric("최대 평균 생중량(데이터 기준)", f"{max_mean:.3f} g", f"EC {max_ec:.1f} ({max_school})")
    colA.metric("최적 EC(도출 결과)", f"{best_ec:.1f}", f"학교: {best_school}")

    if SCHOOL_EC_TARGET["하늘고"] == best_ec:
        colA.success("평균 생중량 기준 최적 EC가 하늘고(EC 2.0) 조건으로 도출되었습니다.")
    else:
        colA.info("평균 생중량 기준 최적 EC는 데이터 결과에 따라 하늘고(EC 2.0)과 다를 수 있습니다.")

    fig_w = go.Figure()
    fig_w.add_trace(
        go.Bar(
            x=growth_sum["EC 목표"].astype(str),
            y=growth_sum["평균 생중량(g)"],
            text=growth_sum["학교"].astype(str),
            marker_color=[SCHOOL_COLOR[str(s)] for s in growth_sum["학교"]],
        )
    )
    fig_w.update_layout(
        title="EC별 평균 생중량(학교별 조건)",
        xaxis_title="EC 목표",
        yaxis_title="평균 생중량(g)",
        font=dict(family=PLOTLY_FONT_FAMILY),
        height=420,
    )
    colB.plotly_chart(fig_w, use_container_width=True)

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수(장)", "평균 지상부 길이(mm)", "개체수 비교"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig2.add_trace(go.Bar(x=growth_sum["EC 목표"].astype(str), y=growth_sum["평균 생중량(g)"], name="평균 생중량"), row=1, col=1)
    fig2.add_trace(go.Bar(x=growth_sum["EC 목표"].astype(str), y=growth_sum["평균 잎 수(장)"], name="평균 잎 수"), row=1, col=2)
    fig2.add_trace(go.Bar(x=growth_sum["EC 목표"].astype(str), y=growth_sum["평균 지상부 길이(mm)"], name="평균 지상부 길이"), row=2, col=1)
    fig2.add_trace(go.Bar(x=growth_sum["EC 목표"].astype(str), y=growth_sum["개체수(n)"], name="개체수"), row=2, col=2)

    fig2.update_layout(
        height=720,
        font=dict(family=PLOTLY_FONT_FAMILY),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("학교별 생중량 분포")

    dist_rows = []
    for s in selected_schools:
        df = growth_by_school[s].copy()
        df["학교"] = s
        df["EC 목표"] = SCHOOL_EC_TARGET[s]
        dist_rows.append(df)
    dist_df = pd.concat(dist_rows, ignore_index=True)

    fig_dist = px.violin(
        dist_df,
        x="학교",
        y="생중량(g)",
        box=True,
        points="all",
        title="학교별 생중량 분포(바이올린+박스)",
    )
    fig_dist.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=520)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.divider()
    st.subheader("상관관계 분석(산점도 2개)")

    c1, c2 = st.columns(2)

    fig_sc1 = px.scatter(dist_df, x="잎 수(장)", y="생중량(g)", color="학교", title="잎 수 vs 생중량")
    fig_sc1.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=480)
    c1.plotly_chart(fig_sc1, use_container_width=True)

    fig_sc2 = px.scatter(dist_df, x="지상부 길이(mm)", y="생중량(g)", color="학교", title="지상부 길이 vs 생중량")
    fig_sc2.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=480)
    c2.plotly_chart(fig_sc2, use_container_width=True)

    with st.expander("학교별 생육 데이터 원본 + XLSX 다운로드"):
        if school_option == "전체":
            for s in SCHOOLS:
                st.markdown(f"**{s} (EC {SCHOOL_EC_TARGET[s]:.1f})**")
                st.dataframe(growth_by_school[s], use_container_width=True)

                buf = df_to_xlsx_bytes(growth_by_school[s], sheet_name="생육데이터")
                st.download_button(
                    label=f"{s} 생육 XLSX 다운로드",
                    data=buf,
                    file_name=f"{s}_생육데이터.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            try:
                raw_bytes = xlsx_path.read_bytes()
                st.download_button(
                    label="원본 4개교 생육결과 엑셀 다운로드",
                    data=raw_bytes,
                    file_name="4개교_생육결과데이터.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                st.error(f"원본 엑셀 다운로드 준비 중 오류: {e}")

        else:
            s = school_option
            st.markdown(f"**{s} (EC {SCHOOL_EC_TARGET[s]:.1f})**")
            st.dataframe(growth_by_school[s], use_container_width=True)

            buf = df_to_xlsx_bytes(growth_by_school[s], sheet_name="생육데이터")
            st.download_button(
                label=f"{s} 생육 XLSX 다운로드",
                data=buf,
                file_name=f"{s}_생육데이터.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
