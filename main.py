import io
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# =============================
# Page config + Korean font
# =============================
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


# =============================
# Constants
# =============================
SCHOOLS: List[str] = ["송도고", "하늘고", "아라고", "동산고"]

# 이번 요청 기준 EC 조건 (변경 반영)
SCHOOL_EC_TARGET: Dict[str, float] = {
    "동산고": 1.0,
    "송도고": 2.0,
    "하늘고": 4.0,  # (최적) 가설/기대치로 표기
    "아라고": 8.0,
}

# 생육 시트별 개체수(요약 표에 사용)
SCHOOL_N_EXPECTED: Dict[str, int] = {"동산고": 58, "송도고": 29, "아라고": 106, "하늘고": 45}

SCHOOL_COLOR: Dict[str, str] = {
    "동산고": "#1f77b4",
    "송도고": "#ff7f0e",
    "하늘고": "#2ca02c",
    "아라고": "#d62728",
}

ENV_REQUIRED_COLS = ["time", "temperature", "humidity", "ph", "ec"]
GROW_REQUIRED_COLS = ["개체번호", "잎 수(장)", "지상부 길이(mm)", "지하부길이(mm)", "생중량(g)"]


# =============================
# NFC/NFD safe filename match
# =============================
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _same_filename(a: str, b: str) -> bool:
    """
    NFC/NFD 양방향 비교 (필수 요구사항)
    """
    return len({_nfc(a), _nfd(a)}.intersection({_nfc(b), _nfd(b)})) > 0


def find_file_by_exact_names(folder: Path, exact_names: List[str]) -> Optional[Path]:
    """
    - pathlib.Path.iterdir() 사용
    - glob-only 방식 금지 준수
    - 파일명 f-string 조합 금지 (exact_names 리스트로만 비교)
    """
    if not folder.exists() or not folder.is_dir():
        return None

    for p in folder.iterdir():
        if not p.is_file():
            continue
        for name in exact_names:
            if _same_filename(p.name, name):
                return p
    return None


def best_match_sheet_name(sheet_names: List[str], school: str) -> Optional[str]:
    """
    시트명 하드코딩 금지:
    - 엑셀 실제 sheet_names에서 학교명 포함(정규화 포함) 시트를 추정 매칭
    """
    school_vars = {_nfc(school), _nfd(school)}
    scored: List[Tuple[int, str]] = []

    for sh in sheet_names:
        sh_vars = {_nfc(sh), _nfd(sh)}
        hit = any((sv in hv) or (hv in sv) for sv in school_vars for hv in sh_vars)
        if not hit:
            continue
        # 짧고 명확한 시트명 선호
        score = 1000 - len(_nfc(sh))
        scored.append((score, sh))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# =============================
# Data loading (cached)
# =============================
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

    # 핵심 값이 비어있으면 제거
    df = df.dropna(subset=["temperature", "humidity", "ph", "ec"])
    return df


@st.cache_data(show_spinner=False)
def load_growth_xlsx_all_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    # 시트명 하드코딩 금지: sheet_name=None로 전체 로딩
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


def df_to_xlsx_buffer(df: pd.DataFrame, sheet_name: str = "data") -> io.BytesIO:
    """
    XLSX 다운로드: BytesIO + ExcelWriter(openpyxl) (필수 요구사항)
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    buffer.seek(0)
    return buffer


# =============================
# Preprocessing / filtering
# =============================
def iqr_filter(df: pd.DataFrame, cols: List[str], k: float = 1.5) -> pd.DataFrame:
    """
    극단값 제외(IQR 기반)
    - 환경데이터: temp/humidity/ph/ec 등
    """
    out = df.copy()
    mask = pd.Series(True, index=out.index)
    for c in cols:
        if c not in out.columns:
            continue
        q1 = out[c].quantile(0.25)
        q3 = out[c].quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        low = q1 - k * iqr
        high = q3 + k * iqr
        mask = mask & out[c].between(low, high, inclusive="both")
    return out.loc[mask].copy()


def apply_env_filters(
    df: pd.DataFrame,
    use_ph_range: bool,
    ph_low: float,
    ph_high: float,
    drop_extremes: bool,
    iqr_k: float,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    반환:
      filtered_df, report(dict: before/after counts, excluded counts)
    """
    before = len(df)
    out = df.copy()

    excluded_ph = 0
    if use_ph_range:
        mask_ph = out["ph"].between(ph_low, ph_high, inclusive="both")
        excluded_ph = int((~mask_ph).sum())
        out = out.loc[mask_ph].copy()

    excluded_iqr = 0
    if drop_extremes:
        before_iqr = len(out)
        out = iqr_filter(out, cols=["temperature", "humidity", "ph", "ec"], k=iqr_k)
        excluded_iqr = before_iqr - len(out)

    after = len(out)
    return out, {"before": before, "after": after, "excluded_ph": excluded_ph, "excluded_iqr": excluded_iqr}


def apply_growth_outlier_filter(df: pd.DataFrame, drop_extremes: bool, iqr_k: float) -> Tuple[pd.DataFrame, int]:
    """
    생육 데이터는 pH 필터가 없으므로, 선택적으로 생중량 기반 IQR 제거만 제공.
    """
    if not drop_extremes:
        return df.copy(), 0
    before = len(df)
    out = iqr_filter(df, cols=["생중량(g)"], k=iqr_k)
    return out, before - len(out)


# =============================
# Summaries
# =============================
def env_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "평균": float(df.mean()),
        "최소": float(df.min()),
        "최대": float(df.max()),
    }


def make_env_summary(env_by_school: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in env_by_school.items():
        rows.append(
            {
                "학교": s,
                "측정 시작": df["time"].min(),
                "측정 종료": df["time"].max(),
                "데이터 개수": int(df.shape[0]),
                "결측치(필수열 기준)": 0,  # 로딩 단계에서 필수열 NaN 제거 후라 0으로 정의
                "평균 온도": df["temperature"].mean(),
                "평균 습도": df["humidity"].mean(),
                "평균 pH": df["ph"].mean(),
                "실측 평균 EC": df["ec"].mean(),
                "실측 EC 변동(표준편차)": df["ec"].std(),
            }
        )
    out = pd.DataFrame(rows)
    out["학교"] = pd.Categorical(out["학교"], categories=SCHOOLS, ordered=True)
    out = out.sort_values("학교")
    return out


def make_env_minmax_table(env_by_school: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in env_by_school.items():
        rows.append(
            {
                "학교": s,
                "온도 평균": df["temperature"].mean(),
                "온도 최소": df["temperature"].min(),
                "온도 최대": df["temperature"].max(),
                "습도 평균": df["humidity"].mean(),
                "습도 최소": df["humidity"].min(),
                "습도 최대": df["humidity"].max(),
                "pH 평균": df["ph"].mean(),
                "pH 최소": df["ph"].min(),
                "pH 최대": df["ph"].max(),
                "EC 평균": df["ec"].mean(),
                "EC 최소": df["ec"].min(),
                "EC 최대": df["ec"].max(),
            }
        )
    out = pd.DataFrame(rows)
    out["학교"] = pd.Categorical(out["학교"], categories=SCHOOLS, ordered=True)
    out = out.sort_values("학교")
    return out


def make_growth_summary(growth_by_school: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, df in growth_by_school.items():
        rows.append(
            {
                "학교": s,
                "EC 조건(설정)": SCHOOL_EC_TARGET.get(s, None),
                "개체수(n)": int(df.shape[0]),
                "평균 생중량(g)": df["생중량(g)"].mean(),
                "평균 잎 수(장)": df["잎 수(장)"].mean(),
                "평균 지상부 길이(mm)": df["지상부 길이(mm)"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out["학교"] = pd.Categorical(out["학교"], categories=SCHOOLS, ordered=True)
    out = out.sort_values("학교")
    return out


def best_ec_by_weight(growth_summary: pd.DataFrame) -> Tuple[float, str, float]:
    idx = growth_summary["평균 생중량(g)"].idxmax()
    row = growth_summary.loc[idx]
    return float(row["EC 조건(설정)"]), str(row["학교"]), float(row["평균 생중량(g)"])


# =============================
# File structure + load all
# =============================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 정확 파일명(요구 구조)
CSV_NAME_CANDIDATES: Dict[str, List[str]] = {
    "송도고": ["송도고_환경데이터.csv"],
    "하늘고": ["하늘고_환경데이터.csv"],
    "아라고": ["아라고_환경데이터.csv"],
    "동산고": ["동산고_환경데이터.csv"],
}
XLSX_NAME_CANDIDATES = ["4개교_생육결과데이터.xlsx"]


@st.cache_data(show_spinner=False)
def load_all_raw() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Path, Dict[str, Path]]:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"data 폴더를 찾지 못했습니다: {DATA_DIR}")

    env_by_school: Dict[str, pd.DataFrame] = {}
    csv_paths: Dict[str, Path] = {}

    for school in SCHOOLS:
        p = find_file_by_exact_names(DATA_DIR, CSV_NAME_CANDIDATES[school])
        if p is None:
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


def build_filtered_views(
    env_raw: Dict[str, pd.DataFrame],
    growth_raw: Dict[str, pd.DataFrame],
    use_ph_range: bool,
    ph_low: float,
    ph_high: float,
    drop_env_extremes: bool,
    drop_growth_extremes: bool,
    iqr_k: float,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    반환:
      env_filtered_by_school, growth_filtered_by_school, filter_report_df
    """
    env_filtered: Dict[str, pd.DataFrame] = {}
    growth_filtered: Dict[str, pd.DataFrame] = {}
    report_rows = []

    for s in SCHOOLS:
        df_env_f, rep = apply_env_filters(
            env_raw[s],
            use_ph_range=use_ph_range,
            ph_low=ph_low,
            ph_high=ph_high,
            drop_extremes=drop_env_extremes,
            iqr_k=iqr_k,
        )
        env_filtered[s] = df_env_f

        df_g_f, excluded_g = apply_growth_outlier_filter(
            growth_raw[s],
            drop_extremes=drop_growth_extremes,
            iqr_k=iqr_k,
        )
        growth_filtered[s] = df_g_f

        report_rows.append(
            {
                "학교": s,
                "환경 데이터(전)": rep["before"],
                "환경 데이터(후)": rep["after"],
                "환경 제외(pH범위)": rep["excluded_ph"],
                "환경 제외(IQR)": rep["excluded_iqr"],
                "생육 데이터(전)": int(growth_raw[s].shape[0]),
                "생육 데이터(후)": int(df_g_f.shape[0]),
                "생육 제외(IQR, 생중량)": int(excluded_g),
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_df["학교"] = pd.Categorical(report_df["학교"], categories=SCHOOLS, ordered=True)
    report_df = report_df.sort_values("학교")
    return env_filtered, growth_filtered, report_df


# =============================
# Sidebar controls
# =============================
st.title("🌱 극지식물 최적 EC 농도 연구")

with st.sidebar:
    st.header("설정")

    school_option = st.selectbox("학교 선택", ["전체"] + SCHOOLS, index=0)

    st.markdown("---")
    st.subheader("전처리(신뢰도 점검)")

    use_ph_range = st.checkbox("pH 5~7 범위만 사용", value=True)
    ph_low, ph_high = 5.0, 7.0

    drop_env_extremes = st.checkbox("환경 데이터 극단값 제외(IQR)", value=False)
    drop_growth_extremes = st.checkbox("생육 데이터 극단값 제외(생중량 IQR)", value=False)
    iqr_k = st.slider("IQR 배수(k)", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    st.caption("해석 기준: 측정 주기 차이가 있으므로 동일 시간 간격 비교가 아니라 학교별 요약 통계 중심으로 비교합니다.")


# =============================
# Load data
# =============================
try:
    with st.spinner("데이터를 불러오는 중입니다..."):
        env_raw_by_school, growth_raw_by_school, xlsx_path, csv_paths = load_all_raw()

        env_f_by_school, growth_f_by_school, filter_report = build_filtered_views(
            env_raw=env_raw_by_school,
            growth_raw=growth_raw_by_school,
            use_ph_range=use_ph_range,
            ph_low=ph_low,
            ph_high=ph_high,
            drop_env_extremes=drop_env_extremes,
            drop_growth_extremes=drop_growth_extremes,
            iqr_k=iqr_k,
        )
except Exception as e:
    st.error(f"데이터 로딩 중 오류가 발생했습니다.\n\n- 원인: {e}")
    st.stop()

selected_schools = SCHOOLS if school_option == "전체" else [school_option]

# 요약 테이블(필터 적용본)
env_summary = make_env_summary(env_f_by_school)
env_minmax = make_env_minmax_table(env_f_by_school)
growth_summary = make_growth_summary(growth_f_by_school)

best_ec_data, best_school_data, best_weight_data = best_ec_by_weight(growth_summary)

# 기대 최적(가설) 표기: 하늘고 EC 4.0
expected_best_school = "하늘고"
expected_best_ec = SCHOOL_EC_TARGET.get(expected_best_school, None)


# =============================
# Tabs
# =============================
tab1, tab2, tab3 = st.tabs(
    [
        "📖데이터 개요 및 신뢰도(전처리)",
        "🌡️ 환경–생육 통합 분석(핵심 화면)",
        "📊 EC별 생육 비교 및 다운로드(결과 공유)",
    ]
)


# =============================
# Tab 1: Data overview & reliability
# =============================
with tab1:
    st.subheader("데이터 출처/구성 요약")
    st.write(
        """
- 환경 데이터: CSV 4개(학교별), 컬럼: time, temperature, humidity, ph, ec
- 생육 결과: XLSX 1개(4개 시트), 컬럼: 개체번호, 잎 수(장), 지상부 길이(mm), 지하부길이(mm), 생중량(g)
"""
    )

    st.subheader("학교별 측정 기간 / 데이터 개수 / 전처리 영향")
    st.dataframe(env_summary[["학교", "측정 시작", "측정 종료", "데이터 개수", "평균 온도", "평균 습도", "평균 pH", "실측 평균 EC", "실측 EC 변동(표준편차)"]], use_container_width=True)

    st.subheader("전처리(신뢰도) 제외 기준 명시")
    cols = st.columns(3)
    cols[0].metric("pH 사용 범위", "5.0 ~ 7.0" if use_ph_range else "미적용")
    cols[1].metric("환경 극단값 제외(IQR)", "적용" if drop_env_extremes else "미적용")
    cols[2].metric("생육 극단값 제외(IQR)", "적용" if drop_growth_extremes else "미적용")
    st.dataframe(filter_report, use_container_width=True)

    st.subheader("EC·온도·습도 요약 통계(평균/최소/최대)")
    st.dataframe(env_minmax, use_container_width=True)

    st.subheader("설정 EC vs 실측 평균 EC (관리 안정성 확인)")
    comp = env_summary[["학교", "실측 평균 EC", "실측 EC 변동(표준편차)"]].copy()
    comp["설정 EC"] = comp["학교"].astype(str).map(SCHOOL_EC_TARGET)
    comp["차이(실측-설정)"] = comp["실측 평균 EC"] - comp["설정 EC"]
    st.dataframe(comp[["학교", "설정 EC", "실측 평균 EC", "차이(실측-설정)", "실측 EC 변동(표준편차)"]], use_container_width=True)

    st.caption("해석 포인트: 설정 EC와 실측 평균 EC의 차이가 크거나, 실측 EC 변동이 큰 경우 ‘관리 안정성’ 리스크로 해석합니다.")


# =============================
# Tab 2: Integrated analysis
# =============================
with tab2:
    st.subheader("학교별 환경 평균 비교 (요약 통계 중심)")

    env_view = env_summary[["학교", "평균 온도", "평균 습도", "평균 pH", "실측 평균 EC"]].copy()
    env_view["목표 EC(설정)"] = env_view["학교"].astype(str).map(SCHOOL_EC_TARGET)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 온도", "평균 습도", "평균 pH", "목표 EC vs 실측 EC(평균)"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(
            x=env_view["학교"],
            y=env_view["평균 온도"],
            name="평균 온도",
            marker_color=[SCHOOL_COLOR[str(s)] for s in env_view["학교"]],
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=env_view["학교"],
            y=env_view["평균 습도"],
            name="평균 습도",
            marker_color=[SCHOOL_COLOR[str(s)] for s in env_view["학교"]],
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=env_view["학교"],
            y=env_view["평균 pH"],
            name="평균 pH",
            marker_color=[SCHOOL_COLOR[str(s)] for s in env_view["학교"]],
        ),
        row=2,
        col=1,
    )

    fig.add_trace(go.Bar(x=env_view["학교"], y=env_view["목표 EC(설정)"], name="목표 EC", opacity=0.7), row=2, col=2)
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
    st.subheader("선택한 학교 시계열(온도/습도/EC)")

    if school_option == "전체":
        st.info("학교별 측정 주기 차이가 있으므로, 전체 선택 시 학교별 시계열을 각각 표시합니다.")
        for s in selected_schools:
            df = env_f_by_school[s]
            st.markdown(f"**{s}**")
            c1, c2, c3 = st.columns(3)

            fig_t = px.line(df, x="time", y="temperature", title="온도 변화")
            fig_t.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=320)
            c1.plotly_chart(fig_t, use_container_width=True)

            fig_h = px.line(df, x="time", y="humidity", title="습도 변화")
            fig_h.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=320)
            c2.plotly_chart(fig_h, use_container_width=True)

            fig_ec = px.line(df, x="time", y="ec", title="EC 변화(목표선 포함)")
            fig_ec.add_hline(y=SCHOOL_EC_TARGET[s], line_dash="dash")
            fig_ec.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=320)
            c3.plotly_chart(fig_ec, use_container_width=True)
    else:
        s = school_option
        df = env_f_by_school[s]
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

    st.divider()
    st.subheader("EC–온도–생중량 통합 산점도(생중량 중심)")

    # 학교별 환경 평균 + 생육 평균을 병합(요약 통계 기반)
    comb = env_summary[["학교", "평균 온도", "실측 평균 EC"]].merge(
        growth_summary[["학교", "평균 생중량(g)"]], on="학교", how="inner"
    )
    comb["설정 EC"] = comb["학교"].astype(str).map(SCHOOL_EC_TARGET)

    fig_combo = px.scatter(
        comb,
        x="실측 평균 EC",
        y="평균 온도",
        size="평균 생중량(g)",
        color="평균 생중량(g)",
        hover_name="학교",
        title="학교별 (실측 평균 EC, 평균 온도) 평면에서 생중량(크기/색) 표현",
    )
    fig_combo.update_layout(font=dict(family=PLOTLY_FONT_FAMILY), height=520)
    st.plotly_chart(fig_combo, use_container_width=True)

    st.subheader("상관관계 해석(조합 효과 중심)")
    # 데이터 기반으로 문장 생성(과도한 단정 방지)
    best_by_data_txt = f"데이터 기준 최대 평균 생중량은 {best_school_data}(설정 EC {best_ec_data:.1f})에서 관찰되었습니다."
    expected_txt = f"가설(기대 최적)은 {expected_best_school}(설정 EC {expected_best_ec:.1f})로 설정되어 있습니다."

    # 간단 리스크 문장(실측 EC 안정성)
    comp2 = env_summary.copy()
    comp2["설정 EC"] = comp2["학교"].astype(str).map(SCHOOL_EC_TARGET)
    comp2["|실측-설정|"] = (comp2["실측 평균 EC"] - comp2["설정 EC"]).abs()
    worst_school = str(comp2.sort_values("|실측-설정|", ascending=False).iloc[0]["학교"])
    worst_gap = float(comp2.sort_values("|실측-설정|", ascending=False).iloc[0]["|실측-설정|"])

    st.write(
        f"""
- {expected_txt}
- {best_by_data_txt}
- 조합 효과 해석은 “EC 단독”이 아니라 **(실측 EC × 평균 온도)**에서 생중량이 크게 나타나는 지점을 중심으로 판단합니다.
- 관리 안정성 관점에서, 설정 EC 대비 실측 평균 EC 편차가 가장 큰 학교는 **{worst_school} (편차 |ΔEC|≈{worst_gap:.2f})**로 확인됩니다.
"""
    )

    st.subheader("필요 시: 히트맵(온도 구간 × EC 구간) 요약")
    # 학교별 점(4개)만 있으므로 "커버리지+평균 생중량" 형태의 요약 히트맵 제공
    # 구간은 과도하게 세분화하지 않음(데이터 희소성 고려)
    temp_bins = pd.IntervalIndex.from_tuples([(0, 10), (10, 15), (15, 20), (20, 25), (25, 35)], closed="left")
    ec_bins = pd.IntervalIndex.from_tuples([(0, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 9.0)], closed="left")

    hm = comb.copy()
    hm["온도 구간"] = pd.cut(hm["평균 온도"], bins=temp_bins)
    hm["EC 구간"] = pd.cut(hm["실측 평균 EC"], bins=ec_bins)

    pivot = hm.pivot_table(index="온도 구간", columns="EC 구간", values="평균 생중량(g)", aggfunc="mean")

    # Plotly heatmap
    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[str(i) for i in pivot.index],
            hoverongaps=False,
        )
    )
    fig_hm.update_layout(
        title="(희소 데이터) 온도 구간 × 실측 EC 구간별 평균 생중량 요약",
        font=dict(family=PLOTLY_FONT_FAMILY),
        height=420,
        xaxis_title="실측 EC 구간",
        yaxis_title="평균 온도 구간",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("주의: 학교 단위(4점) 요약이므로 빈 구간이 많습니다. ‘비어 있는 온도–EC 영역’을 확인하는 목적에 적합합니다.")

    with st.expander("환경 데이터 원본/필터 결과 테이블 + CSV 다운로드"):
        mode = st.radio("표시 데이터", ["필터 적용본", "원본(필터 미적용)"], horizontal=True)
        env_view_dict = env_f_by_school if mode == "필터 적용본" else env_raw_by_school

        if school_option == "전체":
            for s in SCHOOLS:
                st.markdown(f"**{s}**  (파일: {csv_paths[s].name})")
                st.dataframe(env_view_dict[s], use_container_width=True)
                st.download_button(
                    label=f"{s} CSV 다운로드({mode})",
                    data=df_to_csv_bytes(env_view_dict[s]),
                    file_name=f"{s}_환경데이터_{'필터' if mode=='필터 적용본' else '원본'}.csv",
                    mime="text/csv",
                )
        else:
            s = school_option
            st.markdown(f"**{s}**  (파일: {csv_paths[s].name})")
            st.dataframe(env_view_dict[s], use_container_width=True)
            st.download_button(
                label=f"{s} CSV 다운로드({mode})",
                data=df_to_csv_bytes(env_view_dict[s]),
                file_name=f"{s}_환경데이터_{'필터' if mode=='필터 적용본' else '원본'}.csv",
                mime="text/csv",
            )


# =============================
# Tab 3: Growth comparison & downloads
# =============================
with tab3:
    st.subheader("🥇 핵심 결과(생중량 최우선): EC 조건별 평균 생중량")

    # 결과 카드: 데이터 기반 최적 vs 기대 최적(가설)
    c1, c2, c3 = st.columns(3)
    c1.metric("데이터 기반 최적(평균 생중량 최대)", f"EC {best_ec_data:.1f}", f"{best_school_data} / {best_weight_data:.3f} g")

    if expected_best_ec is not None:
        # 기대 최적(하늘고) 결과도 함께 표시(단정 방지)
        exp_row = growth_summary[growth_summary["학교"].astype(str) == expected_best_school]
        if not exp_row.empty:
            exp_w = float(exp_row.iloc[0]["평균 생중량(g)"])
            c2.metric("가설(기대 최적)", f"{expected_best_school} / EC {expected_best_ec:.1f}", f"{exp_w:.3f} g")
        else:
            c2.metric("가설(기대 최적)", f"{expected_best_school} / EC {expected_best_ec:.1f}", "생육 시트 매칭 필요")
    else:
        c2.metric("가설(기대 최적)", "미지정", "")

    c3.metric("전처리 영향(요약)", "pH 필터" if use_ph_range else "미적용", "IQR 제외 적용" if (drop_env_extremes or drop_growth_extremes) else "IQR 미적용")

    # EC 조건 순서(1.0,2.0,4.0,8.0)로 정렬
    gs = growth_summary.copy()
    gs["EC 조건(설정)"] = pd.to_numeric(gs["EC 조건(설정)"], errors="coerce")
    gs = gs.sort_values("EC 조건(설정)")

    # 막대: 평균 생중량(가장 중요) - 하늘고(기대 최적) 강조를 위해 테두리/주석(색은 학교색)
    bar_colors = [SCHOOL_COLOR[str(s)] for s in gs["학교"].astype(str)]

    fig_w = go.Figure()
    fig_w.add_trace(
        go.Bar(
            x=gs["EC 조건(설정)"].astype(str),
            y=gs["평균 생중량(g)"],
            text=gs["학교"].astype(str),
            marker_color=bar_colors,
        )
    )
    fig_w.update_layout(
        title="EC 조건별 평균 생중량(학교별 조건)",
        xaxis_title="EC 조건(설정)",
        yaxis_title="평균 생중량(g)",
        font=dict(family=PLOTLY_FONT_FAMILY),
        height=420,
    )

    # 기대 최적(하늘고) 주석 표시
    try:
        idx_h = gs.index[gs["학교"].astype(str) == expected_best_school][0]
        x_h = str(gs.loc[idx_h, "EC 조건(설정)"])
        y_h = float(gs.loc[idx_h, "평균 생중량(g)"])
        fig_w.add_annotation(
            x=x_h,
            y=y_h,
            text="기대 최적(가설)",
            showarrow=True,
            arrowhead=2,
            yshift=15,
        )
    except Exception:
        pass

    st.plotly_chart(fig_w, use_container_width=True)

    st.divider()
    st.subheader("EC별 생육 비교 (2x2)")

    fig2 = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("평균 생중량(g) ⭐", "평균 잎 수(장)", "평균 지상부 길이(mm)", "개체수 비교"),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig2.add_trace(go.Bar(x=gs["EC 조건(설정)"].astype(str), y=gs["평균 생중량(g)"]), row=1, col=1)
    fig2.add_trace(go.Bar(x=gs["EC 조건(설정)"].astype(str), y=gs["평균 잎 수(장)"]), row=1, col=2)
    fig2.add_trace(go.Bar(x=gs["EC 조건(설정)"].astype(str), y=gs["평균 지상부 길이(mm)"]), row=2, col=1)
    fig2.add_trace(go.Bar(x=gs["EC 조건(설정)"].astype(str), y=gs["개체수(n)"]), row=2, col=2)

    fig2.update_layout(
        height=720,
        font=dict(family=PLOTLY_FONT_FAMILY),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("분포(편차·이상치): 학교별 생중량 분포")

    dist_rows = []
    for s in selected_schools:
        df = growth_f_by_school[s].copy()
        df["학교"] = s
        df["EC 조건(설정)"] = SCHOOL_EC_TARGET[s]
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

    st.divider()
    st.subheader("후속 실험 제안(비어 있는 온도–EC 영역 기반)")

    # 히트맵에서 관측된 구간만 표시해 “빈 구간”을 간단 요약
    comb2 = env_summary[["학교", "평균 온도", "실측 평균 EC"]].merge(
        growth_summary[["학교", "평균 생중량(g)"]], on="학교", how="inner"
    )
    comb2["온도 구간"] = pd.cut(comb2["평균 온도"], bins=[0, 10, 15, 20, 25, 35], right=False)
    comb2["EC 구간"] = pd.cut(comb2["실측 평균 EC"], bins=[0, 1.5, 3.0, 5.0, 9.0], right=False)

    observed_bins = comb2[["온도 구간", "EC 구간"]].dropna().drop_duplicates()

    st.write(
        """
- 목표: 현재 데이터에서 상대적으로 비어 있는 (온도 × EC) 구간을 보완하여, **조합 효과(EC 단독이 아닌 EC×온도)**를 더 명확히 확인합니다.
- 권장:
  - EC 세분화: **1.5 ~ 3.0 구간** 추가(중간 EC 영역의 반응 확인)
  - 온도 단계화: 동일 EC에서 **온도 2~3단계**(예: 15–18–21°C 등)로 반복 측정
  - 반복수/기간 확장: 표본 수와 측정 기간을 늘려 변동성(특히 실측 EC 변동)을 안정적으로 추정
"""
    )

    st.caption(f"관측된 (온도구간×EC구간) 조합 수: {len(observed_bins)}개. 비어 있는 구간을 우선 보강하는 방식이 효율적입니다.")

    with st.expander("원자료(필터 적용 결과 포함) 표 제공 및 XLSX 다운로드"):
        mode = st.radio("표시 데이터", ["필터 적용본", "원본(필터 미적용)"], horizontal=True, key="growth_mode")
        g_view = growth_f_by_school if mode == "필터 적용본" else growth_raw_by_school

        if school_option == "전체":
            for s in SCHOOLS:
                st.markdown(f"**{s} (EC {SCHOOL_EC_TARGET[s]:.1f})**")
                st.dataframe(g_view[s], use_container_width=True)

                buf = df_to_xlsx_buffer(g_view[s], sheet_name="생육데이터")
                st.download_button(
                    label=f"{s} 생육 XLSX 다운로드({mode})",
                    data=buf,
                    file_name=f"{s}_생육데이터_{'필터' if mode=='필터 적용본' else '원본'}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            # 원본 엑셀(업로드 파일 그대로) 다운로드
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
            st.dataframe(g_view[s], use_container_width=True)

            buf = df_to_xlsx_buffer(g_view[s], sheet_name="생육데이터")
            st.download_button(
                label=f"{s} 생육 XLSX 다운로드({mode})",
                data=buf,
                file_name=f"{s}_생육데이터_{'필터' if mode=='필터 적용본' else '원본'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
