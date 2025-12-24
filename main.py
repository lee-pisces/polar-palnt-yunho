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
        df
