"""필터링된 종목 목록 테이블"""
from __future__ import annotations
import pandas as pd
import streamlit as st
from config import COMPARE_DAYS

_LIST_COLS = [
    ("종목명",              "종목명",          "str"),
    ("market",              "시장",            "str"),
    ("종가",                "종가(원)",         "{:,.0f}"),
    ("등락률",              "등락률(%)",        "{:+.2f}%"),
    ("시가총액_억",         "시총(억)",         "{:,.0f}"),
    ("거래대금_억",         "거래대금(억)",     "{:,.1f}"),
    ("시총대비거래대금",    "시총대비(%)",      "{:.3f}%"),
    ("주식수회전율",        "주식수회전율(%)",  "{:.3f}%"),
    ("5일평균대비거래대금", "5일대비(배)",      "{:.2f}x"),
    ("10일평균대비거래대금","10일대비(배)",     "{:.2f}x"),
    ("20일평균대비거래대금","20일대비(배)",     "{:.2f}x"),
    ("기관순매수_억",       "기관(억)",         "{:+.1f}"),
    ("외국인순매수_억",     "외국인(억)",       "{:+.1f}"),
    ("개인순매수_억",       "개인(억)",         "{:+.1f}"),
]

def render_stock_list(df: pd.DataFrame) -> str | None:
    if df.empty:
        st.warning("조건에 맞는 종목이 없습니다.")
        return None
    st.caption(f"**{len(df)}개** 종목")
    display_df = _format_display(df)
    event = st.dataframe(
        display_df,
        use_container_width=True,
        height=480,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = event.selection.get("rows", [])
    if selected_rows:
        return df.index[selected_rows[0]]
    return None

def _format_display(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for internal, display, fmt in _LIST_COLS:
        if internal not in df.columns:
            continue
        col = df[internal]
        if fmt == "str":
            rows[display] = col.fillna("")
        else:
            rows[display] = col.apply(lambda v: fmt.format(v) if pd.notna(v) else "-")
    return pd.DataFrame(rows, index=df.index)
