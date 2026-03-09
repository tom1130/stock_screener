"""주식 스크리너 Streamlit 앱 진입점"""
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="주식 스크리너", page_icon="📈", layout="wide")

from config import COMPARE_DAYS
from data.fetcher import (
    fetch_market_cap, fetch_ohlcv, fetch_investor_net,
    fetch_ticker_names, fetch_historical_market_cap, get_business_dates,
)
from core.metrics import build_metrics_dataframe
from core.screener import apply_filters
from ui.filters import render_sidebar_filters
from ui.stock_list import render_stock_list
from ui.stock_detail import render_stock_detail


def main():
    st.title("📈 주식 스크리너")
    st.caption("종가·거래량·투자자 데이터를 기반으로 주목할 종목을 필터링합니다.")

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None

    conditions, date_str = render_sidebar_filters()

    with st.spinner("데이터 로딩 중…"):
        metrics_df, hist_cap_df = _load_all_data(date_str, conditions.market)

    if metrics_df.empty:
        st.error("데이터를 불러오지 못했습니다. 날짜 또는 시장을 확인하세요.")
        return

    filtered_df = apply_filters(metrics_df, conditions)

    if st.session_state.selected_ticker is not None:
        col_back, _ = st.columns([1, 9])
        with col_back:
            if st.button("← 목록으로"):
                st.session_state.selected_ticker = None
                st.rerun()
        render_stock_detail(st.session_state.selected_ticker, metrics_df, hist_cap_df, date_str)
    else:
        _render_summary_stats(filtered_df, date_str)
        selected = render_stock_list(filtered_df)
        if selected:
            st.session_state.selected_ticker = selected
            st.rerun()


def _load_all_data(date_str, market):
    import pandas as pd
    max_days = max(COMPARE_DAYS) + 1
    biz_dates = get_business_dates(date_str, max_days)
    if not biz_dates:
        return pd.DataFrame(), {}
    today = biz_dates[-1]
    past_dates = biz_dates[:-1]

    prog = st.progress(0, text="시가총액·OHLCV 로딩…")
    today_cap_df = fetch_market_cap(today, market)
    prog.progress(40, text="투자자 데이터 로딩…")
    investor_df = fetch_investor_net(today, market)
    prog.progress(55, text="과거 데이터 로딩…")
    hist_cap_df = fetch_historical_market_cap(past_dates, market)
    prog.progress(80, text="종목명 로딩…")
    ticker_names = fetch_ticker_names(today, market)
    prog.progress(92, text="메트릭 계산…")

    # ohlcv는 fetch_market_cap에 포함 — 컬럼 존재 여부 확인 후 추출
    if not today_cap_df.empty and all(c in today_cap_df.columns for c in ["종가", "등락률", "market"]):
        today_ohlcv_df = today_cap_df[["종가", "등락률", "market"]].copy()
    elif not today_cap_df.empty and "종가" in today_cap_df.columns:
        # market 컬럼 없을 수 있음 — 있는 컬럼만 추출
        keep_cols = [c for c in ["종가", "등락률", "market"] if c in today_cap_df.columns]
        today_ohlcv_df = today_cap_df[keep_cols].copy()
    else:
        # fallback: fetch_ohlcv 별도 호출
        today_ohlcv_df = fetch_ohlcv(today, market)

    metrics_df = build_metrics_dataframe(
        today_cap_df=today_cap_df,
        today_ohlcv_df=today_ohlcv_df,
        investor_df=investor_df,
        hist_cap_df=hist_cap_df,
        ticker_names=ticker_names,
        today_date=today,
    )
    prog.progress(100, text="완료")
    prog.empty()
    return metrics_df, hist_cap_df


def _render_summary_stats(df, date_str):
    import pandas as pd
    st.markdown(f"**기준일: {date_str[:4]}-{date_str[4:6]}-{date_str[6:]}**")
    if df.empty:
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("필터링 종목", f"{len(df):,}개")
    with c2:
        avg = df["시총대비거래대금"].mean()
        st.metric("평균 시총대비거래대금", f"{avg:.3f}%" if pd.notna(avg) else "-")
    with c3:
        inst_pos = (df["기관순매수_억"] > 0).sum() if "기관순매수_억" in df.columns else 0
        st.metric("기관 순매수 종목", f"{inst_pos:,}개")
    with c4:
        fore_pos = (df["외국인순매수_억"] > 0).sum() if "외국인순매수_억" in df.columns else 0
        st.metric("외국인 순매수 종목", f"{fore_pos:,}개")
    with c5:
        if "기관순매수_억" in df.columns and "외국인순매수_억" in df.columns:
            both = ((df["기관순매수_억"] > 0) & (df["외국인순매수_억"] > 0)).sum()
        else:
            both = 0
        st.metric("기관+외국인 동시 순매수", f"{both:,}개")


if __name__ == "__main__":
    main()
