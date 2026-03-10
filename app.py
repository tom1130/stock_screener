"""주식 스크리너 Streamlit 앱 진입점"""
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="주식 스크리너", page_icon="📈", layout="wide")

import pandas as pd

from config import COMPARE_DAYS
from data.fetcher import (
    fetch_market_cap, fetch_etf_tickers,
    fetch_investor_for_tickers, fetch_hist_data_for_tickers,
    get_latest_business_date,
)
from core.metrics import (
    build_metrics_dataframe,
    enrich_with_investor,
    enrich_with_hist_volume,
)
from core.screener import apply_filters
from ui.filters import render_sidebar_filters
from ui.stock_list import render_stock_list
from ui.stock_detail import render_stock_detail

# 결과 임계값: 이 이하면 자동으로 상세 데이터 로드
_AUTO_ENRICH_LIMIT = 300


def main():
    st.title("📈 주식 스크리너")
    st.caption("Naver Finance 기준 · 실시간 시세 데이터")

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None
    if "enriched_df" not in st.session_state:
        st.session_state.enriched_df = None
    if "last_filter_hash" not in st.session_state:
        st.session_state.last_filter_hash = None

    conditions = render_sidebar_filters()

    # ── 1단계: 기본 데이터 로드 ─────────────────────────────────────────────
    with st.spinner("시세 데이터 로딩 중…"):
        base_df = _load_base_data(conditions.market)

    if base_df.empty:
        st.error("데이터를 불러오지 못했습니다.")
        return

    # ETF 제외 (API 목록 + 종목명 패턴 병행)
    if conditions.exclude_etf:
        etf_set = fetch_etf_tickers()
        name_mask = _is_etf_by_name(base_df.get("종목명", pd.Series(dtype=str)))
        base_df = base_df[~(base_df.index.isin(etf_set) | name_mask)]

    # 기본 필터 적용 (투자자·N일 제외)
    pre_filtered = apply_filters(base_df, conditions)

    # ── 2단계: 상세 데이터 (투자자 + N일 거래량) ────────────────────────────
    filter_hash = _make_hash(conditions, len(pre_filtered))

    need_enrich = (
        (conditions.institution_net_min is not None)
        or (conditions.foreigner_net_min is not None)
        or any(v for v in conditions.value_vs_avg_min.values() if v)
        or any(v for v in conditions.turnover_vs_avg_min.values() if v)
        or conditions.require_inst_and_fore
        or len(pre_filtered) <= _AUTO_ENRICH_LIMIT
    )

    if st.session_state.last_filter_hash != filter_hash:
        st.session_state.enriched_df = None
        st.session_state.last_filter_hash = filter_hash

    enriched_df = st.session_state.enriched_df

    if need_enrich and enriched_df is None:
        tickers = pre_filtered.index.tolist()
        if len(tickers) > 0:
            prog = st.progress(0, text=f"투자자·N일 데이터 로딩 중… ({len(tickers)}종목)")
            price_map = pre_filtered["종가"].dropna().to_dict()

            investor_df = fetch_investor_for_tickers(tickers, price_map)
            prog.progress(50, text="N일 거래량 계산 중…")

            hist_data = fetch_hist_data_for_tickers(tickers)
            prog.progress(90, text="메트릭 계산 중…")

            enriched = enrich_with_investor(pre_filtered.copy(), investor_df)
            enriched = enrich_with_hist_volume(enriched, hist_data, COMPARE_DAYS)
            prog.progress(100)
            prog.empty()

            st.session_state.enriched_df = enriched
            enriched_df = enriched

    if enriched_df is not None:
        filtered_df = apply_filters(enriched_df, conditions)
    else:
        filtered_df = pre_filtered

    # ── 화면 렌더링 ──────────────────────────────────────────────────────────
    if st.session_state.selected_ticker is not None:
        col_back, _ = st.columns([1, 9])
        with col_back:
            if st.button("← 목록으로"):
                st.session_state.selected_ticker = None
                st.rerun()
        detail_df = enriched_df if enriched_df is not None else base_df
        render_stock_detail(st.session_state.selected_ticker, detail_df, {}, get_latest_business_date())
    else:
        _render_summary_stats(filtered_df)
        selected = render_stock_list(filtered_df)
        if selected:
            st.session_state.selected_ticker = selected
            st.rerun()


_ETF_PREFIXES = (
    "KODEX", "TIGER", "ARIRANG", "KINDEX", "HANARO", "KOSEF",
    "SOL", "ACE", "TIMEFOLIO", "PLUS", "TREX", "FOCUS",
    "GIANT", "KTOP", "MASTER", "WOORI", "SMART ETF",
)


def _is_etf_by_name(name_series: pd.Series) -> pd.Series:
    """종목명이 ETF 접두어로 시작하면 True."""
    mask = pd.Series(False, index=name_series.index)
    for prefix in _ETF_PREFIXES:
        mask |= name_series.str.startswith(prefix, na=False)
    return mask


def _load_base_data(market: str) -> pd.DataFrame:
    today = get_latest_business_date()
    cap_df = fetch_market_cap(today, market)
    if cap_df.empty:
        return pd.DataFrame()

    ticker_names = cap_df["종목명"].dropna().to_dict() if "종목명" in cap_df.columns else {}
    ohlcv_df = cap_df[["종가", "등락률", "market"]].copy() if "종가" in cap_df.columns else pd.DataFrame()

    return build_metrics_dataframe(
        today_cap_df=cap_df,
        today_ohlcv_df=ohlcv_df,
        investor_df=pd.DataFrame(),
        hist_cap_df={},
        ticker_names=ticker_names,
        today_date=today,
    )


def _make_hash(conditions, n_tickers: int) -> str:
    import hashlib, json
    d = {
        "market": conditions.market,
        "cap_min": conditions.market_cap_min,
        "cap_max": conditions.market_cap_max,
        "price_min": conditions.price_min,
        "price_max": conditions.price_max,
        "chg_min": conditions.change_rate_min,
        "chg_max": conditions.change_rate_max,
        "exclude_etf": conditions.exclude_etf,
        "n": n_tickers,
    }
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


def _render_summary_stats(df: pd.DataFrame):
    today = get_latest_business_date()
    st.markdown(f"**기준: {today[:4]}-{today[4:6]}-{today[6:]} (실시간 Naver 시세)**")
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
