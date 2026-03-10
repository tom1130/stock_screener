"""주식 스크리너 Streamlit 앱 진입점"""
from __future__ import annotations
import streamlit as st

st.set_page_config(page_title="주식 스크리너", page_icon="📈", layout="wide")

import pandas as pd

from config import COMPARE_DAYS
from data.fetcher import (
    fetch_market_cap, fetch_etf_tickers,
    fetch_all_data_for_tickers,
    fetch_hist_volume_avg,
    get_latest_business_date,
)
from core.metrics import (
    build_metrics_dataframe,
    enrich_with_investor,
    enrich_with_hist_volume_avg,
)
from core.screener import apply_filters
from ui.filters import render_sidebar_filters
from ui.stock_list import render_stock_list
from ui.stock_detail import render_stock_detail

# 종목 수 임계값: 이 초과 시 사용자 확인 버튼 표시
_CONFIRM_THRESHOLD = 500


def main():
    st.title("📈 주식 스크리너")
    st.caption("Naver Finance 기준 · 실시간 시세 데이터")

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None
    if "enriched_df" not in st.session_state:
        st.session_state.enriched_df = None
    if "last_filter_hash" not in st.session_state:
        st.session_state.last_filter_hash = None
    if "enrich_confirmed" not in st.session_state:
        st.session_state.enrich_confirmed = False

    conditions, date_str = render_sidebar_filters()

    # ── 1단계: 기본 시세 데이터 로드 (Naver sise, 항상 오늘) ─────────────────
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

    # ── 2단계: 상세 데이터 (투자자 + N일 거래량) ─────────────────────────────
    filter_hash = _make_hash(conditions, date_str, len(pre_filtered))

    # 필터 조건 변경 시 캐시 초기화
    if st.session_state.last_filter_hash != filter_hash:
        st.session_state.enriched_df = None
        st.session_state.last_filter_hash = filter_hash
        st.session_state.enrich_confirmed = False

    enriched_df = st.session_state.enriched_df

    if enriched_df is None:
        tickers = pre_filtered.index.tolist()
        n_tickers = len(tickers)

        if n_tickers == 0:
            pass  # 종목 없음, enrichment 불필요
        elif n_tickers > _CONFIRM_THRESHOLD:
            # 사용자 확인 버튼 표시
            st.warning(
                f"기본 필터 결과가 **{n_tickers}개** 종목입니다. "
                f"투자자·N일 데이터를 로드하면 시간이 걸릴 수 있습니다."
            )
            if st.session_state.enrich_confirmed:
                enriched_df = _run_enrichment(pre_filtered, tickers, date_str)
                st.session_state.enriched_df = enriched_df
            else:
                if st.button(f"투자자·N일 데이터 로드 ({n_tickers}종목)"):
                    st.session_state.enrich_confirmed = True
                    st.rerun()
        else:
            # 500개 이하: 자동 로드
            enriched_df = _run_enrichment(pre_filtered, tickers, date_str)
            st.session_state.enriched_df = enriched_df

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


def _run_enrichment(pre_filtered: pd.DataFrame, tickers: list[str], date_str: str) -> pd.DataFrame:
    """투자자·N일 데이터를 로드하고 enrich된 DataFrame 반환."""
    n_tickers = len(tickers)
    prog = st.progress(0, text=f"투자자 데이터 로딩 중… ({n_tickers}종목)")

    investor_df = fetch_all_data_for_tickers(tickers, date_str=date_str)
    prog.progress(50, text="N일 평균 거래량 계산 중…")

    avg_vol_df = fetch_hist_volume_avg(tickers, date_str=date_str, compare_days=COMPARE_DAYS)
    prog.progress(90, text="메트릭 계산 중…")

    enriched = enrich_with_investor(pre_filtered.copy(), investor_df)
    enriched = enrich_with_hist_volume_avg(enriched, avg_vol_df, COMPARE_DAYS)

    prog.progress(100)
    prog.empty()
    return enriched


_ETF_PREFIXES = (
    "KODEX", "TIGER", "ARIRANG", "KINDEX", "HANARO", "KOSEF",
    "SOL", "ACE", "TIMEFOLIO", "PLUS", "TREX", "FOCUS",
    "GIANT", "KTOP", "MASTER", "WOORI", "SMART ETF",
    "RISE", "TIME ", "WON ", "KB", "NH-Amundi",
)

_ETF_KEYWORDS = ("ETN", "레버리지", "인버스", " ETF")


def _is_etf_by_name(name_series: pd.Series) -> pd.Series:
    """종목명이 ETF/ETN 접두어로 시작하거나 키워드를 포함하면 True."""
    mask = pd.Series(False, index=name_series.index)
    for prefix in _ETF_PREFIXES:
        mask |= name_series.str.startswith(prefix, na=False)
    for kw in _ETF_KEYWORDS:
        mask |= name_series.str.contains(kw, na=False, regex=False)
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


def _make_hash(conditions, date_str: str, n_tickers: int) -> str:
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
        "date_str": date_str,
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
