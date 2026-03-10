"""사이드바 필터 위젯"""
from __future__ import annotations
import streamlit as st
from config import MARKETS, MARKET_CAP_TIERS, COMPARE_DAYS
from core.screener import FilterConditions


def render_sidebar_filters() -> FilterConditions:
    st.sidebar.header("🔍 필터 설정")

    # 1. 시장
    st.sidebar.subheader("시장")
    market = st.sidebar.selectbox("시장 선택", MARKETS, index=0)

    # 2. ETF 제외
    exclude_etf = st.sidebar.checkbox("ETF 제외", value=True)

    # 3. 시가총액 (필수)
    st.sidebar.subheader("📊 시가총액 (필수)")
    tier_labels = list(MARKET_CAP_TIERS.keys())
    selected_tier = st.sidebar.selectbox("구간 빠른 선택", ["직접 입력"] + tier_labels, index=0)
    preset_min, preset_max = MARKET_CAP_TIERS[selected_tier] if selected_tier != "직접 입력" else (300, 9_999_999)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        cap_min = st.number_input("최소(억)", value=float(preset_min), min_value=0.0, step=100.0)
    with col2:
        cap_max = st.number_input("최대(억)", value=float(preset_max), min_value=0.0, step=100.0)

    # 4. 종가
    st.sidebar.subheader("💰 종가 (원)")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        price_min_raw = st.number_input("최소", value=0, min_value=0, step=1000, key="price_min")
    with c2:
        price_max_raw = st.number_input("최대", value=0, min_value=0, step=1000, key="price_max")
    price_min = price_min_raw if price_min_raw > 0 else None
    price_max = price_max_raw if price_max_raw > 0 else None

    # 5. 등락률
    st.sidebar.subheader("📈 등락률 (%)")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        chg_min_raw = st.number_input("최소(%)", value=-30.0, step=0.5, key="chg_min")
    with c2:
        chg_max_raw = st.number_input("최대(%)", value=30.0, step=0.5, key="chg_max")
    chg_min = chg_min_raw if chg_min_raw > -30.0 else None
    chg_max = chg_max_raw if chg_max_raw < 30.0 else None

    # 6. 거래 비율
    st.sidebar.subheader("🔄 거래 비율")
    cap_val_min = st.sidebar.number_input("시총대비거래대금 최소(%)", value=0.0, min_value=0.0, step=0.1)
    turnover_min = st.sidebar.number_input("주식수회전율 최소(%)", value=0.0, min_value=0.0, step=0.1)

    # 7. N일 평균 대비 배수
    st.sidebar.subheader("📅 N일 평균 대비 배수")
    st.sidebar.caption("⚠️ 설정 시 종목별 일별 데이터를 추가 로드합니다")
    value_vs_avg: dict = {}
    turnover_vs_avg: dict = {}
    for n in COMPARE_DAYS:
        col_v, col_t = st.sidebar.columns(2)
        with col_v:
            v = st.number_input(f"거래대금 {n}일대비", value=0.0, min_value=0.0, step=0.5, key=f"val_{n}")
        with col_t:
            t = st.number_input(f"회전율 {n}일대비", value=0.0, min_value=0.0, step=0.5, key=f"tur_{n}")
        value_vs_avg[n] = v if v > 0 else None
        turnover_vs_avg[n] = t if t > 0 else None

    # 8. 투자자
    st.sidebar.subheader("👥 투자자 순매수 (억원)")
    st.sidebar.caption("⚠️ 설정 시 종목별 투자자 데이터를 추가 로드합니다")
    inst_min_raw = st.sidebar.number_input("기관 순매수 최소", value=0.0, step=10.0, key="inst_min")
    fore_min_raw = st.sidebar.number_input("외국인 순매수 최소", value=0.0, step=10.0, key="fore_min")
    indi_min_raw = st.sidebar.number_input("개인 순매수 최소", value=0.0, step=10.0, key="indi_min")
    require_both = st.sidebar.checkbox("기관 + 외국인 동시 순매수")
    inst_min = inst_min_raw if inst_min_raw > 0 else None
    fore_min = fore_min_raw if fore_min_raw > 0 else None
    indi_min = indi_min_raw if indi_min_raw > 0 else None

    # 9. 정렬
    st.sidebar.subheader("🔃 정렬")
    sort_options = [
        "시총대비거래대금", "20일평균대비거래대금", "10일평균대비거래대금",
        "5일평균대비거래대금", "주식수회전율", "기관순매수_억",
        "외국인순매수_억", "거래대금_억", "시가총액_억", "등락률",
    ]
    sort_col = st.sidebar.selectbox("정렬 기준", sort_options, index=0)
    sort_asc = st.sidebar.checkbox("오름차순", value=False)

    return FilterConditions(
        market=market,
        exclude_etf=exclude_etf,
        market_cap_min=cap_min, market_cap_max=cap_max,
        price_min=price_min, price_max=price_max,
        change_rate_min=chg_min, change_rate_max=chg_max,
        cap_value_ratio_min=cap_val_min, turnover_rate_min=turnover_min,
        value_vs_avg_min=value_vs_avg, turnover_vs_avg_min=turnover_vs_avg,
        institution_net_min=inst_min, foreigner_net_min=fore_min, individual_net_min=indi_min,
        require_inst_and_fore=require_both,
        sort_col=sort_col, sort_asc=sort_asc,
    )
