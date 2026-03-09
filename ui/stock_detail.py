"""단일 종목 상세 뷰"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from config import (
    COLOR_INSTITUTION, COLOR_FOREIGNER, COLOR_INDIVIDUAL,
    COLOR_VOLUME, COLOR_AVG_LINE, COMPARE_DAYS, WON_TO_EOKWON,
)
from data.fetcher import fetch_ticker_ohlcv, fetch_ticker_investor


def render_stock_detail(ticker, metrics_df, hist_cap_df, date_str):
    if ticker not in metrics_df.index:
        st.error(f"{ticker} 데이터를 찾을 수 없습니다.")
        return
    row = metrics_df.loc[ticker]
    name = row.get("종목명", ticker)
    st.markdown(f"### {name} `{ticker}`")
    _render_metric_cards(row)
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["거래량 추이", "투자자 순매수", "가격 차트"])
    with tab1:
        _render_volume_trend(ticker, hist_cap_df, date_str)
    with tab2:
        _render_investor_chart(ticker, date_str)
    with tab3:
        _render_price_chart(ticker, date_str)


def _render_metric_cards(row):
    """핵심 지표 카드 4열 표시"""
    c1, c2, c3, c4 = st.columns(4)

    close_val = row.get("종가", None)
    cap_val = row.get("시가총액_억", None)
    with c1:
        st.metric(
            "종가",
            f"{close_val:,.0f}원" if pd.notna(close_val) else "-",
        )
        st.metric(
            "시가총액",
            f"{cap_val:,.0f}억" if pd.notna(cap_val) else "-",
        )

    chg_val = row.get("등락률", None)
    trade_val = row.get("거래대금_억", None)
    with c2:
        st.metric(
            "등락률",
            f"{chg_val:+.2f}%" if pd.notna(chg_val) else "-",
        )
        st.metric(
            "거래대금",
            f"{trade_val:,.1f}억" if pd.notna(trade_val) else "-",
        )

    cap_ratio = row.get("시총대비거래대금", None)
    turnover = row.get("주식수회전율", None)
    with c3:
        st.metric(
            "시총대비거래대금",
            f"{cap_ratio:.3f}%" if pd.notna(cap_ratio) else "-",
        )
        st.metric(
            "주식수회전율",
            f"{turnover:.3f}%" if pd.notna(turnover) else "-",
        )

    inst_val = row.get("기관순매수_억", None)
    fore_val = row.get("외국인순매수_억", None)
    with c4:
        st.metric(
            "기관 순매수",
            f"{inst_val:+.1f}억" if pd.notna(inst_val) else "-",
        )
        st.metric(
            "외국인 순매수",
            f"{fore_val:+.1f}억" if pd.notna(fore_val) else "-",
        )

    # N일 평균 대비 배수 행
    avg_cols = st.columns(len(COMPARE_DAYS) * 2)
    idx = 0
    for n in COMPARE_DAYS:
        val_key = f"{n}일평균대비거래대금"
        tur_key = f"{n}일평균대비회전율"
        val = row.get(val_key, None)
        tur = row.get(tur_key, None)
        with avg_cols[idx]:
            st.metric(
                f"거래대금 {n}일대비",
                f"{val:.2f}x" if pd.notna(val) else "-",
            )
        idx += 1
        with avg_cols[idx]:
            st.metric(
                f"회전율 {n}일대비",
                f"{tur:.2f}x" if pd.notna(tur) else "-",
            )
        idx += 1


def _render_volume_trend(ticker, hist_cap_df, date_str):
    """거래량 추이 차트. hist_cap_df는 {date_str: DataFrame(인덱스=ticker)} 형태."""
    if not hist_cap_df:
        st.info("과거 비교 데이터 없음 (Naver Finance는 당일 데이터만 제공)")
        return

    rows = []
    for d, df in sorted(hist_cap_df.items()):
        if df.empty or ticker not in df.index:
            continue
        r = df.loc[ticker].copy()
        r["date"] = d
        rows.append(r)

    if not rows:
        st.info("과거 비교 데이터 없음")
        return

    hist = pd.DataFrame(rows).set_index("date").sort_index()
    if "거래대금" not in hist.columns:
        st.info("거래대금 데이터 없음")
        return

    trade_eok = pd.to_numeric(hist["거래대금"], errors="coerce") / WON_TO_EOKWON
    x_vals = hist.index

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("거래대금(억)", "시총대비거래대금(%)"),
        row_heights=[0.6, 0.4], vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Bar(x=x_vals, y=trade_eok, name="거래대금", marker_color=COLOR_VOLUME),
        row=1, col=1,
    )
    colors_ma = ["#FF7F0E", "#2CA02C", "#D62728"]
    for n, color in zip(COMPARE_DAYS, colors_ma):
        ma = trade_eok.rolling(n).mean()
        fig.add_trace(
            go.Scatter(x=x_vals, y=ma, name=f"{n}일MA",
                       line=dict(color=color, width=1.5, dash="dot")),
            row=1, col=1,
        )
    if "시가총액" in hist.columns:
        ratio = (pd.to_numeric(hist["거래대금"], errors="coerce") /
                 pd.to_numeric(hist["시가총액"], errors="coerce").replace(0, float("nan")) * 100)
        fig.add_trace(
            go.Scatter(x=x_vals, y=ratio, name="시총대비(%)",
                       line=dict(color=COLOR_AVG_LINE, width=2)),
            row=2, col=1,
        )

    fig.update_layout(height=480, legend=dict(orientation="h", y=1.05),
                      margin=dict(l=40, r=40, t=60, b=40), plot_bgcolor="white")
    fig.update_xaxes(tickformat="%m/%d")
    st.plotly_chart(fig, use_container_width=True)


def _render_investor_chart(ticker, date_str):
    """투자자 순매수 grouped bar 차트 (최근 20 영업일)."""
    from data.fetcher import get_business_dates
    dates = get_business_dates(date_str, 20)
    if not dates:
        st.info("날짜 계산 실패")
        return

    inv_df = fetch_ticker_investor(ticker, dates[0], date_str)
    if inv_df is None or inv_df.empty:
        st.info("투자자 데이터 없음 (KRX API 제한으로 현재 미지원)")
        return

    x_vals = inv_df.index.astype(str)
    inst_col = next((c for c in ["기관합계", "기관", "금융투자"] if c in inv_df.columns), None)
    fore_col = next((c for c in ["외국인합계", "외국인"] if c in inv_df.columns), None)
    indi_col = next((c for c in ["개인"] if c in inv_df.columns), None)

    fig = go.Figure()
    if inst_col:
        fig.add_trace(go.Bar(x=x_vals, y=inv_df[inst_col] / WON_TO_EOKWON, name="기관", marker_color=COLOR_INSTITUTION))
    if fore_col:
        fig.add_trace(go.Bar(x=x_vals, y=inv_df[fore_col] / WON_TO_EOKWON, name="외국인", marker_color=COLOR_FOREIGNER))
    if indi_col:
        fig.add_trace(go.Bar(x=x_vals, y=inv_df[indi_col] / WON_TO_EOKWON, name="개인", marker_color=COLOR_INDIVIDUAL))

    fig.update_layout(
        barmode="group", height=380, title="투자자별 일별 순매수(억원)",
        legend=dict(orientation="h"), plot_bgcolor="white",
        xaxis_tickformat="%m/%d", margin=dict(l=40, r=40, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_price_chart(ticker, date_str):
    """캔들스틱 + 거래량 차트 (최근 60 영업일)."""
    from data.fetcher import get_business_dates
    dates = get_business_dates(date_str, 60)
    if not dates:
        st.info("날짜 계산 실패")
        return

    df = fetch_ticker_ohlcv(ticker, dates[0], date_str)
    if df is None or df.empty:
        st.info("가격 데이터 없음")
        return

    df = df.copy()
    df.index = df.index.astype(str)

    open_col  = next((c for c in ["시가", "Open"]  if c in df.columns), None)
    high_col  = next((c for c in ["고가", "High"]  if c in df.columns), None)
    low_col   = next((c for c in ["저가", "Low"]   if c in df.columns), None)
    close_col = next((c for c in ["종가", "Close"] if c in df.columns), None)
    vol_col   = next((c for c in ["거래량", "Volume"] if c in df.columns), None)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("가격", "거래량"),
        row_heights=[0.7, 0.3], vertical_spacing=0.05,
    )

    if all(c is not None for c in [open_col, high_col, low_col, close_col]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df[open_col], high=df[high_col],
                low=df[low_col],   close=df[close_col],
                name="OHLCV",
                increasing_line_color="#EF553B",
                decreasing_line_color="#636EFA",
            ),
            row=1, col=1,
        )

    if vol_col:
        close_vals = pd.to_numeric(df.get(close_col, pd.Series()), errors="coerce")
        open_vals  = pd.to_numeric(df.get(open_col, pd.Series()),  errors="coerce")
        colors = ["#EF553B" if c >= o else "#636EFA"
                  for c, o in zip(close_vals, open_vals)]
        fig.add_trace(
            go.Bar(x=df.index, y=df[vol_col], name="거래량",
                   marker_color=colors, opacity=0.7),
            row=2, col=1,
        )

    fig.update_layout(
        height=540, xaxis_rangeslider_visible=False,
        plot_bgcolor="white", margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(tickformat="%m/%d")
    st.plotly_chart(fig, use_container_width=True)
