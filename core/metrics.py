"""
core/metrics.py
주식 스크리닝 지표 계산 모듈. 외부 IO 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import WON_TO_EOKWON, SHARE_TO_PCT, COMPARE_DAYS

_INSTITUTION_COLS = ["기관합계", "기관", "금융투자"]
_FOREIGNER_COLS   = ["외국인합계", "외국인"]
_INDIVIDUAL_COLS  = ["개인"]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── 단위 계산 함수 ────────────────────────────────────────────────────────────

def calc_cap_value_ratio(trading_value: pd.Series, market_cap: pd.Series) -> pd.Series:
    """시총 대비 거래대금 (%). 원 단위 입력."""
    return (trading_value / market_cap.replace(0, np.nan) * SHARE_TO_PCT).round(4)


def calc_turnover_rate(volume: pd.Series, listed_shares: pd.Series) -> pd.Series:
    """주식수 회전율 (%). 주 단위 입력."""
    return (volume / listed_shares.replace(0, np.nan) * SHARE_TO_PCT).round(4)


def calc_vs_avg_ratio(today: pd.Series, avg: pd.Series) -> pd.Series:
    """오늘 값 / N일 평균 배수. 평균 0이면 NaN."""
    return (today / avg.replace(0, np.nan)).round(2)


def calc_n_day_avg(
    hist_dict: dict[str, pd.DataFrame],
    col: str,
    n: int,
    today_date: str,
) -> pd.Series:
    """
    hist_dict: {date_str: DataFrame(인덱스=ticker)} 형태.
    today_date 제외 최근 n일 평균. 유효 데이터 없으면 빈 Series.
    """
    past = {d: df for d, df in hist_dict.items() if d != today_date and not df.empty}
    recent = sorted(past.keys(), reverse=True)[:n]

    frames = []
    for d in recent:
        df = past[d]
        if col in df.columns:
            frames.append(pd.to_numeric(df[col], errors="coerce").rename(d))

    if not frames:
        return pd.Series(dtype=float)

    combined = pd.concat(frames, axis=1)
    return combined.mean(axis=1)


# ── 메인 DataFrame 조립 ───────────────────────────────────────────────────────

def build_metrics_dataframe(
    today_cap_df: pd.DataFrame,
    today_ohlcv_df: pd.DataFrame,
    investor_df: pd.DataFrame,
    hist_cap_df: dict[str, pd.DataFrame],
    ticker_names: dict[str, str],
    today_date: str,
) -> pd.DataFrame:
    """
    모든 소스를 합쳐 스크리너용 메트릭 DataFrame 반환.

    반환 컬럼:
        종목명, market, 종가, 등락률,
        시가총액_억, 거래대금_억,
        시총대비거래대금, 주식수회전율,
        기관순매수_억, 외국인순매수_억, 개인순매수_억,
        5/10/20일평균대비거래대금, 5/10/20일평균대비회전율
    """
    if today_cap_df.empty:
        return pd.DataFrame()

    df = today_cap_df.copy()

    # 숫자 강제 변환
    for col in ["종가", "등락률", "시가총액", "상장주식수", "거래량", "거래대금"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 억원 환산
    df["시가총액_억"] = (df["시가총액"] / WON_TO_EOKWON).round(1) if "시가총액" in df.columns else np.nan
    df["거래대금_억"] = (df["거래대금"] / WON_TO_EOKWON).round(1) if "거래대금" in df.columns else np.nan

    # OHLCV 보완 (today_ohlcv_df에 없는 컬럼 채우기)
    if not today_ohlcv_df.empty:
        for col in ["종가", "등락률", "market"]:
            if col not in df.columns and col in today_ohlcv_df.columns:
                df[col] = today_ohlcv_df[col].reindex(df.index)

    if "market" not in df.columns:
        df["market"] = "UNKNOWN"
    if "종가" not in df.columns:
        df["종가"] = np.nan
    if "등락률" not in df.columns:
        df["등락률"] = np.nan

    # 핵심 비율 지표
    if "거래대금" in df.columns and "시가총액" in df.columns:
        df["시총대비거래대금"] = calc_cap_value_ratio(df["거래대금"], df["시가총액"])
    else:
        df["시총대비거래대금"] = np.nan

    if "거래량" in df.columns and "상장주식수" in df.columns:
        df["주식수회전율"] = calc_turnover_rate(df["거래량"], df["상장주식수"])
    else:
        df["주식수회전율"] = np.nan

    # 투자자 순매수
    if not investor_df.empty:
        inst_col = _find_col(investor_df, _INSTITUTION_COLS)
        fore_col = _find_col(investor_df, _FOREIGNER_COLS)
        indi_col = _find_col(investor_df, _INDIVIDUAL_COLS)
        df["기관순매수_억"]   = (pd.to_numeric(investor_df[inst_col], errors="coerce").reindex(df.index) / WON_TO_EOKWON).round(1) if inst_col else np.nan
        df["외국인순매수_억"] = (pd.to_numeric(investor_df[fore_col], errors="coerce").reindex(df.index) / WON_TO_EOKWON).round(1) if fore_col else np.nan
        df["개인순매수_억"]   = (pd.to_numeric(investor_df[indi_col], errors="coerce").reindex(df.index) / WON_TO_EOKWON).round(1) if indi_col else np.nan
    else:
        df["기관순매수_억"] = df["외국인순매수_억"] = df["개인순매수_억"] = np.nan

    # N일 평균 대비 배수
    for n in COMPARE_DAYS:
        avg_val = calc_n_day_avg(hist_cap_df, "거래대금", n, today_date).reindex(df.index)
        df[f"{n}일평균대비거래대금"] = calc_vs_avg_ratio(df.get("거래대금", pd.Series(dtype=float)), avg_val) if "거래대금" in df.columns else np.nan

        # 과거 회전율 평균
        tr_frames = []
        for d, hdf in hist_cap_df.items():
            if d == today_date or hdf.empty:
                continue
            if "거래량" in hdf.columns and "상장주식수" in hdf.columns:
                tr = calc_turnover_rate(
                    pd.to_numeric(hdf["거래량"], errors="coerce"),
                    pd.to_numeric(hdf["상장주식수"], errors="coerce"),
                ).rename(d)
                tr_frames.append(tr)

        if tr_frames:
            avg_turn = pd.concat(tr_frames, axis=1).mean(axis=1).reindex(df.index)
            df[f"{n}일평균대비회전율"] = calc_vs_avg_ratio(df["주식수회전율"], avg_turn)
        else:
            df[f"{n}일평균대비회전율"] = np.nan

    # 종목명
    df["종목명"] = df.index.map(lambda t: ticker_names.get(t, t) if isinstance(ticker_names, dict) else t)

    df.index.name = "ticker"
    output_cols = [
        "종목명", "market", "종가", "등락률",
        "시가총액_억", "거래대금_억",
        "시총대비거래대금", "주식수회전율",
        "기관순매수_억", "외국인순매수_억", "개인순매수_억",
        "5일평균대비거래대금", "10일평균대비거래대금", "20일평균대비거래대금",
        "5일평균대비회전율", "10일평균대비회전율", "20일평균대비회전율",
    ]
    return df[[c for c in output_cols if c in df.columns]]
