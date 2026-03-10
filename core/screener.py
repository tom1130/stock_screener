"""
core/screener.py
주식 스크리닝 필터 조건 정의 및 적용 모듈.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 필터 조건 데이터클래스
# ---------------------------------------------------------------------------

@dataclass
class FilterConditions:
    """스크리닝 필터 조건."""
    market: str = "KOSPI"

    # 시가총액 범위 (억원)
    market_cap_min: float = 300.0
    market_cap_max: float = 9_999_999.0

    # 가격 범위 (원), None이면 조건 없음
    price_min: Optional[float] = None
    price_max: Optional[float] = None

    # 등락률 범위 (%), None이면 조건 없음
    change_rate_min: Optional[float] = None
    change_rate_max: Optional[float] = None

    # 시총 대비 거래대금 최소 (%)
    cap_value_ratio_min: float = 0.0

    # 주식수 회전율 최소 (%)
    turnover_rate_min: float = 0.0

    # N일 평균 대비 거래대금 최소 배수 {n: min_ratio}
    value_vs_avg_min: dict = field(default_factory=dict)

    # N일 평균 대비 회전율 최소 배수 {n: min_ratio}
    turnover_vs_avg_min: dict = field(default_factory=dict)

    # 투자자별 순매수 최소 (억원), None이면 조건 없음
    institution_net_min: Optional[float] = None
    foreigner_net_min: Optional[float] = None
    individual_net_min: Optional[float] = None

    # 기관 & 외국인 동시 순매수 조건
    require_inst_and_fore: bool = False

    # ETF 제외
    exclude_etf: bool = True

    # 정렬 기준
    sort_col: str = "시총대비거래대금"
    sort_asc: bool = False


# ---------------------------------------------------------------------------
# 필터 적용 함수
# ---------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, conditions: FilterConditions) -> pd.DataFrame:
    """
    FilterConditions에 따라 df를 필터링하고 정렬하여 반환.

    Parameters
    ----------
    df : build_metrics_dataframe 결과 DataFrame (인덱스=ticker).
    conditions : FilterConditions 인스턴스.

    Returns
    -------
    필터링 & 정렬된 DataFrame.
    """
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    # 시장 필터
    if conditions.market != "ALL" and "market" in df.columns:
        mask &= df["market"] == conditions.market

    # 시가총액 범위 (억원)
    if "시가총액_억" in df.columns:
        cap = df["시가총액_억"]
        mask &= cap.fillna(0) >= conditions.market_cap_min
        mask &= cap.fillna(0) <= conditions.market_cap_max

    # 가격 범위
    if "종가" in df.columns:
        price = df["종가"]
        if conditions.price_min is not None:
            mask &= price.fillna(0) >= conditions.price_min
        if conditions.price_max is not None:
            mask &= price.fillna(np.inf) <= conditions.price_max

    # 등락률 범위
    if "등락률" in df.columns:
        chg = df["등락률"]
        if conditions.change_rate_min is not None:
            mask &= chg.fillna(-np.inf) >= conditions.change_rate_min
        if conditions.change_rate_max is not None:
            mask &= chg.fillna(np.inf) <= conditions.change_rate_max

    # 시총 대비 거래대금 최소
    if "시총대비거래대금" in df.columns and conditions.cap_value_ratio_min > 0:
        mask &= df["시총대비거래대금"].fillna(0) >= conditions.cap_value_ratio_min

    # 주식수 회전율 최소
    if "주식수회전율" in df.columns and conditions.turnover_rate_min > 0:
        mask &= df["주식수회전율"].fillna(0) >= conditions.turnover_rate_min

    # N일 평균 대비 거래대금 최소 배수
    for n, min_ratio in conditions.value_vs_avg_min.items():
        col = f"{n}일평균대비거래대금"
        if col in df.columns and min_ratio is not None and min_ratio > 0:
            mask &= df[col].fillna(0) >= min_ratio

    # N일 평균 대비 회전율 최소 배수
    for n, min_ratio in conditions.turnover_vs_avg_min.items():
        col = f"{n}일평균대비회전율"
        if col in df.columns and min_ratio is not None and min_ratio > 0:
            mask &= df[col].fillna(0) >= min_ratio

    # 기관 순매수 최소
    if conditions.institution_net_min is not None and "기관순매수_억" in df.columns:
        mask &= df["기관순매수_억"].fillna(-np.inf) >= conditions.institution_net_min

    # 외국인 순매수 최소
    if conditions.foreigner_net_min is not None and "외국인순매수_억" in df.columns:
        mask &= df["외국인순매수_억"].fillna(-np.inf) >= conditions.foreigner_net_min

    # 개인 순매수 최소
    if conditions.individual_net_min is not None and "개인순매수_억" in df.columns:
        mask &= df["개인순매수_억"].fillna(-np.inf) >= conditions.individual_net_min

    # 기관 & 외국인 동시 순매수
    if conditions.require_inst_and_fore:
        if "기관순매수_억" in df.columns and "외국인순매수_억" in df.columns:
            mask &= df["기관순매수_억"].fillna(-np.inf) > 0
            mask &= df["외국인순매수_억"].fillna(-np.inf) > 0

    # 필터 적용
    result = df[mask].copy()

    # 정렬
    if conditions.sort_col in result.columns:
        result = result.sort_values(
            by=conditions.sort_col,
            ascending=conditions.sort_asc,
            na_position="last",
        )

    return result
