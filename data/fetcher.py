"""
data/fetcher.py
Naver Finance 스크래핑 기반 시장 데이터 수집 모듈.

주의: pykrx의 get_market_ohlcv_by_ticker, get_market_cap_by_ticker,
get_market_trading_value_by_ticker 는 KRX HTTP 400 LOGOUT 오류로 동작 불가.
동작 가능: get_market_ohlcv_by_date(start, end, ticker) — Naver 기반.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup

_NAVER_SISE_URL = "https://finance.naver.com/sise/sise_market_sum.naver"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.naver.com/sise/sise_market_sum.naver",
    "Accept-Language": "ko-KR,ko;q=0.9",
}
_SOSOK_MAP = {"KOSPI": 0, "KOSDAQ": 1}


# ── 날짜 유틸 ─────────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> date:
    """'YYYYMMDD' 또는 'YYYY-MM-DD' 모두 파싱."""
    s = date_str.replace("-", "")
    return datetime.strptime(s, "%Y%m%d").date()


# ── 페이지 파싱 ───────────────────────────────────────────────────────────────

def _parse_page(html: str, market: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    # 티커 추출 (검증된 방법: /item/main.naver?code= 링크)
    links = soup.find_all("a", href=re.compile(r"/item/main\.naver\?code="))
    tickers = []
    for a in links:
        m = re.search(r"code=(\d{6})", a["href"])
        if m:
            tickers.append(m.group(1))
    if not tickers:
        return pd.DataFrame()

    try:
        tables = pd.read_html(StringIO(html), encoding="euc-kr")
        t = tables[1].dropna(subset=["종목명"]).reset_index(drop=True)
    except (ValueError, IndexError, KeyError):
        return pd.DataFrame()

    # 길이 불일치 보정
    min_len = min(len(t), len(tickers))
    t = t.iloc[:min_len].reset_index(drop=True)
    tickers = tickers[:min_len]

    t["ticker"] = tickers
    t["market"] = market
    return t


def _fetch_page_html(sosok: int, page: int) -> str:
    r = requests.get(
        _NAVER_SISE_URL,
        headers=_HEADERS,
        params={"sosok": str(sosok), "page": str(page)},
        timeout=12,
    )
    r.encoding = "euc-kr"
    return r.text


def _get_max_page(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=re.compile(r"page=\d+"))
    if not links:
        return 1
    return max(int(re.search(r"page=(\d+)", a["href"]).group(1)) for a in links)


# ── 전체 시장 스크래핑 (캐시 미포함 원시 버전) ────────────────────────────────

def _fetch_naver_all_raw(sosok: int, market: str) -> pd.DataFrame:
    """Naver 시세 페이지를 병렬 스크래핑해 전체 종목 DataFrame 반환."""
    try:
        html1 = _fetch_page_html(sosok, 1)
    except Exception:
        return pd.DataFrame()

    max_page = _get_max_page(html1)
    pages_html: dict[int, str] = {1: html1}

    remaining = list(range(2, max_page + 1))
    if remaining:
        with ThreadPoolExecutor(max_workers=10) as exc:
            futures = {exc.submit(_fetch_page_html, sosok, p): p for p in remaining}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    pages_html[p] = fut.result()
                except Exception:
                    pass

    frames = []
    for p in sorted(pages_html):
        df = _parse_page(pages_html[p], market)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)

    # 컬럼 이름 정규화
    raw = raw.rename(columns={
        "현재가": "종가",
        "시가총액": "_시총억원",
        "상장주식수": "_상장천주",
    })

    def _to_num(s):
        if isinstance(s, str):
            s = s.replace(",", "").replace("+", "").strip()
        return pd.to_numeric(s, errors="coerce")

    if "종가" in raw.columns:
        raw["종가"] = raw["종가"].apply(_to_num)
    if "등락률" in raw.columns:
        raw["등락률"] = raw["등락률"].astype(str).str.replace("%", "", regex=False).apply(_to_num)
    if "_시총억원" in raw.columns:
        raw["시가총액"] = raw["_시총억원"].apply(_to_num) * 1e8
        raw.drop(columns=["_시총억원"], inplace=True)
    if "_상장천주" in raw.columns:
        raw["상장주식수"] = raw["_상장천주"].apply(_to_num) * 1000
        raw.drop(columns=["_상장천주"], inplace=True)
    if "거래량" in raw.columns:
        raw["거래량"] = raw["거래량"].apply(_to_num)

    # 거래대금 추정: 거래량 × 종가
    if "종가" in raw.columns and "거래량" in raw.columns:
        raw["거래대금"] = raw["종가"] * raw["거래량"]

    raw = raw.set_index("ticker")
    keep = ["종목명", "종가", "등락률", "시가총액", "상장주식수", "거래량", "거래대금", "market"]
    return raw[[c for c in keep if c in raw.columns]]


# ── st.cache_data 래퍼 (Streamlit 미사용 시 fallback) ────────────────────────

def _make_cached(fn, **cache_kwargs):
    try:
        import streamlit as st
        return st.cache_data(**cache_kwargs)(fn)
    except Exception:
        return fn


def _raw_fetch_naver(sosok: int) -> pd.DataFrame:
    market = "KOSPI" if sosok == 0 else "KOSDAQ"
    return _fetch_naver_all_raw(sosok, market)


_fetch_naver_all = _make_cached(_raw_fetch_naver, ttl=3600, show_spinner=False)


# ── 공개 API ──────────────────────────────────────────────────────────────────

def _cap_impl(date_str: str, market: str) -> pd.DataFrame:
    if market == "ALL":
        return pd.concat([_fetch_naver_all(0), _fetch_naver_all(1)])
    return _fetch_naver_all(_SOSOK_MAP.get(market, 0))


fetch_market_cap = _make_cached(_cap_impl, ttl=3600, show_spinner=False)


def fetch_ohlcv(date_str: str, market: str) -> pd.DataFrame:
    """fetch_market_cap과 동일 데이터 소스."""
    return fetch_market_cap(date_str, market)


def fetch_investor_net(date_str: str, market: str) -> pd.DataFrame:
    """KRX 기반으로 현재 동작 불가 → 빈 DataFrame."""
    return pd.DataFrame()


def fetch_ticker_names(date_str: str, market: str) -> dict[str, str]:
    """종목코드 → 종목명 딕셔너리."""
    df = fetch_market_cap(date_str, market)
    if df.empty or "종목명" not in df.columns:
        return {}
    return df["종목명"].dropna().to_dict()


def fetch_historical_market_cap(dates: list[str], market: str) -> dict[str, pd.DataFrame]:
    """
    여러 날짜의 시가총액 데이터.
    Naver는 오늘 데이터만 제공 → 과거 날짜는 빈 DataFrame.
    반환: {date_str: DataFrame}
    """
    return {d: pd.DataFrame() for d in dates}


# ── 영업일 계산 ───────────────────────────────────────────────────────────────

def _biz_dates_impl(end_date: str, n: int) -> list[str]:
    """end_date 포함 최근 n 영업일(월~금) 목록. 'YYYYMMDD' 형식."""
    end = _parse_date(end_date)
    result = []
    cur = end
    while len(result) < n:
        if cur.weekday() < 5:  # 월=0 … 금=4
            result.append(cur.strftime("%Y%m%d"))
        cur -= timedelta(days=1)
    return list(reversed(result))


get_business_dates = _make_cached(_biz_dates_impl, ttl=86400, show_spinner=False)


def _latest_biz_impl() -> str:
    today = date.today()
    for delta in range(7):
        candidate = today - timedelta(days=delta)
        if candidate.weekday() < 5:
            return candidate.strftime("%Y%m%d")
    return today.strftime("%Y%m%d")


get_latest_business_date = _make_cached(_latest_biz_impl, ttl=3600, show_spinner=False)


# ── 개별 종목 (차트용) ────────────────────────────────────────────────────────

def _ticker_ohlcv_impl(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        from pykrx import stock as krx
        df = krx.get_market_ohlcv_by_date(start, end, ticker)
        df.index.name = "date"
        return df
    except Exception:
        return pd.DataFrame()


fetch_ticker_ohlcv = _make_cached(_ticker_ohlcv_impl, ttl=3600, show_spinner=False)


def fetch_ticker_investor(ticker: str, start: str, end: str) -> pd.DataFrame:
    """KRX 기반, 현재 동작 불가 → 빈 DataFrame."""
    return pd.DataFrame()
