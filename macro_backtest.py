"""
PTJ Macro Analogue Engine — Historical Validation Backtest
===========================================================
Tests whether the Phase 1+2 macro analogue engine correctly predicts:
  1. SPY market direction (up/down) over the next 6 and 12 months
  2. Which sector outperforms SPY over the next 6 months (Panel 3)

METHODOLOGY — strict no-lookahead
  At each test date T (every 6 months from 2000 to 2024):
    - SPY data sliced to [history_start → T]  (no future data)
    - FRED macro data sliced to [1990 → T]    (no future data)
    - Engine runs exactly as in live mode
    - Forward returns measured AFTER T using known history
    - Direction prediction vs actual direction scored

Panel 3 changes vs prior version:
  - Forward measurement horizon: 12mo → 6mo
    Rationale: non-overlapping test dates at 6-month steps
  - Pre-1999 analogues: now included via proxy stock baskets
    Rationale: prior version silently skipped the engine's best
    analogues (1994-1996) because ETFs didn't exist yet. The
    backtest was validating the wrong inputs. Proxy baskets
    (same as the live app) fix this.
  - PANEL3_END extended to 2023-12-31 (only needs 6mo forward)
  - Test sample roughly doubles vs prior run

This is a walk-forward backtest, not curve-fitted to any parameter.
The same engine parameters used in live trading are used here.

Run:  python macro_backtest.py
Open: http://localhost:5004
"""

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import json
import logging
import time

# =============================================================================
# SECTION 1 — Configuration
# =============================================================================

from dotenv import load_dotenv
import os
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_START    = "1993-01-01"
MACRO_START      = "1990-01-01"
BACKTEST_START   = "2000-01-01"   # first test date
BACKTEST_END     = "2023-06-30"   # last test date (needs 12mo of forward data)
STEP_MONTHS      = 6              # test every 6 months

# Windows to test — we test all three to find which is most predictive
TEST_WINDOWS     = [126, 189, 252]  # 6mo, 9mo, 12mo in trading days
DEFAULT_WINDOW   = 252

MIN_SEPARATION   = 126
MACRO_CANDIDATES = 20
TOP_N            = 3
ETF_INCEPTION    = pd.Timestamp("1999-01-01")

FRED_SERIES = {
    "fed_funds":   "FEDFUNDS",
    "yield_curve": "T10Y2Y",
    "cpi":         "CPIAUCSL",
    "hy_spread":   "BAMLH0A0HYM2",
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Disc",
    "XLP": "Consumer Staples",
    "XLB": "Materials",
    "XLU": "Utilities",
}

# Proxy stock baskets for pre-1999 analogue periods.
# Equal-weighted, large-cap names that existed and traded in the early 1990s.
# Mirrors the SECTORS dict in macro_analogue.py exactly so backtest and live
# app are testing the same thing.
SECTOR_PROXIES = {
    "XLK": ["MSFT", "IBM",  "TXN",  "HPQ",  "AMAT",
             "MU",   "ADI",  "KLAC", "LRCX", "NTAP",
             "GLW",  "CSCO", "ORCL", "SNX",  "CTSH"],
    "XLF": ["JPM",  "BAC",  "WFC",  "C",    "GS",
             "MS",   "AXP",  "USB",  "PNC",  "MET",
             "PRU",  "ALL",  "TRV",  "AFL",  "BK"],
    "XLE": ["XOM",  "CVX",  "SLB",  "HAL",  "COP",
             "OXY",  "DVN",  "APA",  "EOG",  "PSX",
             "VLO",  "MPC",  "PXD"],
    "XLV": ["JNJ",  "PFE",  "MRK",  "ABT",  "LLY",
             "MDT",  "BMY",  "AMGN", "GILD", "BAX",
             "BDX",  "SYK",  "BSX",  "HUM",  "CI"],
    "XLI": ["GE",   "MMM",  "HON",  "CAT",  "EMR",
             "ETN",  "PH",   "ROK",  "DOV",  "ITW",
             "DHR",  "AME",  "ROP",  "FAST", "GWW"],
    "XLY": ["MCD",  "DIS",  "HD",   "LOW",  "TGT",
             "F",    "YUM",  "MAR",  "NKE",  "SBUX",
             "WHR",  "LEN",  "PHM"],
    "XLP": ["PG",   "KO",   "PEP",  "WMT",  "CL",
             "GIS",  "CPB",  "HRL",  "MKC",  "SJM",
             "CAG",  "HSY",  "CHD",  "CLX",  "KMB"],
    "XLB": ["DD",   "PPG",  "SHW",  "NEM",  "FCX",
             "NUE",  "RS",   "VMC",  "MLM",  "ALB",
             "ECL",  "APD",  "LIN"],
    "XLU": ["NEE",  "DUK",  "SO",   "D",    "AEP",
             "EXC",  "XEL",  "ED",   "WEC",  "ES",
             "ETR",  "FE",   "PPL"],
}

# Cache for proxy stock data — fetched once, shared across all test dates
_proxy_cache = {}
_proxy_lock  = __import__("threading").Lock()


def fetch_proxy_data(ticker):
    """
    Fetch daily close price history for a single proxy stock.
    Cached so each stock is only downloaded once across all test dates.
    Returns a pd.Series indexed by date (tz-naive), or None on failure.
    """
    with _proxy_lock:
        if ticker in _proxy_cache:
            return _proxy_cache[ticker]

    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start="1990-01-01", auto_adjust=True)
        if df.empty or len(df) < 60:
            with _proxy_lock:
                _proxy_cache[ticker] = None
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            with _proxy_lock:
                _proxy_cache[ticker] = None
            return None

        series = df["close"].dropna()

        # Normalise to tz-naive — prevents comparison errors with FRED dates
        if series.index.tzinfo is not None:
            series.index = series.index.tz_convert(None)

        with _proxy_lock:
            _proxy_cache[ticker] = series
        log.debug("  Proxy %s: %d rows from %s", ticker, len(series),
                  series.index[0].strftime("%Y-%m-%d"))
        return series

    except Exception as e:
        log.debug("  Proxy %s fetch failed: %s", ticker, e)
        with _proxy_lock:
            _proxy_cache[ticker] = None
        return None


def compute_proxy_forward_return(ticker, from_date, fwd_days):
    """
    Return the fwd_days forward return for a single proxy stock from from_date.
    Returns None if data unavailable or insufficient.
    """
    series = fetch_proxy_data(ticker)
    if series is None:
        return None

    try:
        d = pd.Timestamp(from_date)
        if d.tzinfo is not None:
            d = d.tz_convert(None)

        idx = series.index
        pos_start = idx.searchsorted(d, side="left")
        if pos_start >= len(idx):
            return None

        pos_end = pos_start + fwd_days
        if pos_end >= len(idx):
            # Accept partial window if we have at least 80% of the days
            if (len(idx) - pos_start) < int(fwd_days * 0.8):
                return None
            pos_end = len(idx) - 1

        p0 = float(series.iloc[pos_start])
        p1 = float(series.iloc[pos_end])
        if p0 == 0 or np.isnan(p0) or np.isnan(p1):
            return None

        return round((p1 - p0) / p0, 4)

    except Exception:
        return None


def compute_sector_return_with_proxy(sector_key, from_date, fwd_days,
                                     full_sectors):
    """
    Compute a sector's forward return from from_date over fwd_days trading days.

    Strategy (mirrors macro_analogue.py):
      1. If analogue end_date >= ETF_INCEPTION and ETF data covers the period,
         use the ETF directly.
      2. Otherwise fall back to equal-weighted proxy basket. Require at least
         3 valid proxy stocks; return None if fewer available.

    Returns (return_value, method) or (None, None) on failure.
    """
    d = pd.Timestamp(from_date)
    if d.tzinfo is not None:
        d = d.tz_convert(None)

    # Attempt ETF first if data covers this period
    if d >= SECTOR_ETF_INCEPTION and sector_key in full_sectors:
        sec_df    = full_sectors[sector_key]
        sec_close = sec_df["close"].values.astype(float)
        sec_dates = sec_df.index
        if sec_dates.tzinfo is not None:
            sec_dates = sec_dates.tz_convert(None)

        if d >= sec_dates[0]:
            pos_s   = sec_dates.searchsorted(d, side="left")
            pos_end = pos_s + fwd_days
            if pos_end < len(sec_close):
                p0 = float(sec_close[pos_s])
                p1 = float(sec_close[pos_end])
                if p0 != 0 and not np.isnan(p0) and not np.isnan(p1):
                    return round((p1 - p0) / p0, 4), "etf"

    # Fall back to proxy basket
    proxies = SECTOR_PROXIES.get(sector_key, [])
    proxy_returns = []
    for ticker in proxies:
        ret = compute_proxy_forward_return(ticker, from_date, fwd_days)
        if ret is not None:
            proxy_returns.append(ret)

    if len(proxy_returns) < 3:
        return None, None

    return round(float(np.mean(proxy_returns)), 4), "proxy"

# =============================================================================
# SECTION 2 — App setup
# =============================================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# SECTION 3 — Data fetching (full history, fetched once)
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


_full_spy    = None
_full_macro  = None
_full_sectors = {}


def fetch_full_spy():
    global _full_spy
    if _full_spy is not None:
        return _full_spy
    log.info("Fetching full SPY history...")
    tkr = yf.Ticker("SPY")
    df  = tkr.history(start=HISTORY_START, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    _full_spy = df[["close"]].dropna()
    log.info("SPY: %d rows (%s to %s)", len(_full_spy),
             _full_spy.index[0].date(), _full_spy.index[-1].date())
    return _full_spy


def fetch_full_macro():
    global _full_macro
    if _full_macro is not None:
        return _full_macro
    log.info("Fetching FRED macro data...")
    fred    = Fred(api_key=FRED_API_KEY)
    raw     = {}
    for name, series_id in FRED_SERIES.items():
        s = fred.get_series(series_id, observation_start=MACRO_START)
        s.index = pd.to_datetime(s.index)
        raw[name] = s

    last_date = max(s.index[-1] for s in raw.values())
    daily_idx = pd.date_range(MACRO_START, last_date, freq="D")
    aligned   = {name: s.reindex(daily_idx).ffill() for name, s in raw.items()}
    aligned["cpi_yoy"] = aligned["cpi"].pct_change(periods=365) * 100

    _full_macro = aligned
    log.info("Macro: loaded %d series", len(_full_macro))
    return _full_macro


def fetch_full_sectors():
    global _full_sectors
    if _full_sectors:
        return _full_sectors
    log.info("Fetching sector ETF history...")
    for ticker in SECTOR_ETFS:
        try:
            tkr = yf.Ticker(ticker)
            df  = tkr.history(start="1998-01-01", auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            _full_sectors[ticker] = df[["close"]].dropna()
            log.info("  %s: %d rows", ticker, len(_full_sectors[ticker]))
        except Exception as e:
            log.warning("  %s failed: %s", ticker, e)
    return _full_sectors


# =============================================================================
# SECTION 4 — Engine functions (same as macro_analogue.py, self-contained)
# =============================================================================

def build_return_matrix(close_arr, window):
    close_arr = np.asarray(close_arr, dtype=float)
    n_windows = len(close_arr) - window + 1
    if n_windows <= 0:
        return np.empty((0, window), dtype=float)
    idx   = np.arange(window)[None, :] + np.arange(n_windows)[:, None]
    mat   = close_arr[idx]
    first = mat[:, 0:1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.where(first != 0, mat / first - 1.0, 0.0)
    return mat


def vectorised_pearson(target, matrix):
    t   = target - target.mean()
    t_n = t / (np.sqrt((t ** 2).sum()) + 1e-10)
    m   = matrix - matrix.mean(axis=1, keepdims=True)
    m_n = m / (np.sqrt((m ** 2).sum(axis=1, keepdims=True)) + 1e-10)
    return (m_n @ t_n).clip(-1, 1)


def safe_val(series, date):
    """Point-in-time lookup — strip timezone, use last known value."""
    try:
        cleaned = series.dropna().sort_index()
        if cleaned.empty:
            return np.nan
        d = pd.Timestamp(date)
        if d.tzinfo is not None:
            d = d.tz_convert(None)
        if d >= cleaned.index[-1]:
            return float(cleaned.iloc[-1])
        if d < cleaned.index[0]:
            return np.nan
        pos = cleaned.index.searchsorted(d, side="right") - 1
        return float(cleaned.iloc[pos]) if pos >= 0 else np.nan
    except Exception:
        return np.nan


def get_macro_snapshot(macro, date):
    """Point-in-time macro snapshot — uses only data up to date."""
    try:
        d     = pd.Timestamp(date)
        if d.tzinfo is not None:
            d = d.tz_convert(None)
        d_1yr = d - pd.DateOffset(years=1)

        fed_now  = safe_val(macro["fed_funds"],   d)
        fed_1yr  = safe_val(macro["fed_funds"],   d_1yr)
        yc_now   = safe_val(macro["yield_curve"], d)
        cpi_now  = safe_val(macro["cpi_yoy"],     d)
        cpi_1yr  = safe_val(macro["cpi_yoy"],     d_1yr)
        hy_now   = safe_val(macro["hy_spread"],   d)

        if any(np.isnan(v) for v in [fed_now, fed_1yr, cpi_now, cpi_1yr]):
            return None
        if np.isnan(yc_now): yc_now = 0.0
        if np.isnan(hy_now): hy_now = 4.0

        return {
            "fed_funds":   round(fed_now,          2),
            "fed_change":  round(fed_now - fed_1yr, 2),
            "yield_curve": round(yc_now,            2),
            "cpi_yoy":     round(cpi_now,           2),
            "cpi_change":  round(cpi_now - cpi_1yr, 2),
            "hy_spread":   round(hy_now,            2),
        }
    except Exception:
        return None


def score_regime_similarity(snap_a, snap_b):
    SCALES = {
        "fed_funds": 5.0, "fed_change": 3.0, "yield_curve": 3.0,
        "cpi_yoy": 6.0, "cpi_change": 4.0, "hy_spread": 6.0,
    }
    sq_dist = sum(
        ((snap_a[k] - snap_b[k]) / scale) ** 2
        for k, scale in SCALES.items()
    )
    return round(float(np.exp(-sq_dist / 2.0)), 4)


def find_analogues_at_date(spy_close, spy_dates, macro, test_date, window):
    """
    Run the Phase 1+2 analogue engine using only data up to test_date.
    This is the strict no-lookahead version for backtesting.

    Returns list of top TOP_N analogue dicts, or empty list on failure.
    """
    # Slice SPY to only data up to test_date (strict)
    d = pd.Timestamp(test_date)
    if d.tzinfo is not None:
        d = d.tz_convert(None)

    mask = spy_dates <= d
    if mask.sum() < window * 2:
        return []

    close_arr = spy_close[mask]
    dates_arr = spy_dates[mask]
    n         = len(close_arr)

    # Current pattern = last `window` bars
    cur_window = close_arr[n - window:]
    cur_target = (cur_window / cur_window[0]) - 1.0

    # Search space: must end 2x window before test_date
    search_end = n - window * 2
    if search_end < window:
        return []

    # Stage 1: vectorised price similarity
    return_mat   = build_return_matrix(close_arr[:search_end + window - 1], window)
    return_mat   = return_mat[:search_end]
    if len(return_mat) == 0:
        return []

    similarities = vectorised_pearson(cur_target, return_mat)
    order        = np.argsort(similarities)[::-1]

    candidates         = []
    used_start_indices = []
    for idx in order:
        price_sim = float(similarities[idx])
        if price_sim <= 0:
            break
        if any(abs(int(idx) - used) < MIN_SEPARATION for used in used_start_indices):
            continue
        candidates.append({
            "start_idx": int(idx),
            "end_idx":   int(idx) + window - 1,
            "price_sim": price_sim,
        })
        used_start_indices.append(int(idx))
        if len(candidates) >= MACRO_CANDIDATES:
            break

    if not candidates:
        return []

    # Stage 2: macro regime scoring
    today_snap   = get_macro_snapshot(macro, test_date)
    macro_active = today_snap is not None

    scored = []
    for c in candidates:
        price_sim = c["price_sim"]
        end_date  = dates_arr[c["end_idx"]]

        # Slice macro to only data up to analogue end date
        if macro_active:
            hist_snap  = get_macro_snapshot(macro, end_date)
            regime_sim = score_regime_similarity(today_snap, hist_snap) \
                         if hist_snap else 0.0
            combined   = 0.5 * price_sim + 0.5 * regime_sim
        else:
            regime_sim = None
            combined   = price_sim

        scored.append({**c,
                       "regime_sim": regime_sim,
                       "combined":   combined,
                       "end_date":   end_date})

    scored.sort(key=lambda x: x["combined"], reverse=True)

    analogues = []
    for rank, c in enumerate(scored[:TOP_N], start=1):
        analogues.append({
            "rank":       rank,
            "price_sim":  round(c["price_sim"],  4),
            "regime_sim": round(c["regime_sim"], 4) if c["regime_sim"] else None,
            "combined":   round(c["combined"],   4),
            "start_date": dates_arr[c["start_idx"]].strftime("%Y-%m-%d"),
            "end_date":   c["end_date"].strftime("%Y-%m-%d"),
        })

    return analogues


def predict_direction(analogues, spy_close, spy_dates, full_spy_close, full_spy_dates):
    """
    Given analogues found at test_date, compute the weighted average forward
    return implied by those analogues — this is the direction prediction.

    Returns: predicted_direction ('UP'/'DOWN'), weighted_avg_return, confidence
    """
    if not analogues:
        return None, None, None

    weights  = []
    fwd_rets = []

    for a in analogues:
        end_date = pd.Timestamp(a["end_date"])
        if end_date.tzinfo is not None:
            end_date = end_date.tz_convert(None)

        # Find end_date position in full SPY data
        full_dates_naive = full_spy_dates
        if full_spy_dates.tzinfo is not None:
            full_dates_naive = full_spy_dates.tz_convert(None)

        pos_end = full_dates_naive.searchsorted(end_date, side="left")
        pos_12mo = pos_end + 252

        if pos_12mo >= len(full_spy_close):
            continue

        p0  = float(full_spy_close[pos_end])
        p12 = float(full_spy_close[pos_12mo])
        if p0 == 0:
            continue

        fwd_ret = (p12 - p0) / p0
        fwd_rets.append(fwd_ret)
        weights.append(a["combined"] ** 2)

    if not fwd_rets:
        return None, None, None

    weights = np.array(weights)
    weights /= weights.sum()
    weighted_ret = float(np.dot(weights, fwd_rets))

    direction  = "UP" if weighted_ret > 0 else "DOWN"
    confidence = abs(weighted_ret)

    return direction, round(weighted_ret, 4), round(confidence, 4)


def get_actual_returns(test_date, full_spy, full_sectors):
    """
    Measure actual SPY and sector returns at +6mo and +12mo from test_date.
    Returns dict of actual outcomes — this uses full history (ground truth).
    """
    d = pd.Timestamp(test_date)
    if d.tzinfo is not None:
        d = d.tz_convert(None)

    spy_close  = full_spy["close"].values.astype(float)
    spy_dates  = full_spy.index
    if spy_dates.tzinfo is not None:
        spy_dates = spy_dates.tz_convert(None)

    def get_return(close_arr, dates, from_date, months):
        pos0 = dates.searchsorted(from_date, side="left")
        if pos0 >= len(dates):
            return None
        future_date = from_date + relativedelta(months=months)
        pos1 = dates.searchsorted(future_date, side="left")
        if pos1 >= len(dates):
            pos1 = len(dates) - 1
        p0 = float(close_arr[pos0])
        p1 = float(close_arr[pos1])
        return round((p1 - p0) / p0, 4) if p0 != 0 else None

    spy_6mo  = get_return(spy_close, spy_dates, d, 6)
    spy_12mo = get_return(spy_close, spy_dates, d, 12)

    actual_direction_6mo  = "UP" if spy_6mo  and spy_6mo  > 0 else "DOWN" if spy_6mo  else None
    actual_direction_12mo = "UP" if spy_12mo and spy_12mo > 0 else "DOWN" if spy_12mo else None

    # Sector returns
    sector_returns = {}
    for ticker, name in SECTOR_ETFS.items():
        if ticker not in full_sectors:
            continue
        sec_df    = full_sectors[ticker]
        sec_close = sec_df["close"].values.astype(float)
        sec_dates = sec_df.index
        if sec_dates.tzinfo is not None:
            sec_dates = sec_dates.tz_convert(None)

        if d < sec_dates[0]:
            continue

        r6  = get_return(sec_close, sec_dates, d, 6)
        r12 = get_return(sec_close, sec_dates, d, 12)

        sector_returns[ticker] = {
            "name":       name,
            "return_6mo":  r6,
            "return_12mo": r12,
            "rel_6mo":  round(r6  - spy_6mo,  4) if r6  and spy_6mo  else None,
            "rel_12mo": round(r12 - spy_12mo, 4) if r12 and spy_12mo else None,
            "beat_spy_6mo":  r6  > spy_6mo  if r6  and spy_6mo  else None,
            "beat_spy_12mo": r12 > spy_12mo if r12 and spy_12mo else None,
        }

    # Which sector actually led?
    valid_12 = {k: v for k, v in sector_returns.items()
                if v["rel_12mo"] is not None}
    best_sector_12mo = max(valid_12, key=lambda k: valid_12[k]["rel_12mo"]) \
                       if valid_12 else None

    valid_6 = {k: v for k, v in sector_returns.items()
               if v["rel_6mo"] is not None}
    best_sector_6mo = max(valid_6, key=lambda k: valid_6[k]["rel_6mo"]) \
                      if valid_6 else None

    return {
        "spy_6mo":            spy_6mo,
        "spy_12mo":           spy_12mo,
        "actual_dir_6mo":     actual_direction_6mo,
        "actual_dir_12mo":    actual_direction_12mo,
        "sector_returns":     sector_returns,
        "best_sector_6mo":    best_sector_6mo,
        "best_sector_12mo":   best_sector_12mo,
    }


# =============================================================================
# SECTION 5 — Backtest runner
# =============================================================================

def run_backtest(window=DEFAULT_WINDOW):
    """
    Walk forward from BACKTEST_START to BACKTEST_END in STEP_MONTHS steps.
    At each date, run the full Phase 1+2 engine using only historical data.
    Record predictions and actual outcomes.

    Yields SSE-formatted result dicts as each test date completes.
    """
    # Fetch all data upfront
    full_spy     = fetch_full_spy()
    full_macro   = fetch_full_macro()
    full_sectors = fetch_full_sectors()

    spy_close = full_spy["close"].values.astype(float)
    spy_dates = full_spy.index
    if spy_dates.tzinfo is not None:
        spy_dates_naive = spy_dates.tz_convert(None)
    else:
        spy_dates_naive = spy_dates

    # Generate test dates
    test_dates = []
    current = pd.Timestamp(BACKTEST_START)
    end     = pd.Timestamp(BACKTEST_END)
    while current <= end:
        test_dates.append(current)
        current += relativedelta(months=STEP_MONTHS)

    results = []
    for i, test_date in enumerate(test_dates):
        log.info("Testing %s (%d/%d)...", test_date.date(), i+1, len(test_dates))

        # Slice macro to test_date (strict no-lookahead)
        d_naive = test_date
        macro_sliced = {}
        for name, series in full_macro.items():
            s = series.copy()
            s_idx = s.index
            macro_sliced[name] = s[s_idx <= d_naive]

        # Run analogue engine (no-lookahead)
        analogues = find_analogues_at_date(
            spy_close, spy_dates_naive, macro_sliced, test_date, window)

        # Direction prediction from analogues
        pred_dir, pred_ret, confidence = predict_direction(
            analogues, spy_close, spy_dates_naive,
            spy_close, spy_dates_naive)

        # Actual outcomes (ground truth — uses full history)
        actuals = get_actual_returns(test_date, full_spy, full_sectors)

        # Score the prediction
        correct_6mo  = (pred_dir == actuals["actual_dir_6mo"])  if pred_dir and actuals["actual_dir_6mo"]  else None
        correct_12mo = (pred_dir == actuals["actual_dir_12mo"]) if pred_dir and actuals["actual_dir_12mo"] else None

        # Top predicted sector (from analogue end dates — which sector led
        # historically in those analogue forward periods)
        # We use the sector rotation logic: which sector beat SPY most
        # consistently across the top analogues' forward periods
        predicted_sector = predict_top_sector(analogues, full_sectors, full_spy)

        sector_correct_12mo = None
        if predicted_sector and actuals["best_sector_12mo"]:
            sector_correct_12mo = (predicted_sector == actuals["best_sector_12mo"])

        result = {
            "test_date":          test_date.strftime("%Y-%m-%d"),
            "window_days":        window,
            "n_analogues":        len(analogues),
            "top_analogues":      analogues[:3],
            "predicted_dir":      pred_dir,
            "predicted_return":   pred_ret,
            "confidence":         confidence,
            "predicted_sector":   predicted_sector,
            "actual_spy_6mo":     actuals["spy_6mo"],
            "actual_spy_12mo":    actuals["spy_12mo"],
            "actual_dir_6mo":     actuals["actual_dir_6mo"],
            "actual_dir_12mo":    actuals["actual_dir_12mo"],
            "correct_6mo":        correct_6mo,
            "correct_12mo":       correct_12mo,
            "best_sector_6mo":    actuals["best_sector_6mo"],
            "best_sector_12mo":   actuals["best_sector_12mo"],
            "sector_correct_12mo": sector_correct_12mo,
            "sector_returns":     actuals["sector_returns"],
            "macro_snap":         get_macro_snapshot(macro_sliced, test_date),
        }

        results.append(result)
        yield result

    # Summary stats
    yield compute_summary(results)


def predict_top_sector(analogues, full_sectors, full_spy):
    """
    Given analogues, predict which sector will outperform SPY.
    For each analogue's forward period, check which sector beat SPY.
    Return the sector that most consistently beat SPY across analogues.
    """
    if not analogues:
        return None

    sector_scores = {ticker: 0 for ticker in SECTOR_ETFS}

    spy_close  = full_spy["close"].values.astype(float)
    spy_dates  = full_spy.index
    if spy_dates.tzinfo is not None:
        spy_dates = spy_dates.tz_convert(None)

    for a in analogues:
        end_date = pd.Timestamp(a["end_date"])
        if end_date.tzinfo is not None:
            end_date = end_date.tz_convert(None)

        pos_end  = spy_dates.searchsorted(end_date, side="left")
        pos_12mo = pos_end + 252
        if pos_12mo >= len(spy_close):
            continue

        spy_ret = (spy_close[pos_12mo] - spy_close[pos_end]) / spy_close[pos_end]

        for ticker in SECTOR_ETFS:
            if ticker not in full_sectors:
                continue
            sec_df    = full_sectors[ticker]
            sec_close = sec_df["close"].values.astype(float)
            sec_dates = sec_df.index
            if sec_dates.tzinfo is not None:
                sec_dates = sec_dates.tz_convert(None)

            if end_date < sec_dates[0]:
                continue

            pos_s  = sec_dates.searchsorted(end_date, side="left")
            pos_s12 = pos_s + 252
            if pos_s12 >= len(sec_close):
                continue

            sec_ret = (sec_close[pos_s12] - sec_close[pos_s]) / sec_close[pos_s]
            if sec_ret > spy_ret:
                sector_scores[ticker] += a["combined"]  # weight by combined score

    if not any(sector_scores.values()):
        return None

    return max(sector_scores, key=sector_scores.get)


def compute_summary(results):
    """Compute aggregate accuracy statistics across all test dates."""
    with_pred  = [r for r in results if r["predicted_dir"] is not None]
    with_12mo  = [r for r in with_pred if r["correct_12mo"] is not None]
    with_6mo   = [r for r in with_pred if r["correct_6mo"]  is not None]
    with_sector = [r for r in results if r["sector_correct_12mo"] is not None]

    dir_acc_12mo = np.mean([r["correct_12mo"] for r in with_12mo]) if with_12mo else None
    dir_acc_6mo  = np.mean([r["correct_6mo"]  for r in with_6mo])  if with_6mo  else None
    sec_acc_12mo = np.mean([r["sector_correct_12mo"] for r in with_sector]) if with_sector else None

    # Magnitude calibration: predicted return vs actual return
    mag_pairs = [(r["predicted_return"], r["actual_spy_12mo"])
                 for r in with_12mo if r["predicted_return"] and r["actual_spy_12mo"]]
    calibration = None
    if mag_pairs:
        pred_rets   = [p[0] for p in mag_pairs]
        actual_rets = [p[1] for p in mag_pairs]
        correlation = float(np.corrcoef(pred_rets, actual_rets)[0, 1]) if len(mag_pairs) > 2 else None
        avg_pred    = float(np.mean(pred_rets))
        avg_actual  = float(np.mean(actual_rets))
        calibration = {
            "correlation":  round(correlation, 3) if correlation else None,
            "avg_predicted": round(avg_pred,   3),
            "avg_actual":    round(avg_actual,  3),
            "bias":          round(avg_pred - avg_actual, 3),
            "pairs":         mag_pairs,
        }

    # By predicted direction — accuracy when predicting UP vs DOWN
    up_calls   = [r for r in with_12mo if r["predicted_dir"] == "UP"]
    down_calls = [r for r in with_12mo if r["predicted_dir"] == "DOWN"]

    return {
        "type":             "summary",
        "total_tests":      len(results),
        "tests_with_pred":  len(with_pred),
        "dir_acc_6mo":      round(dir_acc_6mo  * 100, 1) if dir_acc_6mo  else None,
        "dir_acc_12mo":     round(dir_acc_12mo * 100, 1) if dir_acc_12mo else None,
        "sector_acc_12mo":  round(sec_acc_12mo * 100, 1) if sec_acc_12mo else None,
        "up_calls":         len(up_calls),
        "up_accuracy":      round(np.mean([r["correct_12mo"] for r in up_calls]) * 100, 1) if up_calls else None,
        "down_calls":       len(down_calls),
        "down_accuracy":    round(np.mean([r["correct_12mo"] for r in down_calls]) * 100, 1) if down_calls else None,
        "calibration":      calibration,
        "results":          results,
    }


# =============================================================================
# SECTION 5b — Panel 3 validation: sector rotation method
# =============================================================================
#
# METHODOLOGY
# -----------
# At each test date T from 2002-01-01 to 2022-06-30 (needs 12mo forward data
# AND sector ETFs must exist — ETFs launched late 1998, so 2002 gives 3+ years
# of ETF history for the analogue forward periods):
#
#   1. Run full Phase 1+2 engine at T (strict no-lookahead)
#   2. For each of the top 3 analogues, find their historical end dates
#   3. For each analogue end date, measure what each sector ETF actually
#      returned in the 12 months following that date (historical fact)
#   4. Rank sectors by consistency x magnitude — same formula as live Panel 3
#   5. Record: what was Panel 3's top sector recommendation?
#   6. Measure: did that sector actually outperform SPY in the 12 months
#      following T? (ground truth)
#   7. Also measure: did the TOP sector from Panel 3 beat a random sector pick?
#
# Panel 3 methodology (updated):
#   Use proxy stock baskets for analogue periods before ETF inception (pre-1999).
#   This matches what the live app does and ensures the engine's best analogues
#   (1994-1996) actually contribute to sector ranking instead of being skipped.
#   Forward measurement is now 6 months so test dates don't overlap in their
#   measurement windows (step = 6mo, horizon = 6mo → independent samples).

PANEL3_START        = "2000-01-01"   # first test date (same as main backtest)
PANEL3_END          = "2023-12-31"   # last test date (only needs 6mo forward)
PANEL3_FWD_DAYS     = 126            # 6 months in trading days
SECTOR_ETF_INCEPTION = pd.Timestamp("1999-01-01")  # ETFs reliably available


def compute_panel3_sector_score(analogues, full_sectors, full_spy):
    """
    Given analogues found at a test date, compute the Panel 3 sector ranking.

    Changes from prior version:
    - Pre-1999 analogue periods now included via proxy stock baskets
      (previously skipped entirely — this was the core gap in the validation)
    - Forward horizon: 6 months (PANEL3_FWD_DAYS=126) not 12 months
      Reason: test dates step every 6mo, so 6mo horizon = non-overlapping samples

    Ranking: consistency × magnitude (same formula as live app)
      score = weighted_avg_relative_return × consistency_multiplier
      consistency_multiplier: 3/3→1.0, 2/3→0.67, 1/3→0.33, 0/3→0.0

    Returns (scores_dict, valid_analogue_count, method_counts)
    """
    spy_close  = full_spy["close"].values.astype(float)
    spy_dates  = full_spy.index
    if spy_dates.tzinfo is not None:
        spy_dates = spy_dates.tz_convert(None)

    # sector_key → list of {rel_return, outperformed, weight, method}
    sector_data = {k: [] for k in SECTOR_ETFS}
    valid_analogues = 0
    method_counts   = {"etf": 0, "proxy": 0, "skipped": 0}

    for a in analogues:
        end_date = pd.Timestamp(a["end_date"])
        if end_date.tzinfo is not None:
            end_date = end_date.tz_convert(None)

        # SPY forward return from analogue end date over 6 months
        pos_spy = spy_dates.searchsorted(end_date, side="left")
        pos_6mo = pos_spy + PANEL3_FWD_DAYS
        if pos_6mo >= len(spy_close):
            method_counts["skipped"] += 1
            continue

        p0_spy  = float(spy_close[pos_spy])
        p6_spy  = float(spy_close[pos_6mo])
        if p0_spy == 0:
            method_counts["skipped"] += 1
            continue
        spy_ret = (p6_spy - p0_spy) / p0_spy

        # Sector returns — ETF if available, proxy basket otherwise
        analogue_contributed = False
        for sector_key in SECTOR_ETFS:
            sec_ret, method = compute_sector_return_with_proxy(
                sector_key, end_date, PANEL3_FWD_DAYS, full_sectors)

            if sec_ret is None:
                continue

            rel_ret = sec_ret - spy_ret
            sector_data[sector_key].append({
                "rel_return":   round(rel_ret, 4),
                "outperformed": rel_ret > 0,
                "weight":       a["combined"] ** 2,
                "method":       method,
            })
            analogue_contributed = True
            method_counts[method] = method_counts.get(method, 0) + 1

        if analogue_contributed:
            valid_analogues += 1

    if valid_analogues == 0:
        return {}, 0, method_counts

    # Rank sectors: consistency × magnitude (identical to live app scoring)
    scores = {}
    for sector_key, entries in sector_data.items():
        if not entries:
            continue

        n       = len(entries)
        n_out   = sum(1 for e in entries if e["outperformed"])
        weights = np.array([e["weight"] for e in entries])
        rels    = np.array([e["rel_return"] for e in entries])
        weights /= weights.sum()
        avg_rel  = float(np.dot(weights, rels))

        # Consistency multiplier — same as live app
        if   n_out == n:   mult = 1.00
        elif n_out >= n-1: mult = 0.67
        elif n_out == 1:   mult = 0.33
        else:              mult = 0.00

        scores[sector_key] = round(avg_rel * mult, 4)

    # Sort descending
    sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return sorted_scores, valid_analogues, method_counts


def run_panel3_backtest(window=DEFAULT_WINDOW):
    """
    Walk forward from PANEL3_START to PANEL3_END running the full Phase 1+2
    engine, then computing the Panel 3 sector recommendation at each date.

    Now uses proxy baskets for pre-1999 analogue periods and measures sector
    performance at 6 months forward (matching the 6-month step interval so
    test samples are non-overlapping).
    """
    full_spy     = fetch_full_spy()
    full_macro   = fetch_full_macro()
    full_sectors = fetch_full_sectors()

    spy_close = full_spy["close"].values.astype(float)
    spy_dates = full_spy.index
    if spy_dates.tzinfo is not None:
        spy_dates_naive = spy_dates.tz_convert(None)
    else:
        spy_dates_naive = spy_dates

    # Pre-fetch all proxy stocks so they're cached before the main loop
    log.info("Pre-fetching proxy stock data...")
    all_proxies = {t for tickers in SECTOR_PROXIES.values() for t in tickers}
    for ticker in sorted(all_proxies):
        fetch_proxy_data(ticker)
    log.info("Proxy prefetch complete (%d tickers)", len(all_proxies))

    # Generate test dates
    test_dates = []
    current = pd.Timestamp(PANEL3_START)
    end     = pd.Timestamp(PANEL3_END)
    while current <= end:
        test_dates.append(current)
        current += relativedelta(months=STEP_MONTHS)

    results = []

    for i, test_date in enumerate(test_dates):
        log.info("Panel3 [%d/%d] %s ...", i+1, len(test_dates), test_date.date())

        # Strict no-lookahead macro slice
        d_naive = test_date
        macro_sliced = {}
        for name, series in full_macro.items():
            s = series.copy()
            macro_sliced[name] = s[s.index <= d_naive]

        # Phase 1+2 analogue engine
        analogues = find_analogues_at_date(
            spy_close, spy_dates_naive, macro_sliced, test_date, window)

        if not analogues:
            results.append({
                "test_date": test_date.strftime("%Y-%m-%d"),
                "skipped":   True,
                "reason":    "No analogues found",
            })
            yield results[-1]
            continue

        # Panel 3 sector ranking — now includes proxy baskets
        sector_scores, valid_analogues, method_counts = compute_panel3_sector_score(
            analogues, full_sectors, full_spy)

        if not sector_scores or valid_analogues == 0:
            results.append({
                "test_date":  test_date.strftime("%Y-%m-%d"),
                "skipped":    True,
                "reason":     "Insufficient sector data for all analogues",
                "analogues":  analogues,
            })
            yield results[-1]
            continue

        top_sector = next(iter(sector_scores))
        top_score  = sector_scores[top_sector]

        # Actual outcomes: measure at 6mo forward
        actuals = get_actual_returns(test_date, full_spy, full_sectors)

        recommended_rel_6mo  = actuals["sector_returns"].get(top_sector, {}).get("rel_6mo")
        recommended_beat_spy = actuals["sector_returns"].get(top_sector, {}).get("beat_spy_6mo")
        actual_best_sector   = actuals["best_sector_6mo"]
        got_best_sector      = (top_sector == actual_best_sector)

        # Rank of recommended sector among all actual 6mo performances
        valid_6 = {k: v["rel_6mo"] for k, v in actuals["sector_returns"].items()
                   if v.get("rel_6mo") is not None}
        sector_rank = None
        if valid_6 and top_sector in valid_6:
            sorted_secs = sorted(valid_6.items(), key=lambda x: x[1], reverse=True)
            sector_rank = next((j+1 for j, (k, _) in enumerate(sorted_secs)
                                if k == top_sector), None)

        result = {
            "test_date":            test_date.strftime("%Y-%m-%d"),
            "skipped":              False,
            "window_days":          window,
            "n_analogues":          len(analogues),
            "valid_analogues":      valid_analogues,
            "method_counts":        method_counts,   # how many ETF vs proxy
            "top_analogues":        analogues[:3],
            "sector_scores":        sector_scores,
            "top_recommendation":   top_sector,
            "top_score":            top_score,
            "recommended_beat_spy": recommended_beat_spy,
            "recommended_rel_6mo":  recommended_rel_6mo,
            "got_best_sector":      got_best_sector,
            "actual_best_sector":   actual_best_sector,
            "sector_rank":          sector_rank,
            "actual_spy_6mo":       actuals["spy_6mo"],
            "all_sector_actual": {
                k: {"rel_6mo": v["rel_6mo"], "beat_spy": v["beat_spy_6mo"]}
                for k, v in actuals["sector_returns"].items()
                if v.get("rel_6mo") is not None
            },
            "macro_snap": get_macro_snapshot(macro_sliced, test_date),
        }

        results.append(result)
        yield result

    yield compute_panel3_summary(results)


def compute_panel3_summary(results):
    """
    Compute Panel 3 validation summary statistics.
    Now measures at 6-month forward horizon and reports ETF vs proxy usage.
    """
    valid = [r for r in results if not r.get("skipped")]

    if not valid:
        return {"type": "panel3_summary", "error": "No valid test dates"}

    beat_spy = [r for r in valid if r.get("recommended_beat_spy") is True]
    got_best = [r for r in valid if r.get("got_best_sector") is True]

    beat_rate = len(beat_spy) / len(valid)
    best_rate = len(got_best) / len(valid)

    rel_rets = [r["recommended_rel_6mo"] for r in valid
                if r.get("recommended_rel_6mo") is not None]
    avg_rel  = float(np.mean(rel_rets)) if rel_rets else None

    ranks    = [r["sector_rank"] for r in valid if r.get("sector_rank") is not None]
    avg_rank = float(np.mean(ranks)) if ranks else None

    # Per-sector breakdown
    by_sector = {}
    for r in valid:
        s = r.get("top_recommendation")
        if not s:
            continue
        if s not in by_sector:
            by_sector[s] = {"times_recommended": 0, "times_beat_spy": 0, "rel_rets": []}
        by_sector[s]["times_recommended"] += 1
        if r.get("recommended_beat_spy") is True:
            by_sector[s]["times_beat_spy"] += 1
        if r.get("recommended_rel_6mo") is not None:
            by_sector[s]["rel_rets"].append(r["recommended_rel_6mo"])

    for s in by_sector:
        d = by_sector[s]
        d["beat_rate"] = round(d["times_beat_spy"] / d["times_recommended"] * 100, 1)
        d["avg_rel"]   = round(float(np.mean(d["rel_rets"])) * 100, 2) if d["rel_rets"] else None

    # Random baseline: average fraction of sectors that beat SPY each period
    all_beat_rates = []
    for r in valid:
        for k, v in r.get("all_sector_actual", {}).items():
            if v.get("beat_spy") is not None:
                all_beat_rates.append(1 if v["beat_spy"] else 0)
    random_baseline = float(np.mean(all_beat_rates)) * 100 if all_beat_rates else 50.0

    # Aggregate ETF vs proxy method counts across all test dates
    total_etf   = sum(r.get("method_counts", {}).get("etf",   0) for r in valid)
    total_proxy = sum(r.get("method_counts", {}).get("proxy", 0) for r in valid)

    return {
        "type":             "panel3_summary",
        "horizon":          "6mo",
        "total_tests":      len(valid),
        "skipped":          len(results) - len(valid),
        "beat_spy_rate":    round(beat_rate * 100, 1),
        "got_best_rate":    round(best_rate * 100, 1),
        "avg_rel_return":   round(avg_rel * 100, 2) if avg_rel is not None else None,
        "avg_rank":         round(avg_rank, 1) if avg_rank else None,
        "random_baseline":  round(random_baseline, 1),
        "etf_observations": total_etf,
        "proxy_observations": total_proxy,
        "by_sector":        by_sector,
        "results":          valid,
    }



@app.route("/")
def index():
    return render_template("macro_backtest.html")


@app.route("/api/backtest/stream")
def api_backtest_stream():
    """
    SSE endpoint — streams one result per test date as it completes.
    Expect 5-15 minutes for a full run.
    """
    window = int(request.args.get("window", DEFAULT_WINDOW))
    window = max(63, min(window, 504))

    def generate():
        try:
            test_dates = []
            current = pd.Timestamp(BACKTEST_START)
            while current <= pd.Timestamp(BACKTEST_END):
                test_dates.append(current)
                current += relativedelta(months=STEP_MONTHS)

            yield f"data: {json.dumps({'type':'start','total':len(test_dates),'window':window}, cls=NumpyEncoder)}\n\n"

            completed = 0
            t0        = time.time()

            for result in run_backtest(window):
                if result.get("type") == "summary":
                    payload = json.dumps({"type": "summary", "data": result},
                                         cls=NumpyEncoder)
                    yield f"data: {payload}\n\n"
                else:
                    completed += 1
                    elapsed = round(time.time() - t0, 1)
                    payload = json.dumps({
                        "type":      "result",
                        "completed": completed,
                        "total":     len(test_dates),
                        "elapsed":   elapsed,
                        "result":    result,
                    }, cls=NumpyEncoder)
                    yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'type':'done','elapsed':round(time.time()-t0,1)})}\n\n"

        except GeneratorExit:
            log.info("SSE client disconnected")
        except Exception as e:
            log.error("SSE error: %s", e)
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        },
    )


@app.route("/api/panel3/stream")
def api_panel3_stream():
    """
    SSE endpoint for Panel 3 sector rotation validation.
    Tests whether the Panel 3 method (historical sector returns in analogue
    periods) predicts which sector will outperform SPY at the 6-month horizon.

    Updated methodology:
    - Proxy stock baskets used for pre-1999 analogue periods (no more skipping)
    - 6-month forward measurement (non-overlapping with 6-month step interval)
    - Test range: 2000-01-01 to 2023-12-31 (~48 test dates vs prior 37)
    """
    window = int(request.args.get("window", DEFAULT_WINDOW))
    window = max(63, min(window, 504))

    def generate():
        try:
            test_dates = []
            current = pd.Timestamp(PANEL3_START)
            while current <= pd.Timestamp(PANEL3_END):
                test_dates.append(current)
                current += relativedelta(months=STEP_MONTHS)

            yield f"data: {json.dumps({'type':'start','total':len(test_dates),'window':window}, cls=NumpyEncoder)}\n\n"

            completed = 0
            t0        = time.time()

            for result in run_panel3_backtest(window):
                if result.get("type") == "panel3_summary":
                    yield f"data: {json.dumps({'type':'panel3_summary','data':result}, cls=NumpyEncoder)}\n\n"
                else:
                    completed += 1
                    elapsed = round(time.time() - t0, 1)
                    yield f"data: {json.dumps({'type':'result','completed':completed,'total':len(test_dates),'elapsed':elapsed,'result':result}, cls=NumpyEncoder)}\n\n"

            yield f"data: {json.dumps({'type':'done','elapsed':round(time.time()-t0,1)})}\n\n"

        except GeneratorExit:
            log.info("Panel3 SSE client disconnected")
        except Exception as e:
            log.error("Panel3 SSE error: %s", e)
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"},
    )



if __name__ == "__main__":
    log.info("Macro Backtest starting on port 5004")
    log.info("Test window: %s to %s, every %d months",
             BACKTEST_START, BACKTEST_END, STEP_MONTHS)
    app.run(debug=False, port=5004, threaded=True)
