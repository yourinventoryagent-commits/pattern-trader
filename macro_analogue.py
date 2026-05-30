"""
PTJ Macro Analogue Engine — Phase 1 + 2
=========================================
Inspired by Paul Tudor Jones overlaying the 1980s Dow against the 1920s Dow
and using the matched historical period as a forward roadmap.

PHASE 1 — Price pattern matching
  Takes the current SPY chart over a configurable window (default 252 trading
  days / 12 months), normalises to % returns, and slides it across SPY history
  back to 1993. Finds the top historical periods whose price *shape* most
  closely resembles the current market using vectorised Pearson correlation.

PHASE 2 — Macro regime scoring
  For each candidate analogue, scores how closely the macro environment of
  that historical period matches today's macro environment across four
  dimensions pulled from FRED:
    1. Fed Funds rate direction (hiking / cutting / holding)
    2. Yield curve shape      (2y/10y spread — inversion = recession signal)
    3. Inflation trend        (CPI YoY — rising / falling / stable)
    4. Credit stress          (HY spread — wide = stress, tight = risk-on)

  A combined score weights price similarity 50% and regime similarity 50%.
  This suppresses analogues whose price shape matched but whose macro context
  was completely different — solving the core "1999 vs 2011" problem where
  identical chart shapes produced opposite outcomes.

Run:  python macro_analogue.py
Open: http://localhost:5002

Dependencies:
  pip install flask yfinance pandas numpy fredapi
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import json
import logging


# =============================================================================
# SECTION 1 — Configuration
# =============================================================================

# -- Price pattern settings ---------------------------------------------------

TICKER         = "SPY"
DEFAULT_WINDOW = 252        # trading days for pattern window (~12 months)
TOP_N          = 3          # number of final analogues to surface
MIN_SEPARATION = 126        # min trading days between analogue start dates
                            # prevents near-duplicate overlapping windows
HISTORY_START  = "1993-01-01"   # SPY inception

FORWARD_HORIZONS = {        # forward-look periods measured from analogue end
    "6mo":  126,
    "12mo": 252,
    "24mo": 504,
}

# -- Macro regime settings ----------------------------------------------------

from dotenv import load_dotenv
import os
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# FRED series IDs used for regime scoring
FRED_SERIES = {
    "fed_funds":   "FEDFUNDS",       # Effective Fed Funds Rate (monthly, %)
    "yield_curve": "T10Y2Y",         # 10yr minus 2yr Treasury spread (daily, %)
    "cpi":         "CPIAUCSL",       # CPI all items (monthly, index level)
    "hy_spread":   "BAMLH0A0HYM2",  # ICE BofA HY OAS spread (daily, %)
}

MACRO_HISTORY_START = "1990-01-01"  # pull FRED data from here

# Combined score weights — must sum to 1.0
PRICE_WEIGHT  = 0.50    # weight of price pattern similarity
REGIME_WEIGHT = 0.50    # weight of macro regime similarity

# Stage 1 produces this many price-ranked candidates before macro re-ranking.
# Larger = slower but more thorough macro search; 20 is a good balance.
MACRO_CANDIDATES = 20


# =============================================================================
# SECTION 2 — App bootstrap
# =============================================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# SECTION 3 — Data fetching and caching
# =============================================================================

# Module-level caches — populated once per server session.
# FRED data is monthly so session-level caching is appropriate.
_spy_cache   = None   # DataFrame: full SPY OHLCV history
_macro_cache = None   # dict of pd.Series: one per FRED series, daily-reindexed


def fetch_spy():
    """
    Download full SPY daily OHLCV history from 1993 and cache in memory.

    Uses yf.Ticker().history() rather than yf.download() to avoid the
    thread-safety bug where parallel calls bleed into each other's sessions.

    Returns a clean lowercase-column DataFrame, or None on failure.
    """
    global _spy_cache
    if _spy_cache is not None:
        log.info("SPY: using cached data (%d rows)", len(_spy_cache))
        return _spy_cache

    log.info("SPY: fetching full history from %s ...", HISTORY_START)
    try:
        tkr = yf.Ticker(TICKER)
        df  = tkr.history(start=HISTORY_START, auto_adjust=True)

        if df.empty or len(df) < DEFAULT_WINDOW * 2:
            log.error("SPY: insufficient data (%d rows)", len(df))
            return None

        # yfinance >= 0.2.x may return MultiIndex (field, ticker) columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df = df[["open", "high", "low", "close", "volume"]].dropna()
        log.info("SPY: %d rows (%s to %s)",
                 len(df), df.index[0].date(), df.index[-1].date())

        _spy_cache = df
        return df

    except Exception as e:
        log.error("SPY fetch failed: %s", e)
        return None


def fetch_macro():
    """
    Download all four FRED macro series and align them to a common daily index.

    Monthly series (FEDFUNDS, CPI) are forward-filled to daily frequency.
    Forward-fill is the correct treatment: the Fed Funds rate on any given
    day is whatever it was set to at the most recent FOMC meeting, not an
    interpolation from future decisions.

    Also derives CPI year-over-year % change (more useful than the raw index).

    Returns a dict of {name: pd.Series} or None if FRED is unreachable.
    """
    global _macro_cache
    if _macro_cache is not None:
        log.info("Macro: using cached FRED data")
        return _macro_cache

    log.info("Macro: fetching FRED series ...")
    try:
        fred   = Fred(api_key=FRED_API_KEY)
        raw    = {}

        for name, series_id in FRED_SERIES.items():
            log.info("  FRED: %s (%s)", name, series_id)
            s = fred.get_series(series_id, observation_start=MACRO_HISTORY_START)
            s.index = pd.to_datetime(s.index)
            raw[name] = s

        # Reindex to daily and forward-fill — stop at the latest date that
        # has real data across all series to avoid trailing NaN rows
        last_date = max(s.index[-1] for s in raw.values())
        daily_idx = pd.date_range(MACRO_HISTORY_START, last_date, freq="D")
        aligned   = {name: s.reindex(daily_idx).ffill() for name, s in raw.items()}

        # Derive CPI YoY: % change vs same date one year prior
        aligned["cpi_yoy"] = aligned["cpi"].pct_change(periods=365) * 100

        log.info("Macro: all FRED series loaded and aligned")
        _macro_cache = aligned
        return aligned

    except Exception as e:
        log.error("FRED fetch failed: %s", e)
        return None


# =============================================================================
# SECTION 4 — Vectorised price similarity engine (Phase 1)
# =============================================================================

def build_return_matrix(close_arr, window):
    """
    Build a (n_windows x window) matrix of all sliding windows across
    close_arr, where each window is normalised to % return from its own
    first element. Uses NumPy stride indexing — no Python loop.

    Returns empty (0 x window) array if close_arr is shorter than window.
    """
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
    """
    Compute Pearson correlation between 1-D target and every row of matrix
    in a single matrix multiply. No scipy, no Python loop.

    Returns a 1-D array of correlations clipped to [-1, 1].
    """
    t   = target - target.mean()
    t_n = t / (np.sqrt((t ** 2).sum()) + 1e-10)

    m   = matrix - matrix.mean(axis=1, keepdims=True)
    m_n = m / (np.sqrt((m ** 2).sum(axis=1, keepdims=True)) + 1e-10)

    return (m_n @ t_n).clip(-1, 1)


# =============================================================================
# SECTION 5 — Macro regime scoring (Phase 2)
# =============================================================================

def _safe_val(series, date):
    """
    Safely retrieve the most recent non-NaN value in `series` at or before
    `date`. Strips timezone info before comparison so SPY dates (America/New_York)
    and FRED dates (timezone-naive) can be compared without errors.
    Returns float or np.nan if no data exists before this date.
    """
    try:
        cleaned = series.dropna().sort_index()
        if cleaned.empty:
            return np.nan
        # Strip timezone — FRED series are naive, SPY dates are tz-aware
        d = pd.Timestamp(date).tz_localize(None) if pd.Timestamp(date).tzinfo is None \
            else pd.Timestamp(date).tz_convert(None)
        if d >= cleaned.index[-1]:
            return float(cleaned.iloc[-1])
        if d < cleaned.index[0]:
            return np.nan
        pos = cleaned.index.searchsorted(d, side="right") - 1
        if pos < 0:
            return np.nan
        return float(cleaned.iloc[pos])
    except Exception as e:
        log.warning("_safe_val failed for %s: %s", date, e)
        return np.nan


def get_macro_snapshot(macro, date):
    """
    Extract a point-in-time macro snapshot for a given date.

    Uses _safe_val() to retrieve the most recent non-NaN observation at or
    before `date` for each series — robust to gaps in FRED data (e.g. the
    yield curve series has weekends and holidays missing).

    Returns a dict of six scalar values (all in %), or None if critical
    series (fed_funds, cpi_yoy) are unavailable for this date.

    Dict keys:
      fed_funds   : Fed Funds rate level
      fed_change  : 12-month change in Fed Funds (+ve = hiking, -ve = cutting)
      yield_curve : 10y minus 2y spread (-ve = inverted = recession signal)
      cpi_yoy     : CPI year-over-year inflation rate
      cpi_change  : 12-month change in CPI YoY (+ve = inflation accelerating)
      hy_spread   : High yield OAS spread (+ve wide = credit stress)
    """
    try:
        d     = pd.Timestamp(date)
        d_1yr = d - pd.DateOffset(years=1)

        fed_now  = _safe_val(macro["fed_funds"],   d)
        fed_1yr  = _safe_val(macro["fed_funds"],   d_1yr)
        yc_now   = _safe_val(macro["yield_curve"], d)
        cpi_now  = _safe_val(macro["cpi_yoy"],     d)
        cpi_1yr  = _safe_val(macro["cpi_yoy"],     d_1yr)
        hy_now   = _safe_val(macro["hy_spread"],   d)

        # Log what we got so it's visible in the terminal on first run
        log.info("Snapshot %s: fed=%.2f yc=%.2f cpi=%.2f hy=%.2f",
                  d.date(), fed_now, yc_now, cpi_now, hy_now)

        # Fed funds and CPI are critical — reject snapshot if missing
        if any(np.isnan(v) for v in [fed_now, fed_1yr, cpi_now, cpi_1yr]):
            log.debug("Snapshot %s: missing critical series", d.date())
            return None

        # Yield curve and HY spread may have early-history gaps — substitute 0
        # (neutral) rather than rejecting the whole snapshot
        if np.isnan(yc_now): yc_now = 0.0
        if np.isnan(hy_now): hy_now = 4.0   # approximate long-run average

        return {
            "fed_funds":   round(fed_now,           2),
            "fed_change":  round(fed_now - fed_1yr,  2),
            "yield_curve": round(yc_now,             2),
            "cpi_yoy":     round(cpi_now,            2),
            "cpi_change":  round(cpi_now - cpi_1yr,  2),
            "hy_spread":   round(hy_now,             2),
        }

    except Exception as e:
        log.warning("Macro snapshot failed for %s: %s", date, e)
        return None


def score_regime_similarity(snap_today, snap_hist):
    """
    Compute a [0, 1] similarity score between two macro snapshots using a
    normalised Euclidean distance converted via a Gaussian kernel.

    Each dimension is divided by a characteristic scale chosen to reflect
    what constitutes a meaningful difference in that variable:

      fed_funds   / 5%  — covers most of a full hiking cycle
      fed_change  / 3%  — distinguishes aggressive move from hold
      yield_curve / 3%  — flat to deeply inverted to steep
      cpi_yoy     / 6%  — low-inflation era to post-COVID spike
      cpi_change  / 4%  — rapid disinflation vs re-ignition
      hy_spread   / 6%  — tight credit to full crisis

    Gaussian kernel: similarity = exp(-distance^2 / 2)
      distance = 0  -> similarity = 1.0  (identical)
      distance = 1  -> similarity = 0.61 (similar)
      distance = 2  -> similarity = 0.14 (meaningfully different)
    """
    SCALES = {
        "fed_funds":   5.0,
        "fed_change":  3.0,
        "yield_curve": 3.0,
        "cpi_yoy":     6.0,
        "cpi_change":  4.0,
        "hy_spread":   6.0,
    }

    sq_dist = sum(
        ((snap_today[k] - snap_hist[k]) / scale) ** 2
        for k, scale in SCALES.items()
    )

    return round(float(np.exp(-sq_dist / 2.0)), 4)


# =============================================================================
# SECTION 6 — Analogue finder with combined scoring
# =============================================================================

def find_analogues(close_arr, dates, window, macro):
    """
    Two-stage analogue search combining price pattern and macro regime.

    Stage 1 — Price similarity (vectorised, fast):
      Slide window across all history, compute Pearson correlation with the
      current pattern for every position. Collect the top MACRO_CANDIDATES
      non-overlapping results as price-ranked candidates.

    Stage 2 — Macro regime scoring (point-in-time, per candidate):
      For each candidate, fetch a macro snapshot at its end date and compare
      to today's snapshot using score_regime_similarity(). Re-rank by:
        combined = price_sim * PRICE_WEIGHT + regime_sim * REGIME_WEIGHT

    Falls back to price-only ranking if FRED data is unavailable.

    Returns the top TOP_N analogues as a list of dicts.
    """
    n = len(close_arr)

    # Current pattern: last `window` bars normalised to 0% at day 1
    cur_window = close_arr[n - window:]
    cur_target = (cur_window / cur_window[0]) - 1.0

    # Search space ends 2x window before today:
    #   - guarantees no overlap with current window
    #   - guarantees each analogue has at least `window` bars of forward data
    search_end = n - window * 2
    if search_end < window:
        return []

    # --- Stage 1: vectorised price similarity --------------------------------

    hist_close   = close_arr[:search_end + window - 1]
    return_mat   = build_return_matrix(hist_close, window)[:search_end]

    if len(return_mat) == 0:
        return []

    similarities = vectorised_pearson(cur_target, return_mat)
    order        = np.argsort(similarities)[::-1]

    # Greedily collect candidates with MIN_SEPARATION enforced
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

    # --- Stage 2: macro regime scoring ---------------------------------------

    today_snap   = get_macro_snapshot(macro, dates[-1]) if macro else None
    macro_active = macro is not None and today_snap is not None

    if not macro_active:
        log.warning("Macro unavailable — price-only ranking")

    scored = []
    for c in candidates:
        price_sim = c["price_sim"]
        end_date  = dates[c["end_idx"]]

        if macro_active:
            hist_snap  = get_macro_snapshot(macro, end_date)
            regime_sim = score_regime_similarity(today_snap, hist_snap) \
                         if hist_snap else 0.0
            combined   = PRICE_WEIGHT * price_sim + REGIME_WEIGHT * regime_sim
        else:
            regime_sim = None
            combined   = price_sim

        scored.append({**c, "regime_sim": regime_sim, "combined": combined})

    scored.sort(key=lambda x: x["combined"], reverse=True)

    # --- Build output dicts --------------------------------------------------

    max_fwd   = max(FORWARD_HORIZONS.values())
    analogues = []

    for rank, c in enumerate(scored[:TOP_N], start=1):
        s_idx = c["start_idx"]
        e_idx = c["end_idx"]

        # Scalar forward returns at each horizon
        forward = {}
        for label, fwd_days in FORWARD_HORIZONS.items():
            fut_idx = e_idx + fwd_days
            if fut_idx < n:
                t0  = float(close_arr[e_idx])
                fut = float(close_arr[fut_idx])
                forward[label] = round((fut - t0) / t0, 4)
            else:
                forward[label] = None

        # Normalised price series during the matched window (Panel 1 overlay)
        window_prices = close_arr[s_idx: e_idx + 1]
        analogue_norm = ((window_prices / window_prices[0]) - 1.0).tolist()

        # Forward continuation from analogue end point (Panel 2 projection)
        # forward_norm[0] == 0 by construction: all tails start flat at "today"
        fwd_prices   = close_arr[e_idx: min(e_idx + max_fwd + 1, n)]
        forward_norm = ((fwd_prices / fwd_prices[0]) - 1.0).tolist() \
                       if len(fwd_prices) > 0 else []

        # Macro snapshot at analogue end date (for regime detail display in UI)
        macro_snap = get_macro_snapshot(macro, dates[e_idx]) \
                     if macro_active else None

        analogues.append({
            "rank":          rank,
            "price_sim":     round(c["price_sim"],  4),
            "regime_sim":    round(c["regime_sim"], 4) if c["regime_sim"] is not None else None,
            "combined":      round(c["combined"],   4),
            # "similarity" is the combined score — used by existing frontend
            # weighting logic without requiring any UI changes
            "similarity":    round(c["combined"],   4),
            "start_date":    dates[s_idx].strftime("%Y-%m-%d"),
            "end_date":      dates[e_idx].strftime("%Y-%m-%d"),
            "analogue_norm": analogue_norm,
            "forward_norm":  forward_norm,
            "forward":       forward,
            "macro_snap":    macro_snap,
        })

    return analogues


# =============================================================================
# SECTION 7 — Interpretation and summary
# =============================================================================

def build_current_series(close_arr, dates, window):
    """
    Return the normalised current window (last `window` bars) for Panel 1.
    Values are % return from day 1 of the window, same basis as analogue_norm.
    """
    n          = len(close_arr)
    cur_prices = close_arr[n - window:]
    cur_norm   = ((cur_prices / cur_prices[0]) - 1.0).tolist()
    cur_dates  = [d.strftime("%Y-%m-%d") for d in dates[n - window:]]
    return {
        "dates":         cur_dates,
        "norm":          cur_norm,
        "start_date":    cur_dates[0],
        "end_date":      cur_dates[-1],
        "period_return": round((cur_prices[-1] - cur_prices[0]) / cur_prices[0], 4),
    }


def build_interpretation(analogues, macro_active):
    """
    Compute combined-score-squared weighted average forward returns.
    Squaring the weights further emphasises the best-matching analogues.
    Returns expected returns per horizon and a plain-English summary.
    """
    weights = np.array([a["combined"] ** 2 for a in analogues])
    weights /= weights.sum()

    expected = {}
    for label in FORWARD_HORIZONS:
        valid = [(a["forward"][label], w)
                 for a, w in zip(analogues, weights)
                 if a["forward"][label] is not None]
        if valid:
            rets, ws = zip(*valid)
            ws = np.array(ws) / np.array(ws).sum()
            expected[label] = round(float(np.dot(ws, rets)), 4)
        else:
            expected[label] = None

    r6  = expected.get("6mo")
    r12 = expected.get("12mo")

    def fmt(v):
        return "unknown" if v is None \
               else f"{'+' if v >= 0 else ''}{v * 100:.1f}%"

    direction    = "bullish" if (r12 or r6 or 0) > 0 else "bearish"
    magnitude    = "strongly" if abs(r12 or r6 or 0) > 0.10 else "modestly"
    regime_note  = (
        "Ranked by a 50/50 blend of price-pattern and macro-regime similarity "
        "(Fed stance, yield curve, CPI trend, credit spreads)."
        if macro_active else
        "Macro regime scoring unavailable — ranked by price pattern only."
    )

    return {
        "expected_returns": expected,
        "weights":          weights.tolist(),
        "macro_active":     macro_active,
        "summary": (
            f"Weighted expected returns: {fmt(r6)} (6mo) · "
            f"{fmt(expected.get('12mo'))} (12mo) · "
            f"{fmt(expected.get('24mo'))} (24mo). "
            f"Analogues suggest a {magnitude} {direction} outlook. "
            f"{regime_note}"
        ),
    }


# =============================================================================
# SECTION 8 — JSON serialisation
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars and arrays to native Python types for JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


# =============================================================================
# SECTION 9 — Flask routes
# =============================================================================

@app.route("/")
def index():
    return render_template("macro.html")


@app.route("/api/analogues", methods=["POST"])
def api_analogues():
    """
    Main analysis endpoint.

    Request body (JSON, all optional):
      window : int  — pattern window in trading days (default 252, clamped 63-756)

    Response (JSON):
      window_days    : int
      current        : {dates, norm, start_date, end_date, period_return}
      analogues      : list of TOP_N analogue dicts with price_sim, regime_sim,
                       combined, analogue_norm, forward_norm, forward, macro_snap
      interpretation : {expected_returns, weights, summary, macro_active}
      spy_latest     : str  — date of last SPY bar
      macro_active   : bool — whether FRED regime scoring succeeded
      today_macro    : dict | None — today's macro snapshot for UI display
    """
    body   = request.get_json(silent=True) or {}
    window = int(body.get("window", DEFAULT_WINDOW))
    window = max(63, min(window, 756))

    df = fetch_spy()
    if df is None:
        return jsonify({"error": "Failed to fetch SPY data"}), 500

    close_arr = df["close"].values.astype(float)
    dates     = df.index

    if len(close_arr) < window * 2:
        return jsonify({"error": "Not enough history for the requested window"}), 400

    # Macro data is optional — engine degrades gracefully if FRED is unreachable
    macro        = fetch_macro()
    macro_active = macro is not None

    log.info("-- Analysis: window=%d days, macro=%s --", window, macro_active)

    analogues = find_analogues(close_arr, dates, window, macro)
    if not analogues:
        return jsonify({"error": "No analogues found — try a shorter window"}), 400

    current     = build_current_series(close_arr, dates, window)
    interp      = build_interpretation(analogues, macro_active)
    today_macro = get_macro_snapshot(macro, dates[-1]) if macro_active else None

    log.info("-- Done: %d analogues --", len(analogues))

    return app.response_class(
        response=json.dumps({
            "window_days":    window,
            "current":        current,
            "analogues":      analogues,
            "interpretation": interp,
            "spy_latest":     df.index[-1].strftime("%Y-%m-%d"),
            "macro_active":   macro_active,
            "today_macro":    today_macro,
        }, cls=NumpyEncoder),
        mimetype="application/json"
    )


# =============================================================================
# SECTION 10 — Phase 3: Sector rotation engine
# =============================================================================
#
# DESIGN
# ------
# Given the top analogue periods from Phase 1+2, compute how each major sector
# performed relative to SPY in the 6 and 12 months following each analogue's
# end date. Rank sectors by a combined consistency × magnitude score.
#
# DATA STRATEGY
# -------------
# Sector ETFs (XLK, XLF etc.) launched in late 1998, so they only cover
# analogues from 1999 onward. For earlier analogue periods (e.g. 1994-1996),
# we use equal-weighted baskets of large-cap proxy stocks that existed and
# traded in those years. Each basket has 10-20 stocks to smooth idiosyncratic
# noise.
#
# RANKING METRIC
# --------------
# score = avg_relative_outperformance × consistency_multiplier
#   consistency_multiplier: 3/3 → 1.0, 2/3 → 0.67, 1/3 → 0.33, 0/3 → 0.0
#
# This rewards sectors that outperformed consistently across analogues more
# than a sector that had one great period and two flat ones.
#
# =============================================================================

# -- Sector definitions -------------------------------------------------------
#
# Each sector has:
#   etf     : SPDR sector ETF ticker (available from late 1998)
#   name    : display name
#   proxies : equal-weighted basket of stocks for pre-ETF periods
#             chosen for: large-cap, existed in early 1990s, representative
#             of sector, not since acquired/delisted under same ticker

SECTORS = {
    "XLK": {
        "name": "Technology",
        "etf":  "XLK",
        "proxies": [
            "MSFT", "IBM",  "TXN",  "HPQ",  "AMAT",
            "MU",   "ADI",  "KLAC", "LRCX", "NTAP",
            "GLW",  "CSCO", "ORCL", "SNX",  "CTSH",
            "APH",  "TEL",  "KEYS", "ANSS", "CDNS",
        ],
    },
    "XLF": {
        "name": "Financials",
        "etf":  "XLF",
        "proxies": [
            "JPM",  "BAC",  "WFC",  "C",    "GS",
            "MS",   "AXP",  "USB",  "PNC",  "MET",
            "PRU",  "ALL",  "TRV",  "AFL",  "BK",
            "STT",  "FITB", "RF",   "KEY",  "MTB",
        ],
    },
    "XLE": {
        "name": "Energy",
        "etf":  "XLE",
        "proxies": [
            "XOM",  "CVX",  "SLB",  "HAL",  "BKR",
            "COP",  "OXY",  "DVN",  "APA",  "EOG",
            "PSX",  "VLO",  "MPC",  "PXD",  "FANG",
        ],
    },
    "XLV": {
        "name": "Healthcare",
        "etf":  "XLV",
        "proxies": [
            "JNJ",  "PFE",  "MRK",  "ABT",  "LLY",
            "MDT",  "BMY",  "AMGN", "GILD", "BAX",
            "BDX",  "SYK",  "BSX",  "HUM",  "CI",
            "CVS",  "MCK",  "CAH",  "ZBH",  "EW",
        ],
    },
    "XLI": {
        "name": "Industrials",
        "etf":  "XLI",
        "proxies": [
            "GE",   "MMM",  "HON",  "CAT",  "EMR",
            "ETN",  "PH",   "ROK",  "DOV",  "ITW",
            "DHR",  "AME",  "ROP",  "FAST", "GWW",
            "LMT",  "RTX",  "NOC",  "GD",   "TDG",
        ],
    },
    "XLY": {
        "name": "Consumer Discretionary",
        "etf":  "XLY",
        "proxies": [
            "MCD",  "DIS",  "HD",   "LOW",  "TGT",
            "F",    "YUM",  "MAR",  "RCL",  "CCL",
            "WHR",  "LEN",  "PHM",  "NKE",  "SBUX",
        ],
    },
    "XLP": {
        "name": "Consumer Staples",
        "etf":  "XLP",
        "proxies": [
            "PG",   "KO",   "PEP",  "WMT",  "CL",
            "GIS",  "CPB",  "HRL",  "MKC",  "SJM",
            "CAG",  "HSY",  "CHD",  "CLX",  "KMB",
        ],
    },
    "XLB": {
        "name": "Materials",
        "etf":  "XLB",
        "proxies": [
            "DD",   "PPG",  "SHW",  "NEM",  "FCX",
            "NUE",  "RS",   "VMC",  "MLM",  "ALB",
            "ECL",  "APD",  "LIN",  "IFF",  "CE",
        ],
    },
    "XLRE": {
        "name": "Real Estate",
        "etf":  "XLRE",
        "proxies": [
            "PLD",  "AMT",  "CCI",  "SPG",  "O",
            "PSA",  "EQR",  "AVB",  "DLR",  "EQIX",
        ],
    },
    "XLU": {
        "name": "Utilities",
        "etf":  "XLU",
        "proxies": [
            "NEE",  "DUK",  "SO",   "D",    "AEP",
            "EXC",  "XEL",  "ED",   "WEC",  "ES",
            "ETR",  "FE",   "PPL",  "CMS",  "AES",
        ],
    },
    "XLC": {
        "name": "Communication Services",
        "etf":  "XLC",
        "proxies": [
            "T",    "VZ",   "CMCSA","NFLX", "TMUS",
            "CHTR", "META", "GOOGL","DIS",  "EA",
        ],
    },
}

# ETF inception date — use proxy baskets for analogue periods before this
ETF_INCEPTION = pd.Timestamp("1999-01-01")

# Sector data cache — keyed by ticker, value is close price Series
_sector_cache = {}
_sector_lock  = __import__("threading").Lock()


def fetch_sector_data(ticker, start="1990-01-01"):
    """
    Fetch daily close price history for a single ticker.
    Caches results to avoid re-downloading on repeated calls.
    Returns a pd.Series indexed by date, or None on failure.
    """
    with _sector_lock:
        if ticker in _sector_cache:
            return _sector_cache[ticker]

    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start=start, auto_adjust=True)

        if df.empty or len(df) < 60:
            log.warning("Sector %s: insufficient data", ticker)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        if "close" not in df.columns:
            log.warning("Sector %s: no close column", ticker)
            return None

        series = df["close"].dropna()
        log.info("  Sector %s: %d rows from %s", ticker, len(series),
                 series.index[0].strftime("%Y-%m-%d"))

        with _sector_lock:
            _sector_cache[ticker] = series
        return series

    except Exception as e:
        log.warning("Sector %s fetch failed: %s", ticker, e)
        return None


def compute_forward_return(price_series, from_date, fwd_days):
    """
    Compute the % return of price_series from from_date over fwd_days
    trading days. Returns None if data is unavailable for either endpoint.

    Strips timezone from both the query date and the series index to avoid
    comparison failures between tz-aware yfinance data and naive date strings.
    """
    try:
        # Normalise query date — strip timezone regardless of source
        d = pd.Timestamp(from_date)
        if d.tzinfo is not None:
            d = d.tz_convert(None)

        # Normalise series index — strip timezone if present
        idx = price_series.index
        if idx.tzinfo is not None:
            idx = idx.tz_convert(None)

        if len(idx) == 0:
            return None

        # Find nearest trading day at or after from_date
        pos_start = idx.searchsorted(d, side="left")
        if pos_start >= len(idx):
            return None

        pos_end = pos_start + fwd_days
        if pos_end >= len(idx):
            # If we don't have full fwd_days of data, use whatever is available
            # as long as we have at least half the requested period
            if pos_end - len(idx) > fwd_days // 2:
                return None
            pos_end = len(idx) - 1

        p0 = float(price_series.iloc[pos_start])
        p1 = float(price_series.iloc[pos_end])

        if p0 == 0 or np.isnan(p0) or np.isnan(p1):
            return None

        return round((p1 - p0) / p0, 4)

    except Exception as e:
        log.debug("Forward return calc failed for %s: %s", from_date, e)
        return None


def compute_sector_return(sector_key, from_date, fwd_days, use_etf):
    """
    Compute the sector's forward return from from_date over fwd_days days.

    If use_etf is True and ETF data is available for this period, use the ETF.
    Otherwise compute an equal-weighted average of the proxy basket returns,
    skipping any stocks with missing data for this period.

    Returns (return_value, method_used) where method_used is 'etf' or 'proxy'.
    Returns (None, None) if insufficient data.
    """
    sector = SECTORS[sector_key]

    if use_etf:
        etf_series = fetch_sector_data(sector["etf"])
        if etf_series is not None:
            ret = compute_forward_return(etf_series, from_date, fwd_days)
            if ret is not None:
                return ret, "etf"

    # Fall back to proxy basket
    proxy_returns = []
    for ticker in sector["proxies"]:
        series = fetch_sector_data(ticker)
        if series is None:
            continue
        ret = compute_forward_return(series, from_date, fwd_days)
        if ret is not None:
            proxy_returns.append(ret)

    if len(proxy_returns) < 3:   # require at least 3 valid proxies
        log.warning("  %s: only %d valid proxies for %s fwd=%d",
                    sector_key, len(proxy_returns), from_date, fwd_days)
        return None, None

    return round(float(np.mean(proxy_returns)), 4), "proxy"


def score_sectors(analogues):
    """
    Core Phase 3 computation. For each sector, compute performance relative
    to SPY across all analogue periods at 6mo and 12mo horizons.

    Returns a list of sector result dicts sorted by combined score descending.

    Scoring:
      relative_return    = sector_return - spy_return  (per analogue, per horizon)
      avg_relative       = mean of relative returns across analogues (where available)
      consistency        = fraction of analogues where sector outperformed SPY
      combined_score     = avg_relative × consistency_multiplier
        where multiplier = 1.0 (all outperform), 0.67 (2/3), 0.33 (1/3), 0.0 (none)
    """
    log.info("-- Phase 3: computing sector rotation --")

    # Fetch SPY series for benchmark comparison
    spy_series = fetch_sector_data("SPY")
    if spy_series is None:
        log.error("Phase 3: SPY data unavailable")
        return []

    horizons = {"6mo": 126, "12mo": 252}
    results  = []

    for sector_key, sector_info in SECTORS.items():
        log.info("  Sector: %s (%s)", sector_key, sector_info["name"])

        analogue_results = []   # one entry per analogue

        for a in analogues:
            end_date = pd.Timestamp(a["end_date"])
            use_etf  = end_date >= ETF_INCEPTION

            horizon_data = {}
            for label, fwd_days in horizons.items():

                # SPY benchmark return
                spy_ret = compute_forward_return(spy_series, end_date, fwd_days)

                # Sector return
                sec_ret, method = compute_sector_return(
                    sector_key, end_date, fwd_days, use_etf)

                if spy_ret is not None and sec_ret is not None:
                    horizon_data[label] = {
                        "sector_return":   sec_ret,
                        "spy_return":      spy_ret,
                        "relative_return": round(sec_ret - spy_ret, 4),
                        "outperformed":    sec_ret > spy_ret,
                        "method":          method,
                    }
                else:
                    horizon_data[label] = None

            analogue_results.append({
                "analogue_end":   a["end_date"],
                "analogue_rank":  a["rank"],
                "horizons":       horizon_data,
            })

        # Aggregate across analogues for the primary horizon (12mo)
        primary = "12mo"
        valid   = [ar["horizons"][primary]
                   for ar in analogue_results
                   if ar["horizons"].get(primary) is not None]

        if not valid:
            log.warning("  %s: no valid data, skipping", sector_key)
            continue

        rel_returns   = [v["relative_return"] for v in valid]
        outperformed  = [v["outperformed"]    for v in valid]
        avg_relative  = float(np.mean(rel_returns))
        consistency   = float(np.mean(outperformed))   # 0.0 to 1.0

        # Consistency multiplier: penalises sectors that only outperformed
        # in one analogue — we want cross-analogue agreement
        n             = len(outperformed)
        n_out         = sum(outperformed)
        if   n_out == n:   multiplier = 1.00   # all analogues agree
        elif n_out == n-1: multiplier = 0.67   # one disagreement
        elif n_out == 1:   multiplier = 0.33   # one agreement
        else:              multiplier = 0.00   # no outperformance

        combined_score = avg_relative * multiplier

        # Also compute 6mo aggregate for display
        valid_6mo = [ar["horizons"]["6mo"]
                     for ar in analogue_results
                     if ar["horizons"].get("6mo") is not None]
        avg_rel_6mo = float(np.mean([v["relative_return"] for v in valid_6mo])) \
                      if valid_6mo else None

        results.append({
            "sector_key":      sector_key,
            "sector_name":     sector_info["name"],
            "etf":             sector_info["etf"],
            "combined_score":  round(combined_score,  4),
            "avg_relative_12": round(avg_relative,    4),
            "avg_relative_6":  round(avg_rel_6mo,     4) if avg_rel_6mo is not None else None,
            "consistency":     round(consistency,      4),
            "n_analogues":     len(valid),
            "n_outperformed":  n_out,
            "analogue_detail": analogue_results,   # per-analogue breakdown for UI
        })

    # Sort by combined score descending
    results.sort(key=lambda x: x["combined_score"], reverse=True)

    log.info("-- Phase 3 complete: %d sectors ranked --", len(results))
    return results


# =============================================================================
# SECTION 11 — Phase 3 Step 2: Individual stock signal scanner
# =============================================================================
#
# DESIGN
# ------
# For each stock in a sector's watchlist, runs the same 21-day pattern
# matching engine used in the original Pattern Trader (app.py) but
# self-contained here so no cross-app dependency is needed.
#
# The engine slides the current 21-day price+volume pattern across 10 years
# of the stock's own history, finds historical windows with high composite
# similarity, and computes expected forward return, win rate, and EV score.
#
# Only stocks with a genuine signal (meets similarity cutoff, min matches,
# and minimum expected move) are returned. Stocks with no edge are filtered.
# Results are ranked by EV score descending.
#
# PARAMETERS (kept consistent with Pattern Trader v3)
# ---------------------------------------------------
# Pattern window:     21 trading days
# Forward horizon:    10 trading days
# Similarity cutoff:  0.75 composite (70% price + 30% volume)
# Min matches:        8
# Min expected move:  2%
# History:            10 years
# =============================================================================

# -- Stock universe by sector -------------------------------------------------
# Current, liquid, tradeable names. These are the stocks you'd actually trade.
# Different from the proxy baskets (which were for historical sector returns).

SECTOR_STOCKS = {
    "XLK": [
        "NVDA", "MSFT", "AAPL", "AVGO", "META",
        "GOOGL","AMD",  "ORCL","CSCO", "AMAT",
        "NOW",  "ADBE","KLAC","LRCX", "SNPS",
        "CDNS", "TXN", "INTC","HPQ",  "IBM",
    ],
    "XLF": [
        "JPM",  "BAC", "WFC", "GS",   "MS",
        "BLK",  "AXP", "C",   "USB",  "PNC",
        "TRV",  "AIG", "COF", "SCHW", "ICE",
    ],
    "XLE": [
        "XOM",  "CVX", "SLB", "COP",  "EOG",
        "PXD",  "OXY", "DVN", "HAL",  "MPC",
    ],
    "XLY": [
        "AMZN", "TSLA","MCD", "HD",   "NKE",
        "SBUX", "LOW", "TGT", "BKNG", "CMG",
    ],
    "XLV": [
        "LLY",  "UNH", "JNJ", "ABBV","MRK",
        "TMO",  "ABT", "DHR", "BMY", "AMGN",
    ],
}

# -- Pattern scanner config ---------------------------------------------------

STOCK_PATTERN_DAYS  = 21      # current pattern window length (trading days)
STOCK_FORWARD_DAYS  = 10      # forward return horizon (trading days)
STOCK_SIM_CUTOFF    = 0.75    # minimum composite similarity
STOCK_MIN_MATCHES   = 8       # minimum historical matches required
STOCK_MIN_MOVE      = 0.02    # minimum absolute expected move (2%)
STOCK_PRICE_WEIGHT  = 0.70    # weight of price shape in composite score
STOCK_VOL_WEIGHT    = 0.30    # weight of volume profile in composite score
STOCK_HISTORY_YEARS = 10      # years of history to fetch per stock
STOCK_TIME_DECAY    = 365     # analogues from this many days ago get half weight

# Stock data cache — separate from sector cache to avoid confusion
_stock_cache      = {}
_stock_cache_lock = __import__("threading").Lock()


def fetch_stock_ohlcv(ticker):
    """
    Fetch 10 years of daily OHLCV for a stock.
    Uses yf.Ticker().history() for thread safety.
    Caches result in memory for the session.
    Returns DataFrame with lowercase columns or None on failure.
    """
    with _stock_cache_lock:
        if ticker in _stock_cache:
            return _stock_cache[ticker]

    start = (datetime.today() - timedelta(days=STOCK_HISTORY_YEARS * 365)
             ).strftime("%Y-%m-%d")
    try:
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start=start, auto_adjust=True)

        if df.empty or len(df) < STOCK_PATTERN_DAYS * 3:
            log.warning("Stock %s: insufficient data", ticker)
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        required = {"close", "volume", "high", "low"}
        if not required.issubset(set(df.columns)):
            return None

        result = df[["high", "low", "close", "volume"]].dropna()
        log.info("  Stock %s: %d rows", ticker, len(result))

        with _stock_cache_lock:
            _stock_cache[ticker] = result
        return result

    except Exception as e:
        log.warning("Stock %s fetch failed: %s", ticker, e)
        return None


def zscore_matrix(arr, window):
    """
    Sliding z-score normalisation across arr.
    Returns (n_windows × window) matrix.
    Used for volume normalisation — z-score is more robust than % return
    for volume which can have extreme outliers.
    """
    arr       = np.asarray(arr, dtype=float)
    n_windows = len(arr) - window + 1
    if n_windows <= 0:
        return np.empty((0, window), dtype=float)

    idx = np.arange(window)[None, :] + np.arange(n_windows)[:, None]
    mat = arr[idx]
    mu  = mat.mean(axis=1, keepdims=True)
    std = mat.std( axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        mat = np.where(std > 0, (mat - mu) / std, 0.0)
    return mat


def compute_stock_atr(high_arr, low_arr, close_arr, period=20):
    """
    Compute Average True Range (ATR) as a rolling mean of True Range.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)

    Returns the ATR as a % of current price — directly usable as a stop %.
    Uses the same formula as the backtest engine (backtest.py):
      stop_pct = ATR(20) * ATR_MULTIPLIER / current_price

    ATR_MULTIPLIER of 2.0 gives a stop wide enough to survive normal daily
    noise while still exiting if the trade genuinely fails. For a $255 stock
    with ATR ~$3.50, this gives a stop of ~2.7% — far more realistic than
    the median drawdown estimate which can be artificially tight on small
    sample sizes.
    """
    ATR_MULTIPLIER = 2.0

    n     = len(close_arr)
    tr    = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]

    for i in range(1, n):
        tr[i] = max(
            high_arr[i]  - low_arr[i],
            abs(high_arr[i]  - close_arr[i - 1]),
            abs(low_arr[i]   - close_arr[i - 1]),
        )

    atr_series = pd.Series(tr).rolling(period).mean().values
    atr_now    = atr_series[-1]

    if np.isnan(atr_now) or close_arr[-1] == 0:
        return 0.03   # fallback: 3% if ATR unavailable

    return round(float(atr_now * ATR_MULTIPLIER / close_arr[-1]), 4)


    """
    Exponential time decay: analogues from half_life days ago get weight 0.5.
    Recent analogues get weight closer to 1.0.
    This prevents the engine from over-weighting ancient history.
    """
    age_days = current_idx - window_end_idx
    return float(2.0 ** (-age_days / half_life))


def scan_stock(ticker):
    """
    Run the 21-day pattern matching engine on a single stock.

    Process:
      1. Fetch 10 years of OHLCV
      2. Extract current 21-day price+volume pattern
      3. Slide pattern across all history (excluding current window)
      4. Compute composite similarity (70% price Pearson + 30% volume Pearson)
      5. Filter by similarity cutoff and minimum matches
      6. Compute ρ²-weighted, time-decayed expected forward return
      7. Return signal dict if passes all gates, None if no edge

    Returns dict with signal details or None.
    """
    df = fetch_stock_ohlcv(ticker)
    if df is None:
        return {"ticker": ticker, "status": "error", "error": "Data unavailable"}

    close_arr = df["close"].values.astype(float)
    vol_arr   = df["volume"].values.astype(float)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)
    n         = len(close_arr)

    if n < STOCK_PATTERN_DAYS * 3 + STOCK_FORWARD_DAYS:
        return {"ticker": ticker, "status": "no_matches", "error": "Insufficient history"}

    t = n  # current position = end of data

    # Current pattern vectors
    cur_price_mat = build_return_matrix(close_arr[t - STOCK_PATTERN_DAYS: t],
                                        STOCK_PATTERN_DAYS)
    cur_vol_mat   = zscore_matrix(vol_arr[t - STOCK_PATTERN_DAYS: t],
                                  STOCK_PATTERN_DAYS)

    if len(cur_price_mat) == 0 or len(cur_vol_mat) == 0:
        return {"ticker": ticker, "status": "no_matches", "error": "Pattern extraction failed"}

    cur_price = cur_price_mat[0]
    cur_vol   = cur_vol_mat[0]

    # Historical search space — exclude the current pattern window
    search_end = t - STOCK_PATTERN_DAYS
    if search_end < STOCK_PATTERN_DAYS + STOCK_FORWARD_DAYS:
        return {"ticker": ticker, "status": "no_matches", "error": "Not enough history"}

    # Build historical matrices
    price_mat = build_return_matrix(close_arr[:search_end], STOCK_PATTERN_DAYS)
    vol_mat   = zscore_matrix(vol_arr[:search_end],         STOCK_PATTERN_DAYS)

    if len(price_mat) == 0:
        return {"ticker": ticker, "status": "no_matches", "error": "No historical windows"}

    # Vectorised composite similarity
    rp        = vectorised_pearson(cur_price, price_mat).clip(0, 1)
    rv        = vectorised_pearson(cur_vol,   vol_mat  ).clip(0, 1)
    composite = STOCK_PRICE_WEIGHT * rp + STOCK_VOL_WEIGHT * rv

    # Apply similarity cutoff
    mask      = composite >= STOCK_SIM_CUTOFF
    composite = composite[mask]
    starts    = np.where(mask)[0]

    if len(composite) < STOCK_MIN_MATCHES:
        return {"ticker": ticker, "status": "no_edge",
                "match_count": int(len(composite)),
                "best_sim": float(rp.max()) if len(rp) > 0 else 0}

    # Forward returns — ensure future data exists for each match
    t0_idx  = starts + STOCK_PATTERN_DAYS - 1
    fut_idx = t0_idx + STOCK_FORWARD_DAYS
    valid   = fut_idx < n

    t0_idx    = t0_idx[valid]
    fut_idx   = fut_idx[valid]
    composite = composite[valid]
    starts    = starts[valid]

    if len(composite) < STOCK_MIN_MATCHES:
        return {"ticker": ticker, "status": "no_edge",
                "match_count": int(len(composite)), "best_sim": float(rp.max())}

    # Forward returns from each matched window
    t0_prices  = close_arr[t0_idx]
    fut_prices = close_arr[fut_idx]
    fwd_rets   = (fut_prices - t0_prices) / t0_prices

    # Weights: similarity² × time decay (recent analogues weighted more)
    decay_w  = np.array([time_decay_weight(int(t0_idx[i]), t)
                         for i in range(len(t0_idx))])
    sim_w    = composite ** 2
    weights  = sim_w * decay_w
    weights /= weights.sum()

    avg_return = float(np.dot(weights, fwd_rets))

    # Signal gate: minimum expected move
    if abs(avg_return) < STOCK_MIN_MOVE:
        return {"ticker": ticker, "status": "no_edge",
                "match_count": int(len(composite)),
                "best_sim": float(composite.max())}

    direction = "LONG" if avg_return > 0 else "SHORT"
    wins      = fwd_rets > 0
    win_rate  = float(wins.mean())
    avg_win   = float(fwd_rets[wins].mean())   if wins.any()   else 0.0
    avg_loss  = float(fwd_rets[~wins].mean())  if (~wins).any() else 0.0
    ev_score  = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    # ATR-based stop — 2 × ATR(20) / current price.
    # This is the same formula used in the backtest engine and is far more
    # robust than the median drawdown estimate, which can be artificially
    # tight when the match sample is small (e.g. MPC's 8 matches).
    atr_stop = compute_stock_atr(high_arr, low_arr, close_arr, period=20)

    # Median historical drawdown kept as a secondary reference — useful
    # context but not the primary stop. If ATR stop is tighter than the
    # worst historical drawdown, use the drawdown as a floor.
    drawdowns = []
    for i in range(len(t0_idx)):
        end = min(t0_idx[i] + STOCK_FORWARD_DAYS, n)
        fp  = close_arr[t0_idx[i]: end]
        if len(fp) > 1 and close_arr[t0_idx[i]] > 0:
            dd = float(((fp - close_arr[t0_idx[i]]) / close_arr[t0_idx[i]]).min())
            drawdowns.append(dd)

    median_drawdown = abs(float(np.median(drawdowns))) if drawdowns else 0.03

    # Final stop: use ATR stop, but floor it at the median drawdown
    # so we never set a stop tighter than history suggests is needed
    suggested_stop = round(max(atr_stop, median_drawdown), 4)

    # Top 5 analogue dates for display
    top_idx = np.argsort(composite)[::-1][:5]
    top_matches = []
    for i in top_idx:
        date_idx = int(t0_idx[i])
        if date_idx < len(df.index):
            top_matches.append({
                "date":       df.index[date_idx].strftime("%Y-%m-%d"),
                "similarity": round(float(composite[i]), 4),
                "fwd_return": round(float(fwd_rets[i]),  4),
            })

    cur_price_val = float(close_arr[-1])

    return {
        "ticker":        ticker,
        "status":        "signal",
        "price":         round(cur_price_val, 2),
        "direction":     direction,
        "avg_return":    round(avg_return,    4),
        "win_rate":      round(win_rate,      4),
        "avg_win":       round(avg_win,       4),
        "avg_loss":      round(avg_loss,      4),
        "ev_score":      round(ev_score,      4),
        "match_count":   int(len(composite)),
        "best_sim":      round(float(composite.max()), 4),
        "suggested_stop":  round(suggested_stop,    4),
        "atr_stop":        round(atr_stop,          4),
        "median_drawdown": round(median_drawdown,   4),
        "top_matches":   top_matches,
    }


def scan_sector_stocks(sector_key):
    """
    Run the pattern scanner on all stocks in a sector's watchlist.
    Uses ThreadPoolExecutor for parallel execution — same approach as
    the backtest engine for speed.
    Returns results sorted by EV score descending, signals first.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tickers = SECTOR_STOCKS.get(sector_key, [])
    if not tickers:
        return []

    log.info("-- Stock scan: %s (%d stocks) --", sector_key, len(tickers))
    results = []

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(scan_stock, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"ticker": ticker, "status": "error", "error": str(e)}
            results.append(result)
            log.info("  Scanned %s: %s", ticker,
                     result.get("status", "?") +
                     (f" EV={result['ev_score']*100:.1f}%"
                      if result.get("status") == "signal" else ""))

    # Sort: signals first by EV score, then no_edge, then errors
    def sort_key(r):
        if r["status"] == "signal":   return (0, -r.get("ev_score", 0))
        if r["status"] == "no_edge":  return (1, 0)
        return (2, 0)

    results.sort(key=sort_key)
    log.info("-- Stock scan complete: %d signals --",
             sum(1 for r in results if r["status"] == "signal"))
    return results



@app.route("/api/sectors", methods=["POST"])
def api_sectors():
    """
    Phase 3 sector rotation endpoint.

    Accepts the analogue list from the frontend (already computed by
    /api/analogues) and computes sector performance relative to SPY
    across all analogue periods.

    Request body (JSON):
      analogues : list of analogue dicts from /api/analogues response
                  each must have: end_date, rank

    Response (JSON):
      sectors : list of sector result dicts sorted by combined_score desc
        each has: sector_key, sector_name, etf, combined_score,
                  avg_relative_12, avg_relative_6, consistency,
                  n_analogues, n_outperformed, analogue_detail
    """
    body      = request.get_json(silent=True) or {}
    analogues = body.get("analogues", [])

    if not analogues:
        return jsonify({"error": "No analogues provided"}), 400

    log.info("-- Sector rotation: %d analogues --", len(analogues))

    sectors = score_sectors(analogues)
    if not sectors:
        return jsonify({"error": "Sector data unavailable"}), 500

    return app.response_class(
        response=json.dumps({"sectors": sectors}, cls=NumpyEncoder),
        mimetype="application/json"
    )


# =============================================================================
# SECTION 12 — Phase 3 Step 2 Flask route: individual stock scanner
# =============================================================================

@app.route("/api/scan_stocks", methods=["POST"])
def api_scan_stocks():
    """
    Phase 3 Step 2 stock scanning endpoint.

    Runs the 21-day pattern matching engine on all stocks in a sector's
    watchlist and returns signals ranked by EV score.

    Request body (JSON):
      sector_key : str  — e.g. "XLK", "XLF"

    Response (JSON):
      sector_key : str
      tickers    : list of all tickers scanned
      results    : list of result dicts sorted by EV score
        signals have: ticker, status, price, direction, avg_return,
                      win_rate, avg_win, avg_loss, ev_score, match_count,
                      best_sim, suggested_stop, top_matches
        no_edge have: ticker, status, match_count, best_sim
        errors  have: ticker, status, error
      n_signals  : int — number of stocks with active signals
      scan_time  : float — seconds taken
    """
    import time
    body       = request.get_json(silent=True) or {}
    sector_key = body.get("sector_key", "").upper()

    if sector_key not in SECTOR_STOCKS:
        return jsonify({"error": f"Unknown sector: {sector_key}. "
                                 f"Available: {list(SECTOR_STOCKS.keys())}"}), 400

    t0      = time.time()
    results = scan_sector_stocks(sector_key)
    elapsed = round(time.time() - t0, 1)

    return app.response_class(
        response=json.dumps({
            "sector_key": sector_key,
            "tickers":    SECTOR_STOCKS[sector_key],
            "results":    results,
            "n_signals":  sum(1 for r in results if r["status"] == "signal"),
            "scan_time":  elapsed,
        }, cls=NumpyEncoder),
        mimetype="application/json"
    )



if __name__ == "__main__":
    app.run(debug=False, port=5002, threaded=True)
