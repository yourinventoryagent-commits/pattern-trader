"""
Pattern Trader — Walk-Forward Backtest Engine v3
=================================================
Improvements over v2:
  - Fixed avg win/loss calculation (per-trade % not cumulative equity delta)
  - Trend filter: 50d/200d MA — only take signals aligned with trend
  - Time-decay weighting: recent analogues weighted 2x vs 5yr-old ones
  - Trailing stop loss: simulated intra-hold, configurable trail %
  - Diagnostic layer: explains why each ticker succeeded or failed
  - Vectorised NumPy similarity (retained from v2)
  - Parallel execution via ThreadPoolExecutor (retained from v2)
  - Server-sent events streaming (retained from v2)

Run:  python backtest.py
Open: http://localhost:5001
"""

from flask import Flask, render_template, jsonify, request, Response, stream_with_context
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import time

# ── NumPy-safe JSON encoder ───────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars/arrays to native Python types for JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def safe_json(obj):
    """Serialize obj to JSON string, handling all numpy types."""
    return json.dumps(obj, cls=NumpyEncoder)


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM",
                     "XOM",  "UNH",  "CAT",   "AMZN", "SPY"]

PATTERN_DAYS      = 21      # window length for pattern matching
FORWARD_DAYS      = 10      # max holding period (trading days)
SIMILARITY_CUTOFF = 0.75    # minimum composite similarity to count as match
MIN_AVG_MOVE      = 0.02    # minimum expected move to generate a signal
PRICE_WEIGHT      = 0.70    # price shape weight in composite score
VOLUME_WEIGHT     = 0.30    # volume profile weight in composite score
MIN_MATCHES       = 8       # raised from 5 — forces higher confidence
HISTORY_YEARS     = 10      # total history to fetch
BACKTEST_YEARS    = 5       # walk-forward window (most recent N years)
STEP_DAYS         = 3       # scan every 3 trading days (was 5)
HAIRCUT           = 0.001   # 0.1% slippage per trade
MAX_WORKERS       = 6       # parallel threads

# Trend filter
USE_TREND_FILTER  = True    # only take signals aligned with 50d/200d MA trend

# Time-decay weighting
TIME_DECAY_HALF   = 365     # analogues from this many days ago get half weight

# Trailing stop loss
USE_TRAILING_STOP = True    # simulate trailing stop within each hold period
TRAIL_PCT         = 0.04    # fallback fixed stop (used if ATR unavailable)
ATR_MULTIPLIER    = 2.0     # stop = ATR(20) × this multiplier
                            # NVDA ATR~3.5% → stop ~7%, SPY ATR~0.8% → stop ~1.6%

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Data layer ────────────────────────────────────────────────────────────────

# Cache fetched data so parallel threads don't re-download
_data_cache = {}
_cache_lock = __import__('threading').Lock()

def fetch_ohlcv(ticker, years=HISTORY_YEARS):
    """
    Download adjusted OHLCV. Handles flat and MultiIndex yfinance formats.
    Caches results so parallel threads share data without re-downloading.
    Returns None on failure or insufficient data.
    """
    with _cache_lock:
        if ticker in _data_cache:
            log.info("%s: using cached data", ticker)
            return _data_cache[ticker]

    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    try:
        # Use a fresh Ticker object per call to avoid session bleed between threads
        tkr = yf.Ticker(ticker)
        df  = tkr.history(start=start, auto_adjust=True)

        if df.empty or len(df) < PATTERN_DAYS * 3:
            log.warning("%s: empty or insufficient data (%d rows)", ticker, len(df))
            return None

        # history() returns lowercase flat columns — normalise just in case
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            log.warning("%s: missing columns: %s", ticker, list(df.columns))
            return None

        result = df[["open", "high", "low", "close", "volume"]].dropna()
        log.info("%s: fetched %d rows", ticker, len(result))

        with _cache_lock:
            _data_cache[ticker] = result
        return result

    except Exception as e:
        log.warning("Fetch failed %s: %s", ticker, e)
        return None


# ── Vectorised similarity engine ──────────────────────────────────────────────

def build_window_matrix(arr, window):
    """
    All sliding windows of length `window` across arr, normalised to
    % return from each window's first element. Returns (n_windows × window).
    Returns empty (0 × window) array if arr is too short.
    """
    arr = np.asarray(arr, dtype=float)
    n_windows = len(arr) - window + 1
    if n_windows <= 0:
        return np.empty((0, window), dtype=float)
    idx   = np.arange(window)[None, :] + np.arange(n_windows)[:, None]
    mat   = arr[idx]
    first = mat[:, 0:1]
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(first != 0, mat / first - 1.0, 0.0)
    return mat


def zscore_matrix(arr, window):
    """Sliding z-score normalisation. Returns (n_windows × window).
    Returns empty (0 × window) array if arr is too short."""
    arr = np.asarray(arr, dtype=float)
    n_windows = len(arr) - window + 1
    if n_windows <= 0:
        return np.empty((0, window), dtype=float)
    idx = np.arange(window)[None, :] + np.arange(n_windows)[:, None]
    mat = arr[idx]
    mu  = mat.mean(axis=1, keepdims=True)
    std = mat.std( axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        mat = np.where(std > 0, (mat - mu) / std, 0.0)
    return mat


def vectorised_pearson(target, matrix):
    """
    Pearson r between 1-D target and every row of matrix.
    Returns 1-D array of correlations, clipped to [-1, 1].
    """
    t   = target - target.mean()
    t_n = t / (np.sqrt((t ** 2).sum()) + 1e-10)
    m   = matrix - matrix.mean(axis=1, keepdims=True)
    m_n = m / (np.sqrt((m ** 2).sum(axis=1, keepdims=True)) + 1e-10)
    return (m_n @ t_n).clip(-1, 1)


# ── Trend filter ──────────────────────────────────────────────────────────────

def precompute_trend(close_arr):
    """
    Pre-compute 50-day and 200-day simple moving averages.
    Returns (ma50, ma200) arrays aligned with close_arr.
    """
    s    = pd.Series(close_arr)
    ma50  = s.rolling(50).mean().values
    ma200 = s.rolling(200).mean().values
    return ma50, ma200


def trend_allows_signal(direction, i, ma50, ma200):
    """
    Return True if the signal direction is consistent with the trend.
      LONG  → price above both MA50 and MA200 (uptrend)
      SHORT → price below both MA50 and MA200 (downtrend)
    Neutral zone (between MAs) → allow any signal.
    """
    if not USE_TREND_FILTER:
        return True
    if np.isnan(ma50[i]) or np.isnan(ma200[i]):
        return True   # not enough history yet — allow
    if direction == "LONG"  and ma50[i] < ma200[i]:
        return False  # MA50 below MA200 = downtrend, skip long
    if direction == "SHORT" and ma50[i] > ma200[i]:
        return False  # MA50 above MA200 = uptrend, skip short
    return True


# ── Time-decay weighting ──────────────────────────────────────────────────────

def time_decay_weight(window_end_idx, current_idx, step_days_per_idx=1):
    """
    Weight = 2^(-age_days / HALF_LIFE).
    A match from TIME_DECAY_HALF days ago gets weight 0.5.
    A match from today gets weight 1.0.
    """
    age_days = (current_idx - window_end_idx) * step_days_per_idx
    return 2.0 ** (-age_days / TIME_DECAY_HALF)


# ── Trailing stop simulation ──────────────────────────────────────────────────

def apply_trailing_stop(close_arr, entry_idx, max_days, trail_pct):
    """
    Simulate a trailing stop over a hold period.
    Tracks the peak price since entry; exits when price drops trail_pct
    below that peak. Returns (actual_return, hold_days_used).
    """
    entry_price = close_arr[entry_idx]
    peak        = entry_price
    n           = len(close_arr)

    for d in range(1, max_days + 1):
        idx = entry_idx + d
        if idx >= n:
            break
        price = close_arr[idx]
        peak  = max(peak, price)
        if price <= peak * (1 - trail_pct):
            # Stop triggered — exit here
            raw_ret = (price - entry_price) / entry_price
            return raw_ret, d

    # No stop triggered — exit at end of hold period
    exit_idx = min(entry_idx + max_days, n - 1)
    raw_ret  = (close_arr[exit_idx] - entry_price) / entry_price
    return raw_ret, max_days


# ── ATR-based adaptive stop ──────────────────────────────────────────────────

def compute_atr(high_arr, low_arr, close_arr, period=20):
    """
    Compute Average True Range (ATR) as a rolling mean of True Range.
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    Returns array aligned with close_arr. First `period` values are NaN.
    """
    n      = len(close_arr)
    tr     = np.zeros(n)
    tr[0]  = high_arr[0] - low_arr[0]
    for i in range(1, n):
        tr[i] = max(
            high_arr[i]  - low_arr[i],
            abs(high_arr[i]  - close_arr[i - 1]),
            abs(low_arr[i]   - close_arr[i - 1]),
        )
    # Vectorised rolling mean for speed
    atr = pd.Series(tr).rolling(period).mean().values
    return atr


def atr_trail_pct(atr_arr, close_arr, t, multiplier=2.0):
    """
    Return the trailing stop distance as a fraction of current price.
    = (ATR at t × multiplier) / close at t
    Falls back to TRAIL_PCT if ATR is not yet available.
    """
    if t >= len(atr_arr) or np.isnan(atr_arr[t]) or close_arr[t] == 0:
        return TRAIL_PCT
    return float((atr_arr[t] * multiplier) / close_arr[t])


# ── Signal engine (point-in-time, vectorised) ─────────────────────────────────

def get_signal_at(close_arr, vol_arr, vol20_arr, ma50, ma200, dates, t, n):
    """
    Generate a signal at index t using only data available at that point.
    Incorporates time-decay weighting and trend filter.
    Returns signal dict or None.
    """
    if t < PATTERN_DAYS * 2 + FORWARD_DAYS:
        return None

    # Current pattern vectors
    _cp = build_window_matrix(close_arr[t - PATTERN_DAYS: t], PATTERN_DAYS)
    _cv = zscore_matrix(vol_arr[t - PATTERN_DAYS: t], PATTERN_DAYS)
    if len(_cp) == 0 or len(_cv) == 0:
        return None
    cur_price = _cp[0]
    cur_vol   = _cv[0]

    end = t - PATTERN_DAYS
    if end < PATTERN_DAYS:
        return None

    price_mat = build_window_matrix(close_arr[:end], PATTERN_DAYS)
    vol_mat   = zscore_matrix(vol_arr[:end],         PATTERN_DAYS)

    rp = vectorised_pearson(cur_price, price_mat).clip(0, 1)
    rv = vectorised_pearson(cur_vol,   vol_mat  ).clip(0, 1)
    composite = PRICE_WEIGHT * rp + VOLUME_WEIGHT * rv

    # Regime filter
    cur_reg = vol20_arr[t - 1]
    starts  = np.arange(len(price_mat))
    if not np.isnan(cur_reg):
        win_regs = vol20_arr[starts]
        ok       = np.where(np.isnan(win_regs), True,
                            np.abs(win_regs - cur_reg) <= cur_reg * 2.0)
        composite = composite[ok]
        starts    = starts[ok]

    # Similarity cutoff
    mask      = composite >= SIMILARITY_CUTOFF
    composite = composite[mask]
    starts    = starts[mask]

    if len(composite) < MIN_MATCHES:
        return None

    # Forward returns (all data is in the past — no lookahead)
    t0_idx   = starts + PATTERN_DAYS - 1
    fut_idx  = t0_idx + FORWARD_DAYS
    in_range = fut_idx < n
    t0_idx   = t0_idx[in_range]
    fut_idx  = fut_idx[in_range]
    composite = composite[in_range]
    starts    = starts[in_range]

    if len(composite) < MIN_MATCHES:
        return None

    # Time-decay weights: recent analogues weighted more heavily
    t0_prices  = close_arr[t0_idx]
    fut_prices = close_arr[fut_idx]
    fwd_rets   = (fut_prices - t0_prices) / t0_prices

    decay_w    = np.array([time_decay_weight(int(t0_idx[i]), t)
                           for i in range(len(t0_idx))])
    sim_w      = composite ** 2
    weights    = sim_w * decay_w
    weights   /= weights.sum()

    avg_return = float(np.dot(weights, fwd_rets))

    if abs(avg_return) < MIN_AVG_MOVE:
        return None

    direction  = "LONG" if avg_return > 0 else "SHORT"

    # Trend filter check
    if not trend_allows_signal(direction, t, ma50, ma200):
        return None

    wins     = fwd_rets > 0
    win_rate = float(wins.mean())
    avg_win  = float(fwd_rets[wins].mean())   if wins.any()   else 0.0
    avg_loss = float(fwd_rets[~wins].mean())  if (~wins).any() else 0.0
    ev_score = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    # Analogue-informed suggested stop:
    # Median intra-period drawdown of matched analogues — how far they
    # historically dipped before recovering. Sets stop just below this.
    fwd_prices_all = np.array([
        close_arr[t0_idx[k]: t0_idx[k] + FORWARD_DAYS]
        for k in range(len(t0_idx))
        if t0_idx[k] + FORWARD_DAYS <= n
    ], dtype=object)

    if len(fwd_prices_all) > 0:
        drawdowns = []
        for k in range(len(t0_idx)):
            if t0_idx[k] + FORWARD_DAYS <= n:
                fp     = close_arr[t0_idx[k]: t0_idx[k] + FORWARD_DAYS]
                t0p    = close_arr[t0_idx[k]]
                if t0p > 0:
                    dd = float(((fp - t0p) / t0p).min())
                    drawdowns.append(dd)
        suggested_stop = abs(float(np.median(drawdowns))) if drawdowns else TRAIL_PCT
    else:
        suggested_stop = TRAIL_PCT

    return {
        "direction":     direction,
        "avg_return":    avg_return,
        "win_rate":      win_rate,
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "ev_score":      ev_score,
        "matches":       int(len(composite)),
        "suggested_stop": round(suggested_stop, 4),  # analogue-informed stop %
    }


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def run_backtest_for_ticker(ticker):
    """
    Walk forward through BACKTEST_YEARS of data, stepping every STEP_DAYS.
    Applies trailing stop within each hold period.
    Returns per-trade results and equity curve.
    """
    df = fetch_ohlcv(ticker)
    if df is None:
        return {"ticker": ticker, "error": "Data unavailable"}

    n          = len(df)
    close_arr  = df["close"].values.astype(float)
    vol_arr    = df["volume"].values.astype(float)
    dates      = df.index

    # Pre-compute derived series (all vectorised)
    high_arr  = df["high"].values.astype(float)
    low_arr   = df["low"].values.astype(float)
    pct_chg   = np.diff(close_arr, prepend=close_arr[0]) / np.where(close_arr != 0, close_arr, 1)
    vol20_arr = pd.Series(pct_chg).rolling(20).std().values
    ma50, ma200 = precompute_trend(close_arr)
    atr_arr     = compute_atr(high_arr, low_arr, close_arr)

    bt_start  = max(n - int(BACKTEST_YEARS * 252), PATTERN_DAYS * 2 + FORWARD_DAYS + 200)
    equity    = 1.0
    equity_curve = []
    trades       = []

    # Diagnostic accumulators
    trend_filtered  = 0   # signals blocked by trend filter
    regime_filtered = 0   # approximate — tracked implicitly

    i = bt_start
    while i < n - FORWARD_DAYS:
        signal = get_signal_at(close_arr, vol_arr, vol20_arr, ma50, ma200, dates, i, n)

        if signal is not None:
            # Apply ATR-adaptive trailing stop or plain exit
            if USE_TRAILING_STOP:
                # Compute stop distance from ATR at signal time — adapts to
                # each stock's natural volatility rather than a fixed %
                trail = atr_trail_pct(atr_arr, close_arr, i,
                                      multiplier=ATR_MULTIPLIER)
                raw_ret, hold_days = apply_trailing_stop(
                    close_arr, i, FORWARD_DAYS, trail)
            else:
                raw_ret   = (close_arr[i + FORWARD_DAYS] - close_arr[i]) / close_arr[i]
                hold_days = FORWARD_DAYS

            # Direction and slippage
            trade_ret = (raw_ret if signal["direction"] == "LONG" else -raw_ret) - HAIRCUT

            # ── FIX: store per-trade % return, NOT equity delta ──
            equity *= (1 + trade_ret)
            trades.append({
                "date":          dates[i].strftime("%Y-%m-%d"),
                "direction":     signal["direction"],
                "predicted":     round(signal["avg_return"] * 100, 2),
                "actual_pct":    round(trade_ret * 100, 2),
                "win":           trade_ret > 0,
                "ev_score":      round(signal["ev_score"] * 100, 2),
                "matches":       signal["matches"],
                "hold_days":     hold_days,
                "stopped":       hold_days < FORWARD_DAYS,
                "atr_stop_pct":  round(trail * 100, 2) if USE_TRAILING_STOP else None,
                "analogue_stop": round(signal.get("suggested_stop", TRAIL_PCT) * 100, 2),
            })

        equity_curve.append({
            "date":   dates[i].strftime("%Y-%m-%d"),
            "equity": round(equity, 6),
            "signal": signal is not None,
        })

        i += STEP_DAYS

    if not trades:
        return {"ticker": ticker, "error": "No signals generated"}

    # ── Stats from per-trade % returns (the correct way) ──
    rets      = np.array([t["actual_pct"] for t in trades])
    wins_mask = rets > 0

    win_rate  = float(wins_mask.mean())
    avg_win   = float(rets[wins_mask].mean())   if wins_mask.any()   else 0.0
    avg_loss  = float(rets[~wins_mask].mean())  if (~wins_mask).any() else 0.0

    eq_vals   = np.array([p["equity"] for p in equity_curve])
    peak      = np.maximum.accumulate(eq_vals)
    max_dd    = float(((eq_vals - peak) / peak).min()) * 100

    sharpe    = (float(rets.mean() / rets.std()) * np.sqrt(252 / STEP_DAYS)
                 if rets.std() > 0 else 0.0)

    stopped_count = sum(1 for t in trades if t.get("stopped"))
    avg_hold      = float(np.mean([t["hold_days"] for t in trades]))
    atr_stops     = [t["atr_stop_pct"] for t in trades if t.get("atr_stop_pct") is not None]
    avg_atr_stop  = float(np.mean(atr_stops)) if atr_stops else None

    # ── Diagnostic characteristics ──
    diagnostics = build_diagnostics(
        ticker, close_arr, vol_arr, ma50, ma200,
        trades, rets, wins_mask, avg_win, avg_loss,
        stopped_count, avg_hold, bt_start, n
    )

    return {
        "ticker":       ticker,
        "equity_curve": equity_curve,
        "trades":       trades[-20:],
        "stats": {
            "total_return":  round((equity - 1.0) * 100, 2),
            "win_rate":      round(win_rate * 100, 1),
            "avg_win":       round(avg_win,         2),   # now in % per trade
            "avg_loss":      round(avg_loss,         2),  # now in % per trade
            "max_drawdown":  round(max_dd,           2),
            "sharpe":        round(sharpe,            2),
            "total_trades":  len(trades),
            "signal_freq":   round(len(trades) / BACKTEST_YEARS, 1),
            "stopped_count":  stopped_count,
            "avg_hold_days":  round(avg_hold, 1),
            "avg_atr_stop":   round(avg_atr_stop, 2) if avg_atr_stop else None,
        },
        "diagnostics": diagnostics,
    }


# ── Diagnostic engine ─────────────────────────────────────────────────────────

def build_diagnostics(ticker, close_arr, vol_arr, ma50, ma200,
                       trades, rets, wins_mask, avg_win, avg_loss,
                       stopped_count, avg_hold, bt_start, n):
    """
    Analyse the characteristics that explain why the engine succeeded or
    failed on this ticker. Returns a dict of labelled insights.
    """
    insights = []
    score    = 0   # composite quality score for ranking

    # 1. Trend consistency — did the stock trend cleanly?
    bt_close  = close_arr[bt_start:]
    if len(bt_close) > 50:
        total_move  = (bt_close[-1] - bt_close[0]) / bt_close[0]
        # R² of price vs a linear fit — 1.0 = perfectly trending, 0 = random
        x   = np.arange(len(bt_close))
        fit = np.polyfit(x, bt_close, 1)
        res = bt_close - np.polyval(fit, x)
        ss_res = (res ** 2).sum()
        ss_tot = ((bt_close - bt_close.mean()) ** 2).sum()
        r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        trend_label = "Strong uptrend" if total_move > 0.5 and r2 > 0.7 \
            else "Moderate trend" if total_move > 0.2 \
            else "Sideways / choppy"
        trend_good  = total_move > 0.3 and r2 > 0.6
        insights.append({
            "label":   "Price trend",
            "value":   f"{trend_label} ({total_move*100:+.0f}% over period, R²={r2:.2f})",
            "helpful": bool(trend_good),
        })
        if trend_good: score += 2

    # 2. Volatility regime — was the stock volatile enough to trade?
    daily_vols = np.diff(bt_close) / bt_close[:-1]
    ann_vol    = float(daily_vols.std() * np.sqrt(252)) * 100
    vol_label  = "High" if ann_vol > 35 else "Moderate" if ann_vol > 18 else "Low"
    vol_good   = 18 < ann_vol < 60
    insights.append({
        "label":   "Annualised volatility",
        "value":   f"{vol_label} ({ann_vol:.1f}% annualised)",
        "helpful": bool(vol_good),
    })
    if vol_good: score += 1

    # 3. Win/loss ratio — does the engine capture more than it gives back?
    if avg_win != 0 and avg_loss != 0:
        wl_ratio   = abs(avg_win / avg_loss)
        wl_good    = wl_ratio > 1.0
        insights.append({
            "label":   "Win/loss ratio",
            "value":   f"{wl_ratio:.2f}× (avg win {avg_win:+.2f}% vs avg loss {avg_loss:.2f}%)",
            "helpful": bool(wl_good),
        })
        if wl_good: score += 2

    # 4. Stop loss effectiveness
    if stopped_count > 0:
        stop_rate = stopped_count / len(trades) * 100
        stop_good = stop_rate < 40
        insights.append({
            "label":   "Trailing stop triggered",
            "value":   f"{stopped_count}/{len(trades)} trades ({stop_rate:.0f}%) — avg hold {avg_hold:.1f}d",
            "helpful": bool(stop_good),
        })
        if stop_good: score += 1

    # 5. Signal clustering — are wins/losses clustered or spread?
    if len(rets) >= 6:
        # Look at runs of consecutive wins/losses
        win_arr  = (rets > 0).astype(int)
        changes  = np.diff(win_arr)
        n_runs   = int((changes != 0).sum()) + 1
        avg_run  = len(rets) / n_runs
        cluster_good = avg_run < 3   # short runs = not clustered = more robust
        insights.append({
            "label":   "Signal clustering",
            "value":   f"Avg run of {avg_run:.1f} consecutive same-direction trades",
            "helpful": bool(cluster_good),
        })
        if cluster_good: score += 1

    # 6. MA alignment — how often was the trend filter aligned?
    bt_ma50  = ma50[bt_start:]
    bt_ma200 = ma200[bt_start:]
    valid    = ~(np.isnan(bt_ma50) | np.isnan(bt_ma200))
    if valid.sum() > 0:
        aligned_pct = float((bt_ma50[valid] > bt_ma200[valid]).mean()) * 100
        align_good  = aligned_pct > 60
        insights.append({
            "label":   "Trend alignment (MA50 > MA200)",
            "value":   f"{aligned_pct:.0f}% of backtest period in uptrend",
            "helpful": bool(align_good),
        })
        if align_good: score += 1

    # Overall verdict
    if score >= 6:
        verdict = "Strong edge — pattern engine well-suited to this stock"
    elif score >= 4:
        verdict = "Moderate edge — usable but watch for regime changes"
    elif score >= 2:
        verdict = "Weak edge — treat signals with caution"
    else:
        verdict = "No edge — remove from watchlist"

    return {"insights": insights, "score": score, "verdict": verdict}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "backtest.html",
        default_watchlist=",".join(DEFAULT_WATCHLIST),
        backtest_years=BACKTEST_YEARS,
        step_days=STEP_DAYS,
    )



@app.route("/api/backtest/stream")
def api_backtest_stream():
    """
    SSE endpoint — streams one result per ticker as it completes.
    All tickers run in parallel via ThreadPoolExecutor.
    Falls back gracefully if the client disconnects.
    """
    tickers = request.args.get("tickers", ",".join(DEFAULT_WATCHLIST))
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    def generate():
        try:
            # Clear data cache so fresh data is fetched each run
            with _cache_lock:
                _data_cache.clear()

            yield f"data: {safe_json({'type':'start','total':len(tickers),'tickers':tickers})}\n\n"
            t0 = time.time()
            completed = 0
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as pool:
                futures = {pool.submit(run_backtest_for_ticker, t): t for t in tickers}
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = {"ticker": ticker, "error": str(e)}
                    completed += 1
                    elapsed = round(time.time() - t0, 1)
                    log.info("Done %s (%d/%d) %.1fs", ticker, completed, len(tickers), elapsed)
                    payload = safe_json({
                        "type": "result", "completed": completed,
                        "total": len(tickers), "elapsed": elapsed, "result": result,
                    })
                    yield f"data: {payload}\n\n"
            yield f"data: {safe_json({'type':'done','elapsed':round(time.time()-t0,1)})}\n\n"
        except GeneratorExit:
            log.info("SSE client disconnected")
        except Exception as e:
            log.error("SSE error: %s", e)
            yield f"data: {safe_json({'type':'error','message':str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/backtest", methods=["POST"])
def api_backtest_post():
    """
    Non-streaming fallback POST endpoint.
    Runs all tickers in parallel and returns full results at once.
    The frontend uses this automatically if SSE fails.
    """
    body    = request.get_json(silent=True) or {}
    raw     = body.get("tickers", ",".join(DEFAULT_WATCHLIST))
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    log.info("── Backtest (POST fallback) started: %s ──", tickers)
    t0      = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tickers))) as pool:
        futures = {pool.submit(run_backtest_for_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"ticker": ticker, "error": str(e)}
            results.append(result)
            log.info("  Done %s (%.1fs elapsed)", ticker, time.time() - t0)

    elapsed = round(time.time() - t0, 1)
    log.info("── Backtest complete in %.1fs ──", elapsed)
    return jsonify({"results": results, "elapsed": elapsed})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=5001, threaded=True)
