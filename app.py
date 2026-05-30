"""
Pattern Trader — Historical Analogue Engine
============================================
Scans a watchlist for 21-day price/volume patterns that match historical
analogues, ranks signals by expected value, and serves results via Flask.

Run:  python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import logging

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST = ["AAPL", "MSFT", "GOOGL", "NVDA", "JPM",
                     "XOM", "UNH", "CAT", "AMZN", "SPY"]

PATTERN_DAYS      = 21      # length of the pattern window
FORWARD_DAYS      = [10, 21]# forward-return horizons to measure
SIMILARITY_CUTOFF = 0.75    # minimum Pearson r to count as a match
MIN_AVG_MOVE      = 0.02    # minimum absolute avg forward return for a signal
PRICE_WEIGHT      = 0.70    # weight of price correlation in composite score
VOLUME_WEIGHT     = 0.30    # weight of volume correlation in composite score
MIN_MATCHES       = 5       # minimum historical matches needed to score a signal
HISTORY_YEARS     = 10      # years of price history to scan

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Data layer ────────────────────────────────────────────────────────────────

def fetch_ohlcv(ticker, years=HISTORY_YEARS):
    """
    Download adjusted daily OHLCV. Returns None on failure.
    Handles both flat and MultiIndex column formats across yfinance versions.
    """
    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    try:
        df = yf.download(ticker, start=start, auto_adjust=True,
                         progress=False, group_by="column")

        if df.empty or len(df) < PATTERN_DAYS * 3:
            log.warning("%s: insufficient data (%d rows)", ticker, len(df))
            return None

        # Flatten MultiIndex columns — yfinance >=0.2.x returns (field, ticker) tuples
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() for col in df.columns]
        else:
            df.columns = [col.lower() for col in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            log.warning("%s: missing columns after flatten. Got: %s", ticker, list(df.columns))
            return None

        result = df[["open", "high", "low", "close", "volume"]].dropna()
        log.info("%s: fetched %d rows", ticker, len(result))
        return result

    except Exception as e:
        log.warning("Failed to fetch %s: %s", ticker, e)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_numpy(series_or_array):
    """
    Safely extract a clean 1-D float64 numpy array from a pandas Series,
    DataFrame column, or existing numpy array. Handles MultiIndex columns
    and nested structures that yfinance sometimes produces.
    """
    if isinstance(series_or_array, pd.DataFrame):
        series_or_array = series_or_array.iloc[:, 0]
    if isinstance(series_or_array, pd.Series):
        series_or_array = series_or_array.values
    arr = np.array(series_or_array, dtype=float).flatten()
    return arr


def to_return_series(prices):
    """Convert price array to % returns anchored at 0 from day 1."""
    arr = to_numpy(prices)
    if arr[0] == 0:
        return np.zeros_like(arr)
    return (arr / arr[0]) - 1.0


def zscore(series):
    """Z-score normalise a 1-D array. Returns zeros if std == 0."""
    arr = to_numpy(series)
    std = arr.std()
    return (arr - arr.mean()) / std if std > 0 else np.zeros_like(arr)


# ── Similarity ────────────────────────────────────────────────────────────────

def composite_similarity(price_a, price_b, vol_a, vol_b):
    """
    Weighted composite of price-shape Pearson r and volume-shape Pearson r.
    Returns 0.0 if correlation cannot be computed.
    """
    try:
        r_price, _ = pearsonr(price_a, price_b)
        r_vol,   _ = pearsonr(vol_a,   vol_b)
        r_price = max(float(r_price), 0.0)
        r_vol   = max(float(r_vol),   0.0)
        return PRICE_WEIGHT * r_price + VOLUME_WEIGHT * r_vol
    except Exception:
        return 0.0


# ── Analogue scanning ─────────────────────────────────────────────────────────

def find_analogues(df):
    """
    Slide a PATTERN_DAYS window over the full history (excluding the most
    recent PATTERN_DAYS rows which form the current pattern). Record every
    window that meets SIMILARITY_CUTOFF along with its forward returns.

    Returns a list of match dicts sorted by similarity descending.
    """
    n = len(df)
    if n < PATTERN_DAYS * 2 + max(FORWARD_DAYS):
        log.warning("Not enough rows to scan: %d", n)
        return []

    # Current pattern: last PATTERN_DAYS rows
    cur_close  = to_numpy(df["close"].iloc[-PATTERN_DAYS:])
    cur_vol    = to_numpy(df["volume"].iloc[-PATTERN_DAYS:])
    cur_price  = to_return_series(cur_close)
    cur_vol_z  = zscore(cur_vol)

    # Pre-compute rolling 20-day volatility for regime filtering
    pct_changes = to_numpy(df["close"].pct_change())
    vol20 = pd.Series(pct_changes).rolling(20).std().values
    cur_regime  = float(vol20[-1]) if pd.notna(vol20[-1]) else None

    matches  = []
    scan_end = n - PATTERN_DAYS - max(FORWARD_DAYS)

    for i in range(PATTERN_DAYS, scan_end):
        # Regime filter: broad check, only skip extreme outlier regimes
        # Using 2.0x tolerance (was 0.75x — too tight, filtered everything out)
        if cur_regime is not None and pd.notna(vol20[i - 1]):
            win_regime = float(vol20[i - 1])
            if abs(win_regime - cur_regime) > cur_regime * 2.0:
                continue

        hist_close = to_numpy(df["close"].iloc[i - PATTERN_DAYS: i])
        hist_vol   = to_numpy(df["volume"].iloc[i - PATTERN_DAYS: i])
        hist_price = to_return_series(hist_close)
        hist_vol_z = zscore(hist_vol)

        score = composite_similarity(cur_price, hist_price, cur_vol_z, hist_vol_z)
        if score < SIMILARITY_CUTOFF:
            continue

        # Forward returns from t0 (end of this historical window)
        t0_price_val = float(df["close"].iloc[i - 1])
        forward_returns = {}
        for fwd in FORWARD_DAYS:
            future_idx = i - 1 + fwd
            if future_idx < n:
                fut_val = float(df["close"].iloc[future_idx])
                forward_returns[fwd] = (fut_val - t0_price_val) / t0_price_val
            else:
                forward_returns[fwd] = None

        # Max drawdown within the primary forward window
        fwd_primary = FORWARD_DAYS[0]
        fwd_prices  = to_numpy(df["close"].iloc[i - 1: i - 1 + fwd_primary])
        drawdown    = float(((fwd_prices - t0_price_val) / t0_price_val).min())

        matches.append({
            "date":          df.index[i - 1].strftime("%Y-%m-%d"),
            "similarity":    round(score, 4),
            "similarity_sq": round(score ** 2, 4),
            "forward":       forward_returns,
            "max_drawdown":  round(drawdown, 4),
        })

    log.info("  → %d matches found above %.0f%% threshold", len(matches), SIMILARITY_CUTOFF * 100)
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


# ── Signal scoring ─────────────────────────────────────────────────────────────

def score_signal(matches, horizon=None):
    """
    Aggregate rho^2-weighted forward returns across all matches.
    Returns None if too few matches or expected move is below threshold.
    """
    if horizon is None:
        horizon = FORWARD_DAYS[0]

    valid = [m for m in matches if m["forward"].get(horizon) is not None]
    if len(valid) < MIN_MATCHES:
        return None

    weights = np.array([m["similarity_sq"] for m in valid])
    returns = np.array([m["forward"][horizon] for m in valid])
    weights = weights / weights.sum()

    avg_return = float(np.dot(weights, returns))
    wins       = returns > 0
    win_rate   = float(wins.mean())
    avg_win    = float(returns[wins].mean())   if wins.any()   else 0.0
    avg_loss   = float(returns[~wins].mean())  if (~wins).any() else 0.0
    volatility = float(returns.std())
    avg_dd     = float(np.mean([m["max_drawdown"] for m in valid]))
    ev_score   = win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    if abs(avg_return) < MIN_AVG_MOVE:
        log.info("  → Signal suppressed: avg_return %.2f%% below MIN_AVG_MOVE threshold",
                 avg_return * 100)
        return None

    # Analogue-informed suggested stop:
    # Median intra-period drawdown across matched analogues.
    # Tells you how much room to give the trade before stopping out.
    analogue_stops = [abs(m["max_drawdown"]) for m in valid if m["max_drawdown"] < 0]
    suggested_stop = round(float(np.median(analogue_stops)), 4) if analogue_stops else 0.04

    return {
        "direction":     "LONG" if avg_return > 0 else "SHORT",
        "avg_return":    round(avg_return,  4),
        "win_rate":      round(win_rate,    4),
        "avg_win":       round(avg_win,     4),
        "avg_loss":      round(avg_loss,    4),
        "volatility":    round(volatility,  4),
        "avg_drawdown":  round(avg_dd,      4),
        "ev_score":      round(ev_score,    4),
        "match_count":   len(valid),
        "top_matches":   valid[:5],
        "suggested_stop": suggested_stop,
    }


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan(tickers):
    """
    Run the full engine across a watchlist. Returns results sorted by
    EV score descending. All tickers returned so UI shows full state.
    """
    results = []

    for ticker in tickers:
        log.info("Scanning %s ...", ticker)
        df = fetch_ohlcv(ticker)

        if df is None:
            results.append({"ticker": ticker, "status": "error",
                             "error": "Data unavailable"})
            continue

        matches = find_analogues(df)

        if not matches:
            results.append({"ticker": ticker, "status": "no_matches",
                             "match_count": 0})
            continue

        signal = score_signal(matches)

        if signal is None:
            results.append({
                "ticker":      ticker,
                "status":      "no_edge",
                "match_count": len(matches),
                "best_sim":    matches[0]["similarity"],
            })
            continue

        cur_price = None
        try:
            cur_price = round(float(df["close"].iloc[-1]), 2)
        except Exception:
            pass

        results.append({
            "ticker":      ticker,
            "status":      "signal",
            "price":       cur_price,
            "match_count": signal["match_count"],
            "best_sim":    matches[0]["similarity"],
            **signal,
        })

    def sort_key(r):
        if r["status"] == "signal":     return (0, -r["ev_score"])
        if r["status"] == "no_edge":    return (1, 0)
        if r["status"] == "no_matches": return (2, 0)
        return (3, 0)

    results.sort(key=sort_key)
    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html",
                           default_watchlist=",".join(DEFAULT_WATCHLIST))


@app.route("/api/scan", methods=["POST"])
def api_scan():
    body    = request.get_json(silent=True) or {}
    raw     = body.get("tickers", ",".join(DEFAULT_WATCHLIST))
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    log.info("── Scan started for: %s ──", tickers)
    results  = run_scan(tickers)
    statuses = {}
    for r in results:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    log.info("── Scan complete: %s ──", statuses)

    return jsonify({"results": results,
                    "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M")})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=False, port=5000)
