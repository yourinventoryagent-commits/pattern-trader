"""
Pattern Trader — Trade Journal
================================
Standalone trade journal for tracking pattern signal trades.
Logs entries, tracks open positions with live P&L, records exits,
and produces performance analytics comparing actual vs expected returns.

Run:  python journal.py
Open: http://localhost:5003

Data stored in: journal_trades.json (same folder)
"""

from flask import Flask, render_template, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

# =============================================================================
# SECTION 1 — App setup
# =============================================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRADES_FILE = os.path.join(os.path.dirname(__file__), "journal_trades.json")


# =============================================================================
# SECTION 2 — JSON persistence
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def load_trades():
    """Load all trades from the JSON file. Returns empty list if file missing."""
    if not os.path.exists(TRADES_FILE):
        return []
    try:
        with open(TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to load trades: %s", e)
        return []


def save_trades(trades):
    """Save all trades to the JSON file atomically."""
    try:
        tmp = TRADES_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(trades, f, indent=2, cls=NumpyEncoder)
        os.replace(tmp, TRADES_FILE)
        return True
    except Exception as e:
        log.error("Failed to save trades: %s", e)
        return False


def next_id(trades):
    """Generate next sequential trade ID."""
    if not trades:
        return "T001"
    ids  = [int(t["id"][1:]) for t in trades if t["id"].startswith("T")]
    return f"T{max(ids) + 1:03d}"


# =============================================================================
# SECTION 3 — Live price fetching
# =============================================================================

_price_cache     = {}
_price_cache_ts  = {}
PRICE_CACHE_SECS = 60   # refresh prices every 60 seconds


def get_current_price(ticker):
    """
    Fetch current price for a ticker. Caches for 60 seconds to avoid
    hammering yfinance on every page load with many open positions.
    Returns float or None on failure.
    """
    now = datetime.now().timestamp()
    if ticker in _price_cache and (now - _price_cache_ts.get(ticker, 0)) < PRICE_CACHE_SECS:
        return _price_cache[ticker]

    try:
        tkr   = yf.Ticker(ticker)
        hist  = tkr.history(period="2d", auto_adjust=True)
        if hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        _price_cache[ticker]    = price
        _price_cache_ts[ticker] = now
        return price
    except Exception as e:
        log.warning("Price fetch failed %s: %s", ticker, e)
        return None


def enrich_open_positions(trades):
    """
    Add current price and live P&L to all open trades.
    Modifies trades in place, returns the list.
    """
    for t in trades:
        if t["status"] != "open":
            continue
        price = get_current_price(t["ticker"])
        if price is not None:
            t["current_price"] = round(price, 2)
            entry = t["entry_price"]
            if t["direction"] == "LONG":
                t["live_pct"] = round((price - entry) / entry * 100, 2)
            else:
                t["live_pct"] = round((entry - price) / entry * 100, 2)
            t["live_pnl"] = round(t["live_pct"] / 100 * t["dollar_amount"], 2)

            # Days held
            entry_dt = datetime.strptime(t["entry_date"], "%Y-%m-%d")
            t["days_held"] = (datetime.now() - entry_dt).days

            # Stop/target status
            if t["direction"] == "LONG":
                t["stop_breached"]   = price <= t.get("stop_price", 0)
                t["target_reached"]  = price >= t.get("target_price", 9999)
            else:
                t["stop_breached"]   = price >= t.get("stop_price", 9999)
                t["target_reached"]  = price <= t.get("target_price", 0)
        else:
            t["current_price"] = None
            t["live_pct"]      = None
            t["live_pnl"]      = None
            t["days_held"]     = None
    return trades


# =============================================================================
# SECTION 4 — Analytics
# =============================================================================

def compute_analytics(trades):
    """
    Compute performance analytics across all closed trades.
    Returns a dict of summary statistics.
    """
    closed = [t for t in trades if t["status"] == "closed"]
    open_  = [t for t in trades if t["status"] == "open"]

    if not closed:
        return {
            "total_trades":    0,
            "open_trades":     len(open_),
            "win_rate":        None,
            "avg_actual_ev":   None,
            "avg_expected_ev": None,
            "total_pnl":       0,
            "best_trade":      None,
            "worst_trade":     None,
            "avg_hold_days":   None,
            "exit_reasons":    {},
            "by_sector":       {},
            "calibration":     None,
        }

    rets      = [t["actual_pct"] for t in closed]
    exp_rets  = [t.get("expected_ev", 0) * 100 for t in closed]
    wins      = [r for r in rets if r > 0]
    pnls      = [t.get("actual_pnl", 0) for t in closed]
    holds     = [t.get("days_held", 0) for t in closed if t.get("days_held")]

    win_rate  = len(wins) / len(rets) if rets else 0
    avg_ret   = float(np.mean(rets))  if rets else 0
    avg_exp   = float(np.mean(exp_rets)) if exp_rets else 0

    # Exit reason breakdown
    reasons = {}
    for t in closed:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    # By sector
    by_sector = {}
    for t in closed:
        sec = t.get("sector", "unknown")
        if sec not in by_sector:
            by_sector[sec] = {"trades": 0, "wins": 0, "total_pct": 0}
        by_sector[sec]["trades"]    += 1
        by_sector[sec]["total_pct"] += t["actual_pct"]
        if t["actual_pct"] > 0:
            by_sector[sec]["wins"] += 1
    for sec in by_sector:
        d = by_sector[sec]
        d["win_rate"] = round(d["wins"] / d["trades"] * 100, 1)
        d["avg_pct"]  = round(d["total_pct"] / d["trades"], 2)

    # Calibration: actual EV vs expected EV
    calibration = round(avg_ret - avg_exp, 2) if exp_rets else None

    # P&L curve for chart (cumulative)
    sorted_closed = sorted(closed, key=lambda t: t.get("exit_date", ""))
    cumulative    = []
    running       = 0
    for t in sorted_closed:
        running += t.get("actual_pnl", 0)
        cumulative.append({
            "date":   t.get("exit_date", ""),
            "pnl":    round(running, 2),
            "ticker": t["ticker"],
        })

    best  = max(closed, key=lambda t: t["actual_pct"]) if closed else None
    worst = min(closed, key=lambda t: t["actual_pct"]) if closed else None

    return {
        "total_trades":    len(closed),
        "open_trades":     len(open_),
        "win_rate":        round(win_rate * 100, 1),
        "avg_actual_ev":   round(avg_ret, 2),
        "avg_expected_ev": round(avg_exp, 2),
        "calibration":     calibration,
        "total_pnl":       round(sum(pnls), 2),
        "best_trade":      {"ticker": best["ticker"], "pct": best["actual_pct"]} if best else None,
        "worst_trade":     {"ticker": worst["ticker"], "pct": worst["actual_pct"]} if worst else None,
        "avg_hold_days":   round(float(np.mean(holds)), 1) if holds else None,
        "exit_reasons":    reasons,
        "by_sector":       by_sector,
        "pnl_curve":       cumulative,
    }


# =============================================================================
# SECTION 5 — Flask routes
# =============================================================================

@app.route("/")
def index():
    return render_template("journal.html")


@app.route("/api/trades", methods=["GET"])
def api_get_trades():
    """Return all trades with live P&L on open positions."""
    trades = load_trades()
    trades = enrich_open_positions(trades)
    analytics = compute_analytics(trades)
    return app.response_class(
        response=json.dumps({
            "trades":    trades,
            "analytics": analytics,
        }, cls=NumpyEncoder),
        mimetype="application/json"
    )


@app.route("/api/trades", methods=["POST"])
def api_log_trade():
    """
    Log a new trade entry.

    Required fields:
      ticker, direction, entry_date, entry_price, dollar_amount,
      shares, sector, layer (1 or 2)

    Optional signal fields (pre-filled from scanner):
      expected_ev, win_rate, stop_price, target_price,
      match_count, best_sim, ev_score, suggested_stop,
      macro_yield_curve, macro_hy_spread, macro_fed_funds,
      notes
    """
    body = request.get_json(silent=True) or {}

    required = ["ticker", "direction", "entry_date", "entry_price",
                "dollar_amount", "shares", "sector", "layer"]
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    trades = load_trades()
    trade  = {
        "id":              next_id(trades),
        "status":          "open",
        "ticker":          body["ticker"].upper(),
        "direction":       body["direction"].upper(),
        "entry_date":      body["entry_date"],
        "entry_price":     float(body["entry_price"]),
        "dollar_amount":   float(body["dollar_amount"]),
        "shares":          float(body["shares"]),
        "sector":          body.get("sector", "unknown"),
        "layer":           int(body.get("layer", 2)),
        "logged_at":       datetime.now().isoformat(),

        # Signal data at entry — for calibration analysis
        "expected_ev":     float(body.get("expected_ev", 0)),
        "expected_win_rate": float(body.get("win_rate", 0)),
        "ev_score":        float(body.get("ev_score", 0)),
        "stop_price":      float(body.get("stop_price", 0)),
        "target_price":    float(body.get("target_price", 0)),
        "match_count":     int(body.get("match_count", 0)),
        "best_sim":        float(body.get("best_sim", 0)),
        "suggested_stop":  float(body.get("suggested_stop", 0)),

        # Macro context at entry
        "macro_yield_curve": body.get("macro_yield_curve"),
        "macro_hy_spread":   body.get("macro_hy_spread"),
        "macro_fed_funds":   body.get("macro_fed_funds"),

        # Exit fields — populated when closed
        "exit_date":       None,
        "exit_price":      None,
        "exit_reason":     None,
        "actual_pct":      None,
        "actual_pnl":      None,
        "days_held":       None,

        "notes":           body.get("notes", ""),
    }

    trades.append(trade)
    if save_trades(trades):
        log.info("Logged trade %s: %s %s @ $%.2f",
                 trade["id"], trade["direction"], trade["ticker"], trade["entry_price"])
        return jsonify({"success": True, "trade": trade})
    return jsonify({"error": "Failed to save"}), 500


@app.route("/api/trades/<trade_id>/close", methods=["POST"])
def api_close_trade(trade_id):
    """
    Close an open trade and record the outcome.

    Required fields:
      exit_price, exit_reason
      (exit_reason: 'stop', 'target', 'time_exit', 'manual', 'macro_trigger')

    Optional:
      exit_date (defaults to today), notes
    """
    body   = request.get_json(silent=True) or {}
    trades = load_trades()

    trade = next((t for t in trades if t["id"] == trade_id), None)
    if not trade:
        return jsonify({"error": f"Trade {trade_id} not found"}), 404
    if trade["status"] != "open":
        return jsonify({"error": f"Trade {trade_id} is already closed"}), 400

    exit_price  = float(body.get("exit_price", 0))
    exit_date   = body.get("exit_date", datetime.now().strftime("%Y-%m-%d"))
    exit_reason = body.get("exit_reason", "manual")

    entry_price = trade["entry_price"]
    if trade["direction"] == "LONG":
        actual_pct = (exit_price - entry_price) / entry_price * 100
    else:
        actual_pct = (entry_price - exit_price) / entry_price * 100

    actual_pnl = actual_pct / 100 * trade["dollar_amount"]

    entry_dt   = datetime.strptime(trade["entry_date"], "%Y-%m-%d")
    exit_dt    = datetime.strptime(exit_date, "%Y-%m-%d")
    days_held  = (exit_dt - entry_dt).days

    trade.update({
        "status":      "closed",
        "exit_date":   exit_date,
        "exit_price":  round(exit_price, 2),
        "exit_reason": exit_reason,
        "actual_pct":  round(actual_pct, 2),
        "actual_pnl":  round(actual_pnl, 2),
        "days_held":   days_held,
        "notes":       body.get("notes", trade.get("notes", "")),
    })

    if save_trades(trades):
        log.info("Closed trade %s: %s %s → %.2f%% ($%.2f)",
                 trade_id, trade["ticker"], exit_reason, actual_pct, actual_pnl)
        return jsonify({"success": True, "trade": trade})
    return jsonify({"error": "Failed to save"}), 500


@app.route("/api/trades/<trade_id>", methods=["DELETE"])
def api_delete_trade(trade_id):
    """Delete a trade (use sparingly — for data entry errors only)."""
    trades = load_trades()
    before = len(trades)
    trades = [t for t in trades if t["id"] != trade_id]
    if len(trades) == before:
        return jsonify({"error": f"Trade {trade_id} not found"}), 404
    if save_trades(trades):
        return jsonify({"success": True})
    return jsonify({"error": "Failed to save"}), 500


@app.route("/api/export", methods=["GET"])
def api_export():
    """Export all trades as CSV."""
    trades = load_trades()
    if not trades:
        return "No trades to export", 404

    fields = ["id", "ticker", "direction", "sector", "layer", "status",
              "entry_date", "entry_price", "dollar_amount", "shares",
              "expected_ev", "expected_win_rate", "ev_score",
              "stop_price", "target_price", "match_count", "best_sim",
              "macro_yield_curve", "macro_hy_spread", "macro_fed_funds",
              "exit_date", "exit_price", "exit_reason",
              "actual_pct", "actual_pnl", "days_held", "notes"]

    rows = [",".join(fields)]
    for t in trades:
        rows.append(",".join(str(t.get(f, "")) for f in fields))

    csv = "\n".join(rows)
    return app.response_class(
        response=csv,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"}
    )


# =============================================================================
# SECTION 6 — Entry point
# =============================================================================

if __name__ == "__main__":
    log.info("Trade Journal starting on port 5003")
    log.info("Data file: %s", TRADES_FILE)
    app.run(debug=False, port=5003, threaded=True)
