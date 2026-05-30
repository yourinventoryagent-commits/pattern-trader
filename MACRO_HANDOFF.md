# PTJ Macro Analogue Engine — Project Handoff Document
## For use when starting a new Claude conversation

---

## What we built

A standalone Flask app (`macro_analogue.py`, port 5002) that implements a
Paul Tudor Jones-style macro analogue engine. PTJ famously overlaid the 1980s
Dow against the 1920s Dow, identified structural similarity, and used the
historical period as a forward roadmap — shorting into Black Monday 1987.

This tool does the same thing systematically, with a two-phase scoring engine.

---

## File structure

```
PatternTrader\
  macro_analogue.py     ← backend (Flask, port 5002)
  MACRO.bat             ← double-click launcher
  templates\
    macro.html          ← frontend (Chart.js, dark theme)
```

---

## Tech stack

- **Backend**: Python + Flask (port 5002, threaded=True)
- **Price data**: yfinance — `yf.Ticker("SPY").history()` from 1993
- **Macro data**: FRED API via `fredapi` library
  - FEDFUNDS — Fed Funds rate (monthly)
  - T10Y2Y — 10yr minus 2yr yield curve spread (daily)
  - CPIAUCSL — CPI index, converted to YoY % (monthly)
  - BAMLH0A0HYM2 — ICE BofA HY OAS credit spread (daily)
- **Computation**: vectorised NumPy (no Python loops for similarity)
- **Frontend**: Vanilla JS + Chart.js 4.4.1, DM Sans + DM Mono fonts
- **OS**: Windows (launches via .bat file)

---

## Architecture — Two phases

### Phase 1 — Price pattern matching

Takes the current SPY chart over a configurable window (default 252 trading
days / 12 months), normalises to % returns from day 1, slides this window
across all SPY history back to 1993. Computes Pearson correlation between
the current pattern and every historical window using a single vectorised
matrix multiply — no Python loop, handles 30 years of daily data in
milliseconds.

Collects top 20 price-ranked candidates (non-overlapping, MIN_SEPARATION=126
days enforced) as input to Phase 2.

### Phase 2 — Macro regime scoring

For each candidate, fetches a 6-dimensional macro snapshot at the analogue's
end date and compares it to today's macro snapshot:

```
snapshot = {
    fed_funds:   rate level (%)
    fed_change:  12mo change in rate (+ve = hiking, -ve = cutting)
    yield_curve: 10y-2y spread (negative = inverted)
    cpi_yoy:     CPI year-over-year %
    cpi_change:  12mo change in CPI YoY
    hy_spread:   high yield OAS spread (wide = stress)
}
```

Regime similarity uses a normalised Euclidean distance converted via a
Gaussian kernel:

```python
similarity = exp(-sum((delta/scale)^2) / 2)
```

Characteristic scales: fed_funds/5, fed_change/3, yield_curve/3,
cpi_yoy/6, cpi_change/4, hy_spread/6.

Final combined score = price_sim × 0.50 + regime_sim × 0.50

Re-ranks the 20 candidates by combined score and returns top 3.

### Why this matters

Without macro filtering, price-pattern matches are misleading. Example:
the 12-month SPY shape from 2007 (pre-GFC crash) looks identical to 2011
(post-GFC recovery). Same chart shape, opposite outcomes — because the macro
environment was completely different. Phase 2 correctly suppresses the 2007
analogue (inverted yield curve, wide credit spreads, Fed cutting into crisis)
and elevates the 2011 analogue.

In practice, adding macro filtering shifted the top analogues from
post-crisis recovery periods (2009, 2010) to mid-1990s soft-landing periods
(1995-1996) — a fundamentally different and more accurate regime match for
today's environment (Fed cutting from 5.25%, CPI ~3.5%, tight spreads).

---

## Critical implementation lessons

### 1. Timezone mismatch — the hardest bug

yfinance returns SPY dates as timezone-aware (`America/New_York`).
FRED returns dates as timezone-naive. Comparing them raises a silent
exception caught by the try/except, returning NaN and disabling macro scoring.

**Fix**: strip timezone before any comparison:
```python
d = pd.Timestamp(date).tz_convert(None) if pd.Timestamp(date).tzinfo \
    else pd.Timestamp(date)
```

### 2. FRED daily reindex end date

`pd.date_range(start, datetime.today(), freq="D")` creates an index that
extends to today. But FRED's monthly series (FEDFUNDS) lags by ~1 month.
After forward-fill, the last real value is at the last FRED observation date,
but the index has NaN rows from there to today.

**Fix**: end the daily index at the last real data date:
```python
last_date = max(s.index[-1] for s in raw.values())
daily_idx = pd.date_range(MACRO_HISTORY_START, last_date, freq="D")
```

### 3. _safe_val for robust series lookup

Never use `series.asof(date)` on a reindexed daily series — it requires
exact index membership. Use a binary search on the dropna'd series instead:

```python
def _safe_val(series, date):
    cleaned = series.dropna().sort_index()
    d = pd.Timestamp(date).tz_convert(None) if pd.Timestamp(date).tzinfo \
        else pd.Timestamp(date)
    if d >= cleaned.index[-1]:
        return float(cleaned.iloc[-1])   # use last known value
    pos = cleaned.index.searchsorted(d, side="right") - 1
    return float(cleaned.iloc[pos]) if pos >= 0 else np.nan
```

### 4. Vectorised similarity (same as Pattern Trader)

```python
def build_return_matrix(close_arr, window):
    idx   = np.arange(window)[None,:] + np.arange(n_windows)[:,None]
    mat   = close_arr[idx]
    first = mat[:,0:1]
    mat   = np.where(first != 0, mat/first - 1.0, 0.0)
    return mat   # shape: (n_windows, window)

def vectorised_pearson(target, matrix):
    t_n = (target - target.mean()) / (np.sqrt(((target-target.mean())**2).sum()) + 1e-10)
    m   = matrix - matrix.mean(axis=1, keepdims=True)
    m_n = m / (np.sqrt((m**2).sum(axis=1, keepdims=True)) + 1e-10)
    return (m_n @ t_n).clip(-1, 1)
```

### 5. Two-stage candidate selection

Don't run macro scoring on every historical window — too slow.
Run price similarity first (vectorised, fast), take top 20,
then run macro scoring on just those 20:

```python
MACRO_CANDIDATES = 20   # price-rank this many, then macro re-rank
TOP_N            = 3    # final output
```

### 6. FRED API key

Key: `dae8904c34369213374f3700e63d3d25`
Registration: fred.stlouisfed.org (free, instant after email confirmation)
Note: key took several hours to activate on initial registration.

---

## UI — Two-panel chart design

### Panel 1 — Pattern Match
- Shows current SPY (white, 3px) + 3 historical matched windows
- All normalised to 0% at own start date — comparing shape not level
- Y-axis locked to ±25% so matched window detail is visible
- Tells you: "these periods moved like this"

### Panel 2 — What Happened Next
- All three forward tails anchored to 0% at TODAY divider
- Solid white line = similarity²-weighted average of all three
- Dashed coloured lines = individual analogue forward paths
- Tells you: "from this point, here's what those periods did next"

### Analogue cards
- Three score bars: price match, regime match, combined
- Forward returns at 6mo / 12mo / 24mo
- Macro snapshot at analogue end date (Fed, yield curve, CPI, HY spread)
  with interpretation labels (hiking/cutting, inverted/normal, rising/falling)

### Today's macro bar
- Shows current Fed Funds, yield curve, CPI YoY, HY spread
- Appears below Panel 2 when FRED data is active
- Lets you visually compare today vs each analogue's macro context

---

## Current results (12-month window, May 2026)

Today's macro regime:
- Fed Funds: 3.64% (cutting)
- Yield curve: 0.48% (flat)
- CPI YoY: 3.7% (rising)
- HY Spread: 2.82% (tight)

Top analogues after Phase 2 filtering:
| # | Period | Price | Regime | Combined | +12mo | +24mo |
|---|--------|-------|--------|----------|-------|-------|
| 1 | Oct 1995 → Sep 1996 | 91.3% | 88.0% | 89.6% | +41.3% | +57.5% |
| 2 | Mar 1995 → Feb 1996 | 89.8% | 86.1% | 87.9% | +27.4% | +68.8% |
| 3 | Oct 2023 → Oct 2024 | 91.6% | 83.9% | 87.8% | +15.6% | — |

Interpretation: Mid-1990s soft landing is the dominant analogue. Fed cutting
from ~5%, tight spreads, moderate inflation. Market went on to rally strongly
through 1999. Weighted expected returns: +13% (12mo), +27% (24mo).

---

## Phase 3 — What's next (not yet built)

### Sector rotation layer

Given the matched historical analogue periods, which sectors outperformed
SPY in the following 6/12 months?

Sector ETFs to analyse:
- XLK (tech), XLF (financials), XLE (energy), XLV (healthcare)
- XLI (industrials), XLY (consumer disc), XLP (consumer staples)
- XLB (materials), XLRE (real estate), XLU (utilities)

For each analogue period, compute each sector's return relative to SPY.
Surface the top 2-3 sectors that consistently outperformed across analogues.
This filters which stocks are even worth looking at.

### Individual stock signals (Phase 3b)

Within favoured sectors, run the existing Pattern Trader engine (app.py)
but scoped to:
- Only stocks in the favoured sector
- Only the time window corresponding to the analogue period
- Higher similarity threshold (0.82+) for stock-level signals

### The complete signal flow

```
Macro analogue says: "looks like 1995-1996"
    ↓
Sector rotation says: "in 1995-1996, XLK and XLF led SPY by 15%+"
    ↓
Stock engine says: "within XLK, MSFT and ORCL are showing timing signals"
    ↓
All three layers agree → act
```

---

## New conversation briefing for Phase 3

```
I'm building Phase 3 of a PTJ-style macro analogue engine.

Phase 1 (done): SPY price pattern matching — finds top 3 historical periods
whose 12-month chart shape most resembles today using vectorised Pearson.

Phase 2 (done): Macro regime scoring — re-ranks candidates by a 50/50 blend
of price similarity and macro regime similarity (Fed, yield curve, CPI, HY
spread) using FRED data. Solved the "1999 vs 2011" problem where identical
chart shapes had opposite outcomes due to different macro environments.

Phase 3 (now building): Sector rotation layer. Given the top analogue periods
from Phase 2, compute how each major sector ETF (XLK, XLF, XLE, XLV, XLI,
XLY, XLP, XLB, XLRE, XLU) performed relative to SPY in the 6 and 12 months
following each analogue's end date. Surface the sectors that consistently
outperformed across all three analogues. Display as a ranked sector table
and a bar chart showing relative performance.

Stack: same Flask app (macro_analogue.py, port 5002), yfinance for sector
ETF data, same dark UI theme.

Key implementation notes:
- Use yf.Ticker(ticker).history() NOT yf.download()
- Always flatten yfinance MultiIndex columns
- Use NumpyEncoder for JSON serialisation
- Timezone fix: SPY dates are tz-aware (America/New_York), strip with
  pd.Timestamp(date).tz_convert(None) before comparing to FRED dates
- The analogue end dates come from the /api/analogues response —
  sector performance is computed from those dates forward
- Add a /api/sectors POST endpoint that accepts analogue end dates
  and returns sector relative performance
- Add a Panel 3 section to macro.html below the existing panels
```

---

## Design system

```css
:root {
  --bg:        #0d0f12;
  --surface:   #13161b;
  --surface2:  #1a1e25;
  --border:    #22272f;
  --border2:   #2e3540;
  --text:      #e2e6ed;
  --muted:     #5a6478;
  --subtle:    #8896aa;
  --green:     #2ecc8a;
  --green-dim: #1a4a35;
  --red:       #e05555;
  --red-dim:   #4a1f1f;
  --amber:     #f0a040;
  --amber-dim: #3d2a10;
  --blue:      #4a9eff;
  --blue-dim:  #0f2a4a;
  --radius:    8px;
  --mono:      'DM Mono', monospace;
  --sans:      'DM Sans', sans-serif;
}
```

Analogue colour palette:
- Current period: `#ffffff` (white)
- Analogue 1: `#f0a040` (amber)
- Analogue 2: `#2ecc8a` (green)
- Analogue 3: `#b57aff` (purple)

---

## Dependencies

```
pip install flask yfinance pandas numpy fredapi
```

---

*Handoff document generated: May 2026*
*PTJ Macro Analogue Engine — Phase 1 + 2 complete*
*Files: macro_analogue.py, templates/macro.html, MACRO.bat*
