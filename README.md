# PatternTrader — Setup & Usage

## What this does
Scans a watchlist of stocks for 21-day price/volume patterns that match
historical analogues. Ranks signals by expected value so you know where
to deploy capital first.

---

## One-time setup (do this once only)

### 1. Install Python
- Go to https://python.org/downloads
- Download and run the installer
- ✅ CHECK "Add Python to PATH" before clicking Install

### 2. Install required libraries
- Press Windows Key + R, type `cmd`, press Enter
- Paste this and press Enter:

  pip install flask yfinance pandas numpy scipy openpyxl

- Wait for it to finish (takes 1–2 minutes)

---

## Running the app

1. Double-click `START.bat`
2. A browser window opens automatically at http://localhost:5000
3. Click "Run Scan" — the scan takes 30–90 seconds depending on watchlist size

To stop the app: close the black terminal window, or press CTRL+C inside it.

---

## Using the app

- **Watchlist**: Edit the tickers in the input box (comma-separated)
- **Run Scan**: Pulls live data from Yahoo Finance and runs the engine
- **Results table**: Sorted by EV score (highest expected value first)
- **Click any signal row** to open the detail drawer showing top historical analogues

### Column guide
| Column | Meaning |
|--------|---------|
| Direction | LONG (buy) or SHORT (sell) |
| Best sim | Highest pattern similarity found (price + volume) |
| Matches | How many historical windows cleared the 90% threshold |
| Avg return | Expected % move at +10 trading days |
| Win rate | % of historical analogues that moved in the signal direction |
| Drawdown | Median intra-window drawdown — how much pain to expect |
| EV score | Expected value = win rate × avg win − loss rate × avg loss |

### Signal status meanings
| Status | Meaning |
|--------|---------|
| LONG / SHORT | Active signal — both gates cleared |
| NO EDGE | Pattern matched but historical moves were too small to trade |
| NO DATA | Not enough historical data found |
| ERROR | Yahoo Finance could not fetch this ticker |

---

## Adjusting parameters
Open `app.py` in any text editor and change the values at the top:

```
PATTERN_DAYS      = 21      # length of the pattern window in trading days
SIMILARITY_CUTOFF = 0.90    # minimum similarity to count as a match (0–1)
MIN_AVG_MOVE      = 0.02    # minimum expected move to generate a signal (2%)
PRICE_WEIGHT      = 0.70    # how much price shape matters vs volume
HISTORY_YEARS     = 10      # years of history to scan
```

Save the file and restart the app for changes to take effect.

---

## Hosting online (future)
To access from anywhere, deploy to a free cloud service like Railway or Render.
The app is standard Flask — no changes needed to the code.
