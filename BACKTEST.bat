@echo off
echo.
echo  ─────────────────────────────────────
echo   PatternTrader Backtest — starting...
echo  ─────────────────────────────────────
echo.
echo  Opening browser at http://localhost:5001
echo  NOTE: Keep the Pattern Trader (port 5000) closed
echo        or open a separate terminal for that one.
echo  Press CTRL+C to stop this server.
echo.
start "" http://localhost:5001
python backtest.py
pause
