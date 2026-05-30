@echo off
echo.
echo  ─────────────────────────────────────
echo   PatternTrader — starting server...
echo  ─────────────────────────────────────
echo.
echo  Opening browser at http://localhost:5000
echo  Press CTRL+C to stop the server.
echo.
start "" http://localhost:5000
python app.py
pause
