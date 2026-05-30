@echo off
echo.
echo  Macro Backtest -- Port 5004
echo  Walk-forward validation 2000-2023
echo  Open your browser to: http://localhost:5004
echo  WARNING: Full run takes 5-15 minutes.
echo  Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
python macro_backtest.py
pause
