@echo off
echo.
echo  Trade Journal -- Port 5003
echo  Open your browser to: http://localhost:5003
echo  Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
python journal.py
pause
