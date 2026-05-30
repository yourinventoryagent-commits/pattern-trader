@echo off
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║   PTJ Macro Analogue Engine — Phase 1        ║
echo  ║   Port 5002                                  ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  Starting server...
echo  Open your browser to: http://localhost:5002
echo  Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
python macro_analogue.py
pause
