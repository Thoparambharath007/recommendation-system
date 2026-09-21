@echo off
title StyleSense Server
setlocal
cd /d "%~dp0"

echo ========================================
echo   StyleSense - Apparel Recommendation
echo ========================================
echo.

:: Free port 8000 - pure batch (a Python spawn just to parse netstat cost
:: 2-4s per start). /C:":8000 " matches the exact port, so a ":80000"
:: listener can never be killed by mistake like the old substring check did.
echo Checking port 8000...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /C:":8000 " ^| findstr /I "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

:: Use the dev machine's python when it exists, otherwise whatever is on PATH.
set "PY=C:\python\python.exe"
if not exist "%PY%" set "PY=python"

:: HF_HUB_OFFLINE is set BY src/web_app.py itself - only when the two models
:: are already in the local cache (a fresh install must stay ONLINE to
:: download them; run setup.bat once). ONEDNN quirk: see PROGRESS.md.
set TF_ENABLE_ONEDNN_OPTS=0
set TOKENIZERS_PARALLELISM=false
set TRANSFORMERS_VERBOSITY=error

echo.
echo First time on this machine? Run setup.bat once (installs deps + models),
echo and make sure the shared data/processed folder is in place (README).
echo.
echo Starting server... the web page is available IMMEDIATELY at
echo   http://127.0.0.1:8000
echo and the first search waits ~1-2 min while models load in the background
echo (subsequent starts are faster - Windows file cache is warm).
echo Press Ctrl+C in this window to stop the server.
echo.

"%PY%" -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
