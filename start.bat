@echo off
title StyleSense Server
setlocal
cd /d "%~dp0"

echo ========================================
echo   StyleSense - Apparel Recommendation
echo ========================================
echo.

:: Resolve python: dev-machine path when it exists, else whatever is on PATH.
set "PY=C:\python\python.exe"
if not exist "%PY%" set "PY=python"
where python >nul 2>&1
if errorlevel 1 if not exist "C:\python\python.exe" (
    echo [ERROR] Python not found on this PC.
    echo         Install Python 3.11+ from https://www.python.org/downloads/
    echo         ^(tick "Add python.exe to PATH"^) and run start.bat again.
    pause
    exit /b 1
)

:: FIRST RUN = full automatic setup (pip deps + ~2.5 GB model download).
:: LATER RUNS = the same checks, skipped in under a second.
:: All logic lives in scripts\bootstrap.py (plain-English errors there).
echo [1/3] One-time setup check (first run: installs deps + downloads models)...
"%PY%" scripts\bootstrap.py
if errorlevel 1 (
    echo.
    echo [ERROR] Setup did not finish - fix the issue printed above,
    echo         then run start.bat again. Nothing was lost; it resumes.
    pause
    exit /b 1
)

:: Free port 8000 - pure batch. /C:":8000 " matches the exact port, so a
:: ":80000" listener can never be killed by mistake.
echo [2/3] Freeing port 8000...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /C:":8000 " ^| findstr /I "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

:: HF_HUB_OFFLINE is set by src\web_app.py itself - only when the models are
:: already cached (fresh machines must stay online for the first download).
:: TF_ENABLE_ONEDNN_OPTS: PROGRESS.md gotcha. The rest just cuts chatter.
set TF_ENABLE_ONEDNN_OPTS=0
set TOKENIZERS_PARALLELISM=false
set TRANSFORMERS_VERBOSITY=error

echo [3/3] Starting server...
echo.
echo   Web page (available IMMEDIATELY):  http://127.0.0.1:8000
echo   First search waits ~1-2 min while models load in the background.
echo   Subsequent starts are faster - Windows file cache is warm.
echo   Press Ctrl+C in this window to stop the server.
echo.

"%PY%" -m uvicorn src.web_app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
