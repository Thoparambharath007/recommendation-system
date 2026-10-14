@echo off
setlocal
cd /d "%~dp0"
title Shirt Studio Server
set "SHIRT_APP_PORT=8010"
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8010/' -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } } catch {}; exit 1"
if not errorlevel 1 (
    echo Shirt Studio is already running.
    start "" "http://127.0.0.1:8010/"
    exit /b 0
)
echo Starting Shirt Studio at http://127.0.0.1:8010/
echo Keep this window open while using the app.
start "" "http://127.0.0.1:8010/"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" launch.py
) else (
    python launch.py
)
if errorlevel 1 (
    echo.
    echo Server could not start. See the error above.
    pause
)
endlocal
