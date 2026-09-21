@echo off
title StyleSense Setup
setlocal
cd /d "%~dp0"

echo ============================================
echo   StyleSense - setup / health check
echo ============================================
echo.
echo OPTIONAL - start.bat already runs all of this automatically the first
echo time you use it. Use this file when you want an explicit, detailed
echo report (installs anything missing, then prints the full checklist).
echo.

set "PY=C:\python\python.exe"
if not exist "%PY%" set "PY=python"

%PY% scripts\bootstrap.py
if errorlevel 1 (
    echo [ERROR] Setup failed - see the message above.
    pause
    exit /b 1
)

echo.
echo Full install check:
%PY% scripts\check_install.py
echo.
echo Next step (if everything is OK): double-click start.bat, then open
echo http://127.0.0.1:8000 in your browser.
echo.
pause
