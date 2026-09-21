@echo off
title StyleSense Setup
setlocal
cd /d "%~dp0"

echo ============================================
echo   StyleSense - first-time project setup
echo ============================================
echo.
echo This installs the Python dependencies, downloads the two AI models
echo (~2.5 GB, one time, needs internet) and checks the install.
echo Run it ONCE after cloning. Re-running is harmless.
echo.

:: Resolve python: dev-machine path if present, else PATH.
set "PY=python"

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.11+ not found on PATH.
    echo         Install it from https://www.python.org/downloads/
    echo         ^(tick "Add python.exe to PATH" during install^) and re-run this file.
    pause
    exit /b 1
)

echo [1/3] Installing dependencies (requirements.txt)...
%PY% -m pip install --upgrade pip >nul 2>&1
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed - check the output above ^(network?^).
    pause
    exit /b 1
)

echo.
echo [2/3] Downloading the HuggingFace models (skip if already cached)...
%PY% scripts\fetch_models.py
if errorlevel 1 (
    echo [WARN] Model download failed - re-run setup.bat with internet,
    echo        or the server will fail to start with a clear error.
)

echo.
echo [3/3] Checking the install...
%PY% scripts\check_install.py
echo.
echo NOTE: recommendations also need the shared data/processed folder
echo       (~2.8 GB, see README step 3) - check_install.py above tells you
echo       whether it is in place.
echo.
pause
