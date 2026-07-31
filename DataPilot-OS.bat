@echo off
title DataPilot OS — Desktop Launcher
echo ===================================================
echo   Starting DataPilot OS Standalone Desktop App
echo ===================================================
echo.

cd /d "%~dp0"

set "PYTHON_CMD=.venv\Scripts\python.exe"
if not exist "%PYTHON_CMD%" (
    set "PYTHON_CMD=python"
)

echo [1/2] Starting DataPilot OS Backend Engine (%PYTHON_CMD%)...
start /b "" "%PYTHON_CMD%" "server.py"

echo [2/2] Launching Electron UI...
npx electron .

echo.
echo DataPilot OS Session Closed.

