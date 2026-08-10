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

:: Clear leftover port 8000 PIDs (fast silent kill)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :8000') do taskkill /f /pid %%a >nul 2>&1

:: Start Backend Engine
start /b "" "%PYTHON_CMD%" "server.py"

:: Instant Direct Binary Electron Launch (Bypasses npx network check & eliminates forced sleep)
if exist "node_modules\electron\dist\electron.exe" (
    "node_modules\electron\dist\electron.exe" .
) else if exist "node_modules\.bin\electron.cmd" (
    call "node_modules\.bin\electron.cmd" .
) else (
    npx electron .
)

echo.
echo DataPilot OS Session Closed.
