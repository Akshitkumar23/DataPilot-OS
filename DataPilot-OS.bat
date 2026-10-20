@echo off
setlocal enabledelayedexpansion
title DataPilot OS — Desktop Launcher
echo ===================================================
echo   Starting DataPilot OS Standalone Desktop App
echo ===================================================
echo.

:: Determine project directory
if exist "%~dp0package.json" (
    set "PROJECT_DIR=%~dp0"
) else if exist "%~dp0DataPilot OS\package.json" (
    set "PROJECT_DIR=%~dp0DataPilot OS"
) else (
    echo [ERROR] DataPilot OS directory could not be located.
    pause
    exit /b 1
)

cd /d "%PROJECT_DIR%"

:: 1. Auto-start Ollama in background if not already running
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe" >NUL
if errorlevel 1 (
    echo [Launcher] Starting Ollama local neural service in background...
    start /min "" ollama serve
    timeout /t 2 /nobreak >NUL
) else (
    echo [Launcher] Ollama is already running.
)

:: 2. Find the best packaged executable
set "APP_EXE="
if exist "%PROJECT_DIR%\dist-electron\win-unpacked\DataPilot OS.exe" (
    set "APP_EXE=%PROJECT_DIR%\dist-electron\win-unpacked\DataPilot OS.exe"
) else if exist "%PROJECT_DIR%\dist-electron-release-20260905e\win-unpacked\DataPilot OS.exe" (
    set "APP_EXE=%PROJECT_DIR%\dist-electron-release-20260905e\win-unpacked\DataPilot OS.exe"
) else (
    for /d %%D in ("%PROJECT_DIR%\dist-electron*") do (
        if exist "%%D\win-unpacked\DataPilot OS.exe" (
            set "APP_EXE=%%D\win-unpacked\DataPilot OS.exe"
        )
    )
)

if not defined APP_EXE (
    echo.
    echo [ERROR] Packaged DataPilot OS application was not found!
    echo Checked locations:
    echo   - %PROJECT_DIR%\dist-electron\win-unpacked\DataPilot OS.exe
    echo   - %PROJECT_DIR%\dist-electron-release-*\win-unpacked\DataPilot OS.exe
    echo.
    echo To build the packaged app, run:
    echo   cd "%PROJECT_DIR%"
    echo   npm run dist
    echo.
    pause
    exit /b 1
)

echo [Launcher] Launching: "%APP_EXE%"
start "" "%APP_EXE%"

echo.
echo ===================================================
echo   DataPilot OS is running!
echo   Live A-to-Z session logs will appear below.
echo   (Clicks, Queries, Planning, Execution, Results)
echo   Close this window when finished.
echo   Session Log: %APPDATA%\DataPilot OS\logs\session.log
echo ===================================================
echo.

powershell -NoProfile -NoExit -Command "$logDir = Join-Path $env:APPDATA 'DataPilot OS\logs'; $session = Join-Path $logDir 'session.log'; if (-not (Test-Path $session)) { New-Item -ItemType File -Force -Path $session | Out-Null }; Write-Host '[Launcher] Live Observability & Telemetry Stream Initialized.' -ForegroundColor Cyan; Write-Host '[Launcher] All UI clicks, queries, planning stages, and results will appear below:' -ForegroundColor Gray; Get-Content -Path $session -Tail 50 -Wait"
