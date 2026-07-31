#!/usr/bin/env bash
set -e

echo "==================================================="
echo "  Starting DataPilot OS Standalone Desktop App"
echo "==================================================="
echo ""

# Change working directory to script location
CDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CDIR"

PYTHON_EXE=""
if [ -f ".venv/bin/python" ]; then
    PYTHON_EXE=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXE="python"
else
    echo "Error: Python interpreter not found. Please install Python 3 or create a .venv virtual environment."
    exit 1
fi

echo "[1/2] Starting DataPilot OS Backend Engine with $PYTHON_EXE..."
$PYTHON_EXE server.py &
BACKEND_PID=$!

cleanup() {
    echo "Shutting down backend engine (PID: $BACKEND_PID)..."
    kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[2/2] Launching Electron UI..."
npx electron .

echo ""
echo "DataPilot OS Session Closed."
