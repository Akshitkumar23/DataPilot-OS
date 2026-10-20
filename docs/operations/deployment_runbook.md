# DataPilot OS — Deployment Runbook

## 1. System Overview & Architecture
DataPilot OS is an offline, privacy-first, enterprise Data Science Operating System. It combines local Ollama LLM intelligence for intent planning with deterministic Python analytical operators, rigorous dataset grounding, and independent result verification.

### Core Architectural Invariants:
1. **Zero Domain Hardcoding**: No hardcoded column names, dataset-specific heuristics, or user phrase workarounds.
2. **Deterministic Operator Execution**: All calculations, statistical tests, ML evaluations, and charts are executed by typed Python operators.
3. **Explicit Error & Decision States**:
   - `EXECUTE` (`SUCCEEDED`, HTTP 200): Fully grounded, scientifically feasible, verified results.
   - `CLARIFY` (`NEEDS_CLARIFICATION`, HTTP 422): Ambiguous columns, unstated temporal grains, or underspecified options.
   - `UNSUPPORTED` (`UNSUPPORTED`, HTTP 400): Capabilities outside the 22 registered ToolSpecs.
   - `UNSAFE` (`UNSAFE` / `FAILED`, HTTP 400): Scientific infeasibility, target leakage, or ungrounded execution.
4. **Independent Result Verification**: Every output is verified against raw data, numerical invariants, and chart artifact parity before emission.

---

## 2. Prerequisites & System Requirements

### Hardware Requirements:
- **CPU**: 4+ physical cores (x86_64 or ARM64)
- **RAM**: 16 GB minimum (32 GB recommended for local LLMs)
- **GPU**: Optional (NVIDIA GPU with 8GB+ VRAM accelerates Ollama inference)
- **Storage**: 10 GB free space for environment, model weights, and datasets

### Software Requirements:
- **OS**: Windows 10/11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.10 to 3.14 (Virtualenv recommended)
- **Node.js**: v20.x or higher (npm v10+)
- **Ollama**: v0.3.0 or higher

---

## 3. Step-by-Step Installation & Deployment

### Step 1: Clone and Enter Workspace
```bash
git clone https://github.com/Akshitkumar23/DataPilot-OS.git
cd DataPilot-OS
```

### Step 2: Python Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Local Offline Ollama Setup
DataPilot OS requires local Ollama for zero-data-leakage analytical planning.
```bash
# Start Ollama service (terminal 1)
ollama serve

# Pull required models (terminal 2)
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5:3b-instruct
```

### Step 4: Frontend & Desktop App Build
```bash
# Install Node dependencies
npm install

# Run Oxlint validation
npm run lint

# Build production Vite bundle
npm run build
```

### Step 5: Launch DataPilot OS
#### Option A: Desktop Electron App
```bash
npm run electron:dev
```
#### Option B: Web & Backend Development
```bash
# Terminal 1: Backend API
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend Dev Server
npm run dev
```

---

## 4. Operational Health Checks
Verify backend readiness and Ollama connectivity:
```bash
curl http://127.0.0.1:8000/api/health
```
Expected response:
```json
{
  "status": "healthy",
  "ollama": {
    "ready": true,
    "model": "qwen2.5-coder:7b"
  },
  "dataset_connected": false
}
```
