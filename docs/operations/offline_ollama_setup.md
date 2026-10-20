# DataPilot OS — Offline Ollama Setup Guide

## Zero Cloud Dependencies, 100% Data Privacy

DataPilot OS is architected to run entirely on premises without external API calls or telemetry. All user queries, dataset profiles, and conversational turns stay local to the host machine.

---

## 1. Installing Ollama

### Windows:
Download and run the installer from:
```
https://ollama.com/download/OllamaSetup.exe
```

### macOS:
```bash
brew install ollama
```

### Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 2. Model Provisioning

DataPilot OS uses specialized Qwen-series compact instruction models for structured analytical planning:

```bash
# Primary planning model (7B parameter coder)
ollama pull qwen2.5-coder:7b

# Lightweight repair & clarification fallback model (3B parameter instruction)
ollama pull qwen2.5:3b-instruct
```

### Checking Model Availability:
```bash
ollama list
```
Expected output:
```
NAME                    ID              SIZE      MODIFIED
qwen2.5-coder:7b        e1d93d39f75b    4.7 GB    2 hours ago
qwen2.5:3b-instruct     451e0649774a    1.9 GB    3 hours ago
```

---

## 3. Configuring DataPilot OS Environment

Set the following environment variables in `.env` or system environment:

```env
# Ollama endpoint
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=qwen2.5-coder:7b

# Timeouts
OLLAMA_TIMEOUT_SEC=30.0

# Backend Host & Port
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
```

---

## 4. Offline Verification & Testing

Verify that DataPilot OS detects Ollama correctly:

```bash
# Run Ollama readiness integration test
$env:PYTHONPATH="."
pytest tests/test_runtime_contract_alignment.py tests/test_runtime_prompt_parity.py -v
```
