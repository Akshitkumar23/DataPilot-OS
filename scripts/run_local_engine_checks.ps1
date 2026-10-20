<#!
.SYNOPSIS
Runs laptop-safe DataPilot OS checks without requiring a large local LLM.

.DESCRIPTION
Validates the query pipeline contracts, typed operators, feasibility checks,
and native Ollama tool-call adapter. It does not claim that a small planner
model understands every natural-language query.
#>

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment not found: $python"
}

Push-Location $projectRoot
try {
    & $python -m pytest `
        tests/test_planner_tool_retrieval.py `
        tests/test_ollama_native_tool_calling.py `
        tests/test_semantic_feasibility.py `
        tests/test_zero_fallback_execution_contract.py `
        tests/test_ds_os_operators_and_validator.py `
        tests/test_failure_paths_and_guardrails.py `
        -q --disable-warnings --tb=short
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python -m compileall -q backend modules
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    npm run build
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
