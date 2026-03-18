Param(
  [string]$Host = "0.0.0.0",
  [int]$Port = 8000,
  [ValidateSet("stub","transformers")] [string]$Backend = "stub",
  [string]$BaseModel = "Qwen/Qwen3-1.7B-Base",
  [string]$Adapter = "checkpoints\\qwen3-1.7b-dpo_LONG_20260318_1657"
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

# Ensure the `src/` layout is importable
$env:PYTHONPATH = "src"

# Backend selection
$env:TRIAGE_BACKEND = $Backend
$env:BASE_MODEL_NAME_OR_PATH = $BaseModel
if ($Backend -eq 'transformers') {
  $env:ADAPTER_NAME_OR_PATH = $Adapter
}

$python = '.\\.venv312\\Scripts\\python.exe'

& $python -m uvicorn triage_llm.api.app:app --host $Host --port $Port --app-dir src
