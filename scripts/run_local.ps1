param([ValidateSet('auto','cpu','cuda')][string]$Device='auto', [int]$Port=8000)
$ErrorActionPreference='Stop'
$codeRoot=Split-Path -Parent $PSScriptRoot
$projectRoot=Split-Path -Parent $codeRoot
$python=Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Project Python is missing. Follow README environment setup first.' }
$env:CATSTUDIO_DEVICE=$Device
$env:CATSTUDIO_BASE_MODEL=Join-Path $projectRoot '05_Models\base_sd15'
$env:CATSTUDIO_ADAPTER=Join-Path $projectRoot '05_Models\cat_lora'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Push-Location -LiteralPath $codeRoot
try { & $python -m uvicorn catstudio.api:app --host 127.0.0.1 --port $Port --workers 1 } finally { Pop-Location }
