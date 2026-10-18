$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path $PSScriptRoot -Parent)
docker info --format '{{.ServerVersion}}'
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop in Linux-container mode, then run this script again.' }
docker compose build catstudio
if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }
docker compose --profile setup run --rm assets scripts/download_assets.py sd15
if ($LASTEXITCODE -ne 0) { throw 'Base model download or checksum verification failed.' }
docker compose --profile setup run --rm assets scripts/fetch_adapter.py
if ($LASTEXITCODE -ne 0) { throw 'Trained adapter download or verification failed.' }
docker compose up -d catstudio
if ($LASTEXITCODE -ne 0) { throw 'Application startup failed.' }
Write-Output 'Open http://127.0.0.1:8000 after the service becomes healthy. The first generation loads the weights.'
