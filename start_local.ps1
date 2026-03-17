<#
  start_local.ps1
  Windows PowerShell helper to start the app via Docker Compose.
  Usage: Open PowerShell, cd to the repo root (on the USB drive) and run:
    .\start_local.ps1
#>

Write-Host "Starting Hammerthrow Analysis (Docker)..." -ForegroundColor Cyan

# Ensure script runs from the directory it's located in
Set-Location -Path $PSScriptRoot

try {
    docker-compose up --build -d
    Write-Host "Started. Open your browser at: http://localhost:5000" -ForegroundColor Green
    Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Yellow
    Write-Host "To stop the app: docker-compose down" -ForegroundColor Yellow
}
catch {
    Write-Host "Error: Failed to start docker-compose. Is Docker Desktop running?" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
}
