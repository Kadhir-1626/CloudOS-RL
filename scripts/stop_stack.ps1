# =============================================================================
# CloudOS-RL — Stop Docker Stack
# =============================================================================

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "[stack]  Stopping Prometheus container ..." -ForegroundColor Yellow
docker compose down

Write-Host "[stack]  ✅ Docker services stopped." -ForegroundColor Green
Write-Host ""
Write-Host "  Note: Kafka, ZooKeeper, Bridge, and Grafana (native) are still running."
Write-Host "  Stop them manually in their respective terminal windows."
Write-Host ""
