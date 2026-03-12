# =============================================================================
# CloudOS-RL — Start Full Monitoring Stack (Windows PowerShell)
# =============================================================================
# Starts Prometheus in Docker and updates Grafana datasource.
# Assumes Kafka + ZooKeeper + Bridge + Grafana already running natively.
#
# Usage:
#   .\scripts\start_stack.ps1
#   .\scripts\start_stack.ps1 -GrafanaPassword "cloudos123"
#
# Prerequisites:
#   - Docker Desktop running
#   - Kafka bridge running: python -m ai_engine.kafka.kafka_prometheus_bridge
#   - Grafana running at localhost:3000
# =============================================================================

param(
    [string]$GrafanaPassword = "cloudos123",
    [string]$GrafanaUser     = "admin",
    [int]   $GrafanaPort     = 3000,
    [int]   $PrometheusPort  = 9091
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     CloudOS-RL  Stack Startup                    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Docker is running
Write-Host "[docker]  Checking Docker ..." -ForegroundColor Yellow
try {
    $dockerVer = docker --version 2>&1
    Write-Host "[docker]  ✅ $dockerVer" -ForegroundColor Green
} catch {
    Write-Host "[docker]  ❌ Docker not found. Install Docker Desktop." -ForegroundColor Red
    exit 1
}

docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[docker]  ❌ Docker daemon not running. Start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "[docker]  ✅ Docker daemon running" -ForegroundColor Green

# Step 2: Check bridge is running
Write-Host ""
Write-Host "[bridge]  Checking Kafka-Prometheus bridge at localhost:9090 ..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:9090/metrics" -TimeoutSec 3 -UseBasicParsing
    $lines = ($r.Content -split "`n" | Where-Object { $_ -match "^cloudos_" }).Count
    Write-Host "[bridge]  ✅ Running — $lines cloudos_* metrics exposed" -ForegroundColor Green
} catch {
    Write-Host "[bridge]  ⚠️  Not reachable at :9090" -ForegroundColor Yellow
    Write-Host "[bridge]     Start it first:" -ForegroundColor Yellow
    Write-Host "           python -m ai_engine.kafka.kafka_prometheus_bridge" -ForegroundColor White
    Write-Host ""
    Write-Host "  Continue anyway? (bridge required for Prometheus to scrape)" -ForegroundColor Yellow
    $continue = Read-Host "  [y/N]"
    if ($continue -ne "y") { exit 1 }
}

# Step 3: Start Prometheus in Docker
Write-Host ""
Write-Host "[prometheus]  Starting Prometheus server (Docker) ..." -ForegroundColor Yellow

docker compose up -d prometheus
if ($LASTEXITCODE -ne 0) {
    Write-Host "[prometheus]  ❌ docker compose failed." -ForegroundColor Red
    exit 1
}

# Wait for Prometheus to be healthy
Write-Host "[prometheus]  Waiting for Prometheus to be healthy ..."
$maxWait = 30
$waited  = 0
while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 2
    $waited += 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:${PrometheusPort}/-/healthy" -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Write-Host "[prometheus]  ✅ Healthy at http://localhost:${PrometheusPort}" -ForegroundColor Green
            break
        }
    } catch { }
    Write-Host "[prometheus]  ... waiting ($waited/$maxWait s)"
}

if ($waited -ge $maxWait) {
    Write-Host "[prometheus]  ❌ Prometheus did not start in time." -ForegroundColor Red
    Write-Host "              Check: docker compose logs prometheus" -ForegroundColor Yellow
    exit 1
}

# Step 4: Update Grafana datasource to point to Prometheus :9091
Write-Host ""
Write-Host "[grafana]  Updating Grafana datasource → Prometheus :${PrometheusPort} ..." -ForegroundColor Yellow

python scripts/import_dashboard.py `
    --grafana-password $GrafanaPassword `
    --grafana-user     $GrafanaUser `
    --grafana-port     $GrafanaPort `
    --prom-port        $PrometheusPort

# Step 5: Final summary
Write-Host ""
Write-Host "=" * 54 -ForegroundColor Cyan
Write-Host "  Stack Status" -ForegroundColor Cyan
Write-Host "=" * 54 -ForegroundColor Cyan
Write-Host "  Bridge     :9090   /metrics (native)" -ForegroundColor White
Write-Host "  Prometheus :9091   PromQL query API (Docker)" -ForegroundColor White
Write-Host "  Grafana    :3000   Dashboard (native)" -ForegroundColor White
Write-Host ""
Write-Host "  Prometheus UI : http://localhost:${PrometheusPort}" -ForegroundColor Green
Write-Host "  Dashboard     : http://localhost:${GrafanaPort}/d/cloudos-rl-v1" -ForegroundColor Green
Write-Host ""
Write-Host "  Run verification:" -ForegroundColor Yellow
Write-Host "  python scripts/verify_grafana.py --grafana-password $GrafanaPassword" -ForegroundColor White
Write-Host ""