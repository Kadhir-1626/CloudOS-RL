# =============================================================================
# CloudOS-RL - Deploy to Minikube
# =============================================================================
# Applies Kubernetes manifests and verifies deployment health.
#
# Usage:
#   .\scripts\deploy_k8s.ps1
#   .\scripts\deploy_k8s.ps1 -BridgeOnly
#   .\scripts\deploy_k8s.ps1 -Teardown
# =============================================================================

param(
    [switch]$BridgeOnly,
    [switch]$Teardown
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   CloudOS-RL Kubernetes Deploy" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# --- Teardown ---------------------------------------------------------------
if ($Teardown) {
    Write-Host "[teardown] Removing all cloudos-rl resources ..." -ForegroundColor Yellow
    kubectl delete namespace cloudos-rl --ignore-not-found
    kubectl delete clusterrole cloudos-operator-role --ignore-not-found
    kubectl delete clusterrolebinding cloudos-operator-binding --ignore-not-found
    kubectl delete crd cloudworkloads.cloudos.ai --ignore-not-found
    Write-Host "[teardown] OK - Done" -ForegroundColor Green
    exit 0
}

# --- Check Minikube ---------------------------------------------------------
$status = minikube status --format "{{.Host}}" 2>$null
if ($status -ne "Running") {
    Write-Host "[error] Minikube is not running." -ForegroundColor Red
    Write-Host "        Run: .\scripts\minikube_setup.ps1" -ForegroundColor Yellow
    exit 1
}

$minikubeIP = minikube ip
Write-Host "[minikube] OK - Running - IP: $minikubeIP" -ForegroundColor Green

# --- Check bridge on host ---------------------------------------------------
Write-Host ""
Write-Host "[bridge] Checking Kafka bridge at localhost:9090 ..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://localhost:9090/metrics" -TimeoutSec 3 -UseBasicParsing
    Write-Host "[bridge] OK - Bridge running on host" -ForegroundColor Green
}
catch {
    Write-Host "[bridge] WARN - Bridge not reachable." -ForegroundColor Yellow
    Write-Host "              Start it with: python -m ai_engine.kafka.kafka_prometheus_bridge" -ForegroundColor White
}

# --- Apply base infrastructure ---------------------------------------------
Write-Host ""
Write-Host "[k8s] Applying base infrastructure..." -ForegroundColor Yellow

kubectl apply -f infrastructure/k8s/namespace.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

kubectl apply -f infrastructure/k8s/crd.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

kubectl apply -f infrastructure/k8s/rbac.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

kubectl apply -f infrastructure/k8s/configmap.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

kubectl apply -f infrastructure/k8s/secret.yaml
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[k8s] OK - Base infrastructure applied" -ForegroundColor Green

# --- Deploy bridge ----------------------------------------------------------
Write-Host ""
Write-Host "[k8s] Deploying cloudos-bridge ..." -ForegroundColor Yellow
kubectl apply -f infrastructure/k8s/deployment-bridge.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "[k8s] ERROR - Bridge deployment apply failed" -ForegroundColor Red
    exit 1
}
Write-Host "[k8s] OK - Bridge deployment applied" -ForegroundColor Green

# --- Deploy API -------------------------------------------------------------
if (-not $BridgeOnly) {
    Write-Host ""
    Write-Host "[k8s] Deploying cloudos-api ..." -ForegroundColor Yellow
    kubectl apply -f infrastructure/k8s/deployment-api.yaml
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[k8s] ERROR - API deployment apply failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[k8s] OK - API deployment applied" -ForegroundColor Green
}

# --- Wait for deployments ---------------------------------------------------
Write-Host ""
Write-Host "[k8s] Waiting for pods to be ready (up to 120s) ..." -ForegroundColor Yellow

kubectl wait --for=condition=available --timeout=120s deployment/cloudos-bridge -n cloudos-rl

if (-not $BridgeOnly) {
    kubectl wait --for=condition=available --timeout=120s deployment/cloudos-api -n cloudos-rl
}

# --- Show status ------------------------------------------------------------
Write-Host ""
Write-Host "[k8s] Pod status:" -ForegroundColor Yellow
kubectl get pods -n cloudos-rl -o wide

Write-Host ""
Write-Host "[k8s] Services:" -ForegroundColor Yellow
kubectl get services -n cloudos-rl

# --- Port-forward -----------------------------------------------------------
Write-Host ""
Write-Host "[k8s] Starting port-forward for local access ..." -ForegroundColor Yellow
Write-Host "      bridge metrics -> localhost:9095" -ForegroundColor White
Write-Host "      api            -> localhost:8001" -ForegroundColor White
Write-Host ""

Start-Job -Name "pf-bridge" -ScriptBlock {
    kubectl port-forward svc/cloudos-bridge-svc 9095:9090 -n cloudos-rl
} | Out-Null

if (-not $BridgeOnly) {
    Start-Job -Name "pf-api" -ScriptBlock {
        kubectl port-forward svc/cloudos-api-svc 8001:8000 -n cloudos-rl
    } | Out-Null
}

Start-Sleep -Seconds 3

# --- Final summary ----------------------------------------------------------
Write-Host ("=" * 56) -ForegroundColor Cyan
Write-Host "  Deployment complete" -ForegroundColor Cyan
Write-Host ("=" * 56) -ForegroundColor Cyan
Write-Host "  Cluster IP      : $minikubeIP"
Write-Host "  Bridge metrics  : http://localhost:9095/metrics (port-forward)"
Write-Host "  API NodePort    : http://${minikubeIP}:30800 (direct)"
Write-Host "  API local       : http://localhost:8001 (port-forward)"
Write-Host ""
Write-Host "  Verify deployment:" -ForegroundColor Yellow
Write-Host "    python scripts/verify_k8s.py" -ForegroundColor White
Write-Host ""
Write-Host "  View logs:" -ForegroundColor Yellow
Write-Host "    kubectl logs -f deployment/cloudos-bridge -n cloudos-rl" -ForegroundColor White
Write-Host "    kubectl logs -f deployment/cloudos-api -n cloudos-rl" -ForegroundColor White
Write-Host ""