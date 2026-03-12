# =============================================================================
# CloudOS-RL Minikube Setup Script (Windows PowerShell)
# =============================================================================

param(
    [int]$CPUs = 4,
    [int]$Memory = 4096,
    [string]$Driver = "docker",
    [switch]$Reset
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   CloudOS-RL Minikube Setup" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check prerequisites
Write-Host "[prereq] Checking prerequisites ..." -ForegroundColor Yellow

$missing = @()

foreach ($cmd in @("minikube", "kubectl", "docker")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        $missing += $cmd
    }
    else {
        Write-Host "[prereq] OK - $cmd found" -ForegroundColor Green
    }
}

if ($missing.Count -gt 0) {
    Write-Host "[prereq] ERROR - Missing: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}

# Step 2: Check Docker daemon
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[prereq] ERROR - Docker daemon not running. Start Docker Desktop first." -ForegroundColor Red
    exit 1
}
Write-Host "[prereq] OK - Docker daemon running" -ForegroundColor Green

# Step 3: Reset cluster if requested
if ($Reset) {
    Write-Host ""
    Write-Host "[minikube] Deleting existing cluster because Reset flag is set ..." -ForegroundColor Yellow
    minikube delete
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[minikube] ERROR - Failed to delete existing cluster." -ForegroundColor Red
        exit 1
    }
    Write-Host "[minikube] OK - Cluster deleted." -ForegroundColor Green
}

# Step 4: Check current status
$status = minikube status --format "{{.Host}}" 2>$null

if ($status -eq "Running") {
    Write-Host ""
    Write-Host "[minikube] OK - Cluster already running." -ForegroundColor Green
    minikube status
}
else {
    Write-Host ""
    Write-Host "[minikube] Starting cluster with CPUs=$CPUs Memory=${Memory}MB Driver=$Driver ..." -ForegroundColor Yellow
    Write-Host ""

    minikube start `
        --driver=$Driver `
        --cpus=$CPUs `
        --memory=$Memory `
        --kubernetes-version=stable `
        --addons=metrics-server

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[minikube] ERROR - minikube start failed." -ForegroundColor Red
        exit 1
    }
}

# Step 5: Get Minikube IP
$minikubeIP = minikube ip
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($minikubeIP)) {
    Write-Host "[minikube] ERROR - Failed to get Minikube IP." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[minikube] OK - Cluster IP: $minikubeIP" -ForegroundColor Green

# Step 6: Mount project directory
Write-Host ""
Write-Host "[mount] Checking /cloudos-rl mount ..." -ForegroundColor Yellow

$mountCheck = minikube ssh "if [ -d /cloudos-rl ]; then echo EXISTS; fi" 2>$null

if ($mountCheck -match "EXISTS") {
    Write-Host "[mount] OK - /cloudos-rl already mounted" -ForegroundColor Green
}
else {
    Write-Host "[mount] Starting mount in background ..." -ForegroundColor Yellow

    $mountJob = Start-Job -ArgumentList $ProjectRoot -ScriptBlock {
        param($projectRoot)
        $mountPath = "${projectRoot}:/cloudos-rl"
        minikube mount $mountPath --uid=1000 --gid=1000
    }

    Start-Sleep -Seconds 5
    Write-Host "[mount] OK - Mount job started. Job ID: $($mountJob.Id)" -ForegroundColor Green
    Write-Host "[mount] Keep this PowerShell session open to maintain the mount." -ForegroundColor Yellow
}

# Step 7: Apply CRD
Write-Host ""
Write-Host "[k8s] Applying CRD ..." -ForegroundColor Yellow

kubectl apply -f infrastructure/k8s/crd.yaml
if ($LASTEXITCODE -ne 0) {
    Write-Host "[k8s] ERROR - CRD apply failed" -ForegroundColor Red
    exit 1
}

Write-Host "[k8s] OK - CRD registered" -ForegroundColor Green

# Step 8: Verify cluster
Write-Host ""
Write-Host "[k8s] Verifying cluster ..." -ForegroundColor Yellow
kubectl cluster-info
Write-Host ""
kubectl get nodes

if ($LASTEXITCODE -ne 0) {
    Write-Host "[k8s] ERROR - Cluster verification failed" -ForegroundColor Red
    exit 1
}

# Step 9: Finish
Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  Minikube setup complete" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Cluster IP : $minikubeIP"
Write-Host "Next step  : .\scripts\deploy_k8s.ps1" -ForegroundColor Yellow
Write-Host "API URL    : http://${minikubeIP}:30800"
Write-Host ""