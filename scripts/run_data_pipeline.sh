#!/usr/bin/env bash
# =============================================================================
# CloudOS-RL — Data Pipeline Runner
# =============================================================================
# Modes:
#   bash scripts/run_data_pipeline.sh           -> one-shot full refresh
#   bash scripts/run_data_pipeline.sh --daemon  -> background daemon mode
#
# What it does:
#   - Fetches real AWS EC2 on-demand + spot pricing
#   - Fetches real AWS Cost Explorer blended rates
#   - Fetches live or static carbon intensity per region
#   - Writes output to data/pricing/ and data/carbon/
#
# Run BEFORE training to pre-populate real pricing data.
# Run AS DAEMON alongside the API server for live data during inference.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║         CloudOS-RL  Data Pipeline                ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Activate virtual environment
# -----------------------------------------------------------------------------
if   [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
    echo "[env]  Activated: .venv/Scripts/activate"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "[env]  Activated: .venv/bin/activate"
else
    echo "[env]  No .venv found — using system Python"
fi

# -----------------------------------------------------------------------------
# Step 2: Load .env file
# -----------------------------------------------------------------------------
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "[env]  Loaded .env"
else
    echo "[env]  .env not found — using environment variables if set"
fi

# -----------------------------------------------------------------------------
# Step 3: Check AWS credentials
# -----------------------------------------------------------------------------
echo ""
echo "[aws]  Checking credentials ..."
if aws sts get-caller-identity --output text --query 'Account' > /dev/null 2>&1; then
    ACCOUNT_ID=$(aws sts get-caller-identity --output text --query 'Account')
    echo "[aws]  Authenticated — Account: $ACCOUNT_ID"
else
    echo "[aws]  WARNING: AWS credentials not configured."
    echo "       Pricing API and CUR will use fallback values."
    echo "       Run 'aws configure' to fix this."
fi

# -----------------------------------------------------------------------------
# Step 4: Check Electricity Maps API key
# -----------------------------------------------------------------------------
echo ""
if [ -n "${ELECTRICITY_MAPS_API_KEY:-}" ]; then
    echo "[carbon]  Electricity Maps API key found — live carbon data enabled"
else
    echo "[carbon]  ELECTRICITY_MAPS_API_KEY not set — using static carbon values"
    echo "          Get a free key: https://www.electricitymaps.com/free-tier"
fi

# -----------------------------------------------------------------------------
# Step 5: Validate Python dependencies
# -----------------------------------------------------------------------------
echo ""
echo "[deps]  Checking Python dependencies ..."
python3 -c "
import boto3, httpx, yaml
print('[deps]  boto3  OK')
print('[deps]  httpx  OK')
print('[deps]  yaml   OK')
"

# -----------------------------------------------------------------------------
# Step 6: Create output directories
# -----------------------------------------------------------------------------
mkdir -p data/pricing data/carbon
echo "[dirs]  data/pricing/ and data/carbon/ ready"

# -----------------------------------------------------------------------------
# Step 7: Run pipeline
# -----------------------------------------------------------------------------
MODE="${1:-}"

echo ""
if [ "$MODE" = "--daemon" ]; then
    echo "[pipeline]  Starting DAEMON mode (Ctrl+C to stop) ..."
    echo ""
    python3 -m ai_engine.data_pipeline.pipeline_orchestrator --daemon

else
    echo "[pipeline]  Starting ONE-SHOT refresh ..."
    echo ""
    python3 -m ai_engine.data_pipeline.pipeline_orchestrator

    echo ""
    echo "═══════════════════════════════════════════════════"
    echo " Output Files Written:"
    echo "═══════════════════════════════════════════════════"

    if [ -f "data/pricing/aws_pricing.json" ]; then
        SIZE=$(wc -c < data/pricing/aws_pricing.json)
        echo "  [OK]  data/pricing/aws_pricing.json       ($SIZE bytes)"
    else
        echo "  [--]  data/pricing/aws_pricing.json       (not written)"
    fi

    if [ -f "data/pricing/aws_actual_costs.json" ]; then
        SIZE=$(wc -c < data/pricing/aws_actual_costs.json)
        echo "  [OK]  data/pricing/aws_actual_costs.json  ($SIZE bytes)"
    else
        echo "  [--]  data/pricing/aws_actual_costs.json  (not written — CUR may need IAM perms)"
    fi

    if [ -f "data/carbon/carbon_intensity.json" ]; then
        SIZE=$(wc -c < data/carbon/carbon_intensity.json)
        echo "  [OK]  data/carbon/carbon_intensity.json   ($SIZE bytes)"
    else
        echo "  [--]  data/carbon/carbon_intensity.json   (not written)"
    fi

    echo "═══════════════════════════════════════════════════"
    echo ""
    echo "[done]  Data pipeline refresh complete."
fi