#!/usr/bin/env bash
# =============================================================================
# CloudOS-RL — SHAP Background Dataset Generator
# =============================================================================
# Generates the background dataset used by SHAP KernelExplainer.
# Reads Module G output files (aws_pricing.json, carbon_intensity.json)
# for realistic feature distributions.
#
# Run this ONCE before inference or after retraining the RL model.
# Output: data/shap/background_dataset.npy (200 samples x 45 features)
#
# Usage:
#   bash scripts/run_shap_background.sh            # 200 samples (default)
#   bash scripts/run_shap_background.sh --force    # force regeneration
#   bash scripts/run_shap_background.sh --samples 500
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     CloudOS-RL  SHAP Background Generator       ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Activate venv
if   [ -f ".venv/Scripts/activate" ]; then source .venv/Scripts/activate
elif [ -f ".venv/bin/activate"     ]; then source .venv/bin/activate
fi

# Load .env
if [ -f ".env" ]; then set -a; source .env; set +a; fi

# Check shap installed
python3 -c "import shap; print('[deps]  shap OK:', shap.__version__)" 2>/dev/null || {
    echo "[deps]  shap not installed. Fix: pip install shap"
    exit 1
}

# Check Module G data files
echo "[data]  Checking Module G pipeline output files ..."
[ -f "data/pricing/aws_pricing.json"       ] && \
    python3 -c "import json; d=json.load(open('data/pricing/aws_pricing.json')); print(f'[data]  aws_pricing.json OK ({len(d)} regions)')" \
    || echo "[data]  aws_pricing.json MISSING — run data pipeline first"

[ -f "data/carbon/carbon_intensity.json"   ] && \
    python3 -c "import json; d=json.load(open('data/carbon/carbon_intensity.json')); print(f'[data]  carbon_intensity.json OK ({len(d)} regions)')" \
    || echo "[data]  carbon_intensity.json MISSING — run data pipeline first"

mkdir -p data/shap

# Parse arguments
FORCE=""
SAMPLES=200
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)   FORCE="True"; shift ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo ""
echo "[shap]  Generating background dataset (${SAMPLES} samples) ..."
echo ""

python3 - <<PYEOF
import sys, json, yaml
sys.path.insert(0, '.')

with open('config/settings.yaml') as f:
    config = yaml.safe_load(f)

from ai_engine.explainability.background_generator import BackgroundDataGenerator

gen = BackgroundDataGenerator(config)
bg  = gen.generate(n_samples=${SAMPLES}, seed=42, force=${FORCE:-False})

print(f'[shap]  Background dataset shape: {bg.shape}')
print(f'[shap]  dtype:  {bg.dtype}')
print(f'[shap]  mean:   {bg.mean():.4f}')
print(f'[shap]  std:    {bg.std():.4f}')
print(f'[shap]  min:    {bg.min():.4f}')
print(f'[shap]  max:    {bg.max():.4f}')
print()

# Show metadata
import json
with open('data/shap/background_metadata.json') as f:
    meta = json.load(f)
print(f'[shap]  Saved to:        {meta.get(\"shape\")}')
print(f'[shap]  Pricing source:  {meta.get(\"pricing_source\")}')
print(f'[shap]  Carbon source:   {meta.get(\"carbon_source\")}')
print(f'[shap]  Generated at:    {meta.get(\"generated_at\")}')
PYEOF

echo ""
echo "[done]  SHAP background dataset ready at data/shap/background_dataset.npy"
