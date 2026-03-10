#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "==> CloudOS-RL Training Pipeline"
echo "==> Project root: $PROJECT_ROOT"

# Activate venv if present
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Validate environment
python -c "import stable_baselines3, gymnasium, torch; print('Dependencies OK')"

# Run training
python -m ai_engine.training.train_agent

echo "==> Training complete. Model saved to models/"