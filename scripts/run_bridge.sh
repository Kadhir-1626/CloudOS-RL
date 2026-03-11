#!/usr/bin/env bash
# =============================================================================
# CloudOS-RL — Kafka → Prometheus Bridge Runner
# =============================================================================
# What it does:
#   - Consumes Kafka topics: decisions, metrics, alerts, workload events
#   - Exposes Prometheus metrics at http://localhost:9090/metrics
#   - Reads Module G output files for carbon + pricing gauges
#
# Prerequisites:
#   - Kafka running (localhost:9092)
#   - Module G data pipeline has run at least once
#   - pip install confluent-kafka prometheus-client
#
# Usage:
#   bash scripts/run_bridge.sh            # default port 9090
#   bash scripts/run_bridge.sh --port 9091
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     CloudOS-RL  Kafka → Prometheus Bridge        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# -----------------------------------------------------------------------------
# Activate virtual environment
# -----------------------------------------------------------------------------
if   [ -f ".venv/Scripts/activate" ]; then source .venv/Scripts/activate
elif [ -f ".venv/bin/activate"     ]; then source .venv/bin/activate
fi

# -----------------------------------------------------------------------------
# Load .env
# -----------------------------------------------------------------------------
if [ -f ".env" ]; then
    set -a; source .env; set +a
    echo "[env]  Loaded .env"
fi

# -----------------------------------------------------------------------------
# Check dependencies
# -----------------------------------------------------------------------------
echo "[deps] Checking dependencies ..."
python3 -c "
import sys
missing = []
for pkg in ['confluent_kafka', 'prometheus_client', 'yaml']:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    pkg_map = {'confluent_kafka': 'confluent-kafka', 'prometheus_client': 'prometheus-client', 'yaml': 'pyyaml'}
    cmds = ' '.join(pkg_map.get(p, p) for p in missing)
    print(f'[deps] MISSING: {missing}')
    print(f'[deps] Fix: pip install {cmds}')
    sys.exit(1)

print('[deps] All OK: confluent_kafka, prometheus_client, yaml')
"

# -----------------------------------------------------------------------------
# Check Kafka is running
# -----------------------------------------------------------------------------
echo ""
echo "[kafka] Checking Kafka connection ..."
python3 -c "
from confluent_kafka.admin import AdminClient
try:
    a = AdminClient({'bootstrap.servers': 'localhost:9092'})
    t = a.list_topics(timeout=5)
    print(f'[kafka] Connected — {len(t.topics)} topics available')
except Exception as e:
    print(f'[kafka] WARNING: Cannot connect to Kafka: {e}')
    print('[kafka] Bridge will retry on startup. Make sure Kafka is running.')
"

# -----------------------------------------------------------------------------
# Check data files exist
# -----------------------------------------------------------------------------
echo ""
echo "[data]  Checking Module G output files ..."
[ -f "data/carbon/carbon_intensity.json" ] && echo "[data]  carbon_intensity.json  OK" || echo "[data]  carbon_intensity.json  MISSING — run data pipeline first"
[ -f "data/pricing/aws_pricing.json"     ] && echo "[data]  aws_pricing.json       OK" || echo "[data]  aws_pricing.json       MISSING — run data pipeline first"

# -----------------------------------------------------------------------------
# Parse optional port argument
# -----------------------------------------------------------------------------
PORT=9090
if [ "${1:-}" = "--port" ] && [ -n "${2:-}" ]; then
    PORT="$2"
fi

echo ""
echo "[bridge] Starting bridge on port $PORT ..."
echo "[bridge] Prometheus metrics: http://localhost:$PORT/metrics"
echo "[bridge] Press Ctrl+C to stop."
echo ""

python3 -m ai_engine.kafka.kafka_prometheus_bridge --port "$PORT"