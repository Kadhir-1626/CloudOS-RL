#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    --reload
```

---

### `.env.example`
```
CLOUDOS_DEBUG=false
CLOUDOS_API_PORT=8000
CLOUDOS_LOG_LEVEL=INFO
CLOUDOS_KAFKA_BOOTSTRAP=localhost:9092
CLOUDOS_AWS_REGION=us-east-1
CLOUDOS_MODEL_PATH=models/best/best_model
CLOUDOS_VECNORM_PATH=models/vec_normalize.pkl

# AWS credentials — use aws configure instead of hardcoding here
AWS_PROFILE=default
```

---

## 12. Execution Roadmap
```
WEEK 1 — Environment + Training
────────────────────────────────
1. pip install -r requirements.txt
2. mkdir -p data/pricing data/carbon models
3. python -m pytest tests/ -v              ← all green before training
4. bash scripts/run_training.sh            ← 2M timesteps ~2-4 hrs
5. tensorboard --logdir models/tensorboard ← watch reward climb

WEEK 2 — API + Kafka
──────────────────────
6. bash scripts/start_api.sh
7. curl -X POST localhost:8000/api/v1/schedule/ -H "Content-Type: application/json" \
   -d '{"workload_id":"w1","cpu_request":4,"memory_request_gb":8,"expected_duration_hours":2}'
8. Verify Kafka topics created; tail consumer logs

WEEK 3 — Kubernetes
─────────────────────
9.  minikube start --cpus=4 --memory=8192
10. kubectl apply -f infrastructure/k8s/crd.yaml
11. kubectl apply -f infrastructure/k8s/rbac.yaml
12. kubectl get cloudosworkloads -A

WEEK 4 — Grafana Dashboards
─────────────────────────────
13. Configure Prometheus scrape at localhost:8000/metrics
14. Import panels: cost_savings_pct, carbon_savings_pct,
    inference_latency_ms, decisions_per_hour, cloud_dist pie