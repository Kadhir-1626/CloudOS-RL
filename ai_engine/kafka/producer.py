import json
import logging
import time
from typing import Any, Dict

from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

logger = logging.getLogger(__name__)

TOPICS = {
    "decisions": "cloudos.scheduling.decisions",
    "metrics":   "cloudos.metrics",
    "alerts":    "cloudos.alerts",
    "workloads": "cloudos.workload.events",
}


class CloudOSProducer:
    def __init__(self, config: Dict):
        servers = config.get("kafka", {}).get("bootstrap_servers", "localhost:9092")
        self._producer = Producer({
            "bootstrap.servers": servers,
            "client.id":         "cloudos-producer",
            "acks":              "all",
            "retries":           3,
            "retry.backoff.ms":  300,
            "compression.type":  "lz4",
            "linger.ms":         5,
        })
        self._ensure_topics(servers, config)

    def publish_decision(self, decision: Dict):
        self._send(TOPICS["decisions"], decision["workload_id"], decision)

    def publish_metrics(self, metrics: Dict):
        self._send(TOPICS["metrics"], str(int(time.time() * 1000)), metrics)

    def publish_alert(self, kind: str, detail: Dict):
        self._send(TOPICS["alerts"], kind, {"kind": kind, "detail": detail, "ts": time.time()})

    def flush(self, timeout: float = 10.0):
        self._producer.flush(timeout)

    # ── private ─────────────────────────────────────────────────────────────

    def _send(self, topic: str, key: str, payload: Dict):
        try:
            self._producer.produce(
                topic=topic,
                key=key.encode(),
                value=json.dumps(payload, default=str).encode(),
                on_delivery=self._on_delivery,
            )
            self._producer.poll(0)
        except KafkaException as exc:
            logger.error("Kafka produce [%s]: %s", topic, exc)

    @staticmethod
    def _on_delivery(err: Any, msg: Any):
        if err:
            logger.error("Delivery failed: %s", err)
        else:
            logger.debug("Delivered → %s [%d]", msg.topic(), msg.partition())

    def _ensure_topics(self, servers: str, config: Dict):
        admin = AdminClient({"bootstrap.servers": servers})
        try:
            existing = set(admin.list_topics(timeout=10).topics)
        except KafkaException:
            return

        parts  = config.get("kafka", {}).get("partitions", 3)
        rep    = config.get("kafka", {}).get("replication", 1)
        to_add = [
            NewTopic(t, num_partitions=parts, replication_factor=rep)
            for t in TOPICS.values() if t not in existing
        ]
        if to_add:
            for t, f in admin.create_topics(to_add).items():
                try:
                    f.result()
                    logger.info("Created topic: %s", t)
                except Exception as exc:
                    if "already exists" not in str(exc):
                        logger.warning("Topic %s: %s", t, exc)