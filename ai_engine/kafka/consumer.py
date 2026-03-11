"""
CloudOS-RL Kafka Consumer (Base)
==================================
General-purpose Kafka consumer used by components other than the bridge.
The bridge uses its own internal consumer loop for tighter control.

Usage:
  consumer = CloudOSConsumer(config, group_id="my-group", topics=["cloudos.alerts"])
  consumer.on("cloudos.alerts", handle_alert)
  consumer.start()
  ...
  consumer.stop()
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

try:
    from confluent_kafka import Consumer, KafkaError, KafkaException
except ImportError:
    raise ImportError(
        "\n\nconfluent-kafka is not installed.\n"
        "Fix: run   pip install confluent-kafka   then try again.\n"
    )

logger = logging.getLogger(__name__)


class CloudOSConsumer:
    """
    Simple callback-based Kafka consumer.
    Each topic maps to one handler function via .on(topic, handler).
    """

    def __init__(self, config: Dict, group_id: str, topics: List[str]):
        servers = config.get("kafka", {}).get("bootstrap_servers", "localhost:9092")
        self._consumer = Consumer({
            "bootstrap.servers":     servers,
            "group.id":              group_id,
            "auto.offset.reset":     "latest",
            "enable.auto.commit":    False,
            "session.timeout.ms":    30_000,
            "max.poll.interval.ms":  300_000,
        })
        self._topics   = topics
        self._handlers: Dict[str, Callable[[Dict], None]] = {}
        self._running  = False
        self._thread:   Optional[threading.Thread] = None

    def on(self, topic: str, handler: Callable[[Dict], None]) -> "CloudOSConsumer":
        """Register a handler for a topic. Returns self for chaining."""
        self._handlers[topic] = handler
        return self

    def start(self):
        """Start consuming in a background daemon thread."""
        self._consumer.subscribe(self._topics)
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop,
            daemon=True,
            name=f"consumer-{'-'.join(self._topics[:2])}",
        )
        self._thread.start()
        logger.info("CloudOSConsumer started — topics=%s", self._topics)

    def stop(self):
        """Signal the consumer to stop and wait for it."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=8.0)
        self._consumer.close()
        logger.info("CloudOSConsumer stopped.")

    def _loop(self):
        while self._running:
            try:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() != KafkaError._PARTITION_EOF:
                        raise KafkaException(msg.error())
                    continue

                topic   = msg.topic()
                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    logger.warning("Consumer parse error [%s]: %s", topic, exc)
                    continue

                if topic in self._handlers:
                    try:
                        self._handlers[topic](payload)
                    except Exception as exc:
                        logger.exception("Handler error [%s]: %s", topic, exc)

                self._consumer.commit(asynchronous=False)

            except KafkaException as exc:
                logger.error("Consumer KafkaException: %s", exc)
                time.sleep(1.0)
            except Exception as exc:
                logger.exception("Consumer unexpected error: %s", exc)
                time.sleep(1.0)