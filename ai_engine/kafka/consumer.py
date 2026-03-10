import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException

logger = logging.getLogger(__name__)


class CloudOSConsumer:
    def __init__(self, config: Dict, group_id: str, topics: List[str]):
        servers = config.get("kafka", {}).get("bootstrap_servers", "localhost:9092")
        self._consumer = Consumer({
            "bootstrap.servers":    servers,
            "group.id":             group_id,
            "auto.offset.reset":    "latest",
            "enable.auto.commit":   False,
            "session.timeout.ms":   30_000,
            "max.poll.interval.ms": 300_000,
        })
        self._topics   = topics
        self._handlers: Dict[str, Callable[[Dict], None]] = {}
        self._running  = False
        self._thread: Optional[threading.Thread] = None

    def on(self, topic: str, handler: Callable[[Dict], None]):
        self._handlers[topic] = handler
        return self

    def start(self):
        self._consumer.subscribe(self._topics)
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="kafka-consumer")
        self._thread.start()
        logger.info("Consumer started → %s", self._topics)

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._consumer.close()
        logger.info("Consumer stopped.")

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
                payload = json.loads(msg.value().decode())
                if topic in self._handlers:
                    self._handlers[topic](payload)

                self._consumer.commit(asynchronous=False)

            except KafkaException as exc:
                logger.error("Consumer error: %s", exc)
                time.sleep(1.0)
            except Exception as exc:
                logger.exception("Unexpected error: %s", exc)
                time.sleep(1.0)