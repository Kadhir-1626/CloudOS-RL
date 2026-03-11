"""
Entry point for:
  python -m ai_engine.kafka.kafka_prometheus_bridge
"""
import logging
import signal
import sys

from ai_engine.kafka.bridge_config import BridgeConfig
from ai_engine.kafka.kafka_prometheus_bridge import KafkaPrometheusBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("bridge_main")


def main():
    # Parse optional --port argument
    port = 9090
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--port" and i < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except (IndexError, ValueError):
                pass

    config = BridgeConfig.from_yaml("config/settings.yaml")

    # Override port from CLI if provided
    if port != 9090:
        config._raw["prometheus"]["port"] = port

    bridge = KafkaPrometheusBridge(config)

    # Graceful shutdown on Ctrl+C
    def _shutdown(sig, frame):
        logger.info("Shutdown signal received ...")
        bridge.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start background threads (consumer + pipeline + gauge updater)
    bridge.start()

    # Start Prometheus HTTP server in main thread (blocking)
    bridge.run_prometheus_server()


if __name__ == "__main__":
    main()
