"""
Bridge Configuration Loader
=============================
Loads config for the Kafka → Prometheus bridge from settings.yaml.
Falls back to safe defaults if any section is missing.

Config section in settings.yaml:

  prometheus:
    port: 9090
    host: 0.0.0.0

  kafka:
    bootstrap_servers: localhost:9092
    group_id: cloudos-consumers

  bridge:
    poll_timeout_seconds: 1.0
    max_messages_per_poll: 100
    pipeline_metrics_push_interval: 30
    decision_window_seconds: 60
"""

import logging
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML not installed. Run: pip install pyyaml")

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "kafka": {
        "bootstrap_servers":       "localhost:9092",
        "group_id":                "cloudos-consumers",
    },
    "prometheus": {
        "host": "0.0.0.0",
        "port": 9090,
    },
    "bridge": {
        "poll_timeout_seconds":              1.0,
        "max_messages_per_poll":             100,
        "pipeline_metrics_push_interval":    30,
        "decision_window_seconds":           60,
    },
    "data_pipeline": {
        "carbon_output_path":   "data/carbon/carbon_intensity.json",
        "pricing_output_path":  "data/pricing/aws_pricing.json",
    },
}


class BridgeConfig:
    """
    Typed config accessor for the Kafka-Prometheus bridge.
    All fields have safe defaults so the bridge starts even if settings.yaml
    is incomplete.
    """

    def __init__(self, config: Dict):
        self._raw = config

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str = "config/settings.yaml") -> "BridgeConfig":
        try:
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            logger.info("BridgeConfig: loaded from %s", path)
        except FileNotFoundError:
            logger.warning("BridgeConfig: %s not found — using defaults.", path)
            raw = {}
        except Exception as exc:
            logger.warning("BridgeConfig: load error (%s) — using defaults.", exc)
            raw = {}

        # Deep-merge defaults with loaded config (loaded values take priority)
        merged: Dict = {}
        for section, defaults in _DEFAULTS.items():
            merged[section] = {**defaults, **(raw.get(section, {}) or {})}

        # Pass through any extra sections (aws, environment, etc.)
        for k, v in raw.items():
            if k not in merged:
                merged[k] = v

        return cls(merged)

    # -----------------------------------------------------------------------
    # Kafka
    # -----------------------------------------------------------------------

    @property
    def kafka_bootstrap(self) -> str:
        return self._raw["kafka"]["bootstrap_servers"]

    @property
    def kafka_group_id(self) -> str:
        return self._raw["kafka"]["group_id"]

    # -----------------------------------------------------------------------
    # Prometheus
    # -----------------------------------------------------------------------

    @property
    def prometheus_host(self) -> str:
        return self._raw["prometheus"]["host"]

    @property
    def prometheus_port(self) -> int:
        return int(self._raw["prometheus"]["port"])

    # -----------------------------------------------------------------------
    # Bridge behaviour
    # -----------------------------------------------------------------------

    @property
    def poll_timeout(self) -> float:
        return float(self._raw["bridge"]["poll_timeout_seconds"])

    @property
    def max_per_poll(self) -> int:
        return int(self._raw["bridge"]["max_messages_per_poll"])

    @property
    def pipeline_push_interval(self) -> int:
        return int(self._raw["bridge"]["pipeline_metrics_push_interval"])

    @property
    def decision_window(self) -> int:
        return int(self._raw["bridge"]["decision_window_seconds"])

    # -----------------------------------------------------------------------
    # Data paths (for reading pipeline output files)
    # -----------------------------------------------------------------------

    @property
    def carbon_path(self) -> str:
        return self._raw["data_pipeline"]["carbon_output_path"]

    @property
    def pricing_path(self) -> str:
        return self._raw["data_pipeline"]["pricing_output_path"]

    def raw(self) -> Dict:
        return dict(self._raw)