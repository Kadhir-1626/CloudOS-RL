"""
Data Pipeline Orchestrator
============================
Coordinates all three data fetchers on independent refresh timers.

Refresh schedule (configurable in config/settings.yaml):
  - Carbon intensity:  every 15 min  (grid mix changes continuously)
  - EC2 pricing:       every 60 min  (spot prices move hourly)
  - CUR / actual cost: every 60 min  (Cost Explorer lags ~24h anyway)

Thread model:
  - Each refresh type runs in its own daemon thread via threading.Timer.
  - Initial refresh runs in a background thread so startup is non-blocking.
  - DataNormalizer writes are atomic — readers never see partial files.
  - In-memory caches allow zero-disk-read hot path for get_pricing/get_carbon.

CLI Usage:
  python -m ai_engine.data_pipeline.pipeline_orchestrator          # one-shot
  python -m ai_engine.data_pipeline.pipeline_orchestrator --daemon # background

Library Usage:
  from ai_engine.data_pipeline.pipeline_orchestrator import DataPipelineOrchestrator
  pipeline = DataPipelineOrchestrator.from_config("config/settings.yaml")
  pipeline.start()
  pipeline.refresh_now()
  carbon  = pipeline.get_carbon()
  pricing = pipeline.get_pricing()
  pipeline.stop()
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# yaml import with clear error message
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    raise ImportError(
        "\n\nPyYAML is not installed.\n"
        "Fix: run   pip install pyyaml   then try again.\n"
    )

from ai_engine.data_pipeline.aws_cur_ingestor    import AWSCURIngestor
from ai_engine.data_pipeline.aws_pricing_fetcher import AWSPricingFetcher
from ai_engine.data_pipeline.carbon_api_client   import CarbonAPIClient
from ai_engine.data_pipeline.data_normalizer     import DataNormalizer

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_INTERVAL = 3600
_DEFAULT_CARBON_INTERVAL  = 900
_DEFAULT_CUR_INTERVAL     = 3600

# Default config used when settings.yaml is missing or unreadable
_DEFAULT_CONFIG: Dict = {
    "data_pipeline": {
        "pricing_refresh_sec":     3600,
        "carbon_refresh_sec":      900,
        "cur_refresh_sec":         3600,
        "pricing_output_path":     "data/pricing/aws_pricing.json",
        "actual_costs_output_path":"data/pricing/aws_actual_costs.json",
        "carbon_output_path":      "data/carbon/carbon_intensity.json",
        "anomaly_threshold_pct":   50.0,
    },
    "aws": {"region": "us-east-1"},
    "environment": {
        "pricing_fallback_path": "data/pricing/aws_pricing.json",
    },
}


# ---------------------------------------------------------------------------
# Pipeline health metrics
# ---------------------------------------------------------------------------

class PipelineMetrics:
    """Thread-safe in-memory counters for pipeline health monitoring."""

    _COUNT_KEYS = [
        "pricing_fetches", "carbon_fetches",  "cur_fetches",
        "pricing_errors",  "carbon_errors",   "cur_errors",
    ]
    _TS_KEYS = [
        "last_pricing_fetch", "last_carbon_fetch", "last_cur_fetch",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._data.update({k: 0    for k in self._COUNT_KEYS})
        self._data.update({k: None for k in self._TS_KEYS})

    def inc(self, key: str):
        with self._lock:
            self._data[key] = self._data.get(key, 0) + 1

    def set(self, key: str, value: Any):
        with self._lock:
            self._data[key] = value

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DataPipelineOrchestrator:
    """
    Background data pipeline for CloudOS-RL.
    Runs pricing, carbon, and CUR fetchers on independent timers.
    """

    def __init__(self, config: Dict):
        self._config = config
        dp = config.get("data_pipeline", {})

        self._pricing_interval = int(dp.get("pricing_refresh_sec", _DEFAULT_PRICING_INTERVAL))
        self._carbon_interval  = int(dp.get("carbon_refresh_sec",  _DEFAULT_CARBON_INTERVAL))
        self._cur_interval     = int(dp.get("cur_refresh_sec",     _DEFAULT_CUR_INTERVAL))

        self._pricing_fetcher = AWSPricingFetcher(config)
        self._cur_ingestor    = AWSCURIngestor(config)
        self._carbon_client   = CarbonAPIClient(config)
        self._normalizer      = DataNormalizer(config)

        self._metrics     = PipelineMetrics()

        self._pricing_cache: Dict = {}
        self._carbon_cache:  Dict = {}
        self._cur_cache:     Dict = {}
        self._cache_lock         = threading.Lock()

        self._running = False
        self._timers: Dict[str, Optional[threading.Timer]] = {
            "pricing": None,
            "carbon":  None,
            "cur":     None,
        }

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> "DataPipelineOrchestrator":
        """
        Loads config from YAML file.
        Falls back to built-in defaults if file is missing or unreadable.
        """
        try:
            with open(config_path, encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
            if not config:
                raise ValueError("settings.yaml is empty")
            logger.info("Config loaded from %s", config_path)
            return cls(config)
        except FileNotFoundError:
            logger.warning(
                "Config file not found: %s — using built-in defaults.", config_path
            )
            return cls(_DEFAULT_CONFIG)
        except Exception as exc:
            logger.warning(
                "Config load error (%s): %s — using built-in defaults.", config_path, exc
            )
            return cls(_DEFAULT_CONFIG)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        """
        Non-blocking start.
        Runs initial full refresh in background thread, then schedules timers.
        """
        if self._running:
            logger.warning("DataPipelineOrchestrator already running.")
            return

        self._running = True
        logger.info(
            "DataPipelineOrchestrator starting — "
            "pricing=%ds  carbon=%ds  cur=%ds",
            self._pricing_interval, self._carbon_interval, self._cur_interval,
        )

        init = threading.Thread(
            target=self._initial_refresh,
            daemon=True,
            name="pipeline-init",
        )
        init.start()

    def stop(self):
        """Cancels all pending timers and stops the pipeline cleanly."""
        self._running = False
        for name, timer in self._timers.items():
            if timer is not None:
                timer.cancel()
                logger.debug("Cancelled timer: %s", name)
        logger.info("DataPipelineOrchestrator stopped.")

    def refresh_now(self):
        """
        Blocking full refresh of all three data sources.
        Use this for: pre-training data population, CLI one-shot mode, tests.
        """
        logger.info("DataPipelineOrchestrator: starting full blocking refresh ...")
        self._do_refresh_carbon()
        self._do_refresh_pricing()
        self._do_refresh_cur()
        logger.info("DataPipelineOrchestrator: full refresh complete.")

    # -----------------------------------------------------------------------
    # Data access (hot path — no disk reads)
    # -----------------------------------------------------------------------

    def get_pricing(self) -> Dict:
        """Returns latest merged pricing dict. No disk read."""
        with self._cache_lock:
            return dict(self._pricing_cache)

    def get_carbon(self) -> Dict[str, float]:
        """
        Returns {region: gco2_per_kwh} flat dict.
        Compatible with StateBuilder.build() carbon parameter.
        Falls back to disk if in-memory cache is empty.
        """
        with self._cache_lock:
            if self._carbon_cache:
                return {
                    r: float(d.get("gco2_per_kwh", 400.0))
                    for r, d in self._carbon_cache.items()
                }
        return self._normalizer.get_flat_carbon()

    def get_cur(self) -> Dict:
        """Returns latest CUR data (blended_rates + usage_summary)."""
        with self._cache_lock:
            return dict(self._cur_cache)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns pipeline health counters."""
        return self._metrics.snapshot()

    # -----------------------------------------------------------------------
    # Private: initial startup
    # -----------------------------------------------------------------------

    def _initial_refresh(self):
        try:
            self._do_refresh_carbon()
            self._do_refresh_pricing()
            self._do_refresh_cur()
        except Exception as exc:
            logger.error("Initial refresh error: %s", exc)
        finally:
            if self._running:
                self._schedule_carbon()
                self._schedule_pricing()
                self._schedule_cur()

    # -----------------------------------------------------------------------
    # Private: refresh actions
    # -----------------------------------------------------------------------

    def _do_refresh_pricing(self):
        try:
            logger.info("Pipeline [pricing]: fetching ...")
            raw    = self._pricing_fetcher.fetch()
            cur    = self._cur_cache if self._cur_cache else None
            merged = self._normalizer.normalize_pricing(raw, cur)
            with self._cache_lock:
                self._pricing_cache = merged
            self._metrics.inc("pricing_fetches")
            self._metrics.set("last_pricing_fetch", datetime.now(tz=timezone.utc).isoformat())
            logger.info("Pipeline [pricing]: done — %d regions", len(merged))
        except Exception as exc:
            logger.error("Pipeline [pricing]: failed: %s", exc)
            self._metrics.inc("pricing_errors")

    def _do_refresh_carbon(self):
        try:
            logger.info("Pipeline [carbon]: fetching ...")
            raw    = self._carbon_client.fetch()
            normed = self._normalizer.normalize_carbon(raw)
            with self._cache_lock:
                self._carbon_cache = normed
            self._metrics.inc("carbon_fetches")
            self._metrics.set("last_carbon_fetch", datetime.now(tz=timezone.utc).isoformat())
            live = sum(1 for v in normed.values() if "live" in v.get("data_source", ""))
            logger.info("Pipeline [carbon]: done — %d live / %d total", live, len(normed))
        except Exception as exc:
            logger.error("Pipeline [carbon]: failed: %s", exc)
            self._metrics.inc("carbon_errors")

    def _do_refresh_cur(self):
        try:
            logger.info("Pipeline [cur]: fetching Cost Explorer ...")
            raw    = self._cur_ingestor.fetch()
            normed = self._normalizer.normalize_actual_costs(raw)
            with self._cache_lock:
                self._cur_cache = normed
            self._metrics.inc("cur_fetches")
            self._metrics.set("last_cur_fetch", datetime.now(tz=timezone.utc).isoformat())
            logger.info("Pipeline [cur]: done — status=%s", raw.get("status"))
        except Exception as exc:
            logger.error("Pipeline [cur]: failed: %s", exc)
            self._metrics.inc("cur_errors")

    # -----------------------------------------------------------------------
    # Private: timer scheduling
    # -----------------------------------------------------------------------

    def _schedule_pricing(self):
        self._schedule("pricing", self._pricing_interval, self._do_refresh_pricing)

    def _schedule_carbon(self):
        self._schedule("carbon", self._carbon_interval, self._do_refresh_carbon)

    def _schedule_cur(self):
        self._schedule("cur", self._cur_interval, self._do_refresh_cur)

    def _schedule(self, name: str, interval: int, fn):
        if not self._running:
            return

        def _run():
            try:
                fn()
            except Exception as exc:
                logger.error("Pipeline timer [%s] error: %s", name, exc)
            finally:
                self._schedule(name, interval, fn)

        timer = threading.Timer(interval=interval, function=_run)
        timer.daemon = True
        timer.name   = f"pipeline-{name}-timer"
        timer.start()
        self._timers[name] = timer
        logger.debug("Pipeline [%s]: next refresh in %ds", name, interval)


# ---------------------------------------------------------------------------
# CLI entry point  (python -m ai_engine.data_pipeline.pipeline_orchestrator)
# ---------------------------------------------------------------------------

def _print_results(pipeline: DataPipelineOrchestrator):
    """Prints summary after a one-shot refresh."""
    m = pipeline.get_metrics()
    print("\n" + "=" * 52)
    print("  Pipeline Metrics")
    print("=" * 52)
    print(f"  pricing_fetches  : {m['pricing_fetches']}")
    print(f"  carbon_fetches   : {m['carbon_fetches']}")
    print(f"  cur_fetches      : {m['cur_fetches']}")
    print(f"  pricing_errors   : {m['pricing_errors']}")
    print(f"  carbon_errors    : {m['carbon_errors']}")
    print(f"  cur_errors       : {m['cur_errors']}")
    print(f"  last_pricing     : {m['last_pricing_fetch']}")
    print(f"  last_carbon      : {m['last_carbon_fetch']}")
    print(f"  last_cur         : {m['last_cur_fetch']}")
    print("=" * 52)

    carbon = pipeline.get_carbon()
    if carbon:
        print("\n  Top 5 Cleanest Regions (gCO2/kWh):")
        for region, co2 in sorted(carbon.items(), key=lambda x: x[1])[:5]:
            print(f"    {region:<25}  {co2:6.1f}")
    print()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-40s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    mode = sys.argv[1] if len(sys.argv) > 1 else "once"

    pipeline = DataPipelineOrchestrator.from_config("config/settings.yaml")

    if mode == "--daemon":
        print("\nCloudOS-RL Data Pipeline — DAEMON MODE")
        print("Press Ctrl+C to stop.\n")
        pipeline.start()

        # Windows-safe signal handling
        try:
            while True:
                time.sleep(60)
                m = pipeline.get_metrics()
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"pricing={m['pricing_fetches']} "
                    f"carbon={m['carbon_fetches']} "
                    f"cur={m['cur_fetches']} "
                    f"errors={m['pricing_errors']}/{m['carbon_errors']}/{m['cur_errors']}"
                )
        except KeyboardInterrupt:
            print("\nShutting down ...")
            pipeline.stop()
            sys.exit(0)

    else:
        print("\nCloudOS-RL Data Pipeline — ONE-SHOT MODE\n")
        pipeline.refresh_now()
        _print_results(pipeline)