"""Parallel API data fetching for CJA Auto SDR."""

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

import cjapy
import pandas as pd
from tqdm import tqdm

from cja_auto_sdr.api.resilience import CircuitBreaker, make_api_call_with_retry
from cja_auto_sdr.api.tuning import APIWorkerTuner
from cja_auto_sdr.core.config import APITuningConfig
from cja_auto_sdr.core.exceptions import CircuitBreakerOpen
from cja_auto_sdr.core.perf import PerformanceTracker

TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
API_FETCH_TASK_COUNT = 3  # metrics + dimensions + dataview


class ParallelAPIFetcher:
    """Fetch multiple API endpoints in parallel using threading.

    Supports optional API auto-tuning and circuit breaker protection.

    Args:
        cja: CJA API client
        logger: Logger instance
        perf_tracker: Performance tracker for timing metrics
        max_workers: Maximum parallel workers (default: 3)
        quiet: Suppress progress output (default: False)
        tuning_config: Optional API worker auto-tuning configuration
        circuit_breaker: Optional circuit breaker for failure protection
    """

    def __init__(
        self,
        cja: cjapy.CJA,
        logger: logging.Logger,
        perf_tracker: PerformanceTracker,
        max_workers: int = 3,
        quiet: bool = False,
        tuning_config: APITuningConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.cja = cja
        self.logger = logger
        self.perf_tracker = perf_tracker
        self.max_workers = max_workers
        self.quiet = quiet
        self.circuit_breaker = circuit_breaker

        # Initialize API tuner if config provided
        self.tuner: APIWorkerTuner | None = None
        if tuning_config is not None:
            # Each fetch cycle emits one timing per API task. Clamp the window so
            # auto-tuning can make at least one decision per data view run.
            effective_window = max(1, min(tuning_config.sample_window, API_FETCH_TASK_COUNT))
            effective_config = tuning_config
            if effective_window != tuning_config.sample_window:
                effective_config = replace(tuning_config, sample_window=effective_window)
                self.logger.debug(
                    "API tuner sample_window clamped from %s to %s for per-data-view fetch cycle",
                    tuning_config.sample_window,
                    effective_window,
                )

            self.tuner = APIWorkerTuner(config=effective_config, initial_workers=max_workers, logger=logger)
            # Use tuner's worker count as effective max
            self.max_workers = self.tuner.current_workers

    def fetch_all_data(self, data_view_id: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Fetch metrics, dimensions, and data view info in parallel

        Returns:
            Tuple of (metrics_df, dimensions_df, dataview_info)
        """
        self.logger.info("Starting parallel data fetch operations...")
        self.perf_tracker.start("Parallel API Fetch")

        results = {"metrics": None, "dimensions": None, "dataview": None}

        errors = {}

        # Define fetch tasks
        tasks = {
            "metrics": lambda: self._fetch_metrics(data_view_id),
            "dimensions": lambda: self._fetch_dimensions(data_view_id),
            "dataview": lambda: self._fetch_dataview_info(data_view_id),
        }

        # Execute tasks in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_name = {executor.submit(task): name for name, task in tasks.items()}

            # Collect results as they complete with progress indicator
            with tqdm(
                total=len(tasks),
                desc="Fetching API data",
                unit="item",
                bar_format=TQDM_BAR_FORMAT,
                leave=False,
                disable=self.quiet,
            ) as pbar:
                for future in as_completed(future_to_name):
                    task_name = future_to_name[future]
                    try:
                        results[task_name] = future.result()
                        pbar.set_postfix_str(f"\u2713 {task_name}", refresh=True)
                        self.logger.info(f"\u2713 {task_name.capitalize()} fetch completed")
                    except Exception as e:
                        errors[task_name] = str(e)
                        pbar.set_postfix_str(f"\u2717 {task_name}", refresh=True)
                        self.logger.error(f"\u2717 {task_name.capitalize()} fetch failed: {e}")
                    pbar.update(1)

        self.perf_tracker.end("Parallel API Fetch")

        # Log summary
        success_count = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"Parallel fetch complete: {success_count}/3 successful")

        if errors:
            self.logger.warning(f"Errors encountered: {list(errors.keys())}")

        # Return results with proper None checking for DataFrames
        metrics_result = results.get("metrics")
        dimensions_result = results.get("dimensions")
        dataview_result = results.get("dataview")

        # Handle DataFrame None checks properly
        if metrics_result is None or (isinstance(metrics_result, pd.DataFrame) and metrics_result.empty):
            metrics_result = pd.DataFrame()

        if dimensions_result is None or (isinstance(dimensions_result, pd.DataFrame) and dimensions_result.empty):
            dimensions_result = pd.DataFrame()

        if dataview_result is None:
            dataview_result = {}

        return metrics_result, dimensions_result, dataview_result

    def _timed_api_call(self, api_func: Callable, *args, operation_name: str, **kwargs) -> Any:
        """
        Execute an API call with timing for auto-tuning.

        Wraps make_api_call_with_retry with timing measurement and tuner feedback.
        Only records timing on successful calls so retries don't inflate metrics.
        """
        start_time = time.time()
        result = make_api_call_with_retry(
            api_func,
            *args,
            logger=self.logger,
            operation_name=operation_name,
            circuit_breaker=self.circuit_breaker,
            **kwargs,
        )
        # Record response time only on success to avoid retry delays inflating metrics
        if self.tuner is not None:
            duration_ms = (time.time() - start_time) * 1000
            new_workers = self.tuner.record_response_time(duration_ms)
            if new_workers is not None:
                self.max_workers = new_workers
        return result

    def _fetch_metrics(self, data_view_id: str) -> pd.DataFrame:
        """Fetch metrics with error handling and retry"""
        try:
            self.logger.debug(f"Fetching metrics for {data_view_id}")

            # Use retry for transient network errors with circuit breaker support
            metrics = self._timed_api_call(
                self.cja.getMetrics, data_view_id, inclType=True, full=True, operation_name="getMetrics"
            )

            if metrics is None or (isinstance(metrics, pd.DataFrame) and metrics.empty):
                self.logger.warning("No metrics returned from API")
                return pd.DataFrame()

            self.logger.info(f"Successfully fetched {len(metrics)} metrics")
            return metrics

        except CircuitBreakerOpen as e:
            self.logger.warning(f"Circuit breaker open for metrics fetch: {e.message}")
            return pd.DataFrame()
        except AttributeError as e:
            self.logger.error(f"API method error - getMetrics may not be available: {e!s}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch metrics: {e!s}")
            return pd.DataFrame()

    def _fetch_dimensions(self, data_view_id: str) -> pd.DataFrame:
        """Fetch dimensions with error handling and retry"""
        try:
            self.logger.debug(f"Fetching dimensions for {data_view_id}")

            # Use retry for transient network errors with circuit breaker support
            dimensions = self._timed_api_call(
                self.cja.getDimensions, data_view_id, inclType=True, full=True, operation_name="getDimensions"
            )

            if dimensions is None or (isinstance(dimensions, pd.DataFrame) and dimensions.empty):
                self.logger.warning("No dimensions returned from API")
                return pd.DataFrame()

            self.logger.info(f"Successfully fetched {len(dimensions)} dimensions")
            return dimensions

        except CircuitBreakerOpen as e:
            self.logger.warning(f"Circuit breaker open for dimensions fetch: {e.message}")
            return pd.DataFrame()
        except AttributeError as e:
            self.logger.error(f"API method error - getDimensions may not be available: {e!s}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Failed to fetch dimensions: {e!s}")
            return pd.DataFrame()

    def _fetch_dataview_info(self, data_view_id: str) -> dict:
        """Fetch data view information with error handling and retry"""
        try:
            self.logger.debug(f"Fetching data view information for {data_view_id}")

            # Use retry for transient network errors with circuit breaker support
            lookup_data = self._timed_api_call(self.cja.getDataView, data_view_id, operation_name="getDataView")

            if not lookup_data:
                self.logger.error("Data view information returned empty")
                return {"name": "Unknown", "id": data_view_id}

            self.logger.info(f"Successfully fetched data view info: {lookup_data.get('name', 'Unknown')}")
            return lookup_data

        except CircuitBreakerOpen as e:
            self.logger.warning(f"Circuit breaker open for data view fetch: {e.message}")
            return {"name": "Unknown", "id": data_view_id, "circuit_breaker_open": True}
        except Exception as e:
            self.logger.error(f"Failed to fetch data view information: {e!s}")
            return {"name": "Unknown", "id": data_view_id, "error": str(e)}

    def get_tuner_statistics(self) -> dict[str, Any] | None:
        """Get API tuner statistics if tuning is enabled."""
        if self.tuner is not None:
            return self.tuner.get_statistics()
        return None
