"""Tests for ParallelAPIFetcher class"""
import pytest
import pandas as pd
import logging
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from concurrent.futures import ThreadPoolExecutor
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cja_auto_sdr.generator import ParallelAPIFetcher, PerformanceTracker


@pytest.fixture
def mock_logger():
    """Create a mock logger"""
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    return logger


@pytest.fixture
def mock_perf_tracker(mock_logger):
    """Create a mock performance tracker"""
    tracker = Mock(spec=PerformanceTracker)
    tracker.start = Mock()
    tracker.end = Mock()
    return tracker


@pytest.fixture
def mock_cja():
    """Create a mock CJA instance"""
    cja = Mock()
    return cja


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data"""
    return pd.DataFrame([
        {"id": "metric1", "name": "Metric 1", "type": "calculated", "description": "Test"},
        {"id": "metric2", "name": "Metric 2", "type": "standard", "description": "Test 2"}
    ])


@pytest.fixture
def sample_dimensions_data():
    """Sample dimensions data"""
    return pd.DataFrame([
        {"id": "dim1", "name": "Dimension 1", "type": "string", "description": "Test"},
        {"id": "dim2", "name": "Dimension 2", "type": "string", "description": "Test 2"}
    ])


@pytest.fixture
def sample_dataview_info():
    """Sample data view info"""
    return {
        "id": "dv_test_12345",
        "name": "Test Data View",
        "owner": {"name": "Test Owner"}
    }


class TestParallelAPIFetcherInit:
    """Tests for ParallelAPIFetcher initialization"""

    def test_init_with_defaults(self, mock_cja, mock_logger, mock_perf_tracker):
        """Test initialization with default parameters"""
        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)

        assert fetcher.cja == mock_cja
        assert fetcher.logger == mock_logger
        assert fetcher.perf_tracker == mock_perf_tracker
        assert fetcher.max_workers == 3

    def test_init_with_custom_workers(self, mock_cja, mock_logger, mock_perf_tracker):
        """Test initialization with custom worker count"""
        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker, max_workers=5)

        assert fetcher.max_workers == 5

    def test_init_with_minimum_workers(self, mock_cja, mock_logger, mock_perf_tracker):
        """Test initialization with minimum worker count"""
        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker, max_workers=1)

        assert fetcher.max_workers == 1


class TestParallelAPIFetcherFetchAllData:
    """Tests for fetch_all_data method"""

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_fetch_all_data_success(self, mock_tqdm, mock_api_call, mock_cja, mock_logger,
                                     mock_perf_tracker, sample_metrics_data,
                                     sample_dimensions_data, sample_dataview_info):
        """Test successful parallel data fetching"""
        # Setup mock tqdm context manager
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        # Setup API responses
        def api_side_effect(func, *args, **kwargs):
            if 'getMetrics' in kwargs.get('operation_name', ''):
                return sample_metrics_data
            elif 'getDimensions' in kwargs.get('operation_name', ''):
                return sample_dimensions_data
            elif 'getDataView' in kwargs.get('operation_name', ''):
                return sample_dataview_info
            return None

        mock_api_call.side_effect = api_side_effect

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        metrics, dimensions, dataview = fetcher.fetch_all_data("dv_test_12345")

        # Verify results
        assert not metrics.empty
        assert not dimensions.empty
        assert dataview == sample_dataview_info

        # Verify performance tracking
        mock_perf_tracker.start.assert_called_once_with("Parallel API Fetch")
        mock_perf_tracker.end.assert_called_once_with("Parallel API Fetch")

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_fetch_all_data_empty_metrics(self, mock_tqdm, mock_api_call, mock_cja,
                                           mock_logger, mock_perf_tracker,
                                           sample_dimensions_data, sample_dataview_info):
        """Test handling of empty metrics response"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        def api_side_effect(func, *args, **kwargs):
            if 'getMetrics' in kwargs.get('operation_name', ''):
                return None
            elif 'getDimensions' in kwargs.get('operation_name', ''):
                return sample_dimensions_data
            elif 'getDataView' in kwargs.get('operation_name', ''):
                return sample_dataview_info
            return None

        mock_api_call.side_effect = api_side_effect

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        metrics, dimensions, dataview = fetcher.fetch_all_data("dv_test_12345")

        assert metrics.empty
        assert not dimensions.empty
        assert dataview == sample_dataview_info

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_fetch_all_data_empty_dimensions(self, mock_tqdm, mock_api_call, mock_cja,
                                              mock_logger, mock_perf_tracker,
                                              sample_metrics_data, sample_dataview_info):
        """Test handling of empty dimensions response"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        def api_side_effect(func, *args, **kwargs):
            if 'getMetrics' in kwargs.get('operation_name', ''):
                return sample_metrics_data
            elif 'getDimensions' in kwargs.get('operation_name', ''):
                return pd.DataFrame()
            elif 'getDataView' in kwargs.get('operation_name', ''):
                return sample_dataview_info
            return None

        mock_api_call.side_effect = api_side_effect

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        metrics, dimensions, dataview = fetcher.fetch_all_data("dv_test_12345")

        assert not metrics.empty
        assert dimensions.empty
        assert dataview == sample_dataview_info

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_fetch_all_data_empty_dataview(self, mock_tqdm, mock_api_call, mock_cja,
                                            mock_logger, mock_perf_tracker,
                                            sample_metrics_data, sample_dimensions_data):
        """Test handling of empty dataview response"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        def api_side_effect(func, *args, **kwargs):
            if 'getMetrics' in kwargs.get('operation_name', ''):
                return sample_metrics_data
            elif 'getDimensions' in kwargs.get('operation_name', ''):
                return sample_dimensions_data
            elif 'getDataView' in kwargs.get('operation_name', ''):
                return None
            return None

        mock_api_call.side_effect = api_side_effect

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        metrics, dimensions, dataview = fetcher.fetch_all_data("dv_test_12345")

        assert not metrics.empty
        assert not dimensions.empty
        # When dataview info fetch fails, it returns a fallback dict with Unknown name
        assert dataview['name'] == 'Unknown'
        assert dataview['id'] == 'dv_test_12345'


class TestParallelAPIFetcherFetchMetrics:
    """Tests for _fetch_metrics method"""

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_metrics_success(self, mock_api_call, mock_cja, mock_logger,
                                    mock_perf_tracker, sample_metrics_data):
        """Test successful metrics fetching"""
        mock_api_call.return_value = sample_metrics_data

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_metrics("dv_test_12345")

        assert not result.empty
        assert len(result) == 2
        mock_api_call.assert_called_once()

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_metrics_returns_none(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of None response from API"""
        mock_api_call.return_value = None

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_metrics("dv_test_12345")

        assert result.empty
        mock_logger.warning.assert_called()

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_metrics_returns_empty_df(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of empty DataFrame response"""
        mock_api_call.return_value = pd.DataFrame()

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_metrics("dv_test_12345")

        assert result.empty

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_metrics_attribute_error(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of AttributeError (API method not available)"""
        mock_api_call.side_effect = AttributeError("getMetrics not available")

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_metrics("dv_test_12345")

        assert result.empty
        mock_logger.error.assert_called()

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_metrics_generic_exception(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of generic exception"""
        mock_api_call.side_effect = Exception("Network error")

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_metrics("dv_test_12345")

        assert result.empty
        mock_logger.error.assert_called()


class TestParallelAPIFetcherFetchDimensions:
    """Tests for _fetch_dimensions method"""

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dimensions_success(self, mock_api_call, mock_cja, mock_logger,
                                       mock_perf_tracker, sample_dimensions_data):
        """Test successful dimensions fetching"""
        mock_api_call.return_value = sample_dimensions_data

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dimensions("dv_test_12345")

        assert not result.empty
        assert len(result) == 2

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dimensions_returns_none(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of None response"""
        mock_api_call.return_value = None

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dimensions("dv_test_12345")

        assert result.empty

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dimensions_attribute_error(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of AttributeError"""
        mock_api_call.side_effect = AttributeError("getDimensions not available")

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dimensions("dv_test_12345")

        assert result.empty
        mock_logger.error.assert_called()

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dimensions_generic_exception(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of generic exception"""
        mock_api_call.side_effect = Exception("API timeout")

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dimensions("dv_test_12345")

        assert result.empty


class TestParallelAPIFetcherFetchDataviewInfo:
    """Tests for _fetch_dataview_info method"""

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dataview_info_success(self, mock_api_call, mock_cja, mock_logger,
                                          mock_perf_tracker, sample_dataview_info):
        """Test successful dataview info fetching"""
        mock_api_call.return_value = sample_dataview_info

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dataview_info("dv_test_12345")

        assert result == sample_dataview_info
        assert result['name'] == 'Test Data View'

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dataview_info_returns_none(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of None response"""
        mock_api_call.return_value = None

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dataview_info("dv_test_12345")

        assert result['name'] == 'Unknown'
        assert result['id'] == 'dv_test_12345'
        mock_logger.error.assert_called()

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dataview_info_returns_empty_dict(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of empty dict response"""
        mock_api_call.return_value = {}

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dataview_info("dv_test_12345")

        assert result['name'] == 'Unknown'
        assert result['id'] == 'dv_test_12345'

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    def test_fetch_dataview_info_exception(self, mock_api_call, mock_cja, mock_logger, mock_perf_tracker):
        """Test handling of exception"""
        mock_api_call.side_effect = Exception("API error")

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        result = fetcher._fetch_dataview_info("dv_test_12345")

        assert result['name'] == 'Unknown'
        assert result['id'] == 'dv_test_12345'
        assert 'error' in result


class TestParallelAPIFetcherErrorHandling:
    """Tests for error handling scenarios"""

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_partial_failure_continues(self, mock_tqdm, mock_api_call, mock_cja,
                                        mock_logger, mock_perf_tracker,
                                        sample_dimensions_data, sample_dataview_info):
        """Test that partial failures don't stop other fetches"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        call_count = [0]
        def api_side_effect(func, *args, **kwargs):
            call_count[0] += 1
            op_name = kwargs.get('operation_name', '')
            if 'getMetrics' in op_name:
                raise Exception("Metrics fetch failed")
            elif 'getDimensions' in op_name:
                return sample_dimensions_data
            elif 'getDataView' in op_name:
                return sample_dataview_info
            return None

        mock_api_call.side_effect = api_side_effect

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        metrics, dimensions, dataview = fetcher.fetch_all_data("dv_test_12345")

        # Metrics should be empty due to error, but others should succeed
        assert metrics.empty
        assert not dimensions.empty
        assert dataview == sample_dataview_info

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_all_failures_return_empty(self, mock_tqdm, mock_api_call, mock_cja,
                                        mock_logger, mock_perf_tracker):
        """Test that all failures return empty/default values"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        mock_api_call.side_effect = Exception("All calls fail")

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        metrics, dimensions, dataview = fetcher.fetch_all_data("dv_test_12345")

        assert metrics.empty
        assert dimensions.empty
        # On failure, dataview returns fallback dict with error
        assert dataview['name'] == 'Unknown'
        assert 'error' in dataview


class TestParallelAPIFetcherLogging:
    """Tests for logging behavior"""

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_logs_start_message(self, mock_tqdm, mock_api_call, mock_cja,
                                 mock_logger, mock_perf_tracker):
        """Test that starting message is logged"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)
        mock_api_call.return_value = pd.DataFrame()

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        fetcher.fetch_all_data("dv_test_12345")

        # Verify start message logged
        calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("parallel" in c.lower() for c in calls)

    @patch('cja_auto_sdr.generator.make_api_call_with_retry')
    @patch('cja_auto_sdr.generator.tqdm')
    def test_logs_completion_summary(self, mock_tqdm, mock_api_call, mock_cja,
                                      mock_logger, mock_perf_tracker, sample_metrics_data):
        """Test that completion summary is logged"""
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)
        mock_api_call.return_value = sample_metrics_data

        fetcher = ParallelAPIFetcher(mock_cja, mock_logger, mock_perf_tracker)
        fetcher.fetch_all_data("dv_test_12345")

        # Verify completion logged
        calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("complete" in c.lower() for c in calls)
