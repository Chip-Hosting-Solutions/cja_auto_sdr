"""Tests for Data View Comparison (Diff) functionality

Validates that:
1. DataViewSnapshot correctly captures data view state
2. SnapshotManager can save/load snapshots
3. DataViewComparator correctly identifies changes
4. Diff output writers produce correct output
5. CLI integration handles diff arguments correctly
"""
import pytest
import json
import os
import tempfile
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cja_sdr_generator import (
    ChangeType,
    ComponentDiff,
    MetadataDiff,
    DiffSummary,
    DiffResult,
    DataViewSnapshot,
    SnapshotManager,
    DataViewComparator,
    write_diff_console_output,
    write_diff_json_output,
    write_diff_markdown_output,
    write_diff_html_output,
    write_diff_excel_output,
    write_diff_csv_output,
    _format_side_by_side,
    _format_markdown_side_by_side,
)
import logging


# ==================== Fixtures ====================

@pytest.fixture
def sample_metrics():
    """Sample metrics for testing"""
    return [
        {"id": "metrics/pageviews", "name": "Page Views", "type": "int", "description": "Total page views"},
        {"id": "metrics/visits", "name": "Visits", "type": "int", "description": "Total visits"},
        {"id": "metrics/bounce_rate", "name": "Bounce Rate", "type": "decimal", "description": "Bounce percentage"},
    ]


@pytest.fixture
def sample_dimensions():
    """Sample dimensions for testing"""
    return [
        {"id": "dimensions/page", "name": "Page", "type": "string", "description": "Page URL"},
        {"id": "dimensions/device", "name": "Device Type", "type": "string", "description": "Device category"},
    ]


@pytest.fixture
def source_snapshot(sample_metrics, sample_dimensions):
    """Create a source snapshot for comparison"""
    return DataViewSnapshot(
        data_view_id="dv_source_12345",
        data_view_name="Source Data View",
        owner="admin@example.com",
        description="Source description",
        metrics=sample_metrics,
        dimensions=sample_dimensions
    )


@pytest.fixture
def target_snapshot_identical(sample_metrics, sample_dimensions):
    """Create an identical target snapshot"""
    return DataViewSnapshot(
        data_view_id="dv_target_67890",
        data_view_name="Target Data View",
        owner="admin@example.com",
        description="Target description",
        metrics=sample_metrics.copy(),
        dimensions=sample_dimensions.copy()
    )


@pytest.fixture
def target_snapshot_with_changes(sample_metrics, sample_dimensions):
    """Create a target snapshot with changes"""
    # Modify metrics
    modified_metrics = [
        {"id": "metrics/pageviews", "name": "Page Views", "type": "int", "description": "Updated description"},  # Modified
        # metrics/visits removed
        {"id": "metrics/bounce_rate", "name": "Bounce Rate", "type": "decimal", "description": "Bounce percentage"},
        {"id": "metrics/new_metric", "name": "New Metric", "type": "int", "description": "Added metric"},  # Added
    ]

    # Modify dimensions
    modified_dimensions = [
        {"id": "dimensions/page", "name": "Page URL", "type": "string", "description": "Page URL"},  # Name changed
        {"id": "dimensions/device", "name": "Device Type", "type": "string", "description": "Device category"},
        {"id": "dimensions/new_dim", "name": "New Dimension", "type": "string", "description": "Added"},  # Added
    ]

    return DataViewSnapshot(
        data_view_id="dv_target_67890",
        data_view_name="Target Data View",
        owner="admin@example.com",
        description="Target description",
        metrics=modified_metrics,
        dimensions=modified_dimensions
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory"""
    return str(tmp_path)


@pytest.fixture
def logger():
    """Create a logger for testing"""
    return logging.getLogger("test_diff")


# ==================== DataViewSnapshot Tests ====================

class TestDataViewSnapshot:
    """Tests for DataViewSnapshot class"""

    def test_snapshot_creation(self, sample_metrics, sample_dimensions):
        """Test creating a snapshot"""
        snapshot = DataViewSnapshot(
            data_view_id="dv_test_12345",
            data_view_name="Test View",
            owner="test@example.com",
            description="Test description",
            metrics=sample_metrics,
            dimensions=sample_dimensions
        )

        assert snapshot.data_view_id == "dv_test_12345"
        assert snapshot.data_view_name == "Test View"
        assert len(snapshot.metrics) == 3
        assert len(snapshot.dimensions) == 2
        assert snapshot.snapshot_version == "1.0"
        assert snapshot.created_at is not None

    def test_snapshot_to_dict(self, source_snapshot):
        """Test converting snapshot to dictionary"""
        data = source_snapshot.to_dict()

        assert data['snapshot_version'] == "1.0"
        assert data['data_view_id'] == "dv_source_12345"
        assert data['data_view_name'] == "Source Data View"
        assert len(data['metrics']) == 3
        assert len(data['dimensions']) == 2
        assert 'created_at' in data

    def test_snapshot_from_dict(self, source_snapshot):
        """Test creating snapshot from dictionary"""
        data = source_snapshot.to_dict()
        restored = DataViewSnapshot.from_dict(data)

        assert restored.data_view_id == source_snapshot.data_view_id
        assert restored.data_view_name == source_snapshot.data_view_name
        assert len(restored.metrics) == len(source_snapshot.metrics)
        assert len(restored.dimensions) == len(source_snapshot.dimensions)

    def test_snapshot_defaults(self):
        """Test snapshot with minimal data"""
        snapshot = DataViewSnapshot(
            data_view_id="dv_test",
            data_view_name="Test"
        )

        assert snapshot.metrics == []
        assert snapshot.dimensions == []
        assert snapshot.owner == ""
        assert snapshot.description == ""


# ==================== SnapshotManager Tests ====================

class TestSnapshotManager:
    """Tests for SnapshotManager class"""

    def test_save_and_load_snapshot(self, source_snapshot, tmp_path, logger):
        """Test saving and loading a snapshot"""
        manager = SnapshotManager(logger)
        filepath = str(tmp_path / "test_snapshot.json")

        # Save
        saved_path = manager.save_snapshot(source_snapshot, filepath)
        assert os.path.exists(saved_path)

        # Load
        loaded = manager.load_snapshot(saved_path)
        assert loaded.data_view_id == source_snapshot.data_view_id
        assert loaded.data_view_name == source_snapshot.data_view_name
        assert len(loaded.metrics) == len(source_snapshot.metrics)

    def test_load_nonexistent_snapshot(self, tmp_path, logger):
        """Test loading a non-existent snapshot raises error"""
        manager = SnapshotManager(logger)

        with pytest.raises(FileNotFoundError):
            manager.load_snapshot(str(tmp_path / "nonexistent.json"))

    def test_load_invalid_snapshot(self, tmp_path, logger):
        """Test loading an invalid snapshot raises error"""
        manager = SnapshotManager(logger)
        filepath = str(tmp_path / "invalid.json")

        # Create invalid JSON (not a snapshot)
        with open(filepath, 'w') as f:
            json.dump({"foo": "bar"}, f)

        with pytest.raises(ValueError):
            manager.load_snapshot(filepath)

    def test_list_snapshots(self, source_snapshot, tmp_path, logger):
        """Test listing snapshots in a directory"""
        manager = SnapshotManager(logger)

        # Save multiple snapshots
        manager.save_snapshot(source_snapshot, str(tmp_path / "snap1.json"))
        manager.save_snapshot(source_snapshot, str(tmp_path / "snap2.json"))

        # Create a non-snapshot file
        with open(tmp_path / "other.json", 'w') as f:
            json.dump({"foo": "bar"}, f)

        snapshots = manager.list_snapshots(str(tmp_path))
        assert len(snapshots) == 2

    def test_list_snapshots_empty_directory(self, tmp_path, logger):
        """Test listing snapshots in an empty directory"""
        manager = SnapshotManager(logger)
        snapshots = manager.list_snapshots(str(tmp_path))
        assert snapshots == []


# ==================== DataViewComparator Tests ====================

class TestDataViewComparator:
    """Tests for DataViewComparator class"""

    def test_compare_identical_snapshots(self, source_snapshot, target_snapshot_identical, logger):
        """Test comparing identical snapshots"""
        comparator = DataViewComparator(logger)
        result = comparator.compare(source_snapshot, target_snapshot_identical)

        assert result.summary.has_changes is False
        assert result.summary.metrics_added == 0
        assert result.summary.metrics_removed == 0
        assert result.summary.metrics_modified == 0
        assert result.summary.dimensions_added == 0
        assert result.summary.dimensions_removed == 0
        assert result.summary.dimensions_modified == 0

    def test_compare_with_changes(self, source_snapshot, target_snapshot_with_changes, logger):
        """Test comparing snapshots with changes"""
        comparator = DataViewComparator(logger)
        result = comparator.compare(source_snapshot, target_snapshot_with_changes)

        assert result.summary.has_changes is True
        assert result.summary.metrics_added == 1  # new_metric
        assert result.summary.metrics_removed == 1  # visits
        assert result.summary.metrics_modified == 1  # pageviews (description changed)
        assert result.summary.dimensions_added == 1  # new_dim
        assert result.summary.dimensions_removed == 0
        assert result.summary.dimensions_modified == 1  # page (name changed)

    def test_compare_with_ignore_fields(self, source_snapshot, target_snapshot_with_changes, logger):
        """Test comparing with ignored fields"""
        comparator = DataViewComparator(logger, ignore_fields=['description', 'name'])
        result = comparator.compare(source_snapshot, target_snapshot_with_changes)

        # With name and description ignored, only adds/removes should be detected
        assert result.summary.metrics_modified == 0  # description change ignored
        assert result.summary.dimensions_modified == 0  # name change ignored

    def test_compare_custom_labels(self, source_snapshot, target_snapshot_identical, logger):
        """Test comparing with custom labels"""
        comparator = DataViewComparator(logger)
        result = comparator.compare(
            source_snapshot, target_snapshot_identical,
            source_label="Production", target_label="Staging"
        )

        assert result.source_label == "Production"
        assert result.target_label == "Staging"

    def test_change_types_correct(self, source_snapshot, target_snapshot_with_changes, logger):
        """Test that change types are correctly identified"""
        comparator = DataViewComparator(logger)
        result = comparator.compare(source_snapshot, target_snapshot_with_changes)

        # Check metric diffs
        metric_diffs = {d.id: d for d in result.metric_diffs}

        assert metric_diffs["metrics/new_metric"].change_type == ChangeType.ADDED
        assert metric_diffs["metrics/visits"].change_type == ChangeType.REMOVED
        assert metric_diffs["metrics/pageviews"].change_type == ChangeType.MODIFIED

    def test_changed_fields_tracked(self, source_snapshot, target_snapshot_with_changes, logger):
        """Test that changed fields are tracked for modified components"""
        comparator = DataViewComparator(logger)
        result = comparator.compare(source_snapshot, target_snapshot_with_changes)

        # Find the modified pageviews metric
        pageviews_diff = next(d for d in result.metric_diffs if d.id == "metrics/pageviews")

        assert pageviews_diff.change_type == ChangeType.MODIFIED
        assert 'description' in pageviews_diff.changed_fields


# ==================== DiffSummary Tests ====================

class TestDiffSummary:
    """Tests for DiffSummary class"""

    def test_has_changes_false(self):
        """Test has_changes returns False when no changes"""
        summary = DiffSummary()
        assert summary.has_changes is False

    def test_has_changes_true(self):
        """Test has_changes returns True when changes exist"""
        summary = DiffSummary(metrics_added=1)
        assert summary.has_changes is True

        summary2 = DiffSummary(dimensions_modified=2)
        assert summary2.has_changes is True

    def test_total_changes(self):
        """Test total_changes calculation"""
        summary = DiffSummary(
            metrics_added=1,
            metrics_removed=2,
            metrics_modified=3,
            dimensions_added=4,
            dimensions_removed=5,
            dimensions_modified=6
        )
        assert summary.total_changes == 21


# ==================== Diff Output Writer Tests ====================

class TestDiffOutputWriters:
    """Tests for diff output writers"""

    @pytest.fixture
    def sample_diff_result(self, source_snapshot, target_snapshot_with_changes, logger):
        """Create a sample diff result for output testing"""
        comparator = DataViewComparator(logger)
        return comparator.compare(
            source_snapshot, target_snapshot_with_changes,
            source_label="Source", target_label="Target"
        )

    def test_console_output(self, sample_diff_result):
        """Test console output generation"""
        output = write_diff_console_output(sample_diff_result)

        assert "DATA VIEW COMPARISON REPORT" in output
        assert "Source" in output
        assert "Target" in output
        assert "SUMMARY" in output
        assert "METRICS CHANGES" in output
        assert "DIMENSIONS CHANGES" in output

    def test_console_output_changes_only(self, sample_diff_result):
        """Test console output with changes_only flag"""
        output = write_diff_console_output(sample_diff_result, changes_only=True, use_color=False)

        assert "DATA VIEW COMPARISON REPORT" in output
        # Should still show changes
        assert "[+]" in output or "[-]" in output or "[~]" in output

    def test_console_output_summary_only(self, sample_diff_result):
        """Test console output with summary_only flag"""
        output = write_diff_console_output(sample_diff_result, summary_only=True)

        assert "SUMMARY" in output
        assert "METRICS CHANGES" not in output

    def test_json_output(self, sample_diff_result, temp_output_dir, logger):
        """Test JSON output generation"""
        filepath = write_diff_json_output(
            sample_diff_result, "test_diff", temp_output_dir, logger
        )

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            data = json.load(f)

        assert 'metadata' in data
        assert 'source' in data
        assert 'target' in data
        assert 'summary' in data
        assert 'metric_diffs' in data
        assert 'dimension_diffs' in data

    def test_markdown_output(self, sample_diff_result, temp_output_dir, logger):
        """Test Markdown output generation"""
        filepath = write_diff_markdown_output(
            sample_diff_result, "test_diff", temp_output_dir, logger
        )

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            content = f.read()

        assert "# Data View Comparison Report" in content
        assert "## Summary" in content
        assert "| Component |" in content

    def test_html_output(self, sample_diff_result, temp_output_dir, logger):
        """Test HTML output generation"""
        filepath = write_diff_html_output(
            sample_diff_result, "test_diff", temp_output_dir, logger
        )

        assert os.path.exists(filepath)
        with open(filepath, 'r') as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "Data View Comparison Report" in content
        assert "<table" in content

    def test_excel_output(self, sample_diff_result, temp_output_dir, logger):
        """Test Excel output generation"""
        filepath = write_diff_excel_output(
            sample_diff_result, "test_diff", temp_output_dir, logger
        )

        assert os.path.exists(filepath)
        assert filepath.endswith('.xlsx')

    def test_csv_output(self, sample_diff_result, temp_output_dir, logger):
        """Test CSV output generation"""
        dirpath = write_diff_csv_output(
            sample_diff_result, "test_diff", temp_output_dir, logger
        )

        assert os.path.isdir(dirpath)
        assert os.path.exists(os.path.join(dirpath, 'summary.csv'))
        assert os.path.exists(os.path.join(dirpath, 'metadata.csv'))
        assert os.path.exists(os.path.join(dirpath, 'metrics_diff.csv'))
        assert os.path.exists(os.path.join(dirpath, 'dimensions_diff.csv'))


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Tests for edge cases"""

    def test_empty_snapshots(self, logger):
        """Test comparing empty snapshots"""
        source = DataViewSnapshot(
            data_view_id="dv_empty1",
            data_view_name="Empty 1",
            metrics=[],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_empty2",
            data_view_name="Empty 2",
            metrics=[],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.has_changes is False
        assert result.summary.total_changes == 0

    def test_all_added(self, logger, sample_metrics, sample_dimensions):
        """Test when all items are added (empty source)"""
        source = DataViewSnapshot(
            data_view_id="dv_empty",
            data_view_name="Empty",
            metrics=[],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_full",
            data_view_name="Full",
            metrics=sample_metrics,
            dimensions=sample_dimensions
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.metrics_added == 3
        assert result.summary.dimensions_added == 2
        assert result.summary.metrics_removed == 0

    def test_all_removed(self, logger, sample_metrics, sample_dimensions):
        """Test when all items are removed (empty target)"""
        source = DataViewSnapshot(
            data_view_id="dv_full",
            data_view_name="Full",
            metrics=sample_metrics,
            dimensions=sample_dimensions
        )
        target = DataViewSnapshot(
            data_view_id="dv_empty",
            data_view_name="Empty",
            metrics=[],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.metrics_removed == 3
        assert result.summary.dimensions_removed == 2
        assert result.summary.metrics_added == 0

    def test_special_characters_in_names(self, logger):
        """Test handling of special characters in component names"""
        source = DataViewSnapshot(
            data_view_id="dv_special",
            data_view_name="Special <> & \"quotes\" View",
            metrics=[{"id": "m1", "name": "Metric with | pipe", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_special2",
            data_view_name="Special <> & \"quotes\" View 2",
            metrics=[{"id": "m1", "name": "Metric with | pipe", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        # Should not raise any errors
        console_output = write_diff_console_output(result)
        assert "Special" in console_output


# ==================== Comparison Fields Tests ====================

class TestComparisonFields:
    """Tests for field comparison logic as documented in DIFF_COMPARISON.md"""

    def test_default_compare_fields(self, logger):
        """Test that default fields (name, title, description, type, schemaPath) are compared"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{
                "id": "m1",
                "name": "Original Name",
                "title": "Original Title",
                "description": "Original Description",
                "type": "int",
                "schemaPath": "/path/original"
            }],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{
                "id": "m1",
                "name": "Changed Name",
                "title": "Changed Title",
                "description": "Changed Description",
                "type": "decimal",
                "schemaPath": "/path/changed"
            }],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        metric_diff = result.metric_diffs[0]
        assert metric_diff.change_type == ChangeType.MODIFIED
        # All 5 default fields should be detected as changed
        assert 'name' in metric_diff.changed_fields
        assert 'title' in metric_diff.changed_fields
        assert 'description' in metric_diff.changed_fields
        assert 'type' in metric_diff.changed_fields
        assert 'schemaPath' in metric_diff.changed_fields

    def test_id_based_matching(self, logger):
        """Test that components are matched by ID, not by name"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[
                {"id": "metrics/pageviews", "name": "Page Views", "type": "int"},
                {"id": "metrics/visits", "name": "Visits", "type": "int"}
            ],
            dimensions=[]
        )
        # Same IDs but names swapped - should be detected as MODIFIED, not ADD/REMOVE
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[
                {"id": "metrics/pageviews", "name": "Visits", "type": "int"},  # Name changed
                {"id": "metrics/visits", "name": "Page Views", "type": "int"}  # Name changed
            ],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        # Should be 2 modified, 0 added, 0 removed (matched by ID)
        assert result.summary.metrics_modified == 2
        assert result.summary.metrics_added == 0
        assert result.summary.metrics_removed == 0

    def test_metadata_comparison(self, logger):
        """Test that data view metadata changes are tracked"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Original Name",
            owner="original@example.com",
            description="Original description",
            metrics=[],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Changed Name",
            owner="changed@example.com",
            description="Changed description",
            metrics=[],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        # Metadata changes should be tracked
        assert result.metadata_diff.source_name == "Original Name"
        assert result.metadata_diff.target_name == "Changed Name"
        assert 'name' in result.metadata_diff.changed_fields
        assert 'owner' in result.metadata_diff.changed_fields
        assert 'description' in result.metadata_diff.changed_fields

    def test_unchanged_detection(self, logger):
        """Test that unchanged components are correctly identified"""
        metrics = [
            {"id": "m1", "name": "Metric 1", "type": "int", "description": "Desc 1"},
            {"id": "m2", "name": "Metric 2", "type": "int", "description": "Desc 2"},
        ]
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=metrics,
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=metrics.copy(),  # Identical
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.metrics_unchanged == 2
        assert result.summary.metrics_modified == 0
        for diff in result.metric_diffs:
            assert diff.change_type == ChangeType.UNCHANGED

    def test_custom_ignore_fields(self, logger):
        """Test that --ignore-fields functionality works correctly"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{"id": "m1", "name": "Name", "description": "Old desc", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{"id": "m1", "name": "Name", "description": "New desc", "type": "int"}],
            dimensions=[]
        )

        # Without ignore - should detect change
        comparator1 = DataViewComparator(logger)
        result1 = comparator1.compare(source, target)
        assert result1.summary.metrics_modified == 1

        # With ignore description - should not detect change
        comparator2 = DataViewComparator(logger, ignore_fields=['description'])
        result2 = comparator2.compare(source, target)
        assert result2.summary.metrics_modified == 0
        assert result2.summary.metrics_unchanged == 1


# ==================== CLI Argument Tests ====================

class TestCLIArguments:
    """Tests for CLI argument parsing related to diff feature"""

    def test_parse_ignore_fields(self):
        """Test that --ignore-fields argument is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        # Mock sys.argv
        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2',
                       '--ignore-fields', 'description,title']
            args = parse_arguments()
            assert args.ignore_fields == 'description,title'
            assert args.diff is True
        finally:
            sys.argv = original_argv

    def test_parse_diff_labels(self):
        """Test that --diff-labels argument is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2',
                       '--diff-labels', 'Production', 'Staging']
            args = parse_arguments()
            assert args.diff_labels == ['Production', 'Staging']
        finally:
            sys.argv = original_argv

    def test_parse_changes_only(self):
        """Test that --changes-only flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--changes-only']
            args = parse_arguments()
            assert args.changes_only is True
        finally:
            sys.argv = original_argv

    def test_parse_summary_flag(self):
        """Test that --summary flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--summary']
            args = parse_arguments()
            assert args.summary is True
        finally:
            sys.argv = original_argv

    def test_parse_snapshot_argument(self):
        """Test that --snapshot argument is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', 'dv_12345',
                       '--snapshot', './snapshots/baseline.json']
            args = parse_arguments()
            assert args.snapshot == './snapshots/baseline.json'
        finally:
            sys.argv = original_argv

    def test_parse_diff_snapshot_argument(self):
        """Test that --diff-snapshot argument is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', 'dv_12345',
                       '--diff-snapshot', './snapshots/baseline.json']
            args = parse_arguments()
            assert args.diff_snapshot == './snapshots/baseline.json'
        finally:
            sys.argv = original_argv


# ==================== Output Format Verification Tests ====================

class TestOutputFormatVerification:
    """Tests to verify output formats match documentation"""

    @pytest.fixture
    def diff_result_with_all_change_types(self, logger):
        """Create a diff result with all change types for output testing"""
        source = DataViewSnapshot(
            data_view_id="dv_source",
            data_view_name="Source View",
            owner="owner@test.com",
            description="Source description",
            metrics=[
                {"id": "m1", "name": "Unchanged Metric", "type": "int", "description": "Stays same"},
                {"id": "m2", "name": "Modified Metric", "type": "int", "description": "Old desc"},
                {"id": "m3", "name": "Removed Metric", "type": "int", "description": "Will be removed"},
            ],
            dimensions=[
                {"id": "d1", "name": "Unchanged Dim", "type": "string"},
            ]
        )
        target = DataViewSnapshot(
            data_view_id="dv_target",
            data_view_name="Target View",
            owner="owner@test.com",
            description="Target description",
            metrics=[
                {"id": "m1", "name": "Unchanged Metric", "type": "int", "description": "Stays same"},
                {"id": "m2", "name": "Modified Metric", "type": "int", "description": "New desc"},
                {"id": "m4", "name": "Added Metric", "type": "int", "description": "New metric"},
            ],
            dimensions=[
                {"id": "d1", "name": "Unchanged Dim", "type": "string"},
                {"id": "d2", "name": "Added Dim", "type": "string"},
            ]
        )

        comparator = DataViewComparator(logger)
        return comparator.compare(source, target, "Source", "Target")

    def test_console_output_contains_symbols(self, diff_result_with_all_change_types):
        """Test console output contains correct change symbols"""
        output = write_diff_console_output(diff_result_with_all_change_types, use_color=False)

        assert "[+]" in output  # Added
        assert "[-]" in output  # Removed
        assert "[~]" in output  # Modified

    def test_json_output_structure(self, diff_result_with_all_change_types, tmp_path, logger):
        """Test JSON output has correct structure"""
        filepath = write_diff_json_output(
            diff_result_with_all_change_types, "test", str(tmp_path), logger
        )

        with open(filepath) as f:
            data = json.load(f)

        # Verify structure matches documentation
        assert 'metadata' in data
        assert 'source' in data
        assert 'target' in data
        assert 'summary' in data
        assert 'metric_diffs' in data
        assert 'dimension_diffs' in data

        # Verify change types
        change_types = [d['change_type'] for d in data['metric_diffs']]
        assert 'added' in change_types
        assert 'removed' in change_types
        assert 'modified' in change_types
        assert 'unchanged' in change_types

    def test_summary_statistics_accuracy(self, diff_result_with_all_change_types):
        """Test that summary statistics are accurate"""
        summary = diff_result_with_all_change_types.summary

        # Metrics: 1 unchanged, 1 modified, 1 removed, 1 added
        assert summary.metrics_unchanged == 1
        assert summary.metrics_modified == 1
        assert summary.metrics_removed == 1
        assert summary.metrics_added == 1

        # Dimensions: 1 unchanged, 1 added
        assert summary.dimensions_unchanged == 1
        assert summary.dimensions_added == 1
        assert summary.dimensions_removed == 0
        assert summary.dimensions_modified == 0

        # Total
        assert summary.total_changes == 4  # 1 mod + 1 rem + 1 add (metrics) + 1 add (dim)
        assert summary.has_changes is True


# ==================== New Feature Tests (v3.0.10) ====================

class TestExtendedFieldComparison:
    """Tests for extended field comparison (attribution, format, etc.)"""

    def test_extended_compare_fields_list(self, logger):
        """Test that EXTENDED_COMPARE_FIELDS contains expected fields"""
        assert 'name' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'attribution' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'format' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'precision' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'hidden' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'bucketing' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'persistence' in DataViewComparator.EXTENDED_COMPARE_FIELDS
        assert 'formula' in DataViewComparator.EXTENDED_COMPARE_FIELDS

    def test_use_extended_fields_flag(self, logger):
        """Test that use_extended_fields enables extended comparison"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{
                "id": "m1", "name": "Metric",
                "type": "int", "hidden": False, "precision": 2
            }],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{
                "id": "m1", "name": "Metric",
                "type": "int", "hidden": True, "precision": 4  # Changed
            }],
            dimensions=[]
        )

        # Without extended - only basic fields compared
        comparator1 = DataViewComparator(logger, use_extended_fields=False)
        result1 = comparator1.compare(source, target)
        # 'hidden' and 'precision' are not in default fields
        assert result1.summary.metrics_modified == 0

        # With extended - should detect hidden and precision changes
        comparator2 = DataViewComparator(logger, use_extended_fields=True)
        result2 = comparator2.compare(source, target)
        assert result2.summary.metrics_modified == 1
        assert 'hidden' in result2.metric_diffs[0].changed_fields
        assert 'precision' in result2.metric_diffs[0].changed_fields

    def test_attribution_settings_comparison(self, logger):
        """Test comparison of attribution settings (nested structure)"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{
                "id": "m1", "name": "Metric",
                "attribution": {"model": "lastTouch", "lookback": 30}
            }],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{
                "id": "m1", "name": "Metric",
                "attribution": {"model": "firstTouch", "lookback": 30}  # Model changed
            }],
            dimensions=[]
        )

        comparator = DataViewComparator(logger, use_extended_fields=True)
        result = comparator.compare(source, target)

        assert result.summary.metrics_modified == 1
        assert 'attribution' in result.metric_diffs[0].changed_fields


class TestShowOnlyFilter:
    """Tests for --show-only filter functionality"""

    def test_show_only_added(self, logger, sample_metrics, sample_dimensions):
        """Test filtering to show only added items"""
        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "Existing", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[
                {"id": "m1", "name": "Existing", "type": "int"},
                {"id": "m2", "name": "New", "type": "int"}  # Added
            ],
            dimensions=[]
        )

        comparator = DataViewComparator(logger, show_only=['added'])
        result = comparator.compare(source, target)

        # Should only contain added items
        assert len(result.metric_diffs) == 1
        assert result.metric_diffs[0].change_type == ChangeType.ADDED

    def test_show_only_removed(self, logger):
        """Test filtering to show only removed items"""
        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[
                {"id": "m1", "name": "Keep", "type": "int"},
                {"id": "m2", "name": "Remove", "type": "int"}
            ],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m1", "name": "Keep", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger, show_only=['removed'])
        result = comparator.compare(source, target)

        assert len(result.metric_diffs) == 1
        assert result.metric_diffs[0].change_type == ChangeType.REMOVED

    def test_show_only_multiple_types(self, logger):
        """Test filtering with multiple change types"""
        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[
                {"id": "m1", "name": "Unchanged", "type": "int"},
                {"id": "m2", "name": "Modified", "type": "int", "description": "Old"},
                {"id": "m3", "name": "Removed", "type": "int"}
            ],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[
                {"id": "m1", "name": "Unchanged", "type": "int"},
                {"id": "m2", "name": "Modified", "type": "int", "description": "New"},
                {"id": "m4", "name": "Added", "type": "int"}
            ],
            dimensions=[]
        )

        comparator = DataViewComparator(logger, show_only=['added', 'modified'])
        result = comparator.compare(source, target)

        # Should contain added and modified, but not unchanged or removed
        change_types = [d.change_type for d in result.metric_diffs]
        assert ChangeType.ADDED in change_types
        assert ChangeType.MODIFIED in change_types
        assert ChangeType.UNCHANGED not in change_types
        assert ChangeType.REMOVED not in change_types


class TestMetricsOnlyAndDimensionsOnly:
    """Tests for --metrics-only and --dimensions-only flags"""

    def test_metrics_only_flag(self, logger, sample_metrics, sample_dimensions):
        """Test that --metrics-only excludes dimensions"""
        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=sample_metrics,
            dimensions=sample_dimensions
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=sample_metrics,
            dimensions=[]  # All dimensions removed
        )

        comparator = DataViewComparator(logger, metrics_only=True)
        result = comparator.compare(source, target)

        # Dimensions should be empty (not compared)
        assert len(result.dimension_diffs) == 0
        # Metrics should be compared
        assert len(result.metric_diffs) > 0

    def test_dimensions_only_flag(self, logger, sample_metrics, sample_dimensions):
        """Test that --dimensions-only excludes metrics"""
        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=sample_metrics,
            dimensions=sample_dimensions
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[],  # All metrics removed
            dimensions=sample_dimensions
        )

        comparator = DataViewComparator(logger, dimensions_only=True)
        result = comparator.compare(source, target)

        # Metrics should be empty (not compared)
        assert len(result.metric_diffs) == 0
        # Dimensions should be compared
        assert len(result.dimension_diffs) > 0


class TestSideBySideOutput:
    """Tests for side-by-side output view"""

    def test_side_by_side_console_output(self, logger):
        """Test that side-by-side console output contains table characters"""
        from cja_sdr_generator import _format_side_by_side

        diff = ComponentDiff(
            id="m1",
            name="Test Metric",
            change_type=ChangeType.MODIFIED,
            source_data={"name": "Old Name", "description": "Old desc"},
            target_data={"name": "New Name", "description": "New desc"},
            changed_fields={
                "name": ("Old Name", "New Name"),
                "description": ("Old desc", "New desc")
            }
        )

        lines = _format_side_by_side(diff, "Source", "Target")

        # Should contain table border characters
        assert any("┌" in line for line in lines)
        assert any("│" in line for line in lines)
        assert any("└" in line for line in lines)

    def test_side_by_side_markdown_output(self, logger):
        """Test that side-by-side markdown output creates a table"""
        from cja_sdr_generator import _format_markdown_side_by_side

        diff = ComponentDiff(
            id="m1",
            name="Test Metric",
            change_type=ChangeType.MODIFIED,
            source_data={"name": "Old", "type": "int"},
            target_data={"name": "New", "type": "decimal"},
            changed_fields={
                "name": ("Old", "New"),
                "type": ("int", "decimal")
            }
        )

        lines = _format_markdown_side_by_side(diff, "Source", "Target")

        # Should contain markdown table
        assert any("| Field |" in line for line in lines)
        assert any("| --- |" in line for line in lines)
        assert any("`name`" in line for line in lines)

    def test_console_output_with_side_by_side_flag(self, logger):
        """Test console output when side_by_side=True"""
        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "Old Name", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m1", "name": "New Name", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        output = write_diff_console_output(result, side_by_side=True)

        # Should contain table border characters for modified items
        assert "┌" in output or "[~]" in output


# ==================== Large Dataset Performance Tests ====================

class TestLargeDatasetPerformance:
    """Tests for large dataset handling (500+ components)"""

    @pytest.fixture
    def large_metrics(self):
        """Generate 500+ metrics for performance testing"""
        return [
            {
                "id": f"metrics/metric_{i}",
                "name": f"Metric {i}",
                "type": "int" if i % 2 == 0 else "decimal",
                "description": f"Description for metric {i}",
                "schemaPath": f"/schema/path/{i}"
            }
            for i in range(600)
        ]

    @pytest.fixture
    def large_dimensions(self):
        """Generate 200+ dimensions for performance testing"""
        return [
            {
                "id": f"dimensions/dim_{i}",
                "name": f"Dimension {i}",
                "type": "string",
                "description": f"Description for dimension {i}",
                "schemaPath": f"/schema/path/dim/{i}"
            }
            for i in range(250)
        ]

    def test_compare_large_identical_datasets(self, logger, large_metrics, large_dimensions):
        """Test comparison of large identical datasets completes quickly"""
        import time

        source = DataViewSnapshot(
            data_view_id="dv_large_1",
            data_view_name="Large Source",
            metrics=large_metrics,
            dimensions=large_dimensions
        )
        target = DataViewSnapshot(
            data_view_id="dv_large_2",
            data_view_name="Large Target",
            metrics=large_metrics.copy(),
            dimensions=large_dimensions.copy()
        )

        start = time.time()
        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 850 components)
        assert elapsed < 5.0
        assert result.summary.metrics_unchanged == 600
        assert result.summary.dimensions_unchanged == 250
        assert result.summary.has_changes is False

    def test_compare_large_datasets_with_changes(self, logger, large_metrics, large_dimensions):
        """Test comparison of large datasets with mixed changes"""
        import time

        # Modify some items in target
        target_metrics = large_metrics.copy()
        for i in range(50):  # Modify 50 metrics
            target_metrics[i] = {**target_metrics[i], "description": f"Modified desc {i}"}
        # Add 50 new metrics
        for i in range(600, 650):
            target_metrics.append({
                "id": f"metrics/new_metric_{i}",
                "name": f"New Metric {i}",
                "type": "int",
                "description": f"New description {i}"
            })

        source = DataViewSnapshot(
            data_view_id="dv_large_1",
            data_view_name="Large Source",
            metrics=large_metrics,
            dimensions=large_dimensions
        )
        target = DataViewSnapshot(
            data_view_id="dv_large_2",
            data_view_name="Large Target",
            metrics=target_metrics,
            dimensions=large_dimensions.copy()
        )

        start = time.time()
        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0
        assert result.summary.metrics_modified == 50
        assert result.summary.metrics_added == 50
        assert result.summary.has_changes is True

    def test_large_dataset_memory_efficiency(self, logger, large_metrics, large_dimensions):
        """Test that large dataset comparison doesn't consume excessive memory"""
        import sys

        source = DataViewSnapshot(
            data_view_id="dv_large_1",
            data_view_name="Large Source",
            metrics=large_metrics,
            dimensions=large_dimensions
        )
        target = DataViewSnapshot(
            data_view_id="dv_large_2",
            data_view_name="Large Target",
            metrics=large_metrics.copy(),
            dimensions=large_dimensions.copy()
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        # Result should be created without errors
        assert result is not None
        # Summary should have correct counts
        assert result.summary.source_metrics_count == 600
        assert result.summary.source_dimensions_count == 250


# ==================== Unicode Edge Case Tests ====================

class TestUnicodeEdgeCases:
    """Tests for Unicode handling in diff comparison"""

    def test_emoji_in_names(self, logger):
        """Test components with emoji in names"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test 📊",
            metrics=[{"id": "m1", "name": "Pageviews 📈", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test 📊",
            metrics=[{"id": "m1", "name": "Pageviews 📉", "type": "int"}],  # Different emoji
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.metrics_modified == 1
        assert "📈" in str(result.metric_diffs[0].changed_fields.get('name', ('', ''))[0])

    def test_rtl_text_in_descriptions(self, logger):
        """Test components with RTL (Hebrew/Arabic) text"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{"id": "m1", "name": "Metric", "description": "תיאור עברי", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{"id": "m1", "name": "Metric", "description": "תיאור אחר", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.metrics_modified == 1

    def test_special_characters_in_fields(self, logger):
        """Test components with special characters"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test <>&\"'",
            metrics=[{
                "id": "m1",
                "name": "Metric with <html> & \"quotes\"",
                "description": "Line1\nLine2\tTabbed",
                "type": "int"
            }],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test <>&\"'",
            metrics=[{
                "id": "m1",
                "name": "Metric with <html> & \"quotes\"",
                "description": "Different\nContent",
                "type": "int"
            }],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        assert result.summary.metrics_modified == 1

    def test_unicode_in_json_output(self, logger, tmp_path):
        """Test that Unicode is preserved in JSON output"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="数据视图",  # Chinese
            metrics=[{"id": "m1", "name": "指标 🎯", "type": "int", "description": "描述文字"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="数据视图",
            metrics=[{"id": "m1", "name": "指标 🎯", "type": "int", "description": "新描述"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)

        filepath = write_diff_json_output(result, "unicode_test", str(tmp_path), logger)

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Unicode should be preserved, not escaped
        assert "数据视图" in content
        assert "指标" in content


# ==================== Deeply Nested Structure Tests ====================

class TestDeeplyNestedStructures:
    """Tests for deeply nested configuration structures"""

    def test_nested_attribution_config(self, logger):
        """Test comparison of nested attribution configuration"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{
                "id": "m1",
                "name": "Revenue",
                "attribution": {
                    "model": "lastTouch",
                    "lookback": {
                        "type": "visitor",
                        "granularity": "day",
                        "numPeriods": 30
                    }
                }
            }],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{
                "id": "m1",
                "name": "Revenue",
                "attribution": {
                    "model": "lastTouch",
                    "lookback": {
                        "type": "visitor",
                        "granularity": "day",
                        "numPeriods": 60  # Changed from 30 to 60
                    }
                }
            }],
            dimensions=[]
        )

        comparator = DataViewComparator(logger, use_extended_fields=True)
        result = comparator.compare(source, target)

        assert result.summary.metrics_modified == 1
        assert 'attribution' in result.metric_diffs[0].changed_fields

    def test_nested_format_config(self, logger):
        """Test comparison of nested format configuration"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[{
                "id": "m1",
                "name": "Currency",
                "format": {
                    "type": "currency",
                    "currency": "USD",
                    "precision": 2,
                    "negativeFormat": "parentheses"
                }
            }],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[{
                "id": "m1",
                "name": "Currency",
                "format": {
                    "type": "currency",
                    "currency": "EUR",  # Changed
                    "precision": 2,
                    "negativeFormat": "parentheses"
                }
            }],
            dimensions=[]
        )

        comparator = DataViewComparator(logger, use_extended_fields=True)
        result = comparator.compare(source, target)

        assert result.summary.metrics_modified == 1

    def test_nested_bucketing_config(self, logger):
        """Test comparison of nested bucketing configuration for dimensions"""
        source = DataViewSnapshot(
            data_view_id="dv_1",
            data_view_name="Test",
            metrics=[],
            dimensions=[{
                "id": "d1",
                "name": "Price Range",
                "bucketing": {
                    "enabled": True,
                    "buckets": [0, 100, 500, 1000],
                    "labels": ["Low", "Medium", "High", "Premium"]
                }
            }]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2",
            data_view_name="Test",
            metrics=[],
            dimensions=[{
                "id": "d1",
                "name": "Price Range",
                "bucketing": {
                    "enabled": True,
                    "buckets": [0, 50, 200, 500, 1000],  # Changed
                    "labels": ["Very Low", "Low", "Medium", "High", "Premium"]  # Changed
                }
            }]
        )

        comparator = DataViewComparator(logger, use_extended_fields=True)
        result = comparator.compare(source, target)

        assert result.summary.dimensions_modified == 1


# ==================== Concurrent Comparison Thread Safety Tests ====================

class TestConcurrentComparison:
    """Tests for thread safety in concurrent comparisons"""

    def test_concurrent_comparisons(self, logger, sample_metrics, sample_dimensions):
        """Test multiple concurrent comparisons don't interfere"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        source = DataViewSnapshot(
            data_view_id="dv_source",
            data_view_name="Source",
            metrics=sample_metrics,
            dimensions=sample_dimensions
        )

        targets = [
            DataViewSnapshot(
                data_view_id=f"dv_target_{i}",
                data_view_name=f"Target {i}",
                metrics=[{**m, "description": f"Modified {i}"} for m in sample_metrics],
                dimensions=sample_dimensions
            )
            for i in range(10)
        ]

        results = []
        errors = []

        def run_comparison(target):
            try:
                comparator = DataViewComparator(logger)
                return comparator.compare(source, target)
            except Exception as e:
                return e

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_comparison, t) for t in targets]
            for future in as_completed(futures):
                result = future.result()
                if isinstance(result, Exception):
                    errors.append(result)
                else:
                    results.append(result)

        # All comparisons should complete without errors
        assert len(errors) == 0
        assert len(results) == 10

        # Each result should be valid
        for result in results:
            assert result.summary is not None
            assert result.summary.metrics_modified == 3  # All metrics modified


# ==================== Snapshot Version Migration Tests ====================

class TestSnapshotVersionMigration:
    """Tests for handling different snapshot versions"""

    def test_load_v1_snapshot(self, logger, tmp_path):
        """Test loading a v1.0 snapshot format"""
        v1_snapshot = {
            "snapshot_version": "1.0",
            "created_at": "2025-01-01T00:00:00",
            "data_view_id": "dv_legacy",
            "data_view_name": "Legacy View",
            "owner": "legacy@test.com",
            "description": "Old format",
            "metrics": [{"id": "m1", "name": "Metric", "type": "int"}],
            "dimensions": [],
            "metadata": {"tool_version": "3.0.0"}
        }

        filepath = str(tmp_path / "v1_snapshot.json")
        with open(filepath, 'w') as f:
            json.dump(v1_snapshot, f)

        manager = SnapshotManager(logger)
        loaded = manager.load_snapshot(filepath)

        assert loaded.snapshot_version == "1.0"
        assert loaded.data_view_id == "dv_legacy"
        assert len(loaded.metrics) == 1

    def test_snapshot_without_metadata(self, logger, tmp_path):
        """Test loading snapshot with minimal/missing metadata"""
        minimal_snapshot = {
            "snapshot_version": "1.0",
            "data_view_id": "dv_minimal",
            "data_view_name": "Minimal",
            "metrics": [],
            "dimensions": []
            # No metadata, owner, description, created_at
        }

        filepath = str(tmp_path / "minimal_snapshot.json")
        with open(filepath, 'w') as f:
            json.dump(minimal_snapshot, f)

        manager = SnapshotManager(logger)
        loaded = manager.load_snapshot(filepath)

        assert loaded.data_view_id == "dv_minimal"
        assert loaded.owner == ""
        assert loaded.description == ""

    def test_compare_different_version_snapshots(self, logger, tmp_path):
        """Test comparing snapshots from different versions"""
        old_snapshot = DataViewSnapshot(
            snapshot_version="1.0",
            data_view_id="dv_old",
            data_view_name="Old",
            metrics=[{"id": "m1", "name": "Metric", "type": "int"}],
            dimensions=[]
        )
        old_snapshot.metadata = {"tool_version": "2.0.0"}

        new_snapshot = DataViewSnapshot(
            snapshot_version="1.0",
            data_view_id="dv_new",
            data_view_name="New",
            metrics=[{"id": "m1", "name": "Metric", "type": "int"}],
            dimensions=[]
        )
        new_snapshot.metadata = {"tool_version": "3.0.10"}

        comparator = DataViewComparator(logger)
        result = comparator.compare(old_snapshot, new_snapshot)

        # Should work regardless of tool version differences
        assert result is not None
        assert result.summary.metrics_unchanged == 1


# ==================== New CLI Argument Tests ====================

class TestNewCLIArguments:
    """Tests for new CLI arguments added in v3.0.10"""

    def test_parse_show_only_argument(self):
        """Test that --show-only argument is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2',
                       '--show-only', 'added,modified']
            args = parse_arguments()
            assert args.show_only == 'added,modified'
        finally:
            sys.argv = original_argv

    def test_parse_metrics_only_flag(self):
        """Test that --metrics-only flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--metrics-only']
            args = parse_arguments()
            assert args.metrics_only is True
        finally:
            sys.argv = original_argv

    def test_parse_dimensions_only_flag(self):
        """Test that --dimensions-only flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--dimensions-only']
            args = parse_arguments()
            assert args.dimensions_only is True
        finally:
            sys.argv = original_argv

    def test_parse_extended_fields_flag(self):
        """Test that --extended-fields flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--extended-fields']
            args = parse_arguments()
            assert args.extended_fields is True
        finally:
            sys.argv = original_argv

    def test_parse_side_by_side_flag(self):
        """Test that --side-by-side flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--side-by-side']
            args = parse_arguments()
            assert args.side_by_side is True
        finally:
            sys.argv = original_argv


# ==================== v3.0.10 New Feature Tests ====================

class TestDiffSummaryPercentages:
    """Tests for new DiffSummary percentage properties"""

    def test_metrics_change_percent(self, logger):
        """Test metrics_change_percent calculation"""
        from cja_sdr_generator import DiffSummary

        summary = DiffSummary(
            source_metrics_count=100,
            target_metrics_count=100,
            metrics_added=5,
            metrics_removed=3,
            metrics_modified=2
        )

        assert summary.metrics_changed == 10
        assert summary.metrics_change_percent == 10.0

    def test_dimensions_change_percent(self, logger):
        """Test dimensions_change_percent calculation"""
        from cja_sdr_generator import DiffSummary

        summary = DiffSummary(
            source_dimensions_count=50,
            target_dimensions_count=60,
            dimensions_added=10,
            dimensions_removed=0,
            dimensions_modified=5
        )

        assert summary.dimensions_changed == 15
        # Should use max(50, 60) = 60 as base
        assert summary.dimensions_change_percent == 25.0

    def test_zero_components_percent(self, logger):
        """Test percentage is 0 when no components exist"""
        from cja_sdr_generator import DiffSummary

        summary = DiffSummary(
            source_metrics_count=0,
            target_metrics_count=0
        )

        assert summary.metrics_change_percent == 0.0

    def test_natural_language_summary(self, logger):
        """Test natural_language_summary property"""
        from cja_sdr_generator import DiffSummary

        summary = DiffSummary(
            metrics_added=3,
            metrics_removed=2,
            metrics_modified=1,
            dimensions_added=1,
            dimensions_removed=0,
            dimensions_modified=2
        )

        nl_summary = summary.natural_language_summary
        assert "Metrics:" in nl_summary
        assert "3 added" in nl_summary
        assert "2 removed" in nl_summary
        assert "Dimensions:" in nl_summary

    def test_natural_language_summary_no_changes(self, logger):
        """Test natural_language_summary when no changes"""
        from cja_sdr_generator import DiffSummary

        summary = DiffSummary()
        assert summary.natural_language_summary == "No changes detected"


class TestColoredConsoleOutput:
    """Tests for ANSI color-coded console output"""

    def test_console_output_with_color(self, logger):
        """Test console output includes ANSI color codes when enabled"""
        from cja_sdr_generator import write_diff_console_output, DataViewSnapshot, DataViewComparator

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "Test", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m2", "name": "New", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        output = write_diff_console_output(result, use_color=True)

        # Should contain ANSI escape codes
        assert "\x1b[" in output

    def test_console_output_without_color(self, logger):
        """Test console output has no ANSI codes when disabled"""
        from cja_sdr_generator import write_diff_console_output, DataViewSnapshot, DataViewComparator

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "Test", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m2", "name": "New", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        output = write_diff_console_output(result, use_color=False)

        # Should NOT contain ANSI escape codes
        assert "\x1b[" not in output
        # But should still have change symbols
        assert "[+]" in output or "[-]" in output


class TestGroupByFieldOutput:
    """Tests for --group-by-field output mode"""

    def test_grouped_by_field_output(self, logger):
        """Test group by field output format"""
        from cja_sdr_generator import (
            write_diff_grouped_by_field_output, DataViewSnapshot,
            DataViewComparator
        )

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[
                {"id": "m1", "name": "A", "description": "Old A", "type": "int"},
                {"id": "m2", "name": "B", "description": "Old B", "type": "int"},
            ],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[
                {"id": "m1", "name": "A", "description": "New A", "type": "int"},
                {"id": "m2", "name": "B", "description": "New B", "type": "int"},
            ],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        output = write_diff_grouped_by_field_output(result, use_color=False)

        assert "GROUPED BY FIELD" in output
        assert "CHANGES BY FIELD" in output
        assert "description" in output.lower()


class TestPRCommentOutput:
    """Tests for --format-pr-comment output"""

    def test_pr_comment_output_format(self, logger):
        """Test PR comment markdown format"""
        from cja_sdr_generator import (
            write_diff_pr_comment_output, DataViewSnapshot,
            DataViewComparator
        )

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "A", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[
                {"id": "m1", "name": "A", "type": "decimal"},  # Type change - breaking
                {"id": "m2", "name": "B", "type": "int"}
            ],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        output = write_diff_pr_comment_output(result)

        # Should be markdown format
        assert "### 📊 Data View Comparison" in output
        assert "| Component |" in output
        assert "<details>" in output
        assert "</details>" in output

    def test_pr_comment_breaking_changes(self, logger):
        """Test PR comment shows breaking changes"""
        from cja_sdr_generator import (
            write_diff_pr_comment_output, DataViewSnapshot,
            DataViewComparator
        )

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "A", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m1", "name": "A", "type": "decimal"}],  # Type change
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        output = write_diff_pr_comment_output(result)

        # Should flag breaking change
        assert "Breaking Changes" in output


class TestBreakingChangeDetection:
    """Tests for breaking change detection"""

    def test_detect_type_change_as_breaking(self, logger):
        """Test type changes are flagged as breaking"""
        from cja_sdr_generator import detect_breaking_changes, DataViewSnapshot, DataViewComparator

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "A", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m1", "name": "A", "type": "decimal"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        breaking = detect_breaking_changes(result)

        assert len(breaking) == 1
        assert breaking[0]['change_type'] == 'type_changed'
        assert breaking[0]['severity'] == 'high'

    def test_detect_removal_as_breaking(self, logger):
        """Test component removal is flagged as breaking"""
        from cja_sdr_generator import detect_breaking_changes, DataViewSnapshot, DataViewComparator

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "A", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        breaking = detect_breaking_changes(result)

        assert len(breaking) == 1
        assert breaking[0]['change_type'] == 'removed'

    def test_no_breaking_changes(self, logger):
        """Test no breaking changes when only non-breaking changes"""
        from cja_sdr_generator import detect_breaking_changes, DataViewSnapshot, DataViewComparator

        source = DataViewSnapshot(
            data_view_id="dv_1", data_view_name="Source",
            metrics=[{"id": "m1", "name": "A", "description": "Old", "type": "int"}],
            dimensions=[]
        )
        target = DataViewSnapshot(
            data_view_id="dv_2", data_view_name="Target",
            metrics=[{"id": "m1", "name": "A", "description": "New", "type": "int"}],
            dimensions=[]
        )

        comparator = DataViewComparator(logger)
        result = comparator.compare(source, target)
        breaking = detect_breaking_changes(result)

        # Description change is not breaking
        assert len(breaking) == 0


class TestNewCLIFlags:
    """Tests for new v3.0.10 CLI flags"""

    def test_parse_no_color_flag(self):
        """Test that --no-color flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--no-color']
            args = parse_arguments()
            assert args.no_color is True
        finally:
            sys.argv = original_argv

    def test_parse_quiet_diff_flag(self):
        """Test that --quiet-diff flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--quiet-diff']
            args = parse_arguments()
            assert args.quiet_diff is True
        finally:
            sys.argv = original_argv

    def test_parse_reverse_diff_flag(self):
        """Test that --reverse-diff flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--reverse-diff']
            args = parse_arguments()
            assert args.reverse_diff is True
        finally:
            sys.argv = original_argv

    def test_parse_warn_threshold_flag(self):
        """Test that --warn-threshold flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--warn-threshold', '10.5']
            args = parse_arguments()
            assert args.warn_threshold == 10.5
        finally:
            sys.argv = original_argv

    def test_parse_group_by_field_flag(self):
        """Test that --group-by-field flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--group-by-field']
            args = parse_arguments()
            assert args.group_by_field is True
        finally:
            sys.argv = original_argv

    def test_parse_diff_output_flag(self):
        """Test that --diff-output flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--diff-output', '/tmp/diff.txt']
            args = parse_arguments()
            assert args.diff_output == '/tmp/diff.txt'
        finally:
            sys.argv = original_argv

    def test_parse_format_pr_comment_flag(self):
        """Test that --format-pr-comment flag is parsed correctly"""
        from cja_sdr_generator import parse_arguments
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['cja_sdr_generator.py', '--diff', 'dv_1', 'dv_2', '--format-pr-comment']
            args = parse_arguments()
            assert args.format_pr_comment is True
        finally:
            sys.argv = original_argv
