from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta

from cja_auto_sdr.core.colors import ConsoleColors
from cja_auto_sdr.diff.models import DataViewSnapshot


class SnapshotManager:
    """
    Manages data view snapshot lifecycle.

    Handles creating snapshots from live data views, saving/loading to JSON files,
    and listing available snapshots.
    """

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)

    def create_snapshot(
        self,
        cja,
        data_view_id: str,
        quiet: bool = False,
        include_calculated_metrics: bool = False,
        include_segments: bool = False,
    ) -> DataViewSnapshot:
        """Create a snapshot from a live data view."""
        self.logger.info(f"Creating snapshot for data view: {data_view_id}")

        dv_info = cja.getDataView(data_view_id)
        if not dv_info:
            raise ValueError(f"Failed to fetch data view info for {data_view_id}")

        dv_name = dv_info.get("name", "Unknown")
        dv_owner = dv_info.get("owner", {})
        owner_name = dv_owner.get("name", "") if isinstance(dv_owner, dict) else str(dv_owner)
        dv_description = dv_info.get("description", "")

        self.logger.info("Fetching metrics...")
        metrics_df = cja.getMetrics(data_view_id, inclType=True, full=True)
        metrics_list = []
        if metrics_df is not None and not metrics_df.empty:
            metrics_list = metrics_df.to_dict("records")
        self.logger.info(f"  Fetched {len(metrics_list)} metrics")

        self.logger.info("Fetching dimensions...")
        dimensions_df = cja.getDimensions(data_view_id, inclType=True, full=True)
        dimensions_list = []
        if dimensions_df is not None and not dimensions_df.empty:
            dimensions_list = dimensions_df.to_dict("records")
        self.logger.info(f"  Fetched {len(dimensions_list)} dimensions")

        snapshot = DataViewSnapshot(
            data_view_id=data_view_id,
            data_view_name=dv_name,
            owner=owner_name,
            description=dv_description,
            metrics=metrics_list,
            dimensions=dimensions_list,
        )

        if include_calculated_metrics:
            if not quiet:
                print("Fetching calculated metrics inventory...")
            try:
                from cja_auto_sdr.inventory.calculated_metrics import CalculatedMetricsInventoryBuilder

                builder = CalculatedMetricsInventoryBuilder(logger=self.logger)
                inventory = builder.build(cja, data_view_id, dv_name)
                snapshot.calculated_metrics_inventory = [m.to_full_dict() for m in inventory.metrics]
                self.logger.info(f"  Fetched {len(snapshot.calculated_metrics_inventory)} calculated metrics")
                if not quiet:
                    print(f"  Calculated metrics: {len(snapshot.calculated_metrics_inventory)} items")
            except ImportError as e:
                self.logger.warning(f"Could not import calculated metrics inventory module: {e}")
                if not quiet:
                    print(ConsoleColors.warning("  Warning: Calculated metrics module not available"))
            except Exception as e:
                self.logger.warning(f"Failed to fetch calculated metrics inventory: {e}")
                if not quiet:
                    print(ConsoleColors.warning(f"  Warning: Could not fetch calculated metrics: {e}"))

        if include_segments:
            if not quiet:
                print("Fetching segments inventory...")
            try:
                from cja_auto_sdr.inventory.segments import SegmentsInventoryBuilder

                builder = SegmentsInventoryBuilder(logger=self.logger)
                inventory = builder.build(cja, data_view_id, dv_name)
                snapshot.segments_inventory = [s.to_full_dict() for s in inventory.segments]
                self.logger.info(f"  Fetched {len(snapshot.segments_inventory)} segments")
                if not quiet:
                    print(f"  Segments: {len(snapshot.segments_inventory)} items")
            except ImportError as e:
                self.logger.warning(f"Could not import segments inventory module: {e}")
                if not quiet:
                    print(ConsoleColors.warning("  Warning: Segments module not available"))
            except Exception as e:
                self.logger.warning(f"Failed to fetch segments inventory: {e}")
                if not quiet:
                    print(ConsoleColors.warning(f"  Warning: Could not fetch segments: {e}"))

        self.logger.info(f"Snapshot created: {len(metrics_list)} metrics, {len(dimensions_list)} dimensions")
        return snapshot

    def save_snapshot(self, snapshot: DataViewSnapshot, filepath: str) -> str:
        """Save a snapshot to a JSON file."""
        filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Snapshot saved to: {filepath}")
        return filepath

    def load_snapshot(self, filepath: str) -> DataViewSnapshot:
        """Load a snapshot from a JSON file."""
        filepath = os.path.abspath(filepath)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Snapshot file not found: {filepath}")

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        if "snapshot_version" not in data:
            raise ValueError(f"Invalid snapshot file: {filepath} (missing snapshot_version)")

        snapshot = DataViewSnapshot.from_dict(data)
        self.logger.info(f"Loaded snapshot: {snapshot.data_view_name} ({snapshot.data_view_id})")
        self.logger.info(f"  Created: {snapshot.created_at}")
        self.logger.info(f"  Metrics: {len(snapshot.metrics)}, Dimensions: {len(snapshot.dimensions)}")

        return snapshot

    def list_snapshots(self, directory: str) -> list[dict]:
        """List available snapshots in a directory."""
        snapshots = []
        directory = os.path.abspath(directory)

        if not os.path.exists(directory):
            return snapshots

        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)
                    if "snapshot_version" in data:
                        snapshots.append(
                            {
                                "filename": filename,
                                "filepath": filepath,
                                "data_view_id": data.get("data_view_id", ""),
                                "data_view_name": data.get("data_view_name", ""),
                                "created_at": data.get("created_at", ""),
                                "metrics_count": len(data.get("metrics", [])),
                                "dimensions_count": len(data.get("dimensions", [])),
                            }
                        )
                except OSError, json.JSONDecodeError:
                    continue

        return sorted(snapshots, key=lambda x: x.get("created_at", ""), reverse=True)

    def get_most_recent_snapshot(self, directory: str, data_view_id: str) -> str | None:
        """Get the filepath of the most recent snapshot for a specific data view."""
        all_snapshots = self.list_snapshots(directory)
        dv_snapshots = [s for s in all_snapshots if s.get("data_view_id") == data_view_id]

        if not dv_snapshots:
            return None

        return dv_snapshots[0].get("filepath")

    def apply_retention_policy(self, directory: str, data_view_id: str, keep_last: int) -> list[str]:
        """Apply retention policy by deleting old snapshots for a specific data view."""
        if keep_last <= 0:
            return []

        all_snapshots = self.list_snapshots(directory)
        dv_snapshots = [s for s in all_snapshots if s.get("data_view_id") == data_view_id]

        if len(dv_snapshots) <= keep_last:
            return []

        deleted = []
        for snapshot in dv_snapshots[keep_last:]:
            filepath = snapshot.get("filepath")
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    self.logger.info(f"Retention policy: Deleted old snapshot {filepath}")
                    deleted.append(filepath)
                except OSError as e:
                    self.logger.warning(f"Failed to delete snapshot {filepath}: {e}")

        return deleted

    def apply_date_retention_policy(
        self,
        directory: str,
        data_view_id: str,
        keep_since_days: int | None = None,
        delete_older_than_days: int | None = None,
    ) -> list[str]:
        """Apply date-based retention policy by deleting snapshots outside the time window."""
        days = keep_since_days or delete_older_than_days
        if not days or days <= 0:
            return []

        cutoff_date = datetime.now(UTC) - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()

        all_snapshots = self.list_snapshots(directory)

        if data_view_id and data_view_id != "*":
            snapshots_to_check = [s for s in all_snapshots if s.get("data_view_id") == data_view_id]
        else:
            snapshots_to_check = all_snapshots

        deleted = []
        for snapshot in snapshots_to_check:
            created_at = snapshot.get("created_at", "")
            if created_at and created_at < cutoff_str:
                filepath = snapshot.get("filepath")
                if filepath and os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                        self.logger.info(f"Date retention policy: Deleted snapshot older than {days} days: {filepath}")
                        deleted.append(filepath)
                    except OSError as e:
                        self.logger.warning(f"Failed to delete snapshot {filepath}: {e}")

        if deleted:
            self.logger.info(f"Date retention: Deleted {len(deleted)} snapshot(s) older than {days} days")

        return deleted

    def generate_snapshot_filename(self, data_view_id: str, data_view_name: str | None = None) -> str:
        """Generate a timestamped filename for auto-saved snapshots."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        if data_view_name:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in data_view_name)
            safe_name = safe_name[:50]
            return f"{safe_name}_{data_view_id}_{timestamp}.json"
        return f"{data_view_id}_{timestamp}.json"


def parse_retention_period(period_str: str) -> int | None:
    """
    Parse a retention period string into days.

    Supports formats:
    - '7d' or '7D' - 7 days
    - '2w' or '2W' - 2 weeks (14 days)
    - '1m' or '1M' - 1 month (30 days)
    - '30' - 30 days (plain number)
    """
    if not period_str:
        return None

    period_str = period_str.strip().lower()

    if period_str.isdigit():
        return int(period_str)

    try:
        if period_str.endswith("d"):
            return int(period_str[:-1])
        if period_str.endswith("w"):
            return int(period_str[:-1]) * 7
        if period_str.endswith("m"):
            return int(period_str[:-1]) * 30
    except ValueError:
        pass

    return None
