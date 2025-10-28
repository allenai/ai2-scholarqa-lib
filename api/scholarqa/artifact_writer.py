"""
Thread artifact updater for ScholarQA reports.
Updates thread artifacts by adding/updating SQA_REPORT as a nested child under messages.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from scholarqa.artifact_client import ArtifactClient

logger = logging.getLogger(__name__)


class ThreadArtifactUpdater:
    """Updates thread artifacts with ScholarQA reports as nested children."""

    def __init__(self, artifact_path: str, message_id: str, task_id: str):
        """
        Initialize the thread artifact updater.

        Args:
            artifact_path: Full path to the thread artifact file
            message_id: ID of the message to add/update the report under
            task_id: Unique task ID for this report
        """
        self.artifact_path = Path(artifact_path)
        self.message_id = message_id
        self.task_id = task_id
        self.report_id = f"report-{task_id}"
        self.artifact_id = str(self.artifact_path.relative_to(self.artifact_path.parent.parent))
        self.report_version = 0
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Initialize artifact client
        artifacts_dir = self.artifact_path.parent
        self.client = ArtifactClient(str(artifacts_dir))

    def write_artifact(
        self,
        json_summary: List[Dict[str, Any]],
        report_title: str = "ScholarQA Report",
        quotes_metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Update the thread artifact with the report data.

        Args:
            json_summary: List of section dictionaries from ScholarQA
            report_title: Title of the report
            quotes_metadata: Optional metadata about extracted quotes
        """
        self.report_version += 1
        now = datetime.now(timezone.utc).isoformat()

        # Build the report artifact structure
        report = {
            "id": self.report_id,
            "version": self.report_version,
            "createdAt": self.created_at,
            "updatedAt": now,
            "data": {
                "type": "SQA_REPORT",
                "data": {
                    "report_title": report_title,
                    "sections": json_summary,
                    "quotes_metadata": quotes_metadata or {},
                    "generated_at": now,
                }
            },
            "children": []
        }

        # Update the thread artifact
        def updater(thread_artifact: Dict[str, Any]) -> Dict[str, Any]:
            # Check if report already exists in message
            existing_report = self.client.get_report_from_message(
                thread_artifact,
                self.message_id,
                self.report_id
            )

            if existing_report:
                # Update existing report
                return self.client.update_report_in_message(
                    thread_artifact,
                    self.message_id,
                    report
                )
            else:
                # Add new report
                return self.client.add_report_to_message(
                    thread_artifact,
                    self.message_id,
                    report
                )

        success = self.client.update_artifact(str(self.artifact_path.name), updater)

        if success:
            logger.info(
                f"Updated thread artifact {self.artifact_id}: "
                f"report {self.report_id} version {self.report_version}"
            )
        else:
            logger.error(f"Failed to update thread artifact {self.artifact_id}")

    def get_current_version(self) -> int:
        """Get the current report version number."""
        return self.report_version

    def artifact_exists(self) -> bool:
        """Check if the thread artifact file exists."""
        return self.artifact_path.exists()
