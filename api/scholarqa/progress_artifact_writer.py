"""
Progress artifact updater for ScholarQA status updates.
Updates thread artifacts by adding/updating STEP_PROGRESS as a nested child under messages.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from scholarqa.artifact_client import ArtifactClient

logger = logging.getLogger(__name__)


class ProgressArtifactUpdater:
    """Updates thread artifacts with progress steps as nested children."""

    def __init__(self, artifact_path: str, message_id: str, task_id: str):
        """
        Initialize the progress artifact updater.

        Args:
            artifact_path: Full path to the thread artifact file
            message_id: ID of the message to add/update the progress under
            task_id: Unique task ID for this progress tracker
        """
        self.artifact_path = Path(artifact_path)
        self.message_id = message_id
        self.task_id = task_id
        self.progress_id = f"progress-{task_id}"
        self.artifact_id = str(self.artifact_path.relative_to(self.artifact_path.parent.parent))
        self.progress_version = 0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.steps: List[str] = []

        # Initialize artifact client
        artifacts_dir = self.artifact_path.parent
        self.client = ArtifactClient(str(artifacts_dir))

    def add_step(self, step_message: str) -> None:
        """
        Add a progress step and update the thread artifact.

        Args:
            step_message: The progress step message to add
        """
        self.steps.append(step_message)
        self._write_progress_artifact()

    def _write_progress_artifact(self) -> None:
        """Update the thread artifact with the current progress data."""
        self.progress_version += 1
        now = datetime.now(timezone.utc).isoformat()

        # Build the progress artifact structure
        progress = {
            "id": self.progress_id,
            "version": self.progress_version,
            "createdAt": self.created_at,
            "updatedAt": now,
            "data": {
                "type": "STEP_PROGRESS",
                "data": {
                    "steps": self.steps.copy()  # Copy to avoid mutation
                }
            },
            "children": []
        }

        # Update the thread artifact
        def updater(thread_artifact: Dict[str, Any]) -> Dict[str, Any]:
            # Check if progress already exists in message
            existing_progress = self.client.get_report_from_message(
                thread_artifact,
                self.message_id,
                self.progress_id
            )

            if existing_progress:
                # Update existing progress
                return self.client.update_report_in_message(
                    thread_artifact,
                    self.message_id,
                    progress
                )
            else:
                # Add new progress
                return self.client.add_report_to_message(
                    thread_artifact,
                    self.message_id,
                    progress
                )

        success = self.client.update_artifact(str(self.artifact_path.name), updater)

        if success:
            logger.info(
                f"Updated progress artifact {self.artifact_id}: "
                f"progress {self.progress_id} version {self.progress_version} "
                f"with {len(self.steps)} steps"
            )
        else:
            logger.error(f"Failed to update progress artifact {self.artifact_id}")

    def get_current_version(self) -> int:
        """Get the current progress version number."""
        return self.progress_version

    def artifact_exists(self) -> bool:
        """Check if the thread artifact file exists."""
        return self.artifact_path.exists()
