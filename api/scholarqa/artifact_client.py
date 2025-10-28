"""
Artifact client for reading and updating thread artifacts.
Provides atomic operations for modifying artifact files.
"""
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class ArtifactClient:
    """Client for reading and updating artifact files atomically."""

    def __init__(self, artifacts_dir: str):
        """
        Initialize the artifact client.

        Args:
            artifacts_dir: Directory containing artifact files
        """
        self.artifacts_dir = Path(artifacts_dir)

    def read_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        """
        Read an artifact from disk.

        Args:
            artifact_id: The artifact ID (e.g., "reports/thread-123.json")

        Returns:
            The artifact dictionary, or None if not found
        """
        file_path = self.artifacts_dir / artifact_id

        if not file_path.exists():
            logger.warning(f"Artifact not found: {artifact_id}")
            return None

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading artifact {artifact_id}: {e}")
            return None

    def write_artifact(self, artifact_id: str, artifact: Dict[str, Any]) -> bool:
        """
        Write an artifact to disk atomically.

        Args:
            artifact_id: The artifact ID
            artifact: The artifact dictionary

        Returns:
            True if successful, False otherwise
        """
        file_path = self.artifacts_dir / artifact_id

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temp file first
            fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent,
                suffix='.json',
                prefix='.tmp_'
            )

            with os.fdopen(fd, 'w') as f:
                json.dump(artifact, f, indent=2)

            # Atomically replace
            shutil.move(temp_path, file_path)
            logger.info(f"Wrote artifact: {artifact_id} (version {artifact.get('version')})")
            return True

        except Exception as e:
            logger.error(f"Error writing artifact {artifact_id}: {e}")
            # Clean up temp file
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
            return False

    def update_artifact(
        self,
        artifact_id: str,
        updater: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> bool:
        """
        Update an artifact atomically using an updater function.

        Args:
            artifact_id: The artifact ID
            updater: Function that takes the artifact dict and returns the updated artifact

        Returns:
            True if successful, False otherwise
        """
        artifact = self.read_artifact(artifact_id)
        if artifact is None:
            return False

        try:
            updated_artifact = updater(artifact)
            return self.write_artifact(artifact_id, updated_artifact)
        except Exception as e:
            logger.error(f"Error updating artifact {artifact_id}: {e}")
            return False

    def add_report_to_message(
        self,
        thread_artifact: Dict[str, Any],
        message_id: str,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add an SQA_REPORT as a child of a message in a thread artifact.

        Args:
            thread_artifact: The thread artifact dictionary
            message_id: ID of the message to add the report to
            report: The report artifact to add

        Returns:
            Updated thread artifact
        """
        import copy
        from datetime import datetime, timezone

        # Deep copy to avoid mutation
        updated = copy.deepcopy(thread_artifact)

        # Find the message child
        for child in updated.get('children', []):
            if child.get('id') == message_id:
                # Add report as child if not already there
                if not any(c.get('id') == report.get('id') for c in child.get('children', [])):
                    child['children'].append(report)

                    # Update thread version and timestamp
                    updated['version'] = updated.get('version', 1) + 1
                    updated['updatedAt'] = datetime.now(timezone.utc).isoformat()

                    logger.info(f"Added report {report.get('id')} to message {message_id}")
                break
        else:
            logger.warning(f"Message {message_id} not found in thread {updated.get('id')}")

        return updated

    def update_report_in_message(
        self,
        thread_artifact: Dict[str, Any],
        message_id: str,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update an existing SQA_REPORT child in a message.

        Args:
            thread_artifact: The thread artifact dictionary
            message_id: ID of the message containing the report
            report: The updated report artifact

        Returns:
            Updated thread artifact
        """
        import copy
        from datetime import datetime, timezone

        # Deep copy to avoid mutation
        updated = copy.deepcopy(thread_artifact)

        # Find the message child
        for child in updated.get('children', []):
            if child.get('id') == message_id:
                # Find and update the report
                report_id = report.get('id')
                for i, report_child in enumerate(child.get('children', [])):
                    if report_child.get('id') == report_id:
                        child['children'][i] = report

                        # Update thread version and timestamp
                        updated['version'] = updated.get('version', 1) + 1
                        updated['updatedAt'] = datetime.now(timezone.utc).isoformat()

                        logger.info(f"Updated report {report_id} in message {message_id}")
                        break
                else:
                    logger.warning(f"Report {report_id} not found in message {message_id}")
                break
        else:
            logger.warning(f"Message {message_id} not found in thread {updated.get('id')}")

        return updated

    def find_message_in_thread(
        self,
        thread_artifact: Dict[str, Any],
        message_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find a message child in a thread artifact.

        Args:
            thread_artifact: The thread artifact dictionary
            message_id: ID of the message to find

        Returns:
            The message artifact, or None if not found
        """
        for child in thread_artifact.get('children', []):
            if child.get('id') == message_id:
                return child
        return None

    def get_report_from_message(
        self,
        thread_artifact: Dict[str, Any],
        message_id: str,
        report_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific report from a message.

        Args:
            thread_artifact: The thread artifact dictionary
            message_id: ID of the message
            report_id: ID of the report

        Returns:
            The report artifact, or None if not found
        """
        message = self.find_message_in_thread(thread_artifact, message_id)
        if message:
            for child in message.get('children', []):
                if child.get('id') == report_id:
                    return child
        return None
