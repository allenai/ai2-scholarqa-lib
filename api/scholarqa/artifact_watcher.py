"""
File watcher for artifact files.
Monitors artifact directory and triggers query processing for new threads.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Callable, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

logger = logging.getLogger(__name__)


class ArtifactFileHandler(FileSystemEventHandler):
    """Handler for artifact file system events."""

    def __init__(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Initialize the handler.

        Args:
            callback: Function to call when a processable artifact is detected.
                     Signature: callback(artifact_path: str, artifact_data: Dict)
        """
        super().__init__()
        self.callback = callback
        self.processed_artifacts: Set[str] = set()
        self.processing_lock: Set[str] = set()

    def on_created(self, event: FileSystemEvent):
        """Handle file creation events."""
        if event.is_directory:
            return

        if not event.src_path.endswith('.json'):
            return

        # Ignore temp files
        if Path(event.src_path).name.startswith('.tmp_'):
            return

        self._process_artifact(event.src_path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification events."""
        if event.is_directory:
            return

        if not event.src_path.endswith('.json'):
            return

        # Ignore temp files
        if Path(event.src_path).name.startswith('.tmp_'):
            return

        # Only process modifications for artifacts we haven't processed yet
        # (this catches cases where the file was created before watcher started)
        if event.src_path not in self.processed_artifacts:
            self._process_artifact(event.src_path)

    def _process_artifact(self, file_path: str):
        """
        Process an artifact file.

        Args:
            file_path: Path to the artifact file
        """
        # Prevent duplicate processing
        if file_path in self.processing_lock:
            return

        self.processing_lock.add(file_path)

        try:
            # Wait a moment for file write to complete
            time.sleep(0.1)

            # Read the artifact
            with open(file_path, 'r') as f:
                artifact = json.load(f)

            # Check if this is a processable artifact
            if self._should_process(artifact):
                logger.info(f"Processing artifact: {file_path}")
                self.callback(file_path, artifact)
                self.processed_artifacts.add(file_path)
            else:
                logger.debug(f"Skipping non-processable artifact: {file_path}")

        except Exception as e:
            logger.error(f"Error processing artifact {file_path}: {e}")
        finally:
            self.processing_lock.discard(file_path)

    def _should_process(self, artifact: Dict[str, Any]) -> bool:
        """
        Check if an artifact should be processed.

        Criteria:
        - Must be a THREAD type
        - Must have at least one MESSAGE child
        - Message must not already have an SQA_REPORT child

        Args:
            artifact: The artifact dictionary

        Returns:
            True if should process, False otherwise
        """
        # Check if it's a thread
        if not isinstance(artifact, dict):
            return False

        data = artifact.get('data', {})
        if not isinstance(data, dict):
            return False

        artifact_type = data.get('type')
        if artifact_type != 'THREAD':
            return False

        # Check for message children
        children = artifact.get('children', [])
        if not children:
            return False

        # Find first message child
        for child in children:
            child_data = child.get('data', {})
            if child_data.get('type') == 'MESSAGE':
                # Check if message already has a report
                message_children = child.get('children', [])
                for msg_child in message_children:
                    msg_child_data = msg_child.get('data', {})
                    if msg_child_data.get('type') == 'SQA_REPORT':
                        logger.debug(f"Message already has SQA_REPORT, skipping")
                        return False

                # Found a message without a report
                return True

        return False


class ArtifactWatcher:
    """Watcher for artifact files that triggers query processing."""

    def __init__(
        self,
        watch_dir: str,
        callback: Callable[[str, Dict[str, Any]], None],
        recursive: bool = True
    ):
        """
        Initialize the watcher.

        Args:
            watch_dir: Directory to watch for artifacts
            callback: Function to call when a processable artifact is detected
            recursive: Whether to watch subdirectories recursively
        """
        self.watch_dir = Path(watch_dir)
        self.callback = callback
        self.recursive = recursive
        self.observer = None
        self.handler = ArtifactFileHandler(callback)

        if not self.watch_dir.exists():
            logger.warning(f"Watch directory does not exist: {watch_dir}")
            self.watch_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created watch directory: {watch_dir}")

    def start(self):
        """Start watching for artifact changes."""
        self.observer = Observer()
        self.observer.schedule(
            self.handler,
            str(self.watch_dir),
            recursive=self.recursive
        )
        self.observer.start()
        logger.info(f"Started watching artifacts in: {self.watch_dir}")

    def stop(self):
        """Stop watching for changes."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Stopped watching artifacts")

    def run_forever(self):
        """Run the watcher indefinitely."""
        self.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
