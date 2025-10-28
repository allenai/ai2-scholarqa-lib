"""
Main entry point for the artifact watcher.
Monitors artifact files and triggers ScholarQA query processing.
"""
import logging
import os
import sys
from pathlib import Path
from time import time
from typing import Dict, Any
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scholarqa.artifact_watcher import ArtifactWatcher
from scholarqa.scholar_qa import ScholarQA
from scholarqa.rag.retrieval import PaperFinder, PaperFinderWithReranker
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.config.config_setup import read_json_config, LogsConfig
from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
from scholarqa.rag.reranker.reranker_base import RERANKER_MAPPING
from scholarqa.models import ToolRequest, TaskStep
from nora_lib.tasks.state import AsyncTaskState

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = os.environ.get("CONFIG_PATH", "run_configs/default.json")
ARTIFACTS_DIR = os.environ.get("ARTIFACT_OUTPUT_DIR", "/artifacts")
TIMEOUT = 600  # 10 minutes timeout
TASK_STATUSES = {
    "STARTED": "STARTED",
    "COMPLETED": "COMPLETED",
    "FAILED": "FAILED"
}

# Global state
app_config = None
logs_config = None
run_config = None
task_mappings = {}  # task_id -> (artifact_path, message_id)


def initialize_config():
    """Initialize configuration from file."""
    global app_config, logs_config, run_config

    logger.info(f"Loading configuration from {CONFIG_PATH}")
    app_config = read_json_config(CONFIG_PATH)
    logs_config = app_config.logs
    run_config = app_config.run_config
    logger.info("Configuration loaded successfully")


def load_scholarqa(artifact_path: str, message_id: str, task_id: str) -> ScholarQA:
    """
    Load ScholarQA with artifact updater configured.

    Args:
        artifact_path: Path to the thread artifact file
        message_id: ID of the message to update
        task_id: Task ID for this query

    Returns:
        Configured ScholarQA instance
    """
    retriever = FullTextRetriever(**run_config.retriever_args)

    if run_config.reranker_args:
        reranker = RERANKER_MAPPING[run_config.reranker_service](**run_config.reranker_args)
        paper_finder = PaperFinderWithReranker(retriever, reranker, **run_config.paper_finder_args)
    else:
        paper_finder = PaperFinder(retriever, **run_config.paper_finder_args)

    state_mgr = LocalStateMgrClient(logs_config.log_dir, "async_state")

    return ScholarQA(
        paper_finder=paper_finder,
        task_id=task_id,
        state_mgr=state_mgr,
        logs_config=logs_config,
        artifact_path=artifact_path,
        message_id=message_id,
        **run_config.pipeline_args
    )


def process_artifact(artifact_path: str, artifact_data: Dict[str, Any]):
    """
    Process a new thread artifact with message.

    Args:
        artifact_path: Path to the artifact file
        artifact_data: Parsed artifact dictionary
    """
    try:
        logger.info(f"Processing new artifact: {artifact_path}")

        # Extract message content
        children = artifact_data.get('children', [])
        if not children:
            logger.warning("No children found in artifact")
            return

        # Find first MESSAGE child
        message = None
        for child in children:
            child_data = child.get('data', {})
            if child_data.get('type') == 'MESSAGE':
                message = child
                break

        if not message:
            logger.warning("No MESSAGE child found in artifact")
            return

        message_id = message.get('id')
        message_content_data = message.get('data', {}).get('data', {})
        query = message_content_data.get('content')

        if not query:
            logger.warning("No content found in message")
            return

        logger.info(f"Extracted query from message {message_id}: {query[:100]}...")

        # Generate task ID
        task_id = str(uuid4())

        # Store mapping
        task_mappings[task_id] = (artifact_path, message_id)

        # Create tool request
        tool_request = ToolRequest(
            task_id=task_id,
            query=query,
            user_id="watcher_user"
        )

        # Initialize ScholarQA with artifact updater
        scholarqa = load_scholarqa(artifact_path, message_id, task_id)

        # Initialize task state
        state_mgr = scholarqa.state_mgr
        task_state_manager = state_mgr.get_state_mgr(tool_request)
        started_task_step = TaskStep(
            description=TASK_STATUSES["STARTED"],
            start_timestamp=time(),
            estimated_timestamp=time() + TIMEOUT
        )
        task_state = AsyncTaskState(
            task_id=task_id,
            estimated_time="~3 minutes",
            task_status=TASK_STATUSES["STARTED"],
            task_result=None,
            extra_state={
                "query": tool_request.query,
                "start": time(),
                "steps": [started_task_step]
            },
        )
        task_state_manager.write_state(task_state)

        logger.info(f"Starting ScholarQA pipeline for task {task_id}")

        # Run the pipeline (this will update the thread artifact incrementally)
        try:
            result = scholarqa.run_qa_pipeline(tool_request, inline_tags=True)
            logger.info(f"Successfully completed task {task_id}")
        except Exception as e:
            logger.error(f"Error running ScholarQA pipeline for task {task_id}: {e}")
            raise

    except Exception as e:
        logger.error(f"Error processing artifact {artifact_path}: {e}", exc_info=True)


def main():
    """Main entry point."""
    logger.info("Starting Artifact Watcher")

    # Initialize configuration
    initialize_config()

    # Create watcher
    watcher = ArtifactWatcher(
        watch_dir=ARTIFACTS_DIR,
        callback=process_artifact,
        recursive=True
    )

    logger.info(f"Watching for artifacts in: {ARTIFACTS_DIR}")

    # Run forever
    try:
        watcher.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down watcher...")
        watcher.stop()
        logger.info("Watcher stopped")


if __name__ == "__main__":
    main()
