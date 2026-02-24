import logging
import multiprocessing
import os
from json import JSONDecodeError
from time import time
from typing import Union
from uuid import uuid4, uuid5, UUID

from fastapi import FastAPI, HTTPException, Request
from nora_lib.tasks.models import TASK_STATUSES, AsyncTaskState
from nora_lib.tasks.state import NoSuchTaskException

from scholarqa.config.config_setup import read_json_config
from scholarqa.models import (
    AsyncToolResponse,
    TaskResult,
    ToolRequest,
    ToolResponse,
    TaskStep,
    SnippetBasedRequest
)
from scholarqa.rag.reranker.modal_engine import ModalReranker
from scholarqa.rag.reranker.reranker_base import RERANKER_MAPPING
from scholarqa.rag.retrieval import PaperFinderWithReranker, PaperFinder
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.scholar_qa import ScholarQA
from scholarqa.lite import ScholarQALite
from scholarqa.state_mgmt.local_state_mgr import LocalStateMgrClient
from typing import Type, TypeVar

logger = logging.getLogger(__name__)

TIMEOUT = 240

async_context = multiprocessing.get_context("fork")

started_task_step = None

T = TypeVar("T", bound=ScholarQA)


def lazy_load_state_mgr_client():
    return LocalStateMgrClient(logs_config.log_dir, "async_state")


SQA_MODE = os.environ.get("SQA_MODE", "default")


def lazy_load_scholarqa(task_id: str, tool_req: ToolRequest=None, sqa_class: Type[T] = ScholarQA, **sqa_args) -> T:
    retriever = FullTextRetriever(**run_config.retriever_args)
    if run_config.reranker_args:
        reranker = RERANKER_MAPPING[run_config.reranker_service](**run_config.reranker_args)
        paper_finder = PaperFinderWithReranker(retriever, reranker, **run_config.paper_finder_args)
    else:
        paper_finder = PaperFinder(retriever, **run_config.paper_finder_args)

    init_kwargs = {**run_config.pipeline_args, **sqa_args}

    if SQA_MODE == "lite":
        sqa_class = ScholarQALite
        init_kwargs["lite_pipeline_args"] = run_config.lite_pipeline_args

    return sqa_class(paper_finder=paper_finder, task_id=task_id, state_mgr=app_config.state_mgr_client,
                     logs_config=logs_config, **init_kwargs)


# setup logging config and local litellm cache
CONFIG_PATH = os.environ.get("CONFIG_PATH", "run_configs/default.json")

app_config = read_json_config(CONFIG_PATH)
logs_config = app_config.logs
run_config = app_config.run_config
app_config.load_scholarqa = lazy_load_scholarqa


def _do_task(tool_request: ToolRequest, task_id: str) -> TaskResult:
    """
    TODO: BYO logic here. Don't forget to define `ToolRequest` and `TaskResult`
    in `models.py`!

    The meat of whatever it is your tool or task agent actually
    does should be kicked off in here. This will be run synchonrously
    unless `_needs_to_be_async()` above returns True, in which case
    it will be run in a background process.

    If you need to update state for an asynchronously running task, you can
    use `task_state_manager.read_state(task_id)` to retrieve, and `.write_state()`
    to write back.
    """
    scholar_qa = app_config.load_scholarqa(task_id, tool_request)
    return scholar_qa.run_qa_pipeline(tool_request)


def _do_snippet_based_task(tool_request: ToolRequest, task_id: str, request: SnippetBasedRequest) -> TaskResult:
    """
    Execute snippet-based pipeline: reranking + report generation.
    Skips query preprocessing and retrieval steps since client provides snippets.
    """
    from scholarqa.llms.constants import CostReportingArgs, STATUS_SYNTHESIS
    from scholarqa.trace.event_traces import EventTrace
    from scholarqa.utils import get_paper_metadata, NUMERIC_META_FIELDS, CATEGORICAL_META_FIELDS

    # Load ScholarQA instance (with reranker configured)
    # Override n_rerank if client specified a different value
    paper_finder_args = {}
    if request.n_rerank and request.n_rerank != run_config.paper_finder_args.get("n_rerank", 50):
        paper_finder_args["n_rerank"] = request.n_rerank

    scholar_qa = app_config.load_scholarqa(task_id, tool_request, **paper_finder_args)

    query = request.query
    user_id = request.user_id or "snippet_based_user"

    logger.info(f"Processing snippet-based query: {query} with {len(request.snippets)} snippets")

    # Initialize event trace for logging
    event_trace = EventTrace(
        task_id,
        0,  # No retrieval performed
        scholar_qa.paper_finder.n_rerank,
        tool_request,
        user_id=user_id
    )

    cost_args = CostReportingArgs(
        task_id=task_id,
        user_id=user_id,
        description="Snippet-based corpus QA",
        model=scholar_qa.llm_model,
        msg_id=task_id
    )

    # Use client-provided snippets directly (skip retrieval)
    retrieved_candidates = request.snippets
    if not retrieved_candidates:
        raise Exception("No snippets provided")

    event_trace.trace_retrieval_event(retrieved_candidates)

    # Extract metadata from snippets if available
    s2_srch_metadata = []
    for snippet in retrieved_candidates:
        if any(field in snippet for field in CATEGORICAL_META_FIELDS + NUMERIC_META_FIELDS):
            metadata = {k: v for k, v in snippet.items() if
                       k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            s2_srch_metadata.append(metadata)

    # Combine client-provided metadata with snippet metadata
    initial_paper_metadata = {str(paper["corpus_id"]): paper for paper in s2_srch_metadata}
    if request.paper_metadata:
        # Client-provided metadata takes precedence
        initial_paper_metadata.update({str(k): v for k, v in request.paper_metadata.items()})

    # Rerank and aggregate snippets at paper level
    scholar_qa.update_task_state("Reranking and aggregating snippets", step_estimated_time=10)
    reranked_df, paper_metadata = scholar_qa.rerank_and_aggregate(
        query, retrieved_candidates, initial_paper_metadata
    )

    if reranked_df.empty:
        raise Exception("No relevant papers found after reranking")

    event_trace.trace_rerank_event(reranked_df.to_dict(orient="records"))

    # Generate report using multi-step pipeline
    scholar_qa.update_task_state(STATUS_SYNTHESIS)
    report_data = scholar_qa.generate_report(
        query=query,
        reranked_df=reranked_df,
        paper_metadata=paper_metadata,
        cost_args=cost_args,
        event_trace=event_trace,
        user_id=user_id,
        inline_tags=False
    )

    # Postprocess and trace
    scholar_qa.postprocess_json_output(report_data.json_summary, quotes_meta=report_data.quotes_metadata)
    event_trace.trace_summary_event(report_data.json_summary, report_data.cost_result, report_data.tcosts)
    event_trace.persist_trace(scholar_qa.logs_config)

    return TaskResult(
        report_title=report_data.report_title,
        sections=report_data.sections,
        cost=event_trace.total_cost,
        tokens=event_trace.tokens
    )


def _estimate_task_length(tool_request: ToolRequest) -> str:
    """

    For telling the user how long to wait before asking for a status
    update on async tasks. This can just be a static guess, but you
    have access to the request if you want to do something fancier.
    """
    return (
        "~3 minutes"
    )


###########################################################################
### BELOW THIS LINE IS ALL TEMPLATE CODE THAT SHOULD NOT NEED TO CHANGE ###
###########################################################################


def create_app() -> FastAPI:
    app = FastAPI(root_path="/api")

    @app.get("/")
    def root(request: Request):
        return {"message": "Hello World", "root_path": request.scope.get("root_path")}

    @app.get("/health", status_code=204)
    def health():
        return "OK"

    @app.post("/query_corpusqa")
    def use_tool(
            tool_request: ToolRequest,
    ) -> Union[AsyncToolResponse, ToolResponse]:
        if not app_config.state_mgr_client:
            app_config.state_mgr_client = lazy_load_state_mgr_client()
        # Caller is asking for a status update of long-running request
        if tool_request.task_id:
            return _handle_async_task_check_in(tool_request)

        # New task
        task_id = str(uuid4())
        logs_config.task_id = task_id
        logger.info("New task")
        app_config.state_mgr_client.init_task(task_id, tool_request)
        estimated_time = _start_async_task(task_id, tool_request)

        return AsyncToolResponse(
            task_id=task_id,
            query=tool_request.query,
            estimated_time=estimated_time,
            task_status=TASK_STATUSES["STARTED"],
            task_result=None,
            steps=[started_task_step]
        )

    @app.post("/query_corpusqa_sync_with_snippets")
    def use_tool_sync_with_snippets(
            request: SnippetBasedRequest,
    ) -> ToolResponse:
        """
        Synchronous endpoint for snippet-based report generation.
        Client provides pre-retrieved snippets, service performs reranking + synthesis.
        """
        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query required and cannot be empty")
        if not request.snippets:
            raise HTTPException(status_code=400, detail="Snippets list cannot be empty")
        if len(request.snippets) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 snippets allowed")

        # Initialize state manager
        if not app_config.state_mgr_client:
            app_config.state_mgr_client = lazy_load_state_mgr_client()

        # Create new task
        task_id = str(uuid4())
        logs_config.task_id = task_id
        logger.info(f"New snippet-based sync task with {len(request.snippets)} snippets")

        # Create ToolRequest for internal processing
        tool_request = ToolRequest(
            task_id=task_id,
            query=request.query,
            user_id=request.user_id,
            opt_in=request.opt_in
        )

        # Initialize task tracking
        app_config.state_mgr_client.init_task(task_id, tool_request)

        start_time = time()
        try:
            # Execute synchronous pipeline
            task_result = _do_snippet_based_task(tool_request, task_id, request)

            elapsed = time() - start_time
            logger.info(f"Snippet-based task completed in {elapsed:.2f} seconds, cost: ${task_result.cost}")

            return ToolResponse(
                task_id=task_id,
                query=request.query,
                task_result=task_result
            )
        except Exception as e:
            elapsed = time() - start_time
            logger.exception(f"Snippet-based task failed after {elapsed:.2f} seconds: {e}")
            raise HTTPException(status_code=500, detail=f"Task failed: {str(e)}")

    app.state.use_tool_fn = use_tool
    return app


def _start_async_task(task_id: str, tool_request: ToolRequest) -> str:
    global started_task_step
    estimated_time = _estimate_task_length(tool_request)
    tool_request.task_id = task_id
    task_state_manager = app_config.state_mgr_client.get_state_mgr(tool_request)
    started_task_step = TaskStep(description=TASK_STATUSES["STARTED"], start_timestamp=time(),
                                 estimated_timestamp=time() + TIMEOUT)
    task_state = AsyncTaskState(
        task_id=task_id,
        estimated_time=estimated_time,
        task_status=TASK_STATUSES["STARTED"],
        task_result=None,
        extra_state={"query": tool_request.query, "start": time(),
                     "steps": [started_task_step]},
    )
    task_state_manager.write_state(task_state)

    def _do_task_and_write_result():
        extra_state = {}
        try:
            task_result = _do_task(tool_request, task_id)
            task_status = TASK_STATUSES["COMPLETED"]
            extra_state["end"] = time()
        except Exception as e:
            task_result = None
            task_status = TASK_STATUSES["FAILED"]
            extra_state["error"] = str(e)

        state = task_state_manager.read_state(task_id)
        state.task_result = task_result
        state.task_status = task_status
        state.extra_state.update(extra_state)
        state.estimated_time = "--"
        task_state_manager.write_state(state)

    async_context.Process(
        target=_do_task_and_write_result,
        name=f"Async Task {task_id}",
        args=(),
    ).start()

    return estimated_time


def _handle_async_task_check_in(
        tool_req: ToolRequest,
) -> Union[ToolResponse | AsyncToolResponse]:
    """
    For tasks that will take a while to complete, we issue a task id
    that can be used to request status updates and eventually, results.

    This helper function is responsible for checking the state store
    and returning either the current state of the given task id, or its
    final result.
    """
    task_id = tool_req.task_id
    logs_config.task_id = task_id
    task_state_manager = app_config.state_mgr_client.get_state_mgr(tool_req)
    try:
        task_state = task_state_manager.read_state(task_id)
    except NoSuchTaskException:
        raise HTTPException(
            status_code=404, detail=f"Referenced task {task_id} does not exist."
        )
    except JSONDecodeError as e:
        logger.warning("state file is corrupted, should be updated on next poll: {e}")
        return AsyncToolResponse(
            task_id=task_id,
            query="",
            estimated_time="~3 minutes",
            task_status=f"{time()}: Processing user query",
            task_result=None,
        )

    # Retrieve data, which is just on local disk for now
    if task_state.task_status == TASK_STATUSES["FAILED"]:
        if task_state.extra_state and "error" in task_state.extra_state:
            msg = f"\nError: {task_state.extra_state['error']}"
            logger.exception(msg)
        else:
            msg = f"Referenced task failed."
        raise HTTPException(status_code=500, detail=f"{msg}")

    if task_state.task_status == TASK_STATUSES["COMPLETED"]:
        if not task_state.task_result:
            msg = f"Task marked completed but has no result."
            logger.error(msg)
            raise HTTPException(
                status_code=500,
                detail=msg,
            )

        if "start" in task_state.extra_state and "end" in task_state.extra_state:
            try:
                cost = task_state.task_result["cost"] if type(
                    task_state.task_result) == dict else task_state.task_result.cost
            except Exception as e:
                logger.warning(f"Error occurred while parsing cost from the response: {e}")
                cost = 0.0
            logger.info(
                f"completed in {task_state.extra_state['end'] - task_state.extra_state['start']} seconds, "
                f"cost: ${cost}")
        return ToolResponse(
            task_id=task_state.task_id,
            query=task_state.extra_state["query"],
            task_result=task_state.task_result,
        )

    if task_state.task_status not in {TASK_STATUSES["COMPLETED"],
                                      TASK_STATUSES["FAILED"]} and "start" in task_state.extra_state:
        elapsed = time() - task_state.extra_state["start"]
        if elapsed > TIMEOUT:
            task_state.task_status = TASK_STATUSES["FAILED"]
            task_state.extra_state["error"] = f"Task timed out after {TIMEOUT} seconds"
            task_state_manager.write_state(task_state)
            logger.info(f"timed out after {time() - task_state.extra_state['start']} seconds.")
            raise HTTPException(
                status_code=500,
                detail=f"Task timed out after {TIMEOUT} seconds.")

    return AsyncToolResponse(
        task_id=task_state.task_id,
        query=task_state.extra_state["query"],
        estimated_time=task_state.estimated_time,
        task_status=task_state.task_status,
        task_result=task_state.task_result,
        steps=task_state.extra_state.get("steps", []),
    )
