from typing import List, Optional, Any, Dict

from nora_lib.tasks.models import AsyncTaskState as BaseAsyncTaskState
from pydantic import BaseModel, Field
from scholarqa.table_generation.table_model import TableWidget


class Author(BaseModel):
    name: str = Field(description="The name of the author")
    authorId: Optional[str] = Field(description="The Semantic Scholar id of the author")


class PaperDetails(BaseModel):
    corpus_id: int = Field(allow_none=False, description=(
        "The Semantic Scholar id of the cited paper"))
    title: str = Field(description="Title of the paper")
    year: int = Field(description="Year of publication")
    venue: Optional[str] = Field(description="Venue of publication", default=None)
    authors: List[Author] = Field(description="Authors of the paper", default=None)
    n_citations: Optional[int] = Field(default=0, description=(
        "The number of times the source paper has been cited"
    ))


# TODO: define your request data
class ToolRequest(BaseModel):
    task_id: Optional[str] = Field(default=None, description=(
        "Reference to a long-running task. Provide this argument to receive an update on its"
        "status and possibly its result."
    ))
    query: str = Field(default=None, description=(
        "A scientific query posed to scholar qa by a user"
    ))
    opt_in: Optional[bool] = Field(default=True, description=(
        "Flag to indicate whether to include the query and response in public release"))
    user_id: Optional[str] = Field(default=None, description="The user id of the user who posed the query")


class CitationSrc(BaseModel):
    id: str = Field(default=None, description=(
        "The id of the citation which is of the format (index, author_ref_string, year)"
    ))
    paper: PaperDetails = Field(description="Metadata of the cited paper")
    snippets: Optional[List[str]] = Field(default=[], description=(
        "A list of all the relevant snippets from the cited paper"
    ))
    score: float = Field(description=("Relevance score of the snippet for the query"))


class GeneratedSection(BaseModel):
    title: str = Field(default=None, description=(
        "header for the generated section text"
    ))
    tldr: str = Field(default=None, description=(
        "A short summary of the generated section"
    ))
    text: str = Field(default=None, description=(
        "The generated section text"
    ))
    citations: List[CitationSrc] = Field(default=None, description=(
        "The citations used in the generated section"
    ))
    table: Optional[TableWidget] = Field(default=None, description=("Table widget object for sections with list format"))


# TODO: define your result data
class TaskResult(BaseModel):
    """The outcome of running a Task to completion"""
    sections: List[GeneratedSection] = Field(
        description="The generated iterations of the answer"
    )
    cost: float = Field(description="The overall cost of the task", default=0.0)
    tokens: Dict[str, int] = Field(description="The tokens used to generate the answer")


class ToolResponse(BaseModel):
    task_id: str = Field(description="Unique identifiers for the invocation of the tool.")
    query: str = Field(description="The query that was posed to the tool.")
    task_result: TaskResult


class TaskStep(BaseModel):
    description: str = Field(description="The step in the task")
    start_timestamp: float = Field(description="The timestamp when the step was started")
    estimated_timestamp: float = Field(description="The estimated timestamp for the step to complete", default=None)


class AsyncTaskState(BaseAsyncTaskState[TaskResult]):
    pass


class AsyncToolResponse(BaseModel):
    task_id: str = Field(
        "Identifies the long-running task so that its status and eventual result"
        "can be checked in follow-up calls."
    )
    query: str = Field(description="The query that was posed to the tool.")
    estimated_time: str = Field(description="How long we expect this task to take from start to finish")
    task_status: str = Field(description="Current human-readable status of the task.")
    task_result: Optional[TaskResult] = Field(description="Final result of the task.")
    steps: List[TaskStep] = Field(description="The steps processed in the task so far.", default=[])


if __name__ == "__main__":
    import json

    # task_result_schema = TaskResult.model_json_schema()
    # with open("task_result.schema.json", "w") as f:
    #     json.dump(task_result_schema, f, indent=2)

    async_task_state_schema = AsyncTaskState.model_json_schema()
    with open("async_task_state.schema.json", "w") as f:
        json.dump(async_task_state_schema, f, indent=2)
