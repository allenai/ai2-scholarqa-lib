import json
import logging
from typing import Callable, Literal

from pydantic import BaseModel, Field

from scholarqa.state_mgmt.local_state_mgr import AbsStateMgrClient
from scholarqa.utils import TaskIdAwareLogFormatter, init_settings

logger = logging.getLogger(__name__)


class LogsConfig(BaseModel):
    log_dir: str = Field(default="logs", description="Directory to store logs, event traces and litellm cache")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO",
                                                                                description="Logging level")
    llm_cache_dir: str = Field(default="llm_cache", description="Sub directory to cache llm calls")
    event_trace_loc: str = Field(default="scholarqa_traces", description="Sub directory to store event traces"
                                                                         "OR the GCS bucket name")
    extras: dict = Field(default=None, description="Extra arguments for the logs")
    tracing_mode: Literal["local", "gcs"] = Field(default="local",
                                                  description="Mode to store event traces (local or gcs)")
    tid_log_formatter: TaskIdAwareLogFormatter = Field(default=None,
                                                       description="Task Id aware log formatter which prepends the current task id to every log message")

    @property
    def task_id(self):
        return self.tid_log_formatter.task_id

    @task_id.setter
    def task_id(self, task_id: str):
        self.tid_log_formatter.task_id = task_id

    def init_formatter(self):
        self.tid_log_formatter = init_settings(logs_dir=self.log_dir, log_level=self.log_level,
                                               litellm_cache_dir=self.llm_cache_dir)

    class Config:
        arbitrary_types_allowed = True


class RunConfig(BaseModel):
    retrieval_service: str = Field(default="public_api", description="Service to use for paper retrieval")
    retriever_args: dict = Field(default=None, description="Arguments for the retrieval service")
    reranker_service: str = Field(default="modal", description="Service to use for paper reranking")
    reranker_args: dict = Field(default=None, description="Arguments for the reranker service")
    paper_finder_args: dict = Field(default=None, description="Arguments for the paper finder service")
    pipeline_args: dict = Field(default=None, description="Arguments for the Scholar QA pipeline service")
    lite_pipeline_args: dict = Field(default=None, description="Arguments for ScholarQALite pipeline")
    extras: dict = Field(default=None, description="Extra arguments for the run config")


class AppConfig(BaseModel):
    logs: LogsConfig = Field(default=None, description="Configuration for logs and event traces")
    run_config: RunConfig = Field(default=None, description="Configuration for components of the ScholarQA pipeline")
    state_mgr_client: AbsStateMgrClient = Field(default=None,
                                                description="State manager client for managing async task states and cost reporting of llm calls")
    load_scholarqa: Callable = Field(default=None,
                                     description="Function to lazy load ScholarQA instance, should be overridden by the app using this config class")

    class Config:
        arbitrary_types_allowed = True


def read_json_config(config_path: str, model_class: AppConfig = AppConfig) -> AppConfig:
    logger.info(f"Reading app config from {config_path}")
    json_data = json.load(open(config_path, "r"))
    app_config = model_class.model_validate(json_data)
    app_config.logs.tid_log_formatter = init_settings(logs_dir=app_config.logs.log_dir,
                                                        log_level=app_config.logs.log_level,
                                                      litellm_cache_dir=app_config.logs.llm_cache_dir)
    return app_config
