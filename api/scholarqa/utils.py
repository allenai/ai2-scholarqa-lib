import logging
import os
import sys
from collections import namedtuple
from logging import Formatter
from typing import Any, Dict, Optional, Set, List

import requests
from fastapi import HTTPException
from google.cloud import storage

from scholarqa import glog
from scholarqa.llms.litellm_helper import setup_llm_cache

logger = logging.getLogger(__name__)

S2_APIKEY = os.getenv("S2_API_KEY", "")
S2_HEADERS = {"x-api-key": S2_APIKEY}
S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1/"
NUMERIC_META_FIELDS = {"year", "citationCount", "referenceCount", "influentialCitationCount"}
CATEGORICAL_META_FIELDS = {"title", "abstract", "corpusId", "authors", "venue", "isOpenAccess", "openAccessPdf"}
METADATA_FIELDS = ",".join(CATEGORICAL_META_FIELDS.union(NUMERIC_META_FIELDS))


class TaskIdAwareLogFormatter(Formatter):
    def __init__(self, task_id: str = ""):
        super().__init__("%(asctime)s - %(name)s - %(levelname)s")
        self.task_id = task_id

    def format(self, record):
        og_message = super().format(record)
        task_id_part = f"[{self.task_id}] " if self.task_id else ""
        return f"{og_message} - {task_id_part}- {record.getMessage()}"


def init_settings(logs_dir: str, log_level: str = "INFO",
                  litellm_cache_dir: str = "litellm_cache") -> TaskIdAwareLogFormatter:
    def setup_logging() -> TaskIdAwareLogFormatter:
        # If LOG_FORMAT is "google:json" emit log message as JSON in a format Google Cloud can parse
        loggers = [
            "LiteLLM Proxy",
            "LiteLLM Router",
            "LiteLLM"
        ]

        for logger_name in loggers:
            litellm_logger = logging.getLogger(logger_name)
            litellm_logger.setLevel(logging.WARNING)

        fmt = os.getenv("LOG_FORMAT")
        tid_log_fmt = TaskIdAwareLogFormatter()
        if fmt == "google:json":
            handlers = [glog.Handler()]
            for handler in handlers:
                handler.setFormatter(glog.Formatter(tid_log_fmt))
        else:
            handlers = []
            # log lower levels to stdout
            stdout_handler = logging.StreamHandler(stream=sys.stdout)
            stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
            handlers.append(stdout_handler)

            # log higher levels to stderr (red)
            stderr_handler = logging.StreamHandler(stream=sys.stderr)
            stderr_handler.addFilter(lambda rec: rec.levelno > logging.INFO)
            handlers.append(stderr_handler)
            for handler in handlers:
                handler.setFormatter(tid_log_fmt)

        level = log_level
        logging.basicConfig(level=level, handlers=handlers)
        return tid_log_fmt

    def setup_local_llm_cache():
        # Local logs directory for litellm caching, event traces and state management
        local_cache_dir = f"{logs_dir}/{litellm_cache_dir}"
        # create parent and subdirectories for the local cache
        os.makedirs(local_cache_dir, exist_ok=True)
        setup_llm_cache(cache_type="disk", disk_cache_dir=local_cache_dir)

    tid_log_fmt = setup_logging()
    setup_local_llm_cache()
    return tid_log_fmt


def make_int(x: Optional[Any]) -> int:
    try:
        return int(x)
    except:
        return 0


def get_ref_author_str(authors: List[Dict[str, str]]) -> str:
    if not authors:
        return "NULL"
    f_author_lname = authors[0]["name"].split()[-1]
    return f_author_lname if len(authors) == 1 else f"{f_author_lname} et al."


def query_s2_api(
        end_pt: str = "paper/batch",
        params: Dict[str, Any] = None,
        payload: Dict[str, Any] = None,
        method="get",
):
    url = S2_API_BASE_URL + end_pt
    req_method = requests.get if method == "get" else requests.post
    response = req_method(url, headers=S2_HEADERS, params=params, json=payload)
    if response.status_code != 200:
        logging.exception(f"S2 API request to end point {end_pt} failed with status code {response.status_code}")
        raise HTTPException(
            status_code=500,
            detail=f"S2 API request failed with status code {response.status_code}",
        )
    return response.json()


def get_paper_metadata(corpus_ids: Set[str], fields=METADATA_FIELDS) -> Dict[str, Any]:
    if not corpus_ids:
        return {}
    paper_data = query_s2_api(
        end_pt="paper/batch",
        params={
            "fields": fields
        },
        payload={"ids": ["CorpusId:{0}".format(cid) for cid in corpus_ids]},
        method="post",
    )
    paper_metadata = {
        str(pdata["corpusId"]): {k: make_int(v) if k in NUMERIC_META_FIELDS else pdata.get(k) for k, v in pdata.items()}
        for pdata in paper_data if pdata and "corpusId" in pdata
    }
    return paper_metadata


def push_to_gcs(text: str, bucket: str, file_path: str):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket)
        blob = bucket.blob(file_path)
        blob.upload_from_string(text)
        logging.info(f"Pushed event trace: {file_path} to GCS")
    except Exception as e:
        logging.info(f"Error pushing {file_path} to GCS: {e}")
