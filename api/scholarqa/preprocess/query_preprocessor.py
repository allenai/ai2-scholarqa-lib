import json
import logging
import re
from collections import namedtuple
from multiprocessing import Queue
from typing import Tuple, List, Optional, Union

from litellm import moderation
from pydantic import BaseModel, Field

from scholarqa.llms.litellm_helper import llm_completion
from scholarqa.llms.constants import CompletionResult
from scholarqa.llms.prompts import QUERY_DECOMPOSER_PROMPT

logger = logging.getLogger(__name__)

LLMProcessedQuery = namedtuple("LLMProcessedQuery",
                               ["rewritten_query", "keyword_query", "search_filters"])


class DecomposedQuery(BaseModel):
    earliest_search_year: str = Field(description="The earliest year to search for papers")
    latest_search_year: str = Field(description="The latest year to search for papers")
    venues: str = Field(description="Comma separated list of venues to search for papers")
    authors: Union[List[str]|str] = Field(description="List of authors to search for papers", default=[])
    field_of_study: str = Field(description="Comma separated list of field of study to search for papers")
    rewritten_query: str = Field(description="The rewritten simplified query")
    rewritten_query_for_keyword_search: str = Field(description="The rewritten query for keyword search")


def moderation_api(text: str) -> bool:
    response = moderation(text, model="omni-moderation-latest")
    return response.results[0].flagged


def validate(query: str) -> None:
    # self.update_task_state(task_id, "Validating the query")
    logger.info(
        "Checking query for malicious content with moderation api..."
    )
    try:
        if moderation_api(query):
            raise Exception(
                "The input query contains harmful content."
            )
    except Exception as e:
        logger.error(f"Query validation failed, {e}")
        raise e
    logger.info(f"{query} is valid")


def decompose_query(query: str, decomposer_llm_model: str, **llm_kwargs) -> Tuple[LLMProcessedQuery, CompletionResult]:
    search_filters = dict()
    decomp_query_res = None
    try:
        # decompose query to get llm re-written and keyword query with filters
        decomp_query_res = llm_completion(user_prompt=query, system_prompt=QUERY_DECOMPOSER_PROMPT,
                                          model=decomposer_llm_model, response_format=DecomposedQuery, **llm_kwargs)
        decomposed_query = json.loads(decomp_query_res.content)
        decomposed_query = {k: str(v) if type(v) == int else v for k, v in decomposed_query.items()}
        decomposed_query = DecomposedQuery(**decomposed_query)
        logger.info(f"Decomposed query: {decomposed_query}")
        rewritten_query, keyword_query = decomposed_query.rewritten_query, decomposed_query.rewritten_query_for_keyword_search
        if decomposed_query.earliest_search_year or decomposed_query.latest_search_year:
            search_filters["year"] = f"{decomposed_query.earliest_search_year}-{decomposed_query.latest_search_year}"
        if decomposed_query.venues:
            search_filters["venue"] = decomposed_query.venues
        if decomposed_query.field_of_study:
            search_filters["fieldsOfStudy"] = decomposed_query.field_of_study
    except Exception as e:
        logger.error(f"Error while decomposing query: {e}")
        rewritten_query = query
        keyword_query = ""
        decomp_query_res = decomp_query_res._replace(model=f"error-{decomp_query_res.model}")

    return LLMProcessedQuery(rewritten_query=rewritten_query, keyword_query=keyword_query,
                             search_filters=search_filters), decomp_query_res
