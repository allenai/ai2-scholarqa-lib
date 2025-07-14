import json
import logging
import re
from enum import Enum
from typing import Tuple, Dict, List, Any, Generator

import pandas as pd
from pydantic import BaseModel, Field

from scholarqa.llms.constants import GPT_4o
from scholarqa.llms.litellm_helper import batch_llm_completion, llm_completion
from scholarqa.llms.prompts import USER_PROMPT_PAPER_LIST_FORMAT, USER_PROMPT_QUOTE_LIST_FORMAT, \
    PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY
from scholarqa.llms.constants import CompletionResult

logger = logging.getLogger(__name__)


class DimFormat(str, Enum):
    SYNTHESIS = "synthesis"
    LIST = "list"


class Dimension(BaseModel):
    name: str = Field(default=None, description=(
        "The name of the dimension"
    ))
    format: DimFormat = Field(default=None, description=(
        "The generation format of the dimension - can be either list of synthesis"
    ))
    quotes: List[int] = Field(default=None, description=(
        "A list of indices of paper quotes in the dimension, can be empty if no relevant quotes are found"
    ))


class ClusterPlan(BaseModel):
    cot: str = Field(default=None, description=(
        "The justification for every dimension name and its format"
    ))
    dimensions: List[Dimension] = Field(default=None, description=(
        "The list of dimensions along with the associated quote indices as per the cot plan"
    ))


class MultiStepQAPipeline:
    def __init__(self, llm_model: str, fallback_llm: str = GPT_4o, batch_workers: int = 20,
                 **llm_kwargs):
        self.llm_model = llm_model
        self.fallback_llm = fallback_llm
        self.batch_workers = batch_workers
        self.llm_kwargs = {"max_tokens": 4096*4}
        if llm_kwargs:
            self.llm_kwargs.update(llm_kwargs)


    def step_select_quotes(self, query: str, scored_df: pd.DataFrame, sys_prompt: str) -> Tuple[
        Dict[str, str], List[CompletionResult]]:

        logger.info(
            f"Querying {self.llm_model} to extract quotes from these papers with {self.batch_workers} parallel workers")
        tup_items = {k: v for k, v in
                     zip(scored_df["reference_string"], scored_df["relevance_judgment_input_expanded"])}
        messages = [USER_PROMPT_PAPER_LIST_FORMAT.format(query, v) for k, v in tup_items.items()]
        completion_results = batch_llm_completion(self.llm_model, messages=messages, system_prompt=sys_prompt,
                                                  max_workers=self.batch_workers, fallback=self.fallback_llm,
                                                  **self.llm_kwargs)
        quotes = [
            cr.content if cr.content != "None" and not cr.content.startswith("None\n") and not cr.content.startswith(
                "None ")
            else "" for cr in completion_results]
        per_paper_summaries = {t[0]: quote for t, quote in zip(tup_items.items(), quotes) if len(quote) > 10}
        per_paper_summaries = dict(sorted(per_paper_summaries.items(), key=lambda x: x[0]))
        return per_paper_summaries, completion_results

    def step_clustering(self, query: str, per_paper_summaries: Dict[str, str],
                        sys_prompt: str) -> Tuple[Dict[str, Any], CompletionResult]:
        def make_prompt(query: str, paper_paper_quotes_dict: Dict[str, str]) -> str:
            # paper_paper_quotes_dict is a dictionary with keys being the paper titles and values being the quotes
            # need to make a single string with all of the quotes
            quotes = ""
            for idx, (paper, quotes_str) in enumerate(paper_paper_quotes_dict.items()):
                # there are multiple quotes per paper
                quotes_str = quotes_str.replace("\n", "")
                quotes += f"[{idx}]\t{quotes_str}" + "\n"
            prompt = USER_PROMPT_QUOTE_LIST_FORMAT.format(query, quotes)
            return prompt

        user_prompt = make_prompt(query, per_paper_summaries)
        try:
            #params for reasoning mode: max_completion_tokens=4096, max_tokens=4096+1024, reasoning_effort="low"
            response = llm_completion(user_prompt=user_prompt,
                                      system_prompt=sys_prompt, fallback=self.fallback_llm, model=self.llm_model,
                                      response_format= ClusterPlan, **self.llm_kwargs
                                      )
            return json.loads(response.content), response
        except Exception as e:
            logger.warning(f"Error while clustering with {self.llm_model}: {e}")
            raise e

    def generate_iterative_summary(self, query: str, per_paper_summaries_extd: Dict[str, Dict[str, Any]],
                                   plan: Dict[str, Any],
                                   sys_prompt: str) -> Generator[CompletionResult, None, None]:
        # first, we need to make a map from the index to the quotes because the llm is using index only

        # now fill in the prompt
        per_paper_summaries_tuples = [(ref_string, response) for ref_string, response in
                                      per_paper_summaries_extd.items()]
        # only use the section headings from the plan, discard the quote indices
        plan_str = "\n".join([k for k in plan])
        existing_sections = []
        i = 0
        for section_name, inds in plan.items():
            # inds are a string like this: "[1, 2, 3]"
            # get the quotes for each index
            quotes = ""
            for ind in inds:
                if ind < len(per_paper_summaries_tuples):
                    quotes += (
                            per_paper_summaries_tuples[ind][0] + ": " + str(per_paper_summaries_tuples[ind][
                                                                                1]) + "\n"
                    )
                else:
                    logger.warning(f"index {ind} out of bounds")
            # existing sections should have their summaries removed because they are confusing.
            # remove anything in []
            already_written = "\n\n".join(existing_sections)
            already_written = re.sub(r"\[.*?\]", "", already_written)
            fill_in_prompt_args = {
                "query": query,
                "plan": plan_str,
                "already_written": already_written,
                "section_name": section_name}
            if quotes:
                fill_in_prompt_args["section_references"] = quotes
                filled_in_prompt = sys_prompt.format(**fill_in_prompt_args)
            else:
                logger.warning(f"No quotes for section {section_name}")
                filled_in_prompt = PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY.format(**fill_in_prompt_args)

            response = llm_completion(user_prompt=filled_in_prompt, model=self.llm_model, fallback=self.fallback_llm,
                                      **self.llm_kwargs)
            existing_sections.append(response.content)
            yield response
