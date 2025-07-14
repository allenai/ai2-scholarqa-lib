import logging
import json
from pydantic import BaseModel
from typing import List, Dict

from scholarqa.table_generation.prompts import ATTRIBUTE_PROMPT, SYSTEM_PROMPT
from scholarqa.utils import get_paper_metadata
from scholarqa.llms.litellm_helper import CostAwareLLMCaller, CostReportingArgs, llm_completion
from scholarqa.llms.constants import GPT_4o

logger = logging.getLogger(__name__)

# Definition of output format to be provided during json mode
class Column(BaseModel):
    name: str
    definition: str
    is_metadata: bool

class ColumnSuggestions(BaseModel):
    columns: List[Column]

def retrieve_paper_info(corpus_ids: List[str]) -> Dict:
    """ 
    Given a set of corpus IDs for papers to be added to the table,
    retrieve titles and abstracts for each paper using the
    Semantic Scholar batch querying API.
    """
    paper_metadata = get_paper_metadata(corpus_ids)
    paper_info = {corpus_id: paper_metadata[corpus_id] if corpus_id in paper_metadata else {} for corpus_id in corpus_ids}
    return paper_info


def format_paper_info(paper_info: Dict) -> str:
    """
    Given titles and abstracts of all papers in the table,
    format this information to be appended to the column suggestion prompt.
    """
    formatted_paper_info = ""
    for index, corpus_id in enumerate(paper_info):
        paper = paper_info[corpus_id]
        title = paper["title"] if "title" in paper else None
        abstract = paper["abstract"].strip() if "abstract" in paper and paper["abstract"] else None
        formatted_paper_info += f'Paper {index+1} title: {title}\nPaper {index+1} abstract: {abstract}\n\n'
    return formatted_paper_info


def generate_final_prompt(query: str, formatted_paper_info: str, column_num: str) -> str:
    """
    Given the formatted paper information, and an optional user query,
    generate the final column suggestion prompt to be sent to the LLM.
    """
    final_prompt = ATTRIBUTE_PROMPT.format(query, column_num, formatted_paper_info)
    return final_prompt


def generate_attribute_suggestions(
        corpus_ids: List[str], 
        model: str = GPT_4o, 
        query: str = None,
        column_num: int = 10,
        llm_caller: CostAwareLLMCaller = None,
        cost_args: CostReportingArgs = None,
    ) -> Dict:
    """
    Entry point to the column suggestion generation process.
    """
    # Step 1: Retrieve user query or backoff to the default query
    default_user_query = "Brief Overview and Comparison of Following Papers"
    user_query = query if query is not None else default_user_query

    # Step 2: Retrieve titles and abstracts for all provided papers
    paper_info = retrieve_paper_info(corpus_ids)
    
    # Step 3: Format all paper titles and abstracts to add to prompt
    formatted_paper_info = format_paper_info(paper_info)
    
    # Step 4: Produce final column generation prompt from papers and user query
    final_prompt = generate_final_prompt(user_query, formatted_paper_info, column_num)

    # Step 5: Prompt the LLM to produce column suggestions
    column_suggestion_params = {
        "user_prompt": final_prompt,
        "system_prompt": SYSTEM_PROMPT,
        "response_format": ColumnSuggestions,
        "model": model,
    }
    output = llm_caller.call_method(
        cost_args=cost_args,
        method=llm_completion,
        **column_suggestion_params,
    )
    column_suggestions = json.loads(output.result.content)["columns"]
    cost_dict = {
        "cost_value": output.result.cost,
        "tokens": {
            "total": output.result.total_tokens,
            "prompt": output.result.input_tokens,
            "completion": output.result.output_tokens,
            "reasoning": output.result.reasoning_tokens
        },
        "model": output.result.model,
    }

    return {"columns": column_suggestions, "cost": cost_dict}