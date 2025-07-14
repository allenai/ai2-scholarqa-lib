import os
import json
import requests
import time
import logging

from typing import List, Dict
from pydantic import BaseModel
import itertools
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from scholarqa.table_generation.prompts import *
from scholarqa.utils import get_paper_metadata
from scholarqa.llms.constants import *
from scholarqa.rag.retrieval import PaperFinder
from scholarqa.llms.litellm_helper import CostAwareLLMCaller, CostReportingArgs, llm_completion

logger = logging.getLogger(__name__)

class PaperQAAnswer(BaseModel):
    answer: str
    exceprts: List[str]

def get_cost_object(completion: CompletionResult) -> dict:
    cost_dict = {
        "cost_value": completion.cost,
        "tokens": {
            "total": completion.total_tokens,
            "prompt": completion.input_tokens,
            "completion": completion.output_tokens,
            "reasoning": completion.reasoning_tokens
        },
        "model": completion.model,
    }
    return cost_dict

def get_metadata_columns(
        question: str, 
        metadata: dict, 
        model: str,
        llm_caller: CostAwareLLMCaller = None,
        cost_args: CostReportingArgs = None,
    ):
    """
    Given a question and metadata from a research paper, prompt
    an LLM to answer the question using the metadata provided. 
    We use this to populate metadata columns in tables (e.g., venue).
    """
    prompt = VALUE_GENERATION_FROM_METADATA.format(question)
    prompt += f"Metadata: {metadata}"
    cur_cost_args = deepcopy(cost_args)
    corpus_id = metadata.get("corpusId", None)
    cur_cost_args = CostReportingArgs(
        task_id=cost_args.task_id,
        user_id=cost_args.user_id,
        msg_id=cost_args.msg_id,
        description=cost_args.description + f" for corpus ID {corpus_id}",
        model=cost_args.model,
    )
    value_generation_params = {
        "user_prompt": prompt,
        "system_prompt": SYSTEM_PROMPT,
        "model": model,
        "fallback": GPT_4o,
    }
    output = llm_caller.call_method(
        cost_args=cur_cost_args,
        method=llm_completion,
        **value_generation_params,
    )
    response = output.result.content
    cost_dict = get_cost_object(output.result)
    response_simplified = {
        "question": question,
        "answer": response,
        "corpusId": metadata.get("corpusId", None),
        "source": "metadata",
        "evidenceId": None,
        "cost": cost_dict,
    }
    return response_simplified


def get_value_from_abstract(
        question: str, 
        corpus_id: str, 
        model: str,
        llm_caller: CostAwareLLMCaller = None,
        cost_args: CostReportingArgs = None,
    ):
    """
    Given a query and a paper's corpus ID, retrieve an answer
    to the query based on the paper abstract. We use this as a
    backoff strategy for papers without full-text access.
    """
    # Step 1: Retrieve abstract for provided corpus ID from Semantic Scholar API
    response = None
    retry_num = 0
    while response is None:
        try:
            response = get_paper_metadata([corpus_id])
        except Exception as e:
            logger.error(f"Error while retrieving paper metadata for corpus ID {corpus_id}: {str(e)}")
        retry_num += 1
        time.sleep(retry_num * 5)
    response_content = response[corpus_id]
    title = response_content["title"] if "title" in response_content else None
    abstract = response_content["abstract"] if "abstract" in response_content and response_content["abstract"] else None
    # Step 2: Prompt LLM to produce a cell value using the paper abstract
    prompt = VALUE_GENERATION_FROM_ABSTRACT + f"Paper title:{title}\nPaper abstract: {abstract}\nQuestion: {question}\nAnswer:"
    cur_cost_args = CostReportingArgs(
        task_id=cost_args.task_id,
        user_id=cost_args.user_id,
        msg_id=cost_args.msg_id,
        description=cost_args.description + f" for corpus ID {corpus_id}",
        model=cost_args.model,
    )
    value_generation_params = {
        "user_prompt": prompt,
        "system_prompt": SYSTEM_PROMPT,
        "model": model,
        "fallback": GPT_4o,
    }
    output = llm_caller.call_method(
        cost_args=cur_cost_args,
        method=llm_completion,
        **value_generation_params,
    )
    value = output.result.content
    cost_dict = get_cost_object(output.result)
    return value, cost_dict


def run_paper_qa(
        question: str, 
        corpus_id: str, 
        model: str,
        paper_finder: PaperFinder = None,
        llm_caller: CostAwareLLMCaller = None,
        cost_args: CostReportingArgs = None,
    ):
    """
    Given a query and a paper's corpus ID, retrieve an answer
    to the query from the paper full-text. This function relies
    on Vespa snippet search utility to first retrieve relevant
    passages from the paper full-text to produce answers and 
    evidence snippets backing them. If we are unable to find
    relevant snippets, we back off to generating an answer from
    the paper's abstract.
    """
    try:
        # Restrict snippet search only to the paper we're currently 
        # generating values for. Also drop formatting instructions
        # from the question for the retrieval function.
        filter_kwargs = {
            "paperIds" : f"CorpusId:{corpus_id}"
        }
        snippets = paper_finder.retrieve_passages(
            query=question.split("Only return the answer. ")[0], 
            **filter_kwargs,
        )
        if snippets:
            paper_title = snippets[0]["title"]
            concatenated_snippets = ""
            for i, snippet in enumerate(snippets):
                concatenated_snippets += f"Snippet {i+1}: {snippet['text']}\n\n"
            prompt = VESPAQA_PROMPT.replace('[TITLE]', paper_title)
            prompt = prompt.replace('[SNIPPETS]', concatenated_snippets)
            prompt = prompt.replace('[QUESTION]', question)
            cur_cost_args = CostReportingArgs(
                task_id=cost_args.task_id,
                user_id=cost_args.user_id,
                msg_id=cost_args.msg_id,
                description=cost_args.description + f" for corpus ID {corpus_id}",
                model=cost_args.model,
            )
            value_generation_params = {
                "user_prompt": prompt,
                "system_prompt": SYSTEM_PROMPT,
                "model": model,
                "fallback": GPT_4o,
                "response_format": PaperQAAnswer, 
            }
            output = llm_caller.call_method(
                cost_args=cur_cost_args,
                method=llm_completion,
                **value_generation_params,
            )
            # print(json.loads(output.result.content))
            response_simplified = {
                "question": question,
                "answer": json.loads(output.result.content)["answer"],
                "corpusId": corpus_id,
                "source": "vespa-snippets",
                "evidenceId": json.loads(output.result.content).get("exceprts", []),
                "cost": get_cost_object(output.result),
            }
            # print(response_simplified)
        else:
            response, cost = get_value_from_abstract(
                question=question, 
                corpus_id=corpus_id, 
                model=model,
                llm_caller=llm_caller,
                cost_args=cost_args,
            )
            response_simplified = {
                "question": question,
                "answer": response,
                "corpusId": corpus_id,
                "source": "abstract",
                "evidenceId": None,
                "cost": cost,
            }
    except Exception as e:
        logger.error(f"Exception while hitting vespa snippet search endpoint: {str(e)}")
        response_simplified = {"error": f"Exception while hitting vespa snippet search endpoint: {str(e)}"}
    return response_simplified


def generate_value_suggestions(
        column_name: str,
        column_def: str,
        corpus_ids: List[str],
        is_metadata: bool = False,
        model: str = "openai/gpt-4o-2024-08-06",
        paper_finder: PaperFinder = None,
        llm_caller: CostAwareLLMCaller = None,
        cost_args: CostReportingArgs = None,
) -> Dict:
    """
    Entry point to the cell value generation process.
    Given the name and definition of a column, we will
    produce values to be added to that column for each 
    paper in the table.
    """
    # Variables to track generated values, evidence and costs.
    cell_values = []
    evidence_ids = {}
    total_cost = 0.0

    # Setting #threads to parallelize value generation
    MAX_THREADS = 1
    # Setting snippeet search retrieval limit to 10 passages per paper
    paper_finder.retriever.n_retrieval = 10

    # Step 1: First, we check if the column to be populated is metadata-based.
    if is_metadata == "True":
        # If yes, we call the Semantic Scholar API to retrieve all metadata 
        # for each paper and construct a JSON blob containing this data.
        results = get_paper_metadata(corpus_ids)
        results = [results[x] if x in results else {} for x in corpus_ids]
        # We produce a query from the provided column name and definition.
        question = f"{column_name}, defined as {column_def}"
        raw_values = {}
        non_na_corpus_ids = []
    
        # We call our metadata-based value generation function with this query.
        # This is executed in parallel for all papers in the table for speed.
        # In addition to answers and costs, we also store corpus IDs for all papers
        # that have answers for the query (i.e., non-N/A values).
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            responses = list(executor.map(
                get_metadata_columns, 
                itertools.repeat(question), 
                results, 
                itertools.repeat(model),
                itertools.repeat(llm_caller),
                itertools.repeat(cost_args),
            ))
            raw_values = {y: x["answer"] if "answer" in x else "N/A" for x,y in zip(responses, corpus_ids)}
            non_na_corpus_ids = [x for x, y in raw_values.items() if y != 'N/A']
            per_cell_costs = {y: x.get("cost", None) for x,y in zip(responses, corpus_ids)}
    else:
        # For non-metadata column to be populated, we run value extraction 
        # on full-texts (backing off to abstracts) for all papers.
        # Step 1: Construct a query for value extraction using the column description.
        paperqa_query = f"Summarize this paper on the aspect of \"{column_name}\""
        if column_def != '':
            paperqa_query += f" by describing \"{column_def}\". "
            paperqa_query += "Only return the answer. Do not repeat the question or add any surrounding text. "
            paperqa_query += "The answer should be brief (fewer than 20 words) and need not be a complete sentence."
            paperqa_query += "Do not start the answer with references like \"this paper describes...\"."
        else:
            paperqa_query += f"."

        # Step 2: We call our QA-based value generation function with this query.
        # This is also executed in parallel for all papers, with storage of answers, evidence,
        # costs and corpus IDs for all papers that have answers for the query (i.e., non-N/A values).
        raw_values = {}
        non_na_corpus_ids = []
    
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            responses = list(executor.map(
                run_paper_qa, 
                itertools.repeat(paperqa_query), 
                corpus_ids, 
                itertools.repeat(model),
                itertools.repeat(paper_finder),
                itertools.repeat(llm_caller),
                itertools.repeat(cost_args),
            ))
            raw_values = {y: x["answer"] if "answer" in x else "No response" for x,y in zip(responses, corpus_ids)}
            non_na_corpus_ids = [x for x, y in raw_values.items() if y != 'N/A']
            evidence_ids = {y: x["evidenceId"] for x,y in zip(responses, corpus_ids) if "evidenceId" in x}
            per_cell_costs = {y: x.get("cost", None) for x,y in zip(responses, corpus_ids)}

    # Step 3: Construct final JSON blobs for each cell value containing answers
    # and evidence which can both be displayed on the UI.
    for k in non_na_corpus_ids:
        cell_value = {
            "corpusId": k,
            "displayValue": raw_values[k],
        }
        if evidence_ids and k in evidence_ids:
            cell_value['metadata'] = {
                "evidence": evidence_ids[k],
            }
        cell_values.append(cell_value)
    
    for k in corpus_ids:
        if  k not in non_na_corpus_ids:
            cell_value = {
            "corpusId": k,
            "displayValue": "N/A",
            }
            if evidence_ids and k in evidence_ids:
                cell_value['metadata'] = {
                    "evidence": evidence_ids[k],
                }
            cell_values.append(cell_value)

    return {"cell_values": cell_values, "cost": per_cell_costs}