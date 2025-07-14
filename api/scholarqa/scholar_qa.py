import logging
import os
import re
from threading import Thread
from time import time
from typing import List, Any, Dict, Tuple, Generator
from uuid import uuid4

import pandas as pd
from anyascii import anyascii
from langsmith import traceable

from scholarqa.config.config_setup import LogsConfig
from scholarqa.llms.constants import CostAwareLLMResult, GPT_4o
from scholarqa.llms.litellm_helper import CLAUDE_37_SONNET, CostAwareLLMCaller, CostReportingArgs
from scholarqa.llms.prompts import SYSTEM_PROMPT_QUOTE_PER_PAPER, SYSTEM_PROMPT_QUOTE_CLUSTER, PROMPT_ASSEMBLE_SUMMARY
from scholarqa.models import GeneratedSection, TaskResult, ToolRequest, CitationSrc
from scholarqa.postprocess.json_output_utils import get_json_summary
from scholarqa.preprocess.query_preprocessor import validate, decompose_query, LLMProcessedQuery
from scholarqa.rag.multi_step_qa_pipeline import MultiStepQAPipeline
from scholarqa.rag.retrieval import PaperFinder
from scholarqa.state_mgmt.local_state_mgr import AbsStateMgrClient, LocalStateMgrClient
from scholarqa.trace.event_traces import EventTrace
from scholarqa.utils import get_paper_metadata, NUMERIC_META_FIELDS, CATEGORICAL_META_FIELDS, get_ref_author_str, \
    make_int
from scholarqa.table_generation.table_model import TableWidget
from scholarqa.table_generation.table_generator import TableGenerator

logger = logging.getLogger(__name__)

# Regular expressions to fix weird formatting issues cause after citation linking in the evidences
CLOSE_BRACKET_PATTERN = r'(?<![\[|,\s*\d])(\d+\])'  # (Doe et al., 2024)10] --> (Doe et al., 2024)[10]
OPEN_BRACKET_PATTERN = r"(\[[\d+,]+),(?=[^\[]*$)"  # [8,9,(Doe et al., 2024) --> [8,9](Doe et al., 2024)


class ScholarQA:
    def __init__(
            self,
            # Required for webapp since a new process is created for each request, for library task_id can be None initially and assigned for each request as below
            paper_finder: PaperFinder,
            task_id: str = None,
            llm_model: str = CLAUDE_37_SONNET,
            multi_step_pipeline: MultiStepQAPipeline = None,
            state_mgr: AbsStateMgrClient = None,
            logs_config: LogsConfig = None,
            run_table_generation: bool = True,
            llm_kwargs: Dict[str, Any] = None,
            **kwargs
    ):
        if logs_config:
            self.logs_config = logs_config
        else:
            logger.info("initializing the log configs")
            self.logs_config = LogsConfig(llm_cache_dir="lib_llm_cache")
            self.logs_config.init_formatter()

        self.task_id = task_id
        self.paper_finder = paper_finder
        self.llm_model = llm_model
        fallback_llm = kwargs.get("fallback_llm", GPT_4o)
        self.validate = kwargs.get("validate", "OPENAI_API_KEY" in os.environ)
        if not self.validate:
            logger.warning("Validation of the query for harmful content is turned off")
        self.decomposer_llm = kwargs.get("decomposer_llm", self.llm_model)
        self.state_mgr = state_mgr if state_mgr else LocalStateMgrClient(self.logs_config.log_dir)
        self.llm_caller = CostAwareLLMCaller(self.state_mgr)
        self.llm_kwargs = llm_kwargs if llm_kwargs else dict()
        if not multi_step_pipeline:
            logger.info(f"Creating a new MultiStepQAPipeline with model: {llm_model} for all the steps")
            self.multi_step_pipeline = MultiStepQAPipeline(self.llm_model, fallback_llm=fallback_llm, **self.llm_kwargs)
        else:
            self.multi_step_pipeline = multi_step_pipeline

        self.tool_request = None
        self.table_llm = kwargs.get("table_llm", self.llm_model)
        self.table_generator = TableGenerator(paper_finder=paper_finder, llm_caller=self.llm_caller)
        self.run_table_generation = run_table_generation

    def update_task_state(
            self,
            status: str,
            step_estimated_time: int = 0,
            curr_response: List[GeneratedSection] = None,
            task_estimated_time: str = None,
    ):
        logger.info(status)
        if self.task_id and self.tool_request:
            self.state_mgr.update_task_state(
                self.task_id,
                self.tool_request,
                status,
                step_estimated_time,
                curr_response,
                task_estimated_time,
            )

    @traceable(name="Preprocessing: Validate and decompose user query")
    def preprocess_query(self, query: str, cost_args: CostReportingArgs=None, ) -> CostAwareLLMResult:
        if self.validate:
            # Validate the query for harmful/unanswerable content
            validate(query)
        # Decompose the query to get filters like year, venue, fos, citations, etc along with a re-written
        # version of the query and a query suitable for keyword search.
        llm_args = {"max_tokens": 4096*2}
        if self.llm_kwargs:
            llm_args.update(self.llm_kwargs)
        return self.llm_caller.call_method(
            cost_args=cost_args, method=decompose_query, query=query, decomposer_llm_model=self.decomposer_llm,
            **llm_args
        )

    @traceable(name="Retrieval: Find relevant paper passages for the query")
    def find_relevant_papers(self, llm_processed_query: LLMProcessedQuery, **kwargs) -> Tuple[
        List[Dict[str, Any]], List[Dict[str, Any]]]:
        # retrieval from vespa index
        start = time()
        rewritten_query = llm_processed_query.rewritten_query
        keyword_query = llm_processed_query.keyword_query
        self.update_task_state(
            f"Retrieving relevant passages from a corpus of 8M+ open access papers",
            step_estimated_time=5
        )
        # Get relevant paper passages from the Semantic Scholar index for the llm rewritten query
        snippet_results = self.paper_finder.retrieve_passages(query=rewritten_query,
                                                              **llm_processed_query.search_filters,
                                                              **kwargs)
        snippet_corpus_ids = {snippet["corpus_id"] for snippet in snippet_results}
        self.update_task_state(f"Retrieved {len(snippet_results)} highly relevant passages", step_estimated_time=1)

        if keyword_query:
            # Get additional papers from the Semantic Scholar api via keyword search
            search_api_results = self.paper_finder.retrieve_additional_papers(keyword_query,
                                                                              **llm_processed_query.search_filters)
            search_api_results = [item for item in search_api_results if item["corpus_id"] not in snippet_corpus_ids]
            self.update_task_state(
                f"Retrieved {len(search_api_results)} more papers from Semantic Scholar abstracts using keyword search",
                step_estimated_time=1)
        else:
            search_api_results = []
        logger.info("Retrieval time: %.2f", time() - start)

        return snippet_results, search_api_results

    @traceable(name="Retrieval: Rerank the passages and aggregate at paper level")
    def rerank_and_aggregate(self, user_query: str, retrieved_candidates: List[Dict[str, Any]], filter_paper_metadata: [
        Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.paper_finder.n_rerank > 0:
            self.update_task_state(
                f"Further re-rank and aggregate passages to focus on up to top {self.paper_finder.n_rerank} papers",
                step_estimated_time=10)
        start = time()
        reranked_candidates = self.paper_finder.rerank(user_query, retrieved_candidates)
        logger.info("Reranking time: %.2f", time() - start)
        paper_metadata = filter_paper_metadata
        paper_metadata.update(get_paper_metadata(
            {snippet["corpus_id"] for snippet in reranked_candidates if
             snippet["corpus_id"] not in filter_paper_metadata}))
        agg_df = self.paper_finder.aggregate_into_dataframe(reranked_candidates, paper_metadata)
        self.update_task_state(
            f"Found {len(agg_df)} highly relevant papers after re-ranking and aggregating",
            step_estimated_time=1)
        logger.info("Reranking w. formatting time: %.2f", time() - start)
        return agg_df, paper_metadata

    @traceable(name="Generation: Extract relevant quotes from paper passages or filter")
    def step_select_quotes(self, query: str, scored_df: pd.DataFrame, cost_args: CostReportingArgs = None,
                           sys_prompt: str = SYSTEM_PROMPT_QUOTE_PER_PAPER) -> CostAwareLLMResult:
        logger.info("Running Step 1 - quote extraction")
        self.update_task_state("Extracting salient key statements from papers",
                               step_estimated_time=15)
        logger.info(
            f"{scored_df.shape[0]} papers with relevance_judgement >= {self.paper_finder.context_threshold} to start with.")
        start = time()
        cost_args = cost_args._replace(model=self.multi_step_pipeline.llm_model)._replace(
            description="Corpus QA Step 1: Quote extraction")
        per_paper_summaries = self.llm_caller.call_method(cost_args, self.multi_step_pipeline.step_select_quotes,
                                                          query=query, scored_df=scored_df,
                                                          sys_prompt=sys_prompt)
        api_corpus_ids = set(
            scored_df[scored_df.sentences.apply(lambda x: not x)].corpus_id.astype(str))
        ref_strs = {rs.split(" | ")[0][1:] for rs in per_paper_summaries.result}
        logger.info(f"Paper abstracts used from s2 api: {api_corpus_ids.intersection(ref_strs)}")

        logger.info(
            f"Step 1 done - {len(per_paper_summaries.result)} papers with quotes extracted, cost: {per_paper_summaries.tot_cost}, "
            f"time: {time() - start:.2f}")
        return per_paper_summaries

    @traceable(name="Generation: Cluster quotes to generate an organization plan")
    def step_clustering(self, query: str, per_paper_summaries: Dict[str, str], cost_args: CostReportingArgs = None,
                        sys_prompt: str = SYSTEM_PROMPT_QUOTE_CLUSTER) -> CostAwareLLMResult:
        logger.info("Running Step 2: Clustering the extracted quotes into meaningful dimensions")
        self.update_task_state("Synthesizing an answer outline based on extracted quotes", step_estimated_time=15)
        start = time()
        cost_args = cost_args._replace(model=self.multi_step_pipeline.llm_model)._replace(
            description="Corpus QA Step 2: Clustering quotes into dimensions")
        cluster_json = self.llm_caller.call_method(cost_args, self.multi_step_pipeline.step_clustering,
                                                   query=query, per_paper_summaries=per_paper_summaries,
                                                   sys_prompt=sys_prompt)
        logger.info(f"Step 2 done - {cluster_json.result}, cost: {cluster_json.tot_cost}, time: {time() - start:.2f}")
        return cluster_json

    @traceable(name="Generation: Generate an iterative summary")
    def step_gen_iterative_summary(self, query: str, per_paper_summaries: Dict[str, str],
                                   plan_json: Dict[str, Any], cost_args: CostReportingArgs = None,
                                   sys_prompt: str = PROMPT_ASSEMBLE_SUMMARY) -> Generator[
        str, None, CostAwareLLMResult]:
        logger.info("Running Step 3: Assemble the summary with the links (takes ~2 mins)")
        start = time()

        cost_args = cost_args._replace(model=self.multi_step_pipeline.llm_model)._replace(
            description="Corpus QA Step 3: Generating summarized answer")
        sec_generator = self.llm_caller.call_iter_method(cost_args, self.multi_step_pipeline.generate_iterative_summary,
                                                         query=query, per_paper_summaries_extd=per_paper_summaries,
                                                         plan=plan_json, sys_prompt=sys_prompt)
        try:
            while True:
                response = next(sec_generator)
                yield response.content
        except StopIteration as e:
            return_val = e.value
        if return_val:
            logger.info(f"Step 3 done, cost: {return_val.tot_cost}, time: {time() - start:.2f}")
        return return_val

    @staticmethod
    def passage_to_quotes_metadata(retrieval_df: pd.DataFrame, per_paper_summaries: Dict[str, str],
                                   plan_json: Dict[str, List[int]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Map the quotes extracted in step 1 `per_paper_summaries` to their actual full length versions
        in retrieval_df and make a map of quote ref string --> [inline citations].

        i) First parse the plan json to get the required paper indices to be sent for the final generation in step 3.
        ii) For those papers, find the quotes from per_paper_summaries and passages from retrieval_df.
        iii) For each quote, first check if the paper metadata from retrieval_df consists of passages, if not, then the quotes are taken from
        abstracts of keyword results.
        iii) Else, try to match the paper quotes as substrings to the corresponding paper passages to get the inline citations along with
        offsets.
        iv) First try matching the raw string (high precision), if unsuccessful, try matching only the alphabet characters (high recall).
        v) Once a match is found, use its offsets in the passage and iterate over the sentence offsets/inline citations to find any that occur within the snippet.
        vi) In case of a raw string match, replace any citation mentions (if possible), with the paper id of the corresponding inline citation to be linked later.
        """
        # get all the ref strings for the clutering plan generated in step 2
        ref_str_list = [k for k in per_paper_summaries]
        # paper identifiers for the selected quotes in the plan obtained from their corresponding index
        req_ref_strs = {ref_str_list[item] for sublist in plan_json.values() for item in sublist if
                        item < len(ref_str_list)}
        quotes_metadata = dict()
        if req_ref_strs:
            # filter the quotes according to the plan
            reqd_paper_summaries = {k: v for k, v in per_paper_summaries.items() if k in req_ref_strs}
            # filter the dataframe according to the plan
            reqd_ref_df = retrieval_df[retrieval_df["reference_string"].apply(lambda x: x in req_ref_strs)].copy()
            # remove all special characters from the passages in the dataframe for approximate match
            reqd_ref_df["sentence_alpha"] = reqd_ref_df["sentences"].apply(
                lambda x: [re.sub(r'[^a-zA-Z]', '', sentence["text"]).lower() for sentence in x])
            # iterate over the reqd_ref_df and get the snippets for each row from reqd_paper_summaries
            for row_idx, row in reqd_ref_df.iterrows():
                ref_str, sentences, sent_alpha = row["reference_string"], row["sentences"], row["sentence_alpha"]
                mapped_quotes = []

                curr_reqd_quotes = reqd_paper_summaries[ref_str].split("...")
                curr_reqd_quotes_reg = [re.sub(r'[^a-zA-Z]', '', quote).lower() for quote in curr_reqd_quotes]
                for idx, (quote, quote_reg) in enumerate(zip(curr_reqd_quotes, curr_reqd_quotes_reg)):
                    new_quote = quote.strip()
                    curr_quote_map = {"quote": new_quote, "section_title": "abstract",
                                      "pdf_hash": "", } if not sentences else dict()

                    shift = 0  # keep track of changes to the quote offsets when the inline citations are modified
                    for sidx, sentence in enumerate(sentences):
                        # can lookup exact string now since we prompt the llm to include the citations in the quotes
                        lookup_idx = sentence["text"].lower().find(quote.lower().strip())
                        raw_match = lookup_idx >= 0
                        if not raw_match:
                            lookup_idx = sent_alpha[sidx].find(quote_reg)
                        if lookup_idx >= 0:
                            lookup_end = lookup_idx + len(quote)
                            curr_quote_map["section_title"] = sentence["section_title"]
                            curr_quote_map["pdf_hash"] = sentence["pdf_hash"]
                            curr_quote_map["start"], curr_quote_map["end"] = lookup_idx, lookup_end
                            curr_quote_map["sentence_offsets"], curr_quote_map["ref_mentions"] = [], []
                            if sentence.get("sentence_offsets"):
                                for sidx, soff in enumerate(sentence["sentence_offsets"]):
                                    # check if the sentence offset is within the range of the quote
                                    # the sentence can be completely or partially inside the quote
                                    if (lookup_idx < soff["end"] <= lookup_end) or (
                                            lookup_idx <= soff["start"] < lookup_end)\
                                            or (soff["start"] <= lookup_idx and lookup_end <= soff["end"]):
                                        curr_quote_map["sentence_offsets"].append(soff)
                            if sentence.get("ref_mentions"):
                                for sref in sentence["ref_mentions"]:
                                    if sref.get("matchedPaperCorpusId") and lookup_idx <= sref.get(
                                            "start") and sref.get("end") <= lookup_end:
                                        curr_quote_map["ref_mentions"].append(sref["matchedPaperCorpusId"])
                                        if raw_match:
                                            new_start, new_end = sref["start"] - lookup_idx, sref["end"] - lookup_idx
                                            cite_str = f"({sref['matchedPaperCorpusId']})"
                                            new_quote = new_quote[:new_start + shift] + cite_str + new_quote[
                                                                                                   new_end + shift:]
                                            shift += (len(cite_str) - sref["end"] + sref["start"])
                                # curr_inline_citations.update(
                                #     [sref["matchedPaperCorpusId"] for sref in sentence["ref_mentions"] if
                                #      sref.get("start") >= lookup_idx and sref.get("end") <= lookup_end])
                            break
                    curr_quote_map["quote"] = new_quote
                    if "section_title" not in curr_quote_map:
                        curr_quote_map["pdf_hash"] = ""
                        for field in ["title", "abstract"]:
                            if row[field] and new_quote.lower() in row[field].lower():
                                curr_quote_map["section_title"] = field
                    mapped_quotes.append(curr_quote_map)
                quotes_metadata[ref_str] = mapped_quotes
                updated_quotes = "...".join([mq["quote"] for mq in mapped_quotes])
                # fix weird formatting
                updated_quotes = re.sub(CLOSE_BRACKET_PATTERN, r'[\1',
                                        updated_quotes)  # (Doe et al., 2024)10] --> (Doe et al., 2024)[10]
                updated_quotes = re.sub(OPEN_BRACKET_PATTERN, r'\1]',
                                        updated_quotes)  # [8,9,(Doe et al., 2024) --> [8,9](Doe et al., 2024)
                per_paper_summaries[ref_str] = updated_quotes

        return quotes_metadata

    def populate_citations_metadata(self, avl_paper_metadata: Dict[str, Dict[str, Any]],
                                    paper_inline_cites: Dict[str, List],
                                    per_paper_summaries: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        retrieve the metadata of the quote inline citations if not already present and update the quotes from string,
        to a dict of {"quote": quote, "inline_citations": {ref_str: abstract, ... }}.
        Also, link the citation mentions to modify them from (corpus_id) to (Doe et al., 2024)
        """
        corpus_id_ref_str_map = {ref_str[1:-1].split(" | ")[0]: ref_str for ref_str in per_paper_summaries}
        additional_citation_ids = {item for sublist in paper_inline_cites.values() for item in sublist if
                                   item not in avl_paper_metadata}
        if additional_citation_ids:
            logger.info(f"Fetching metadata for {len(additional_citation_ids)} additional inline citations")
            additional_metadata = get_paper_metadata(additional_citation_ids)
        else:
            additional_metadata = dict()
        per_paper_summaries = {k: {"quote": quote, "inline_citations": dict()} for k, quote in
                               per_paper_summaries.items()}
        for ref_str, cite_ids in paper_inline_cites.items():
            quote_cid = ref_str[1:-1].split(" | ")[0]
            # 2 sets of ref mentions, one where paper metadata is already available and the rest for which metadata was requested from s2 api
            curr_metadata = [avl_paper_metadata[cite_id] for cite_id in cite_ids if
                             cite_id in avl_paper_metadata]
            curr_metadata += [additional_metadata[cite_id] for cite_id in cite_ids if
                              cite_id in additional_metadata]
            for idx, mdata in enumerate(curr_metadata):
                mref_str = corpus_id_ref_str_map.get(mdata["corpusId"], f"[{mdata['corpusId']} | "
                                                                        f"{get_ref_author_str(mdata['authors'])} | "
                                                                        f"{make_int(mdata.get('year'))} "
                                                                        f"| Citations: {make_int(mdata['citationCount'])}]")
                mref_str = anyascii(mref_str)
                per_paper_summaries[ref_str]["quote"] = per_paper_summaries[ref_str]["quote"].replace(
                    f"({mdata['corpusId']})",
                    f"({get_ref_author_str(mdata['authors'])}, {make_int(mdata.get('year'))})")
                if mdata["corpusId"] in additional_metadata:
                    additional_metadata[mdata["corpusId"]]["relevance_judgement"] = max(
                        additional_metadata[mdata["corpusId"]].get("relevance_judgement", 0),
                        avl_paper_metadata[quote_cid]["relevance_judgement"])
                if mdata.get("abstract"):
                    per_paper_summaries[ref_str]["inline_citations"][mref_str] = mdata["abstract"]
            per_paper_summaries[ref_str]["quote"] = per_paper_summaries[ref_str]["quote"].replace("NULL, ", "")
        avl_paper_metadata.update(additional_metadata)
        return per_paper_summaries

    def extract_quote_citations(self, score_df: pd.DataFrame, per_paper_summaries: Dict[str, str],
                                plan_json: Dict[str, List[int]], paper_metadata: Dict[str, Any]) -> Tuple[
        Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        quotes_metadata = self.passage_to_quotes_metadata(score_df, per_paper_summaries, plan_json)
        per_paper_inline_cites = {
            ref_str: set(ref for q in qmeta if q.get("ref_mentions") for ref in q["ref_mentions"])
            for ref_str, qmeta in quotes_metadata.items()
        }
        per_paper_inline_cites = {k: sorted(v) for k, v in per_paper_inline_cites.items() if v}
        per_paper_summaries_extd = self.populate_citations_metadata(paper_metadata, per_paper_inline_cites,
                                                                    per_paper_summaries)
        for ref_str, quote_map in per_paper_summaries_extd.items():
            if quotes_metadata.get(ref_str):
                quote_parts = quote_map["quote"].split("...")
                for idx, qmap in enumerate(quotes_metadata[ref_str]):
                    qmap["quote"] = quote_parts[idx]
        return per_paper_summaries_extd, quotes_metadata

    @staticmethod
    def get_gen_sections_from_json(section: Dict[str, Any]) -> GeneratedSection:
        try:
            citations = [CitationSrc(**citation) for citation in section["citations"]]
            generated_section = GeneratedSection(title=section["title"],
                                                 tldr=section["tldr"],
                                                 text=section["text"],
                                                 citations=citations)
            return generated_section
        except Exception as e:
            logger.error(f"Error while converting json to TaskResult: {e}")
            raise e

    def postprocess_json_output(self, json_summary: List[Dict[str, Any]], **kwargs) -> None:
        pass

    def answer_query(self, query: str, inline_tags: bool = True) -> Dict[str, Any]:
        task_id = str(uuid4())
        self.logs_config.task_id = task_id
        logger.info("New task")
        tool_request = ToolRequest(task_id=task_id, query=query, user_id="lib_user")
        try:
            task_result = self.run_qa_pipeline(tool_request, inline_tags)
        except Exception as e:
            logger.warning(f"Error while running task: {e}, invalidating llm cache and retrying")
            self.multi_step_pipeline.llm_kwargs["cache"] = {"no-cache": True}
            task_result = self.run_qa_pipeline(tool_request, inline_tags)
        return task_result.model_dump()

    def gen_table_thread(self, user_id: str, query: str, dim: Dict[str, Any],
                         cit_ids: List[int], tlist: List[Any]) -> Thread:
        def call_table_generator(didx: int, payload: Dict[str, Any]):
            logger.info(
                "Received table generation request for topic: " + payload["section_title"]
            )
            table, costs = self.table_generator.run_table_generation(
                thread_id=payload["task_id"],
                user_id=payload["user_id"],
                original_query=payload["query"],
                section_title=payload["section_title"],
                corpus_ids=payload["cit_ids"],
                column_model=payload["column_model"],
                value_model=payload["value_model"],
            )
            tlist[dim["idx"]] = (table, costs)
            
        task_id = self.task_id if self.task_id else self.tool_request.task_id
        payload = {
            "task_id": task_id,
            "user_id": user_id,
            "query": query,
            "section_title": dim["name"],
            "cit_ids": cit_ids,
            "column_model": self.table_llm,
            "value_model": self.table_llm,
        }
        tthread = Thread(target=call_table_generator, args=(dim["idx"], payload,))
        tthread.start()
        return tthread

    def get_user_msg_id(self):
        return self.tool_request.user_id, self.task_id

    @traceable(run_type="tool", name="ai2_scholar_qa_trace")
    def run_qa_pipeline(self, req: ToolRequest, inline_tags=False) -> TaskResult:
        """
                This function takes a query and returns a response.
                Goes through the following steps:
                0) Decompose the query to get filters like year, venue, fos, citations, etc along with a re-written
                version of the query and a query suitable for keyword search.
                1) Query retrieval to get the relevant snippets from the index (n_retrieval)
                1.1) Query semantic scholar with the keyword search query to get the relevant papers.(n_keyword_srch)
                2) Re-rank the snippets based on the query with a cross encoder (n_rerank)
                3) Get exact relevant quotes from an LLM
                4) Generate outline and cluster the quotes from (3)
                4.1) The quotes cluster in the outline have inline citations associated with them. Map the quotes to
                their inline citations and include them with the quotes.
                5) Generate the summarized output using the quotes and outline in (3) and (4)

                :param req: A scientific query posed to scholar qa by a user, consists of the string query, task id and user id
                :param inline_tags: Whether to include inline <paper> tags in the output or not
                :return: A response to the query

        """
        self.tool_request = req
        self.update_task_state("Processing user query", task_estimated_time="~3 minutes", step_estimated_time=5)
        task_id = self.task_id if self.task_id else req.task_id
        user_id, msg_id = self.get_user_msg_id()
        msg_id = task_id if not msg_id else msg_id
        query = req.query
        logger.info(
            f"Received query: {query} from user_id: {user_id} with opt_in: {req.opt_in}"
        )
        event_trace = EventTrace(
            task_id,
            self.paper_finder.retriever.n_retrieval if hasattr(self.paper_finder.retriever, "n_retrieval") else 0,
            # noqa
            self.paper_finder.n_rerank,
            req,
            user_id=user_id
        )
        cost_args = CostReportingArgs(
            task_id=task_id,
            user_id=user_id,
            description="Step 0: Query decomposition",
            model=self.llm_model,
            msg_id=msg_id
        )
        llm_processed_query = self.preprocess_query(query, cost_args)
        event_trace.trace_decomposition_event(llm_processed_query)

        # Paper finder step - retrieve relevant paper passages from semantic scholar index and api
        snippet_srch_res, s2_srch_res = self.find_relevant_papers(llm_processed_query.result)
        retrieved_candidates = snippet_srch_res + s2_srch_res
        if not retrieved_candidates:
            raise Exception(
                f"There is no relevant information in the retrieved snippets for query: {query}.")
        event_trace.trace_retrieval_event(retrieved_candidates)

        # Rerank the retrieved candidates based on the query with a cross encoder
        s2_srch_metadata = [{k: v for k, v in paper.items() if
                             k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS} for paper in
                            s2_srch_res]
        reranked_df, paper_metadata = self.rerank_and_aggregate(query, retrieved_candidates,
                                                                {str(paper["corpus_id"]): paper for paper in
                                                                 s2_srch_metadata})
        if reranked_df.empty:
            raise Exception(
                "No relevant papers found for the query post reranking, skipping quote extraction.")
        event_trace.trace_rerank_event(reranked_df.to_dict(orient="records"))

        # Step 1 - quote extraction
        per_paper_summaries = self.step_select_quotes(query, reranked_df, cost_args)
        if not per_paper_summaries.result:
            raise Exception(
                "No relevant quotes extracted for the query, can't proceed further.")
        event_trace.trace_quote_event(per_paper_summaries)

        # step 2: outline planning and clustering
        cluster_json = self.step_clustering(query, per_paper_summaries.result, cost_args)
        # Changing to expected format in the summary generation prompt
        plan_json = {f'{dim["name"]} ({dim["format"]})': dim["quotes"] for dim in cluster_json.result["dimensions"]}
        if not any([len(d) for d in plan_json.values()]):
            raise Exception("The planning step failed to cluster the relevant documents.")
        event_trace.trace_clustering_event(cluster_json, plan_json)

        # step 2.1: extend the clustered snippets with their inline citations
        per_paper_summaries_extd, quotes_metadata = self.extract_quote_citations(reranked_df,
                                                                                 per_paper_summaries.result,
                                                                                 plan_json, paper_metadata)
        event_trace.trace_inline_citation_following_event(per_paper_summaries_extd, quotes_metadata)

        # step 3: generating output as per the outline
        section_titles = [dim["name"] for dim in cluster_json.result["dimensions"]]
        gen_sections_iter = self.step_gen_iterative_summary(query, per_paper_summaries_extd,
                                                            plan_json, cost_args)

        json_summary, generated_sections, table_threads = [], [], []
        tables = [None for _ in cluster_json.result["dimensions"]]
        citation_ids = dict()

        task_estimated_time = 30 + 15 * len(plan_json)
        task_estimated_time = max((task_estimated_time + task_estimated_time % 60) // 60, 1)
        outline = '\n    - ' + '\n    - '.join(section_titles)
        self.update_task_state(f"Start generating each section in the answer outline: {outline}",
                               task_estimated_time=f"~{task_estimated_time} minutes" if task_estimated_time > 1 else "~1 minute",
                               step_estimated_time=15)

        try:
            gen_iter = gen_sections_iter
            idx = 0
            while True:
                if idx < len(plan_json):
                    self.update_task_state(
                        f"Iteratively generating section: {(idx + 1)} of {len(plan_json)} - {section_titles[idx]}",
                        curr_response=generated_sections, step_estimated_time=15)
                section_text = next(gen_iter)
                section_json = \
                    get_json_summary(self.multi_step_pipeline.llm_model, [section_text], per_paper_summaries_extd,
                                     paper_metadata,
                                     citation_ids, inline_tags)[0]
                section_json["format"] = cluster_json.result["dimensions"][idx]["format"]

                json_summary.append(section_json)
                self.postprocess_json_output(json_summary, quotes_meta=quotes_metadata)
                if section_json["format"] == "list" and section_json["citations"] and self.run_table_generation:
                    cluster_json.result["dimensions"][idx]["idx"] = idx
                    cit_ids = [int(c["paper"]["corpus_id"]) for c in section_json["citations"]]
                    tthread = self.gen_table_thread(user_id, query, cluster_json.result["dimensions"][idx], cit_ids,
                                                    tables)
                    if tthread:
                        table_threads.append(tthread)
                gen_sec = self.get_gen_sections_from_json(section_json)
                generated_sections.append(gen_sec)
                idx += 1
        except StopIteration as e:
            all_sections = e.value

        self.update_task_state(f"Generating comparison tables", curr_response=generated_sections,
                               step_estimated_time=20)

        start = time()
        for tthread in table_threads:
            tthread.join()
        logger.info(f"Adhoc Table generation wait time: {time() - start:.2f}")
        tcosts = []
        for sidx in range(len(json_summary)):
            tables_val = None
            if tables[sidx]:
                if type(tables[sidx]) == tuple:
                    tables_val, tcost = tables[sidx]
                    tcosts.append(tcost)
                else:
                    tables_val = tables[sidx]
            json_summary[sidx]["table"] = tables_val.to_dict() if tables_val else None
            generated_sections[sidx].table = tables_val if tables_val else None
        self.postprocess_json_output(json_summary, quotes_meta=quotes_metadata)
        event_trace.trace_summary_event(json_summary, all_sections, tcosts)
        event_trace.persist_trace(self.logs_config)
        return TaskResult(sections=generated_sections, cost=event_trace.total_cost, tokens=event_trace.tokens)
