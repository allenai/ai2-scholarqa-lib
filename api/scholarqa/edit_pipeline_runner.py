import json
import logging
import re
from time import time
from typing import List, Any, Dict, Tuple, Optional, Union

import pandas as pd
from langsmith import traceable

from scholarqa.llms.constants import CostAwareLLMResult
from scholarqa.llms.litellm_helper import CostReportingArgs
from scholarqa.models import TaskResult, ToolRequest, ReportEditRequest
from scholarqa.postprocess.json_output_utils import get_json_summary
from scholarqa.preprocess.query_preprocessor import LLMProcessedQuery
from scholarqa.preprocess.edit_intent_analyzer import analyze_edit_intent, EditIntentAnalysis
from scholarqa.rag.edit_pipeline import EditAction, EditPipeline
from scholarqa.scholar_qa import ScholarQA
from scholarqa.trace.event_traces import EventTrace
from scholarqa.utils import get_paper_metadata, NUMERIC_META_FIELDS, CATEGORICAL_META_FIELDS

logger = logging.getLogger(__name__)


class EditPipelineRunner(ScholarQA):
    """
    Extends ScholarQA with edit-specific pipeline methods for modifying existing reports.

    Inherits all shared functionality from ScholarQA (retrieval, reranking, task state,
    table generation, citation extraction) and adds edit-specific steps:
    - Retrieve existing report from thread
    - Analyze edit intent (determines search needs, paper add/remove operations)
    - Conditional search with intent-derived query and filters
    - Edit-aware quote extraction, clustering, and section generation
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fallback_llm = kwargs.get("fallback_llm", None)
        self.edit_pipeline = EditPipeline(
            self.llm_model, fallback_llm=fallback_llm, **self.llm_kwargs
        )

    @traceable(name="Edit: Retrieve current report from thread")
    def retrieve_report_from_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the current report JSON from the state manager using thread_id.

        Returns the raw dict (matching the summary JSON format) to avoid
        unnecessary TaskResult conversion — KEEP sections reuse dict sections directly.

        Returns:
            Report dict with 'sections', 'report_title', etc. or None if not found
        """
        try:
            state = self.state_mgr.read_state(thread_id)
            if state and state.task_result:
                if isinstance(state.task_result, TaskResult):
                    return state.task_result.model_dump()
                elif isinstance(state.task_result, dict):
                    return state.task_result
            return None
        except Exception as e:
            logger.error(f"Error retrieving report for thread {thread_id}: {e}")
            return None

    @traceable(name="Edit: Analyze edit intent")
    def analyze_intent(
            self,
            req: ReportEditRequest,
            report_context: str,
            current_report: TaskResult,
    ) -> EditIntentAnalysis:
        """
        Analyze the edit instruction to determine search needs and operations.

        Args:
            req: The edit request containing edit_instruction, query, mentioned_papers, section_titles
            report_context: Pre-formatted report context string
            current_report: The current report being edited (for citation extraction)

        Returns:
            EditIntentAnalysis with search query, constraints, and operation details
        """
        self.update_task_state("Analyzing edit instruction", step_estimated_time=3)

        return analyze_edit_intent(
            req=req,
            report_context=report_context,
            current_report=current_report,
            llm_model=self.llm_model,
            fallback_llm=self.edit_pipeline.fallback_llm,
            **self.llm_kwargs
        )

    @traceable(name="Edit: Retrieve mentioned papers via vespa + metadata")
    def _retrieve_mentioned_papers(
            self,
            papers_to_add: List[str],
            original_query: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieve user-mentioned papers on a best-effort basis.

        1. Fetches metadata for all papers (always needed as fallback)
        2. Tries full-text snippet retrieval via vespa using paperIds filter
        3. For papers without snippets, creates metadata-only candidates (abstract as text)

        Args:
            papers_to_add: Corpus IDs to retrieve
            original_query: The original query that generated the report

        Returns:
            Tuple of (retrieved_candidates, paper_metadata)
        """
        paper_ids = {str(p) for p in papers_to_add if p}
        retrieved_candidates = []

        self.update_task_state(
            f"Fetching {len(paper_ids)} user-mentioned papers",
            step_estimated_time=5
        )
        logger.info(f"Fetching metadata for papers to add: {paper_ids}")

        # Always fetch metadata (used as fallback if full-text retrieval misses papers,
        # and passed to s2_srch_metadata to avoid duplicate fetches in rerank_and_aggregate)
        paper_metadata = get_paper_metadata(paper_ids)
        for corpus_id, mdata in paper_metadata.items():
            mdata["corpus_id"] = corpus_id

        # Try full-text snippet retrieval via vespa using paperIds filter
        search_query = original_query or "relevant research"
        paper_ids_param = ",".join(f"CorpusId:{cid}" for cid in paper_ids)
        logger.info(f"Searching vespa with paperIds={paper_ids_param} for mentioned papers")

        try:
            snippet_results = self.paper_finder.retrieve_passages(
                query=search_query,
                paperIds=paper_ids_param,
            )
            if snippet_results:
                logger.info(f"Got {len(snippet_results)} snippets for mentioned papers via paperIds filter")
                retrieved_candidates.extend(snippet_results)
            else:
                logger.info("No snippets from vespa for mentioned papers, will use metadata fallback")
        except Exception as e:
            logger.warning(f"Vespa search with paperIds failed: {e}, will use metadata fallback")

        # Add metadata-only candidates for papers that didn't get snippets from vespa
        # Mirrors the keyword_search() pattern in retriever_base.py: copy the S2 metadata
        # dict and add snippet-like fields on top so it's compatible with aggregate_into_dataframe
        snippet_corpus_ids = {str(c["corpus_id"]) for c in retrieved_candidates}
        for corpus_id in paper_ids - snippet_corpus_ids:
            if corpus_id in paper_metadata:
                candidate = dict(paper_metadata[corpus_id])
                candidate["text"] = candidate.get("abstract", "")
                candidate["section_title"] = "abstract"
                candidate["char_start_offset"] = 0
                candidate["sentence_offsets"] = []
                candidate["ref_mentions"] = []
                candidate["score"] = 1.0
                candidate["stype"] = "public_api"
                candidate["pdf_hash"] = ""
                retrieved_candidates.append(candidate)
                logger.info(f"Added metadata-only candidate for paper {corpus_id}")

        return retrieved_candidates, paper_metadata

    @traceable(name="Edit: Search for new papers")
    def _search_for_new_papers(
            self,
            intent_analysis: EditIntentAnalysis,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Search for new papers using the intent-derived query and filters.

        Delegates to parent's find_relevant_papers() for vespa + S2 API retrieval.

        Args:
            intent_analysis: Result of analyzing the edit instruction

        Returns:
            Tuple of (snippet_results, search_api_results)
        """
        search_query = intent_analysis.search_query
        search_filters = intent_analysis.to_search_filters()
        logger.info(f"Searching with query: {search_query}, filters: {search_filters}")

        processed_query = LLMProcessedQuery(
            rewritten_query=search_query,
            keyword_query=search_query,
            search_filters=search_filters,
        )
        return self.find_relevant_papers(processed_query)

    @traceable(name="Edit: Conditional search for papers")
    def find_relevant_papers_for_edit(
            self,
            intent_analysis: EditIntentAnalysis,
            original_query: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Find papers relevant to the edit based on intent analysis.

        Orchestrates two sources:
        1. User-mentioned papers - best-effort full-text via vespa paperIds filter + metadata fallback
        2. Search results - vespa + S2 API via parent's find_relevant_papers()

        Args:
            intent_analysis: Result of analyzing the edit instruction
            original_query: The original query that generated the report

        Returns:
            Tuple of (retrieved_candidates, paper_metadata)
        """
        retrieved_candidates = []
        paper_metadata = {}

        if intent_analysis.papers_to_add:
            mentioned_candidates, paper_metadata = self._retrieve_mentioned_papers(
                papers_to_add=intent_analysis.papers_to_add,
                original_query=original_query,
            )
            retrieved_candidates.extend(mentioned_candidates)
        elif intent_analysis.needs_search:
            snippet_results, search_api_results = self._search_for_new_papers(intent_analysis)
            retrieved_candidates.extend(snippet_results + search_api_results)

        return retrieved_candidates, paper_metadata

    def _inject_abstract_fallbacks(
            self,
            per_paper_summaries: Dict[str, str],
            papers_to_add: List[str],
            reranked_df: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        For papers_to_add that were filtered out during quote extraction,
        inject their abstracts as fallback context from reranked_df.

        This ensures user-mentioned papers still contribute to clustering and
        section generation even when the LLM couldn't extract specific quotes.
        Papers are assumed to always be present in reranked_df after the
        rerank step (they were either retrieved via vespa or added as
        metadata-only candidates).

        Args:
            per_paper_summaries: Quotes dict keyed by reference_string
            papers_to_add: Corpus IDs the user wants added
            reranked_df: Reranked DataFrame with reference_string, corpus_id, abstract columns

        Returns:
            Updated per_paper_summaries with abstract fallbacks added
        """
        if reranked_df.empty:
            return per_paper_summaries

        papers_to_add_ids = {str(p) for p in papers_to_add if p}

        # Find which papers_to_add are already in per_paper_summaries
        present_ids = set()
        for ref_str in per_paper_summaries.keys():
            match = re.match(r'\[(\d+)\s*\|', ref_str)
            if match:
                present_ids.add(match.group(1))

        missing_ids = papers_to_add_ids - present_ids
        if not missing_ids:
            return per_paper_summaries

        logger.info(
            f"Papers_to_add filtered after quote extraction: {missing_ids}. "
            f"Injecting abstracts as fallback context."
        )

        for corpus_id in missing_ids:
            corpus_id_int = int(corpus_id)
            if corpus_id_int not in reranked_df["corpus_id"].values:
                logger.warning(f"Paper {corpus_id} not found in reranked_df, skipping fallback")
                continue

            row = reranked_df[reranked_df["corpus_id"] == corpus_id_int].iloc[0]
            ref_str = row["reference_string"]
            abstract = row.get("abstract", "")

            if abstract:
                per_paper_summaries[ref_str] = f"[Abstract] {abstract}"
                logger.info(f"Added abstract fallback for paper {corpus_id}")
            else:
                logger.warning(f"No abstract available for paper {corpus_id}")

        return per_paper_summaries

    @traceable(name="Edit: Extract quotes from new papers")
    def step_select_quotes_edit(
            self,
            original_query: str,
            search_query: str,
            report_context: str,
            scored_df: pd.DataFrame,
            cost_args: CostReportingArgs = None,
    ) -> CostAwareLLMResult:
        """
        Extract relevant quotes from new papers with edit context.

        Returns:
            CostAwareLLMResult with .result = per_paper_summaries dict
        """
        logger.info("Running Edit Step 1 - quote extraction with edit context")
        self.update_task_state(
            "Extracting salient key statements from papers",
            step_estimated_time=15
        )
        logger.info(
            f"{scored_df.shape[0]} papers with relevance_judgement >= "
            f"{self.paper_finder.context_threshold} to start with."
        )

        start = time()
        cost_args = cost_args._replace(
            model=self.edit_pipeline.llm_model
        )._replace(description="Edit Step 1: Quote extraction")

        per_paper_summaries = self.llm_caller.call_method(
            cost_args,
            self.edit_pipeline.step_select_quotes_edit,
            original_query=original_query,
            search_query=search_query,
            report_context=report_context,
            scored_df=scored_df,
        )

        logger.info(
            f"Edit Step 1 done - {len(per_paper_summaries.result)} papers with quotes extracted, "
            f"cost: {per_paper_summaries.tot_cost}, time: {time() - start:.2f}"
        )

        return per_paper_summaries

    @traceable(name="Edit: Generate edit plan with actions")
    def step_clustering_edit(
            self,
            edit_instruction: str,
            report_context: str,
            per_paper_summaries: Dict[str, str],
            intent_analysis: Optional[EditIntentAnalysis] = None,
            cost_args: CostReportingArgs = None,
    ) -> CostAwareLLMResult:
        """
        Generate edit plan with actions (KEEP, REWRITE, DELETE, NEW).

        Returns:
            CostAwareLLMResult with .result = cluster_json_result dict
        """
        logger.info("Running Edit Step 2: Generating edit plan with actions")
        self.update_task_state("Generating edit plan", step_estimated_time=15)

        start = time()
        cost_args = cost_args._replace(
            model=self.edit_pipeline.llm_model
        )._replace(description="Edit Step 2: Clustering/Planning")

        cluster_result = self.llm_caller.call_method(
            cost_args,
            self.edit_pipeline.step_clustering_edit,
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=per_paper_summaries,
            intent_analysis=intent_analysis,
        )

        logger.info(
            f"Edit Step 2 done - {len(cluster_result.result.get('dimensions', []))} dimensions, "
            f"{len(cluster_result.result.get('papers_to_remove', []))} papers to remove, "
            f"cost: {cluster_result.tot_cost}, time: {time() - start:.2f}"
        )

        return cluster_result

    @traceable(name="Edit: Generate/edit sections iteratively")
    def step_gen_iterative_summary_edit(
            self,
            edit_instruction: str,
            current_report: TaskResult,
            per_paper_summaries_extd: Dict[str, Any],
            plan: List[Dict[str, Any]],
            papers_to_remove: List[str] = None,
            cost_args: CostReportingArgs = None,
    ):
        """
        Generate or edit sections iteratively based on edit actions.

        Mirrors the base class step_gen_iterative_summary pattern: wraps the
        pipeline generator with call_iter_method for cost tracking, yields
        response.content to the runner, and returns CostAwareLLMResult via
        generator return value.

        Yields None for DELETE/KEEP (content of noop CompletionResult),
        string for REWRITE/NEW.
        """
        logger.info("Running Edit Step 3: Editing sections with actions")
        start = time()

        cost_args = cost_args._replace(
            model=self.edit_pipeline.llm_model
        )._replace(description="Edit Step 3: Section generation/editing")

        sec_generator = self.llm_caller.call_iter_method(
            cost_args,
            self.edit_pipeline.generate_iterative_summary_edit,
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan,
            papers_to_remove=papers_to_remove,
        )

        try:
            while True:
                response = next(sec_generator)
                yield response.content
        except StopIteration as e:
            return_val = e.value

        if return_val:
            logger.info(
                f"Edit Step 3 done, cost: {return_val.tot_cost}, time: {time() - start:.2f}"
            )

        return return_val

    @traceable(run_type="tool", name="ai2_scholar_qa_edit_trace")
    def run_edit_pipeline(
            self,
            req: ReportEditRequest,
            inline_tags: bool = False,
    ) -> TaskResult:
        """
        Edit an existing report based on user instructions.

        Pipeline steps:
        0) Retrieve current report from thread_id
        0.5) Analyze edit intent (determines search needs, operations)
        1) Conditional search + rerank (includes mentioned_papers)
        2) Quote extraction from new papers (WITH EDIT CONTEXT)
        3) Edit planning/clustering (WITH CURRENT REPORT)
        4) Quote citation mapping (inherited from ScholarQA)
        5) Section generation/editing (WITH CURRENT SECTIONS + ACTIONS)
        6) Table generation

        Args:
            req: Edit request with thread_id, intent, corpus_ids, section_titles
            inline_tags: Whether to include inline <paper> tags in the output

        Returns:
            Updated TaskResult
        """
        self.tool_request = req
        self.update_task_state(
            "Processing edit request",
            task_estimated_time="~3 minutes",
            step_estimated_time=5
        )

        # Validate edit request
        if not req.intent:
            raise ValueError("edit_instruction is required for edit workflow")
        if not req.thread_id:
            raise ValueError("thread_id is required to fetch the current report")

        task_id = self.task_id if self.task_id else req.task_id
        user_id, msg_id = self.get_user_msg_id()
        msg_id = task_id if not msg_id else msg_id
        edit_instruction = req.intent

        logger.info(
            f"Received edit request for thread {req.thread_id}: {edit_instruction} "
            f"from user_id: {user_id}"
        )

        # ====================================================================
        # STEP 0: Retrieve current report (EDIT-SPECIFIC)
        # ====================================================================
        self.update_task_state("Retrieving current report", step_estimated_time=2)
        current_report = self.retrieve_report_from_thread(req.thread_id)
        req.query = req.query if req.query else (current_report.get("query") if current_report else "")
        if not current_report:
            raise ValueError(f"No report found for thread_id: {req.thread_id}")

        report_sections = current_report.get("sections", [])
        report_title = current_report.get("report_title", "")
        logger.info(
            f"Retrieved report with {len(report_sections)} sections: {report_title}"
        )

        self.report_title = report_title

        # Initialize event trace
        event_trace = EventTrace(
            task_id,
            self.paper_finder.retriever.n_retrieval if hasattr(self.paper_finder.retriever, "n_retrieval") else 0,
            self.paper_finder.n_rerank,
            req,
            user_id=user_id
        )

        cost_args = CostReportingArgs(
            task_id=task_id,
            user_id=user_id,
            description="Edit workflow",
            model=self.llm_model,
            msg_id=msg_id
        )

        # Format report context once for all steps
        report_context = self.edit_pipeline.format_report_context(current_report)

        # ====================================================================
        # STEP 0.5: Analyze Edit Intent
        # ====================================================================

        intent_analysis = self.analyze_intent(
            req=req,
            report_context=report_context,
            current_report=current_report,
        )

        logger.info(f"Intent analysis output: {intent_analysis.model_dump_json()}")

        # ====================================================================
        # STEP 1: Search and Rerank (CONDITIONAL, based on intent analysis)
        # ====================================================================
        if intent_analysis.needs_search:
            retrieved_candidates, paper_metadata = self.find_relevant_papers_for_edit(
                intent_analysis=intent_analysis,
                original_query=req.query or "",
            )

            if not retrieved_candidates:
                logger.info("No new papers to add, will work with existing report content")
                reranked_df = pd.DataFrame()
            else:
                event_trace.trace_retrieval_event(retrieved_candidates)

                # Build pre-fetched metadata map to avoid duplicate get_paper_metadata calls.
                # Include paper_metadata (from _retrieve_mentioned_papers) so vespa-found
                # mentioned papers also have their metadata pre-populated and won't be
                # re-fetched in rerank_and_aggregate.
                s2_srch_metadata = [
                    {k: v for k, v in paper.items() if
                     k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
                    for paper in retrieved_candidates + list(paper_metadata.values())
                    if "s2FieldsOfStudy" in paper
                ]

                # Use inherited rerank_and_aggregate from ScholarQA
                reranked_df, paper_metadata = self.rerank_and_aggregate(
                    edit_instruction,
                    retrieved_candidates,
                    {str(paper["corpus_id"]): paper for paper in s2_srch_metadata}
                )

                if reranked_df.empty:
                    logger.warning("Reranking produced no results")

                event_trace.trace_rerank_event(
                    reranked_df.to_dict(orient="records") if not reranked_df.empty else []
                )

            # ====================================================================
            # STEP 2: Quote Extraction (EDIT-SPECIFIC context)
            # ====================================================================

            per_paper_summaries = {}

            if not reranked_df.empty:
                quote_result = self.step_select_quotes_edit(
                    original_query=req.query or "",
                    search_query=intent_analysis.search_query or "",
                    report_context=report_context,
                    scored_df=reranked_df,
                    cost_args=cost_args,
                )
                per_paper_summaries = quote_result.result

                if not per_paper_summaries:
                    logger.warning("No quotes extracted from new papers")
                else:
                    logger.info(f"Extracted quotes from {len(per_paper_summaries)} papers")
                    event_trace.trace_quote_event(quote_result)

            # Inject abstract fallbacks for papers_to_add that were filtered during quote extraction
            if intent_analysis.papers_to_add:
                per_paper_summaries = self._inject_abstract_fallbacks(
                    per_paper_summaries=per_paper_summaries,
                    papers_to_add=intent_analysis.papers_to_add,
                    reranked_df=reranked_df,
                )
        else:
            logger.info("Intent analysis indicates no search needed, skipping to planning with existing report content")
            per_paper_summaries = {}
            paper_metadata = {}
            reranked_df = pd.DataFrame()
        # ====================================================================
        # STEP 3: Clustering/Planning (EDIT-SPECIFIC)
        # ====================================================================

        self.update_task_state("Generating edit plan", step_estimated_time=15)

        cluster_result = self.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=per_paper_summaries,
            intent_analysis=intent_analysis,
            cost_args=cost_args,
        )
        cluster_json_result = cluster_result.result
        logger.info(f"Clustering/planning output: {json.dumps(cluster_json_result)}")
        event_trace.trace_clustering_event(cluster_result, {})

        if cluster_json_result.get("report_title"):
            self.report_title = cluster_json_result["report_title"]

        plan_dimensions = cluster_json_result["dimensions"]
        plan_json = {
            f'{dim["name"]} ({dim["format"]})': dim.get("quotes", [])
            for dim in plan_dimensions
        }

        # ====================================================================
        # STEP 4: Quote Citation Mapping (inherited from ScholarQA)
        # ====================================================================

        per_paper_summaries_extd = {}
        quotes_metadata = {}

        if per_paper_summaries and not reranked_df.empty:
            per_paper_summaries_extd, quotes_metadata = self.extract_quote_citations(
                reranked_df,
                per_paper_summaries,
                plan_json,
                paper_metadata
            )
            event_trace.trace_inline_citation_following_event(per_paper_summaries_extd, quotes_metadata)

        # Ensure abstract-fallback papers are included in per_paper_summaries_extd
        # (they may not be in reranked_df so extract_quote_citations won't process them)
        for ref_str, quote in per_paper_summaries.items():
            if ref_str not in per_paper_summaries_extd:
                per_paper_summaries_extd[ref_str] = {
                    "quote": quote,
                    "inline_citations": {}
                }

        # ====================================================================
        # STEP 4.5: Build existing_paper_summaries from current report citations
        # and merge into per_paper_summaries_extd / paper_metadata / quotes_metadata
        # for get_json_summary() resolution. New quotes take priority for duplicates.
        # ====================================================================
        for section in report_sections:
            for cit in section.get("citations", []):
                ref_key, per_paper_entry, paper_meta_entry = self.edit_pipeline.citation_to_ref_data(cit)
                corpus_id_str = str(cit["paper"]["corpus_id"])
                if ref_key not in per_paper_summaries_extd:
                    per_paper_summaries_extd[ref_key] = per_paper_entry
                if corpus_id_str not in paper_metadata:
                    paper_metadata[corpus_id_str] = paper_meta_entry
                snippet_meta = cit.get("snippetMetadata") or cit.get("snippet_metadata")
                if snippet_meta and ref_key not in quotes_metadata:
                    quotes_metadata[ref_key] = [
                        {
                            "quote": sm.get("quote", ""),
                            "section_title": sm.get("sectionTitle", sm.get("section_title", "")),
                            "pdf_hash": sm.get("pdfHash", sm.get("pdf_hash", "")),
                            "sentence_offsets": sm.get("sentenceOffsets", sm.get("sentence_offsets", [])),
                            "ref_mentions": sm.get("refMentions", sm.get("ref_mentions", [])),
                        }
                        for sm in snippet_meta
                    ]

        # ====================================================================
        # STEP 5: Section Generation/Editing (EDIT-SPECIFIC)
        # ====================================================================

        section_titles = [dim["name"] for dim in plan_dimensions]
        self.update_task_state(
            f"Editing report sections",
            step_estimated_time=15 * len(section_titles)
        )

        papers_to_remove = cluster_json_result.get("papers_to_remove", [])
        if papers_to_remove:
            logger.info(f"Papers to remove from report: {papers_to_remove}")

        gen_sections_iter = self.step_gen_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan_dimensions,
            papers_to_remove=papers_to_remove,
            cost_args=cost_args,
        )

        json_summary, generated_sections, table_threads = [], [], []
        tables = [None for _ in plan_dimensions]
        citation_ids = dict()
        current_sections_map = {s["title"]: s for s in report_sections}

        for idx, dim in enumerate(plan_dimensions):
            action = dim.get("action", EditAction.NEW)
            section_name = dim["name"]

            # DELETE: generator yielded None (content of noop), skip entirely
            if action == EditAction.DELETE:
                self.update_task_state(
                    f"Section {idx + 1} of {len(plan_dimensions)}: {section_titles[idx]} ({action})",
                    curr_response=generated_sections,
                    step_estimated_time=15
                )

            else:
                # REWRITE/NEW: section_result is a string (response.content from call_iter_method)
                self.update_task_state(
                    f"Section {idx + 1} of {len(plan_dimensions)}: {section_titles[idx]} ({action})",
                    curr_response=generated_sections,
                    step_estimated_time=15
                )
            section_result = next(gen_sections_iter)
            if action == EditAction.DELETE:
                continue
            # KEEP: generator yielded None (content of noop), reuse existing section dict directly
            if action == EditAction.KEEP and section_name in current_sections_map:
                section_json = current_sections_map[section_name]
            else:
                section_json = get_json_summary(
                    self.llm_model,
                    [section_result],
                    per_paper_summaries_extd,
                    paper_metadata,
                    citation_ids,
                    inline_tags
                )[0]
            section_json["format"] = dim["format"]

            json_summary.append(section_json)
            self.postprocess_json_output(json_summary, quotes_meta=quotes_metadata)

            existing_section = current_sections_map.get(section_name)
            existing_format = "synthesis" if not existing_section or existing_section.get("table") is None else "list"
            format_changed = dim["format"] != existing_format
            section_edited = action in (EditAction.REWRITE, EditAction.NEW)
            if section_json["format"] == "list" and section_json["citations"] and self.run_table_generation \
                    and (section_edited or format_changed):
                dim["idx"] = idx
                cit_ids = [int(c["paper"]["corpus_id"]) for c in section_json["citations"]]
                tthread = self.gen_table_thread(user_id, edit_instruction, dim, cit_ids, tables)
                if tthread:
                    table_threads.append(tthread)

            gen_sec = self.get_gen_sections_from_json(section_json)
            generated_sections.append(gen_sec)

        # Capture CostAwareLLMResult from generator return value
        try:
            next(gen_sections_iter)
        except StopIteration as e:
            all_sections = e.value

        # ====================================================================
        # STEP 6: Table Generation (inherited gen_table_thread from ScholarQA)
        # ====================================================================

        self.update_task_state(
            "Generating comparison tables",
            curr_response=generated_sections,
            step_estimated_time=20
        )

        start = time()
        for tthread in table_threads:
            tthread.join()
        logger.info(f"Table generation wait time: {time() - start:.2f}")

        tcosts = []
        for sidx in range(len(json_summary)):
            tables_val = None
            if sidx < len(tables) and tables[sidx]:
                if type(tables[sidx]) == tuple:
                    tables_val, tcost = tables[sidx]
                    tcosts.append(tcost)
                else:
                    tables_val = tables[sidx]
            json_summary[sidx]["table"] = tables_val.to_dict() if tables_val else None
            generated_sections[sidx].table = tables_val if tables_val else None

        self.postprocess_json_output(json_summary, quotes_meta=quotes_metadata)

        # ====================================================================
        # Finalize and return
        # ====================================================================

        # Build section_models matching json_summary (KEEP + REWRITE/NEW, no DELETE)
        # all_sections.models has one entry per dimension from call_iter_method
        if all_sections:
            section_models = [
                model for dim, model in zip(plan_dimensions, all_sections.models)
                if dim.get("action", EditAction.NEW) != EditAction.DELETE
            ]
            summary_cost_result = CostAwareLLMResult(
                result=all_sections.result,
                tot_cost=all_sections.tot_cost,
                models=section_models,
                tokens=all_sections.tokens,
            )
            event_trace.trace_summary_event(json_summary, summary_cost_result, tcosts)
        event_trace.persist_trace(self.logs_config)

        return TaskResult(
            report_title=self.report_title,
            sections=generated_sections,
            cost=event_trace.total_cost,
            tokens=event_trace.tokens
        )
