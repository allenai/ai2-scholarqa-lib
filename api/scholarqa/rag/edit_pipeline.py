"""
Edit pipeline that mirrors MultiStepQAPipeline but handles report editing.

This pipeline follows the same 4-step structure as the original:
1. Quote extraction from new papers (mirrors step_select_quotes)
2. Planning/clustering with edit context (mirrors step_clustering)
3. Section generation with edit actions (mirrors generate_iterative_summary)
"""

import json
import logging
import re
from enum import Enum
from typing import Dict, List, Any, Tuple, Generator

import pandas as pd
from anyascii import anyascii
from pydantic import Field

from scholarqa.llms.constants import CompletionResult
from scholarqa.llms.edit.prompts import (
    SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT,
    USER_PROMPT_PAPER_LIST_FORMAT_EDIT,
    SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
    USER_PROMPT_QUOTE_LIST_FORMAT_EDIT,
    PROMPT_ASSEMBLE_SUMMARY_EDIT,
    PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY_EDIT,
)
from scholarqa.llms.litellm_helper import batch_llm_completion, llm_completion
from scholarqa.rag.multi_step_qa_pipeline import Dimension, ClusterPlan
from scholarqa.utils import get_ref_author_str, make_int

logger = logging.getLogger(__name__)

# Dummy result for DELETE/KEEP actions (no LLM call, zero cost)
_NOOP_COMPLETION = CompletionResult(
    content=None, model="", cost=0,
    input_tokens=0, output_tokens=0, total_tokens=0, reasoning_tokens=0,
)


class EditAction(str, Enum):
    KEEP = "KEEP"
    REWRITE = "REWRITE"
    DELETE = "DELETE"
    NEW = "NEW"


class EditDimension(Dimension):
    """Dimension with an edit action."""
    action: EditAction = Field(
        default=EditAction.KEEP,
        description="The edit action to perform on this section"
    )


class EditClusterPlan(ClusterPlan):
    """Edit-aware version of ClusterPlan that includes edit actions and paper removals."""
    papers_to_remove: List[str] = Field(
        default=[],
        description="List of corpus_ids to remove from the entire report"
    )
    dimensions: List[EditDimension] = Field(
        description="The list of dimensions with edit actions and associated quote indices"
    )


class EditPipeline:
    """
    Pipeline for editing existing reports.

    Mirrors the structure of MultiStepQAPipeline but with edit-specific logic.
    """

    def __init__(
        self,
        llm_model: str,
        fallback_llm: str = None,
        batch_workers: int = 20,
        **llm_kwargs
    ):
        self.llm_model = llm_model
        self.fallback_llm = fallback_llm
        self.batch_workers = batch_workers
        self.llm_kwargs = {"max_tokens": 4096 * 4}
        if llm_kwargs:
            self.llm_kwargs.update(llm_kwargs)

    def step_select_quotes_edit(
        self,
        original_query: str,
        search_query: str,
        report_context: str,
        scored_df: pd.DataFrame,
    ) -> Tuple[Dict[str, str], List[CompletionResult]]:
        """
        STEP 1: Extract quotes from papers (mirrors MultiStepQAPipeline.step_select_quotes)

        Extended with edit context: original query, search query, and current report sections.

        Args:
            original_query: The original query that generated the report
            search_query: The rewritten search query from intent analysis
            report_context: Pre-formatted report context string
            scored_df: DataFrame with papers (from search or mentioned_papers)

        Returns:
            Tuple of (per_paper_summaries dict, completion_results list)
        """
        logger.info(
            f"Querying {self.llm_model} to extract quotes from papers for edit task "
            f"with {self.batch_workers} parallel workers"
        )

        # Create system prompt with query and report context
        sys_prompt = SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT.format(
            original_query=original_query,
            search_query=search_query,
            report_context=report_context,
        )

        # Prepare messages for each paper
        tup_items = {
            k: v for k, v in zip(
                scored_df["reference_string"],
                scored_df["relevance_judgment_input_expanded"]
            )
        }

        messages = [
            USER_PROMPT_PAPER_LIST_FORMAT_EDIT.format(paper_content=v)
            for k, v in tup_items.items()
        ]

        # Batch LLM completion (same as original)
        completion_results = batch_llm_completion(
            self.llm_model,
            messages=messages,
            system_prompt=sys_prompt,
            max_workers=self.batch_workers,
            fallback=self.fallback_llm,
            **self.llm_kwargs
        )

        # Filter out "None" responses (same as original)
        quotes = [
            cr.content if cr.content != "None" and
            not cr.content.startswith("None\n") and
            not cr.content.startswith("None ")
            else ""
            for cr in completion_results
        ]

        per_paper_summaries = {
            t[0]: quote
            for t, quote in zip(tup_items.items(), quotes)
            if len(quote) > 10
        }
        per_paper_summaries = dict(sorted(per_paper_summaries.items(), key=lambda x: x[0]))

        logger.info(f"Extracted quotes from {len(per_paper_summaries)} papers")
        return per_paper_summaries, completion_results

    def step_clustering_edit(
        self,
        edit_instruction: str,
        report_context: str,
        per_paper_summaries: Dict[str, str],
        intent_analysis: "EditIntentAnalysis" = None,
    ) -> Tuple[Dict[str, Any], CompletionResult]:
        """
        STEP 2: Plan edits (mirrors MultiStepQAPipeline.step_clustering)

        Extended with edit context: current report, edit instruction, and intent analysis.

        Args:
            edit_instruction: The user's edit instruction
            report_context: Pre-formatted report context string
            per_paper_summaries: Quotes extracted from new papers
            intent_analysis: Analysis of edit intent (papers to add/remove, constraints)

        Returns:
            Tuple of (edit_plan dict, completion_result)
        """
        logger.info("Generating edit plan based on new quotes and current report")

        current_report_str = report_context

        # Format quotes with paper reference for section assignment
        quotes = ""
        for idx, (paper, quotes_str) in enumerate(per_paper_summaries.items()):
            quotes_str = quotes_str.replace("\n", "")
            quotes += f"[{idx}] ({paper})\t{quotes_str}" + "\n"

        # Extract intent analysis info
        papers_to_add = []
        papers_to_remove = []
        is_stylistic = False
        target_sections = []
        affects_all_sections = True

        if intent_analysis:
            papers_to_add = intent_analysis.papers_to_add
            papers_to_remove = intent_analysis.papers_to_remove
            is_stylistic = intent_analysis.is_stylistic
            target_sections = intent_analysis.target_sections
            affects_all_sections = intent_analysis.affects_all_sections

        # Create user prompt with edit context and intent analysis
        user_prompt = USER_PROMPT_QUOTE_LIST_FORMAT_EDIT.format(
            edit_instruction=edit_instruction,
            current_report=current_report_str,
            quotes=quotes if quotes else "None - no new papers",
            papers_to_add=", ".join(papers_to_add) if papers_to_add else "None",
            papers_to_remove=", ".join(papers_to_remove) if papers_to_remove else "None",
            is_stylistic=is_stylistic,
            target_sections=", ".join(target_sections) if target_sections else "All sections",
            affects_all_sections=affects_all_sections,
        )

        try:
            response = llm_completion(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
                fallback=self.fallback_llm,
                model=self.llm_model,
                response_format=EditClusterPlan,
                **self.llm_kwargs
            )

            parsed_result = json.loads(response.content)

            # Merge papers_to_remove from intent analysis if not already included
            if papers_to_remove:
                existing_remove = set(parsed_result.get("papers_to_remove", []))
                for pid in papers_to_remove:
                    if pid not in existing_remove:
                        parsed_result.setdefault("papers_to_remove", []).append(pid)

            logger.info(f"Edit plan generated with {len(parsed_result['dimensions'])} dimensions, "
                       f"{len(parsed_result.get('papers_to_remove', []))} papers to remove")
            return parsed_result, response

        except Exception as e:
            logger.warning(f"Error while generating edit plan with {self.llm_model}: {e}")
            raise e

    def generate_iterative_summary_edit(
        self,
        edit_instruction: str,
        current_report: Dict[str, Any],
        per_paper_summaries_extd: Dict[str, Dict[str, Any]],
        plan: List[Dict[str, Any]],
        papers_to_remove: List[str] = None,
    ) -> Generator[Any | None, None, None]:
        """
        STEP 3: Generate/edit sections (mirrors MultiStepQAPipeline.generate_iterative_summary)

        Extended with edit context: current sections, edit actions, and paper removals.

        Args:
            edit_instruction: The user's edit instruction
            current_report: The current report dict being edited
            per_paper_summaries_extd: Extended quotes with inline citations
                (includes both new papers and existing citations merged by the runner)
            plan: Edit plan dimensions list with actions
            papers_to_remove: List of corpus_ids to remove from the report

        Yields:
            CompletionResult for each section
        """
        logger.info("Executing edit plan section by section")

        papers_to_remove = papers_to_remove or []
        papers_to_remove_str = ", ".join(papers_to_remove) if papers_to_remove else "None"

        # Build map from index to quotes (same as original)
        per_paper_summaries_tuples = [
            (ref_string, response)
            for ref_string, response in per_paper_summaries_extd.items()
        ]

        # Build map from section name to current section dict
        current_sections_map = {
            section["title"]: section
            for section in current_report.get("sections", [])
        }

        # Extract plan structure (same format as original but with actions)
        plan_dimensions = plan  # This is the dimensions list with actions
        plan_str = "\n".join([
            f"{dim['name']} ({dim['format']})"
            for dim in plan_dimensions
        ])

        existing_sections = []

        for idx, dim in enumerate(plan_dimensions):
            section_name = dim["name"]
            section_format = dim["format"]
            quote_indices = dim.get("quotes", [])
            action = dim.get("action", EditAction.NEW)

            # Get current section content if it exists
            current_section = current_sections_map.get(section_name)

            # DELETE: yield noop, skip entirely
            if action == EditAction.DELETE:
                logger.info(f"Skipping deleted section: {section_name}")
                yield _NOOP_COMPLETION
                continue

            current_section_content = ""
            if current_section:
                current_section_content = f"{current_section['title']}\n\n"
                if current_section.get("tldr"):
                    current_section_content += f"TLDR: {current_section['tldr']}\n"
                current_section_content += current_section.get("text", "")

            # KEEP: yield noop, runner reuses existing section dict directly
            if action == EditAction.KEEP and current_section:
                logger.info(f"Keeping section unchanged: {section_name}")
                yield _NOOP_COMPLETION
                existing_sections.append(current_section.get("text", ""))
                continue

            # Build new quotes for this section (from search/quote extraction)
            quotes = ""
            for ind in quote_indices:
                if ind < len(per_paper_summaries_tuples):
                    quotes += (
                        per_paper_summaries_tuples[ind][0] + ": " +
                        str(per_paper_summaries_tuples[ind][1]) + "\n"
                    )
                else:
                    logger.warning(f"Quote index {ind} out of bounds")

            # Build existing citations for REWRITE by looking up per_paper_summaries_extd
            # (existing citations were merged into it by the runner in Step 4.5)
            existing_citations_str = ""
            if current_section and current_section.get("citations") and action == EditAction.REWRITE:
                for cit in current_section["citations"]:
                    ref_key = self.citation_ref_key(cit)
                    if ref_key in per_paper_summaries_extd:
                        existing_citations_str += f"{ref_key}: {per_paper_summaries_extd[ref_key]}\n"

            # Format already written sections (same as original)
            already_written = "\n\n".join(existing_sections)
            already_written = re.sub(r"\[.*?\]", "", already_written)

            # Prepare prompt arguments with edit context
            fill_in_prompt_args = {
                "edit_instruction": edit_instruction,
                "plan": plan_str,
                "already_written": already_written,
                "section_name": f"{section_name} ({section_format})",
                "current_section_content": current_section_content,
                "action": action,
                "papers_to_remove": papers_to_remove_str,
                "existing_section_references": existing_citations_str if existing_citations_str else "None",
            }

            # Choose prompt based on whether we have quotes
            if quotes:
                fill_in_prompt_args["section_references"] = quotes
                filled_in_prompt = PROMPT_ASSEMBLE_SUMMARY_EDIT.format(**fill_in_prompt_args)
            else:
                logger.info(f"No quotes for section {section_name}, using no-quotes prompt")
                filled_in_prompt = PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY_EDIT.format(**fill_in_prompt_args)

            # Generate section (same as original)
            response = llm_completion(
                user_prompt=filled_in_prompt,
                model=self.llm_model,
                fallback=self.fallback_llm,
                **self.llm_kwargs
            )

            existing_sections.append(response.content)
            yield response

    # ========================================================================
    # Helper methods for formatting
    # ========================================================================

    @staticmethod
    def format_report_context(report: Dict[str, Any]) -> str:
        """Format report dict for edit prompts (intent analysis, quote extraction, clustering)."""
        lines = []
        title = report.get("report_title", "")
        if title:
            lines.append(f"Title: {title}\n")

        lines.append("Sections:")
        for i, section in enumerate(report.get("sections", [])):
            fmt = "synthesis" if section.get("table") is None else "list"
            lines.append(f"\n{i+1}. {section['title']} ({fmt})")
            if section.get("tldr"):
                lines.append(f"   TLDR: {section['tldr']}")
            text = section.get("text", "")
            text_preview = text[:300] + "..." if len(text) > 300 else text
            lines.append(f"   Content preview: {text_preview}")
            lines.append(f"   Papers cited: {len(section.get('citations', []))}")

        return "\n".join(lines)

    @staticmethod
    def citation_ref_key(citation: Dict[str, Any]) -> str:
        """Generate the [ID | Author | Year | Citations: N] reference key for a citation dict."""
        paper = citation["paper"]
        authors = paper.get("authors") or []
        return anyascii(
            f"[{make_int(paper['corpus_id'])} | {get_ref_author_str(authors)} | "
            f"{make_int(paper.get('year', 0))} | Citations: {make_int(paper.get('n_citations', 0) or 0)}]"
        )

    @staticmethod
    def citation_to_ref_data(citation: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Convert a citation dict into ref_key, per_paper_summaries entry, and paper_metadata entry.

        Called once per citation in the runner to merge existing citations.
        """
        ref_key = EditPipeline.citation_ref_key(citation)
        paper = citation["paper"]
        authors = paper.get("authors") or []
        snippets = citation.get("snippets") or []

        per_paper_entry = {
            "quote": "...".join(snippets),
            "inline_citations": {},
        }

        paper_meta_entry = {
            "corpusId": make_int(paper["corpus_id"]),
            "title": paper.get("title", ""),
            "year": make_int(paper.get("year", 0)),
            "authors": authors,
            "venue": paper.get("venue", ""),
            "citationCount": make_int(paper.get("n_citations", 0) or 0),
            "relevance_judgement": citation.get("score", 0),
        }

        return ref_key, per_paper_entry, paper_meta_entry
