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
from typing import Dict, List, Any, Tuple, Generator

import pandas as pd
from pydantic import BaseModel, Field

from scholarqa.llms.constants import CompletionResult
from scholarqa.llms.litellm_helper import batch_llm_completion, llm_completion
from scholarqa.llms.edit_prompts import (
    SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT,
    USER_PROMPT_PAPER_LIST_FORMAT_EDIT,
    SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
    USER_PROMPT_QUOTE_LIST_FORMAT_EDIT,
    PROMPT_ASSEMBLE_SUMMARY_EDIT,
    PROMPT_ASSEMBLE_NO_QUOTES_SUMMARY_EDIT,
)
from scholarqa.models import TaskResult, GeneratedSection

logger = logging.getLogger(__name__)


class EditClusterPlan(BaseModel):
    """Edit-aware version of ClusterPlan that includes edit actions."""
    cot: str = Field(description="Chain of thought for edit plan")
    report_title: str = Field(description="Report title (existing or updated)")
    dimensions: List[Dict[str, Any]] = Field(
        description="List of dimensions with edit actions (KEEP, EXPAND, ADD_TO, REPLACE, DELETE, NEW)"
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
        edit_instruction: str,
        current_report: TaskResult,
        scored_df: pd.DataFrame,
    ) -> Tuple[Dict[str, str], List[CompletionResult]]:
        """
        STEP 1: Extract quotes from papers (mirrors MultiStepQAPipeline.step_select_quotes)

        Extended with edit context: current report sections and edit instruction.

        Args:
            edit_instruction: The user's edit instruction
            current_report: The current report being edited
            scored_df: DataFrame with papers (from search or mentioned_papers)

        Returns:
            Tuple of (per_paper_summaries dict, completion_results list)
        """
        logger.info(
            f"Querying {self.llm_model} to extract quotes from papers for edit task "
            f"with {self.batch_workers} parallel workers"
        )

        # Format current report context for the prompt
        current_sections_summary = self._format_sections_for_quote_extraction(
            current_report.sections
        )
        current_report_summary = self._format_report_summary(current_report)

        # Create system prompt with edit context
        sys_prompt = SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT.format(
            current_sections_summary=current_sections_summary,
            edit_instruction=edit_instruction
        )

        # Prepare messages for each paper (same as original but with edit context)
        tup_items = {
            k: v for k, v in zip(
                scored_df["reference_string"],
                scored_df["relevance_judgment_input_expanded"]
            )
        }

        messages = [
            USER_PROMPT_PAPER_LIST_FORMAT_EDIT.format(
                edit_instruction=edit_instruction,
                current_report_summary=current_report_summary,
                paper_content=v
            )
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
        current_report: TaskResult,
        per_paper_summaries: Dict[str, str],
    ) -> Tuple[Dict[str, Any], CompletionResult]:
        """
        STEP 2: Plan edits (mirrors MultiStepQAPipeline.step_clustering)

        Extended with edit context: current report and edit instruction.

        Args:
            edit_instruction: The user's edit instruction
            current_report: The current report being edited
            per_paper_summaries: Quotes extracted from new papers

        Returns:
            Tuple of (edit_plan dict, completion_result)
        """
        logger.info("Generating edit plan based on new quotes and current report")

        # Format current report
        current_report_str = self._format_report_for_clustering(current_report)

        # Format quotes (same as original)
        quotes = ""
        for idx, (paper, quotes_str) in enumerate(per_paper_summaries.items()):
            quotes_str = quotes_str.replace("\n", "")
            quotes += f"[{idx}]\t{quotes_str}" + "\n"

        # Create user prompt with edit context
        user_prompt = USER_PROMPT_QUOTE_LIST_FORMAT_EDIT.format(
            edit_instruction=edit_instruction,
            current_report=current_report_str,
            quotes=quotes
        )

        # Create system prompt with edit context
        sys_prompt = SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT.format(
            current_report=current_report_str,
            edit_instruction=edit_instruction,
            quotes=quotes
        )

        try:
            response = llm_completion(
                user_prompt=user_prompt,
                system_prompt=sys_prompt,
                fallback=self.fallback_llm,
                model=self.llm_model,
                response_format=EditClusterPlan,
                **self.llm_kwargs
            )

            parsed_result = json.loads(response.content)
            logger.info(f"Edit plan generated with {len(parsed_result['dimensions'])} dimensions")
            return parsed_result, response

        except Exception as e:
            logger.warning(f"Error while generating edit plan with {self.llm_model}: {e}")
            raise e

    def generate_iterative_summary_edit(
        self,
        edit_instruction: str,
        current_report: TaskResult,
        per_paper_summaries_extd: Dict[str, Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> Generator[CompletionResult, None, None]:
        """
        STEP 3: Generate/edit sections (mirrors MultiStepQAPipeline.generate_iterative_summary)

        Extended with edit context: current sections and edit actions.

        Args:
            edit_instruction: The user's edit instruction
            current_report: The current report being edited
            per_paper_summaries_extd: Extended quotes with inline citations
            plan: Edit plan from step_clustering_edit

        Yields:
            CompletionResult for each section
        """
        logger.info("Executing edit plan section by section")

        # Build map from index to quotes (same as original)
        per_paper_summaries_tuples = [
            (ref_string, response)
            for ref_string, response in per_paper_summaries_extd.items()
        ]

        # Build map from section name to current section content
        current_sections_map = {
            section.title: section
            for section in current_report.sections
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
            action = dim.get("action", "NEW")

            # Get current section content if it exists
            current_section = current_sections_map.get(section_name)
            current_section_content = ""
            if current_section:
                current_section_content = f"Title: {current_section.title}\n\n"
                if current_section.tldr:
                    current_section_content += f"TLDR: {current_section.tldr}\n\n"
                current_section_content += current_section.text

            # Skip deleted sections
            if action == "DELETE":
                logger.info(f"Skipping deleted section: {section_name}")
                continue

            # For KEEP action, return current content as-is
            if action == "KEEP" and current_section:
                logger.info(f"Keeping section unchanged: {section_name}")
                # Create a dummy completion result with existing content
                class KeepResult:
                    content = current_section.text
                yield KeepResult()
                existing_sections.append(current_section.text)
                continue

            # Build quotes for this section (same as original)
            quotes = ""
            for ind in quote_indices:
                if ind < len(per_paper_summaries_tuples):
                    quotes += (
                        per_paper_summaries_tuples[ind][0] + ": " +
                        str(per_paper_summaries_tuples[ind][1]) + "\n"
                    )
                else:
                    logger.warning(f"Quote index {ind} out of bounds")

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
            }

            # Choose prompt based on whether we have quotes
            if quotes:
                fill_in_prompt_args["section_references"] = quotes
                filled_in_prompt = PROMPT_ASSEMBLE_SUMMARY_EDIT.format(**fill_in_prompt_args)
            else:
                logger.warning(f"No quotes for section {section_name}, using no-quotes prompt")
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

    def _format_sections_for_quote_extraction(self, sections: List[GeneratedSection]) -> str:
        """Format section titles and summaries for quote extraction prompt."""
        lines = []
        for i, section in enumerate(sections):
            lines.append(f"{i+1}. {section.title}")
            if section.tldr:
                lines.append(f"   {section.tldr}")
        return "\n".join(lines)

    def _format_report_summary(self, report: TaskResult) -> str:
        """Format report summary for prompts."""
        lines = []
        if report.report_title:
            lines.append(f"Title: {report.report_title}\n")

        for i, section in enumerate(report.sections):
            lines.append(f"Section {i+1}: {section.title}")
            if section.tldr:
                lines.append(f"  {section.tldr}")

        return "\n".join(lines)

    def _format_report_for_clustering(self, report: TaskResult) -> str:
        """Format full report for clustering/planning prompt."""
        lines = []
        if report.report_title:
            lines.append(f"Title: {report.report_title}\n")

        lines.append("Sections:")
        for i, section in enumerate(report.sections):
            lines.append(f"\n{i+1}. {section.title} ({'synthesis' if section.table is None else 'list'})")
            if section.tldr:
                lines.append(f"   TLDR: {section.tldr}")
            # Include snippet of text
            text_preview = section.text[:300] + "..." if len(section.text) > 300 else section.text
            lines.append(f"   Content preview: {text_preview}")
            lines.append(f"   Papers cited: {len(section.citations)}")

        return "\n".join(lines)
