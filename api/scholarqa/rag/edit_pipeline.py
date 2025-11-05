"""
Edit pipeline for modifying existing reports based on user instructions.

This pipeline implements the three-step edit workflow:
1. Decide if new search is needed
2. Generate per-section edit plan
3. Execute edits section by section
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from scholarqa.llms.constants import CompletionResult
from scholarqa.llms.litellm_helper import llm_completion
from scholarqa.llms.edit_prompts import (
    PROMPT_DECIDE_SEARCH,
    PROMPT_GENERATE_EDIT_PLAN,
    PROMPT_EXECUTE_SECTION_EDIT,
    PROMPT_CREATE_NEW_SECTION,
)
from scholarqa.models_edit import (
    SearchDecision,
    EditPlan,
    SectionEditPlan,
    EditAction,
)
from scholarqa.models import TaskResult, GeneratedSection

logger = logging.getLogger(__name__)


class EditPipeline:
    """Pipeline for editing existing reports based on user instructions."""

    def __init__(
        self,
        llm_model: str,
        fallback_llm: str = None,
        **llm_kwargs
    ):
        self.llm_model = llm_model
        self.fallback_llm = fallback_llm
        self.llm_kwargs = {"max_tokens": 4096 * 4}
        if llm_kwargs:
            self.llm_kwargs.update(llm_kwargs)

    def step_decide_search(
        self,
        current_report: TaskResult,
        edit_instruction: str,
        mentioned_papers: List[int] = None,
    ) -> Tuple[SearchDecision, CompletionResult]:
        """
        Step 1a: Decide if we need to search for new papers.

        Args:
            current_report: The current report to be edited
            edit_instruction: The decontextualized edit instruction
            mentioned_papers: List of corpus_ids mentioned by the user

        Returns:
            Tuple of (SearchDecision, CompletionResult)
        """
        logger.info("Running Edit Step 1a: Deciding if search is needed")

        # Format the current report for the prompt
        report_summary = self._format_report_summary(current_report)

        # Format mentioned papers
        papers_str = "None"
        if mentioned_papers:
            papers_str = ", ".join(str(pid) for pid in mentioned_papers)

        # Create the prompt
        prompt = PROMPT_DECIDE_SEARCH.format(
            current_report=report_summary,
            edit_instruction=edit_instruction,
            mentioned_papers=papers_str,
        )

        try:
            response = llm_completion(
                user_prompt=prompt,
                model=self.llm_model,
                fallback=self.fallback_llm,
                response_format=SearchDecision,
                **self.llm_kwargs
            )

            decision = SearchDecision(**json.loads(response.content))
            logger.info(
                f"Search decision: needs_search={decision.needs_search}, "
                f"query={decision.search_query}"
            )
            return decision, response

        except Exception as e:
            logger.error(f"Error in step_decide_search: {e}")
            raise

    def step_generate_edit_plan(
        self,
        current_report: TaskResult,
        edit_instruction: str,
        mentioned_papers: List[int] = None,
        available_papers: Dict[str, Any] = None,
    ) -> Tuple[EditPlan, CompletionResult]:
        """
        Step 2: Generate a per-section edit plan.

        Args:
            current_report: The current report to be edited
            edit_instruction: The decontextualized edit instruction
            mentioned_papers: List of corpus_ids mentioned by the user
            available_papers: Dict of corpus_id -> paper info (from search or mentioned)

        Returns:
            Tuple of (EditPlan, CompletionResult)
        """
        logger.info("Running Edit Step 2: Generating edit plan")

        # Format sections summary
        sections_summary = self._format_sections_summary(current_report.sections)

        # Format mentioned papers
        papers_str = "None"
        if mentioned_papers:
            papers_str = ", ".join(str(pid) for pid in mentioned_papers)

        # Format available papers
        available_papers_str = "None"
        if available_papers:
            available_papers_str = self._format_available_papers(available_papers)

        # Create the prompt
        prompt = PROMPT_GENERATE_EDIT_PLAN.format(
            report_title=current_report.report_title or "Research Report",
            sections_summary=sections_summary,
            edit_instruction=edit_instruction,
            mentioned_papers=papers_str,
            available_papers=available_papers_str,
        )

        try:
            response = llm_completion(
                user_prompt=prompt,
                model=self.llm_model,
                fallback=self.fallback_llm,
                response_format=EditPlan,
                **self.llm_kwargs
            )

            plan = EditPlan(**json.loads(response.content))
            logger.info(
                f"Edit plan generated with {len(plan.section_plans)} section plans "
                f"and {len(plan.new_sections)} new sections"
            )
            return plan, response

        except Exception as e:
            logger.error(f"Error in step_generate_edit_plan: {e}")
            raise

    def step_execute_section_edit(
        self,
        section: GeneratedSection,
        section_index: int,
        section_plan: SectionEditPlan,
        full_report: TaskResult,
        papers_data: Dict[int, Dict[str, Any]] = None,
    ) -> Tuple[Optional[GeneratedSection], CompletionResult]:
        """
        Step 3: Execute the edit for a single section.

        Args:
            section: The section to edit
            section_index: Index of the section in the report
            section_plan: The plan for editing this section
            full_report: The full report for context
            papers_data: Dict of corpus_id -> paper data (includes quotes/snippets)

        Returns:
            Tuple of (edited section or None if deleted, CompletionResult)
        """
        logger.info(
            f"Executing edit for section {section_index}: "
            f"{section.title} (action: {section_plan.action})"
        )

        # If action is "keep", return the original section unchanged
        if section_plan.action == EditAction.KEEP:
            logger.info(f"Keeping section {section_index} unchanged")
            return section, None

        # If action is "delete", return None
        if section_plan.action == EditAction.DELETE:
            logger.info(f"Deleting section {section_index}")
            return None, None

        # For all other actions, we need to generate new content
        # Format the section references
        section_references = self._format_section_references(
            section,
            section_plan.new_papers,
            papers_data,
        )

        # Format full report context (other sections)
        full_report_context = self._format_report_context(
            full_report,
            exclude_section_index=section_index,
        )

        # Create the prompt
        prompt = PROMPT_EXECUTE_SECTION_EDIT.format(
            section_title=section.title,
            section_content=section.text,
            edit_action=section_plan.action.value,
            specific_instruction=section_plan.specific_instruction or section_plan.reasoning,
            full_report_context=full_report_context,
            section_references=section_references,
        )

        try:
            response = llm_completion(
                user_prompt=prompt,
                model=self.llm_model,
                fallback=self.fallback_llm,
                **self.llm_kwargs
            )

            # Create updated section
            edited_section = GeneratedSection(
                title=section.title,
                tldr=section.tldr,  # Will be regenerated in post-processing
                text=response.content,
                citations=section.citations,  # Will be updated in post-processing
                table=section.table if section_plan.action != EditAction.REPLACE else None,
            )

            logger.info(f"Successfully edited section {section_index}")
            return edited_section, response

        except Exception as e:
            logger.error(f"Error in step_execute_section_edit: {e}")
            raise

    def step_create_new_section(
        self,
        section_title: str,
        section_instruction: str,
        full_report: TaskResult,
        papers_data: Dict[int, Dict[str, Any]] = None,
        paper_ids: List[int] = None,
    ) -> Tuple[GeneratedSection, CompletionResult]:
        """
        Create a brand new section to add to the report.

        Args:
            section_title: Title for the new section
            section_instruction: Instruction for what to include
            full_report: The full report for context
            papers_data: Dict of corpus_id -> paper data
            paper_ids: List of corpus_ids to use for this section

        Returns:
            Tuple of (new section, CompletionResult)
        """
        logger.info(f"Creating new section: {section_title}")

        # Format references
        section_references = "None"
        if paper_ids and papers_data:
            refs = []
            for pid in paper_ids:
                if pid in papers_data:
                    paper_info = papers_data[pid]
                    refs.append(
                        f"[{pid} | {paper_info.get('author_str', 'Unknown')} | "
                        f"Citations: {paper_info.get('n_citations', 0)}]\n"
                        f"{paper_info.get('content', '')}"
                    )
            section_references = "\n\n".join(refs)

        # Format full report
        full_report_str = self._format_report_summary(full_report)

        # Create the prompt
        prompt = PROMPT_CREATE_NEW_SECTION.format(
            full_report=full_report_str,
            section_title=section_title,
            section_instruction=section_instruction,
            section_references=section_references,
        )

        try:
            response = llm_completion(
                user_prompt=prompt,
                model=self.llm_model,
                fallback=self.fallback_llm,
                **self.llm_kwargs
            )

            # Create new section
            new_section = GeneratedSection(
                title=section_title,
                tldr="",  # Will be generated in post-processing
                text=response.content,
                citations=[],  # Will be populated in post-processing
                table=None,
            )

            logger.info(f"Successfully created new section: {section_title}")
            return new_section, response

        except Exception as e:
            logger.error(f"Error in step_create_new_section: {e}")
            raise

    # Helper methods for formatting

    def _format_report_summary(self, report: TaskResult) -> str:
        """Format a report for inclusion in prompts."""
        lines = []
        if report.report_title:
            lines.append(f"Title: {report.report_title}\n")

        lines.append("Sections:")
        for i, section in enumerate(report.sections):
            lines.append(f"\n{i+1}. {section.title}")
            if section.tldr:
                lines.append(f"   Summary: {section.tldr}")
            # Include first 200 chars of text
            text_preview = section.text[:200] + "..." if len(section.text) > 200 else section.text
            lines.append(f"   Preview: {text_preview}")
            lines.append(f"   Citations: {len(section.citations)} papers")

        return "\n".join(lines)

    def _format_sections_summary(self, sections: List[GeneratedSection]) -> str:
        """Format sections for the edit plan prompt."""
        lines = []
        for i, section in enumerate(sections):
            lines.append(f"\n[Section {i}]")
            lines.append(f"Title: {section.title}")
            if section.tldr:
                lines.append(f"Summary: {section.tldr}")
            lines.append(f"Citations: {len(section.citations)} papers")
            # Include corpus IDs
            if section.citations:
                corpus_ids = [str(cit.paper.corpus_id) for cit in section.citations[:5]]
                lines.append(f"Sample papers: {', '.join(corpus_ids)}")

        return "\n".join(lines)

    def _format_available_papers(self, papers: Dict[int, Any]) -> str:
        """Format available papers for the edit plan prompt."""
        lines = []
        for corpus_id, paper_info in papers.items():
            if isinstance(paper_info, dict):
                title = paper_info.get('title', 'Unknown')
                author = paper_info.get('author_str', 'Unknown')
                lines.append(f"[{corpus_id}] {author} - {title}")
            else:
                lines.append(f"[{corpus_id}] {paper_info}")

        return "\n".join(lines)

    def _format_section_references(
        self,
        section: GeneratedSection,
        new_paper_ids: List[int],
        papers_data: Dict[int, Dict[str, Any]],
    ) -> str:
        """Format references for a section edit."""
        refs = []

        # Include existing citations
        for cit in section.citations:
            paper = cit.paper
            snippets = "\n".join(cit.snippets) if cit.snippets else ""
            refs.append(
                f"[{paper.corpus_id} | {', '.join(a.name for a in paper.authors[:2])}, "
                f"{paper.year} | Citations: {paper.n_citations}]\n"
                f"Title: {paper.title}\n"
                f"Snippets: {snippets}"
            )

        # Include new papers if provided
        if new_paper_ids and papers_data:
            for pid in new_paper_ids:
                if pid in papers_data:
                    paper_info = papers_data[pid]
                    refs.append(
                        f"[{pid} | {paper_info.get('author_str', 'Unknown')} | "
                        f"Citations: {paper_info.get('n_citations', 0)}]\n"
                        f"Content: {paper_info.get('content', '')}"
                    )

        return "\n\n".join(refs) if refs else "No references available"

    def _format_report_context(
        self,
        report: TaskResult,
        exclude_section_index: int = None,
    ) -> str:
        """Format the full report context, optionally excluding one section."""
        lines = []
        if report.report_title:
            lines.append(f"Report Title: {report.report_title}\n")

        lines.append("Other sections in the report:")
        for i, section in enumerate(report.sections):
            if i == exclude_section_index:
                continue
            lines.append(f"\n{section.title}")
            if section.tldr:
                lines.append(f"{section.tldr}")

        return "\n".join(lines)
