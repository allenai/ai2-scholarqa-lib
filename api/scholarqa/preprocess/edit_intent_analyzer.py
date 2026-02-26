"""
Edit Intent Analyzer - Analyzes edit instructions to determine search needs and operations.

Similar to query_preprocessor.py but specialized for edit operations.
"""

import json
import logging
from typing import List, Dict, Any

from pydantic import BaseModel, Field, field_validator

from scholarqa.llms.litellm_helper import llm_completion
from scholarqa.models import ReportEditRequest

logger = logging.getLogger(__name__)


class EditIntentAnalysis(BaseModel):
    """
    Result of analyzing an edit instruction.

    Redundant fields removed - infer from values:
    - needs_search: True if search_query is non-empty
    - is_addition: True if papers_to_add is non-empty OR search_query is non-empty
    - is_removal: True if papers_to_remove is non-empty
    """

    # Chain of thought reasoning
    cot: str = Field(description="Chain of thought reasoning for the analysis")

    # Search query (empty = no search needed)
    search_query: str = Field(default="", description="Composed search query. Empty means no search needed.")

    # Search constraints (applied when search_query is non-empty)
    earliest_year: str = Field(default="", description="Earliest publication year filter")
    latest_year: str = Field(default="", description="Latest publication year filter")
    venues: str = Field(default="", description="Comma-separated venue names filter")
    authors: List[str] = Field(default=[], description="List of author names filter")
    field_of_study: str = Field(default="", description="Fields of study filter")
    min_citations: int = Field(default=0, description="Minimum citation count filter")

    # Stylistic flag (no paper changes, just rewriting)
    is_stylistic: bool = Field(description="Whether making stylistic/structural changes only (no paper changes)")

    # Paper operations (non-empty = that operation is requested)
    # NOTE: papers_to_remove should contain RESOLVED corpus_ids from constraint-based removal
    papers_to_add: List[str] = Field(default=[], description="corpus_ids to add to the report")
    papers_to_remove: List[str] = Field(default=[],
                                        description="corpus_ids to remove from the report (resolved from constraints)")

    # Section targeting
    target_sections: List[str] = Field(default=[], description="Specific section titles mentioned in instruction")
    affects_all_sections: bool = Field(description="Whether edit affects entire report vs specific sections")

    # Validators to handle LLM returning empty strings instead of empty lists
    @field_validator('authors', 'papers_to_add', 'papers_to_remove', 'target_sections', mode='before')
    @classmethod
    def convert_empty_to_list(cls, v):
        """Convert empty strings, None, or invalid types to empty lists."""
        if v is None or v == "" or v == "None" or v == "[]":
            return []
        if isinstance(v, str):
            # Try to parse as comma-separated if it looks like a list
            if v.strip():
                return [item.strip() for item in v.split(',') if item.strip()]
            return []
        if isinstance(v, list):
            # Filter out empty strings and None values from the list
            return [str(item) for item in v if item is not None and item != "" and item != "None"]
        return []

    @field_validator('is_stylistic', 'affects_all_sections', mode='before')
    @classmethod
    def convert_to_bool(cls, v):
        """Convert string booleans to actual booleans."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ('true', 'yes', '1')
        return bool(v) if v is not None else False

    @field_validator('min_citations', mode='before')
    @classmethod
    def convert_to_int(cls, v):
        """Convert string integers to actual integers."""
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            if v.strip() == "" or v.lower() == "none":
                return 0
            try:
                return int(v)
            except ValueError:
                return 0
        return 0

    # Computed properties
    @property
    def needs_search(self) -> bool:
        """Whether to search for new papers."""
        return bool(self.search_query)

    @property
    def is_addition(self) -> bool:
        """Whether adding papers (via search or explicit corpus_ids)."""
        return bool(self.search_query) or bool(self.papers_to_add)

    @property
    def is_removal(self) -> bool:
        """Whether removing papers."""
        return bool(self.papers_to_remove)

    def to_search_filters(self) -> Dict[str, Any]:
        """Convert analysis to search filters matching the format from decompose_query / LLMProcessedQuery."""
        filters = {}
        if self.earliest_year or self.latest_year:
            filters['year'] = f"{self.earliest_year}-{self.latest_year}"
        if self.venues:
            filters['venue'] = self.venues
        if self.authors:
            filters['authors'] = self.authors
        if self.field_of_study:
            filters['fieldsOfStudy'] = self.field_of_study
        if self.min_citations:
            filters['minCitationCount'] = self.min_citations
        return filters


def _format_citations_for_prompt(current_report: Dict[str, Any]) -> str:
    """
    Format all citations from the current report dict for the intent analyzer prompt.
    This allows the LLM to resolve constraint-based removal to actual corpus_ids.
    """
    citations_list = []
    seen_corpus_ids = set()

    for section in current_report.get("sections", []):
        for citation in section.get("citations", []):
            paper = citation.get("paper", {})
            corpus_id = str(paper.get("corpus_id", ""))
            if corpus_id in seen_corpus_ids:
                continue
            seen_corpus_ids.add(corpus_id)

            authors = paper.get("authors") or []
            citation_info = {
                "corpus_id": corpus_id,
                "title": paper.get("title", ""),
                "year": paper.get("year", 0),
                "venue": paper.get("venue", ""),
                "authors": [a["name"] if isinstance(a, dict) else a.name for a in authors],
                "citations": paper.get("n_citations", 0),
                "section": section.get("title", ""),
                "relevance_score": paper.get("score", 0)
            }
            citations_list.append(citation_info)

    if not citations_list:
        return "No papers currently cited in the report."

    return json.dumps(citations_list, indent=2)


def analyze_edit_intent(
        req: ReportEditRequest,
        report_context: str,
        current_report: Dict[str, Any],
        llm_model: str,
        fallback_llm: str = None,
        **llm_kwargs
) -> EditIntentAnalysis:
    """
    Analyze an edit instruction to determine search needs and operations.

    Args:
        req: The edit request (ToolRequest or ReportEditRequest) containing:
            - query: Original query that generated the report
            - edit_instruction/intent: The user's edit instruction
            - mentioned_papers/corpus_ids: List of corpus_ids mentioned by the user
            - section_titles: List of section titles mentioned by the user
        report_context: Pre-formatted report context string
        current_report: The current report being edited (for citation extraction)
        llm_model: LLM model to use for analysis
        fallback_llm: Fallback LLM model
        **llm_kwargs: Additional LLM arguments

    Returns:
        EditIntentAnalysis with all decisions and extracted information
    """
    edit_instruction = req.intent
    corpus_ids = req.corpus_ids or []
    section_titles = req.section_titles or []

    original_query = req.query or ""

    logger.info(f"Analyzing edit intent: {edit_instruction[:100]}...")

    # Format current report citations for constraint resolution
    citations_json = _format_citations_for_prompt(current_report)

    # Import prompt here to avoid circular imports
    from scholarqa.llms.edit.prompts import EDIT_INTENT_ANALYZER_PROMPT

    # Format prompt
    prompt = EDIT_INTENT_ANALYZER_PROMPT.format(
        original_query=original_query or "Not specified",
        report_context=report_context,
        edit_instruction=edit_instruction,
        corpus_ids=", ".join(str(c) for c in corpus_ids) if corpus_ids else "None",
        section_titles=", ".join(section_titles) if section_titles else "None",
        current_citations=citations_json
    )

    # Set default kwargs
    kwargs = {"max_tokens": 4096}
    if llm_kwargs:
        kwargs.update(llm_kwargs)

    try:
        response = llm_completion(
            user_prompt=prompt,
            system_prompt="You are an expert at analyzing research report edit instructions. Output valid JSON only.",
            model=llm_model,
            fallback=fallback_llm,
            response_format=EditIntentAnalysis,
            **kwargs
        )

        result = json.loads(response.content)
        analysis = EditIntentAnalysis(**result)

        logger.info(
            f"Intent analysis complete"
        )

        # Validate: if corpus_ids were provided by user and this is an addition, include them
        if corpus_ids and not analysis.is_removal and not analysis.is_stylistic:
            existing = set(analysis.papers_to_add)
            for cid in corpus_ids:
                if str(cid) not in existing:
                    analysis.papers_to_add.append(str(cid))

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing edit intent: {e}")
        # Return a safe default that will trigger search
        raise e
