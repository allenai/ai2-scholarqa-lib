"""
Data models for the edit workflow.
These models support editing existing reports based on user instructions.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class SearchDecision(BaseModel):
    """Decision from Step 1a about whether to perform a new search."""
    needs_search: bool = Field(
        description="Whether a new search is needed to fulfill the edit instruction"
    )
    search_query: Optional[str] = Field(
        default=None,
        description="The search query to use if needs_search is True"
    )
    reasoning: str = Field(
        description="Chain-of-thought reasoning for the search decision"
    )


class EditAction(str, Enum):
    """Types of actions that can be taken on a section."""
    KEEP = "keep"  # Keep section unchanged
    EXPAND = "expand"  # Expand section with more detail
    ADD_PAPERS = "add_papers"  # Add specific papers to section
    DELETE = "delete"  # Remove section entirely
    GO_DEEPER = "go_deeper"  # Go deeper on the topic with more analysis
    REPLACE = "replace"  # Replace section with new content
    MODIFY = "modify"  # Modify section based on specific instruction


class SectionEditPlan(BaseModel):
    """Plan for editing a specific section."""
    section_index: int = Field(
        description="Index of the section in the current report (0-based)"
    )
    section_title: str = Field(
        description="Title of the section to edit"
    )
    action: EditAction = Field(
        description="The action to take on this section"
    )
    reasoning: str = Field(
        description="Reasoning for this action"
    )
    new_papers: List[int] = Field(
        default=[],
        description="Corpus IDs of papers to add to this section"
    )
    specific_instruction: Optional[str] = Field(
        default=None,
        description="Specific instruction for how to modify this section"
    )


class EditPlan(BaseModel):
    """Overall plan for editing the report."""
    cot: str = Field(
        description="Chain-of-thought reasoning for the overall edit plan"
    )
    section_plans: List[SectionEditPlan] = Field(
        description="List of plans for each section"
    )
    new_sections: List[Dict[str, Any]] = Field(
        default=[],
        description="New sections to add (each with 'title', 'position', 'papers', 'instruction')"
    )


class EditContext(BaseModel):
    """Context information for performing an edit."""
    thread_id: str = Field(
        description="ID of the thread containing the report to edit"
    )
    edit_instruction: str = Field(
        description="Decontextualized natural language description of the edit"
    )
    mentioned_papers: List[int] = Field(
        default=[],
        description="Corpus IDs of papers mentioned by the user"
    )
    current_report: Optional[Dict[str, Any]] = Field(
        default=None,
        description="The current report to be edited (TaskResult as dict)"
    )


class EditResult(BaseModel):
    """Summary of changes made during the edit."""
    summary: str = Field(
        description="High-level summary of what was changed"
    )
    sections_modified: List[int] = Field(
        default=[],
        description="Indices of sections that were modified"
    )
    sections_added: List[int] = Field(
        default=[],
        description="Indices of sections that were added"
    )
    sections_deleted: List[int] = Field(
        default=[],
        description="Indices of sections that were deleted"
    )
    papers_added: List[int] = Field(
        default=[],
        description="Corpus IDs of papers that were added"
    )
