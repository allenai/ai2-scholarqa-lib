"""
Unit tests for edit workflow.

These tests verify structure and data flow WITHOUT calling real LLMs.
All LLM calls are mocked to return realistic data structures.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

from scholarqa.models import TaskResult, GeneratedSection, CitationSrc, PaperDetails, Author, ToolRequest
from scholarqa.rag.edit_pipeline import EditPipeline, EditClusterPlan


# ============================================================================
# Mock Data Factories
# ============================================================================

def create_mock_paper_details(corpus_id: int) -> PaperDetails:
    """Create a mock PaperDetails object."""
    return PaperDetails(
        corpus_id=corpus_id,
        title=f"Test Paper {corpus_id}",
        year=2024,
        venue="Test Conference",
        authors=[Author(name="Test Author", authorId="123")],
        n_citations=100
    )


def create_mock_citation(corpus_id: int) -> CitationSrc:
    """Create a mock CitationSrc object."""
    return CitationSrc(
        id=f"[{corpus_id} | Test Author | 2024 | Citations: 100]",
        paper=create_mock_paper_details(corpus_id),
        snippets=["Test snippet from paper"],
        score=0.9
    )


def create_mock_section(title: str, corpus_ids: List[int]) -> GeneratedSection:
    """Create a mock GeneratedSection object."""
    return GeneratedSection(
        title=title,
        tldr=f"Summary of {title}",
        text=f"This is the content of {title}. [123 | Test | 2024 | Citations: 100]",
        citations=[create_mock_citation(cid) for cid in corpus_ids],
        table=None
    )


def create_mock_report(num_sections: int = 3) -> TaskResult:
    """Create a mock TaskResult (report) object."""
    sections = [
        create_mock_section(f"Section {i+1}", [100 + i, 200 + i])
        for i in range(num_sections)
    ]
    return TaskResult(
        report_title="Test Report",
        sections=sections,
        cost=0.5,
        tokens={"total": 1000}
    )


def create_mock_scored_df(num_papers: int = 5) -> pd.DataFrame:
    """Create a mock scored DataFrame like rerank produces."""
    data = []
    for i in range(num_papers):
        corpus_id = 1000 + i
        data.append({
            "corpus_id": corpus_id,
            "reference_string": f"[{corpus_id} | Author {i} et al. | 2024 | Citations: 50]",
            "relevance_judgment_input_expanded": f"Abstract text for paper {corpus_id}. This contains important content.",
            "title": f"Paper Title {i}",
            "score": 0.9 - (i * 0.1)
        })
    return pd.DataFrame(data)


def create_mock_llm_response(content: str):
    """Create a mock LLM completion result."""
    mock_response = Mock()
    mock_response.content = content
    return mock_response


# ============================================================================
# Test EditPipeline Methods
# ============================================================================

class TestEditPipeline:
    """Test the EditPipeline class methods."""

    def test_init(self):
        """Test EditPipeline initialization."""
        pipeline = EditPipeline(
            llm_model="test-model",
            fallback_llm="fallback-model",
            batch_workers=10
        )

        assert pipeline.llm_model == "test-model"
        assert pipeline.fallback_llm == "fallback-model"
        assert pipeline.batch_workers == 10
        assert pipeline.llm_kwargs["max_tokens"] == 4096 * 4

    @patch('scholarqa.rag.edit_pipeline.batch_llm_completion')
    def test_step_select_quotes_edit(self, mock_batch_llm):
        """Test quote extraction with edit context."""
        # Setup
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report(num_sections=2)
        scored_df = create_mock_scored_df(num_papers=3)

        # Mock LLM responses (one per paper)
        mock_responses = [
            create_mock_llm_response("Quote 1 from paper A... More content here."),
            create_mock_llm_response("Quote 2 from paper B... Additional text."),
            create_mock_llm_response("Quote 3 from paper C... Final quote."),
        ]
        mock_batch_llm.return_value = mock_responses

        # Execute
        quotes, completions = pipeline.step_select_quotes_edit(
            edit_instruction="Add papers about X",
            current_report=current_report,
            scored_df=scored_df
        )

        # Verify
        assert len(quotes) == 3
        assert all(len(quote) > 10 for quote in quotes.values())
        assert mock_batch_llm.called

        # Verify prompts included edit context
        call_args = mock_batch_llm.call_args
        system_prompt = call_args[1]["system_prompt"]
        assert "edit" in system_prompt.lower() or "EDIT" in system_prompt

    @patch('scholarqa.rag.edit_pipeline.llm_completion')
    def test_step_clustering_edit(self, mock_llm):
        """Test edit planning with current report context."""
        # Setup
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report(num_sections=2)
        per_paper_summaries = {
            "[1000 | Author A | 2024]": "Quote from paper 1000",
            "[1001 | Author B | 2024]": "Quote from paper 1001",
        }

        # Mock LLM response with edit plan
        mock_plan = {
            "cot": "Reasoning for edit plan...",
            "report_title": "Updated Test Report",
            "dimensions": [
                {
                    "name": "Section 1",
                    "format": "synthesis",
                    "quotes": [0],
                    "action": "EXPAND"
                },
                {
                    "name": "Section 2",
                    "format": "synthesis",
                    "quotes": [],
                    "action": "KEEP"
                },
                {
                    "name": "New Section",
                    "format": "list",
                    "quotes": [1],
                    "action": "NEW"
                }
            ]
        }
        import json
        mock_llm.return_value = create_mock_llm_response(json.dumps(mock_plan))

        # Execute
        plan, _ = pipeline.step_clustering_edit(
            edit_instruction="Add more detail to section 1",
            current_report=current_report,
            per_paper_summaries=per_paper_summaries
        )

        # Verify
        assert plan["report_title"] == "Updated Test Report"
        assert len(plan["dimensions"]) == 3
        assert plan["dimensions"][0]["action"] == "EXPAND"
        assert plan["dimensions"][1]["action"] == "KEEP"
        assert plan["dimensions"][2]["action"] == "NEW"
        assert mock_llm.called

    @patch('scholarqa.rag.edit_pipeline.llm_completion')
    def test_generate_iterative_summary_edit_keep(self, mock_llm):
        """Test section generation with KEEP action."""
        # Setup
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report(num_sections=1)

        plan_dimensions = [
            {
                "name": "Section 1",
                "format": "synthesis",
                "quotes": [],
                "action": "KEEP"
            }
        ]

        # Execute
        sections = list(pipeline.generate_iterative_summary_edit(
            edit_instruction="Keep everything",
            current_report=current_report,
            per_paper_summaries_extd={},
            plan=plan_dimensions
        ))

        # Verify - KEEP should not call LLM, just return existing content
        assert len(sections) == 1
        assert mock_llm.call_count == 0  # No LLM calls for KEEP

    @patch('scholarqa.rag.edit_pipeline.llm_completion')
    def test_generate_iterative_summary_edit_expand(self, mock_llm):
        """Test section generation with EXPAND action."""
        # Setup
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report(num_sections=1)

        per_paper_summaries_extd = {
            "[1000 | Author | 2024]": {
                "quote": "New quote to add",
                "inline_citations": {}
            }
        }

        plan_dimensions = [
            {
                "name": "Section 1",
                "format": "synthesis",
                "quotes": [0],
                "action": "EXPAND"
            }
        ]

        # Mock LLM response
        mock_llm.return_value = create_mock_llm_response(
            "Section 1\n\nTLDR: Expanded section\n\nOriginal content plus new content [1000 | Author | 2024 | Citations: 100]"
        )

        # Execute
        sections = list(pipeline.generate_iterative_summary_edit(
            edit_instruction="Expand with more detail",
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan_dimensions
        ))

        # Verify
        assert len(sections) == 1
        assert mock_llm.call_count == 1

        # Check that prompt included current section and action
        call_args = mock_llm.call_args[1]
        prompt = call_args["user_prompt"]
        assert "EXPAND" in prompt
        assert "current_section_content" in prompt.lower() or "Section 1" in prompt

    @patch('scholarqa.rag.edit_pipeline.llm_completion')
    def test_generate_iterative_summary_edit_delete(self, mock_llm):
        """Test that DELETE action skips section generation."""
        # Setup
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report(num_sections=2)

        plan_dimensions = [
            {
                "name": "Section 1",
                "format": "synthesis",
                "quotes": [],
                "action": "DELETE"
            },
            {
                "name": "Section 2",
                "format": "synthesis",
                "quotes": [],
                "action": "KEEP"
            }
        ]

        # Execute
        sections = list(pipeline.generate_iterative_summary_edit(
            edit_instruction="Delete first section",
            current_report=current_report,
            per_paper_summaries_extd={},
            plan=plan_dimensions
        ))

        # Verify - DELETE skipped, only KEEP section returned
        assert len(sections) == 1
        assert mock_llm.call_count == 0

    def test_format_sections_for_quote_extraction(self):
        """Test section formatting for prompts."""
        pipeline = EditPipeline(llm_model="test-model")
        report = create_mock_report(num_sections=2)

        formatted = pipeline._format_sections_for_quote_extraction(report.sections)

        assert "1. Section 1" in formatted
        assert "2. Section 2" in formatted
        assert "Summary of Section 1" in formatted

    def test_format_report_summary(self):
        """Test report summary formatting."""
        pipeline = EditPipeline(llm_model="test-model")
        report = create_mock_report(num_sections=2)

        summary = pipeline._format_report_summary(report)

        assert "Title: Test Report" in summary
        assert "Section 1: Section 1" in summary
        assert "Section 2: Section 2" in summary

    def test_format_report_for_clustering(self):
        """Test full report formatting for clustering."""
        pipeline = EditPipeline(llm_model="test-model")
        report = create_mock_report(num_sections=2)

        formatted = pipeline._format_report_for_clustering(report)

        assert "Title: Test Report" in formatted
        assert "Section 1 (synthesis)" in formatted
        assert "Section 2 (synthesis)" in formatted
        assert "Papers cited:" in formatted


# ============================================================================
# Test Error Handling
# ============================================================================

class TestEditPipelineErrors:
    """Test error handling in edit pipeline."""

    @patch('scholarqa.rag.edit_pipeline.batch_llm_completion')
    def test_step_select_quotes_edit_handles_none_responses(self, mock_batch_llm):
        """Test that None responses are filtered out."""
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report()
        scored_df = create_mock_scored_df(num_papers=3)

        # Mock some papers returning "None"
        mock_responses = [
            create_mock_llm_response("Valid quote here..."),
            create_mock_llm_response("None"),  # Should be filtered
            create_mock_llm_response("None\nExtra text"),  # Should be filtered
        ]
        mock_batch_llm.return_value = mock_responses

        quotes, _ = pipeline.step_select_quotes_edit(
            edit_instruction="Test",
            current_report=current_report,
            scored_df=scored_df
        )

        # Only one valid quote should remain
        assert len(quotes) == 1

    @patch('scholarqa.rag.edit_pipeline.llm_completion')
    def test_step_clustering_edit_raises_on_llm_error(self, mock_llm):
        """Test that LLM errors are propagated."""
        pipeline = EditPipeline(llm_model="test-model")
        current_report = create_mock_report()

        # Mock LLM raising an error
        mock_llm.side_effect = Exception("LLM API Error")

        with pytest.raises(Exception) as exc_info:
            pipeline.step_clustering_edit(
                edit_instruction="Test",
                current_report=current_report,
                per_paper_summaries={"[123 | A | 2024]": "quote"}
            )

        assert "LLM API Error" in str(exc_info.value)


# ============================================================================
# Test Data Flow
# ============================================================================

class TestEditWorkflowDataFlow:
    """Test data flow through edit workflow components."""

    def test_mock_report_structure(self):
        """Verify mock report has correct structure."""
        report = create_mock_report(num_sections=3)

        assert isinstance(report, TaskResult)
        assert report.report_title == "Test Report"
        assert len(report.sections) == 3
        assert all(isinstance(s, GeneratedSection) for s in report.sections)
        assert all(len(s.citations) > 0 for s in report.sections)

    def test_mock_scored_df_structure(self):
        """Verify mock DataFrame has correct structure."""
        df = create_mock_scored_df(num_papers=5)

        assert len(df) == 5
        assert "corpus_id" in df.columns
        assert "reference_string" in df.columns
        assert "relevance_judgment_input_expanded" in df.columns
        assert all(df["score"] <= 1.0)

    def test_quote_to_plan_to_section_flow(self):
        """Test that data flows correctly through the pipeline stages."""
        # Stage 1: Quotes extracted
        quotes = {
            "[1000 | A | 2024]": "Quote from paper 1000",
            "[1001 | B | 2024]": "Quote from paper 1001",
        }

        # Stage 2: Plan references quotes by index
        plan = [
            {"name": "Section 1", "format": "synthesis", "quotes": [0], "action": "EXPAND"},
            {"name": "Section 2", "format": "list", "quotes": [1], "action": "NEW"},
        ]

        # Verify indices map correctly
        quote_list = list(quotes.items())
        assert quote_list[0][0] == "[1000 | A | 2024]"
        assert plan[0]["quotes"] == [0]  # References first quote
        assert quote_list[1][0] == "[1001 | B | 2024]"
        assert plan[1]["quotes"] == [1]  # References second quote


# ============================================================================
# Test Integration with ToolRequest
# ============================================================================

class TestToolRequestIntegration:
    """Test that ToolRequest model works with edit workflow."""

    def test_tool_request_edit_mode(self):
        """Test ToolRequest with edit fields."""
        req = ToolRequest(
            task_id="test-123",
            query="Original query",
            edit_existing=True,
            thread_id="thread-456",
            edit_instruction="Add these papers",
            mentioned_papers=[123, 456, 789]
        )

        assert req.edit_existing is True
        assert req.thread_id == "thread-456"
        assert req.edit_instruction == "Add these papers"
        assert len(req.mentioned_papers) == 3
        assert 123 in req.mentioned_papers

    def test_tool_request_normal_mode(self):
        """Test ToolRequest without edit fields (normal generation)."""
        req = ToolRequest(
            task_id="test-123",
            query="New query",
            edit_existing=False
        )

        assert req.edit_existing is False
        assert req.thread_id is None
        assert req.edit_instruction is None
        assert req.mentioned_papers is None


# ============================================================================
# Summary Test
# ============================================================================

def test_all_components_importable():
    """Verify all edit workflow components can be imported."""
    from scholarqa.rag.edit_pipeline import EditPipeline, EditClusterPlan
    from scholarqa.llms.edit_prompts import (
        SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT,
        SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
        PROMPT_ASSEMBLE_SUMMARY_EDIT
    )
    from scholarqa.models import ToolRequest

    # Verify classes are importable and constructible
    assert EditPipeline is not None
    assert EditClusterPlan is not None

    # Verify prompts exist and are strings
    assert isinstance(SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT, str)
    assert isinstance(SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT, str)
    assert isinstance(PROMPT_ASSEMBLE_SUMMARY_EDIT, str)

    # Verify edit prompts mention edit-specific concepts
    assert "edit" in SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT.lower()
    assert "edit" in SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT.lower()
    assert "action" in PROMPT_ASSEMBLE_SUMMARY_EDIT.lower()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
