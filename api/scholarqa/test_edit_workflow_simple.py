"""
Simple structural tests for edit workflow (no pytest required).

Run with: python3 test_edit_workflow_simple.py
"""

import sys
import pandas as pd
from unittest.mock import Mock, patch
import json

# Add parent dir to path
sys.path.insert(0, '..')

from scholarqa.models import TaskResult, GeneratedSection, CitationSrc, PaperDetails, Author, ToolRequest
from scholarqa.rag.edit_pipeline import EditPipeline, EditClusterPlan


def create_mock_section(title: str) -> GeneratedSection:
    """Create a mock section."""
    return GeneratedSection(
        title=title,
        tldr=f"Summary of {title}",
        text=f"Content of {title}",
        citations=[],
        table=None
    )


def create_mock_report() -> TaskResult:
    """Create a mock report."""
    return TaskResult(
        report_title="Test Report",
        sections=[
            create_mock_section("Section 1"),
            create_mock_section("Section 2"),
        ],
        cost=0.5,
        tokens={"total": 1000}
    )


def create_mock_scored_df() -> pd.DataFrame:
    """Create mock scored DataFrame."""
    return pd.DataFrame([
        {
            "corpus_id": 1000,
            "reference_string": "[1000 | Author A | 2024 | Citations: 50]",
            "relevance_judgment_input_expanded": "Abstract text for paper 1000",
            "score": 0.9
        },
        {
            "corpus_id": 1001,
            "reference_string": "[1001 | Author B | 2024 | Citations: 30]",
            "relevance_judgment_input_expanded": "Abstract text for paper 1001",
            "score": 0.8
        }
    ])


def test_edit_pipeline_init():
    """Test EditPipeline can be initialized."""
    print("Testing EditPipeline initialization...")
    pipeline = EditPipeline(
        llm_model="test-model",
        fallback_llm="fallback",
        batch_workers=5
    )
    assert pipeline.llm_model == "test-model"
    assert pipeline.fallback_llm == "fallback"
    assert pipeline.batch_workers == 5
    print("✅ PASS: EditPipeline initializes correctly")


def test_mock_data_structures():
    """Test that mock data has correct structure."""
    print("\nTesting mock data structures...")

    # Test mock report
    report = create_mock_report()
    assert isinstance(report, TaskResult)
    assert report.report_title == "Test Report"
    assert len(report.sections) == 2
    print("✅ PASS: Mock report structure valid")

    # Test mock DataFrame
    df = create_mock_scored_df()
    assert len(df) == 2
    assert "corpus_id" in df.columns
    assert "reference_string" in df.columns
    print("✅ PASS: Mock DataFrame structure valid")


def test_tool_request_edit_fields():
    """Test ToolRequest with edit fields."""
    print("\nTesting ToolRequest edit fields...")

    req = ToolRequest(
        task_id="test-123",
        query="Test query",
        edit_existing=True,
        thread_id="thread-456",
        edit_instruction="Add papers",
        mentioned_papers=[123, 456]
    )

    assert req.edit_existing is True
    assert req.thread_id == "thread-456"
    assert req.edit_instruction == "Add papers"
    assert len(req.mentioned_papers) == 2
    print("✅ PASS: ToolRequest edit fields work correctly")


def test_format_methods():
    """Test EditPipeline formatting methods."""
    print("\nTesting EditPipeline formatting methods...")

    pipeline = EditPipeline(llm_model="test")
    report = create_mock_report()

    # Test section formatting
    formatted_sections = pipeline._format_sections_for_quote_extraction(report.sections)
    assert "1. Section 1" in formatted_sections
    assert "2. Section 2" in formatted_sections
    print("✅ PASS: _format_sections_for_quote_extraction works")

    # Test report summary
    summary = pipeline._format_report_summary(report)
    assert "Title: Test Report" in summary
    assert "Section 1" in summary
    print("✅ PASS: _format_report_summary works")

    # Test full report formatting
    full = pipeline._format_report_for_clustering(report)
    assert "Test Report" in full
    assert "synthesis" in full
    print("✅ PASS: _format_report_for_clustering works")


@patch('scholarqa.rag.edit_pipeline.batch_llm_completion')
def test_step_select_quotes_edit_mock(mock_batch_llm):
    """Test quote extraction with mocked LLM."""
    print("\nTesting step_select_quotes_edit with mock...")

    pipeline = EditPipeline(llm_model="test-model")
    report = create_mock_report()
    df = create_mock_scored_df()

    # Mock LLM responses
    mock_responses = []
    for _ in range(len(df)):
        mock_resp = Mock()
        mock_resp.content = "This is a valid quote from the paper... More content here."
        mock_responses.append(mock_resp)

    mock_batch_llm.return_value = mock_responses

    # Call the method
    quotes, completions = pipeline.step_select_quotes_edit(
        edit_instruction="Add papers about X",
        current_report=report,
        scored_df=df
    )

    # Verify
    assert len(quotes) == 2  # Should have 2 valid quotes
    assert mock_batch_llm.called
    print("✅ PASS: step_select_quotes_edit works with mocks")


@patch('scholarqa.rag.edit_pipeline.llm_completion')
def test_step_clustering_edit_mock(mock_llm):
    """Test clustering with mocked LLM."""
    print("\nTesting step_clustering_edit with mock...")

    pipeline = EditPipeline(llm_model="test-model")
    report = create_mock_report()
    quotes = {
        "[1000 | Author A | 2024]": "Quote from paper 1000",
        "[1001 | Author B | 2024]": "Quote from paper 1001",
    }

    # Mock plan response
    mock_plan = {
        "cot": "Reasoning...",
        "report_title": "Test Report",
        "dimensions": [
            {"name": "Section 1", "format": "synthesis", "quotes": [0], "action": "EXPAND"},
            {"name": "Section 2", "format": "synthesis", "quotes": [], "action": "KEEP"},
        ]
    }

    mock_resp = Mock()
    mock_resp.content = json.dumps(mock_plan)
    mock_llm.return_value = mock_resp

    # Call the method
    plan, _ = pipeline.step_clustering_edit(
        edit_instruction="Expand section 1",
        current_report=report,
        per_paper_summaries=quotes
    )

    # Verify
    assert len(plan["dimensions"]) == 2
    assert plan["dimensions"][0]["action"] == "EXPAND"
    assert plan["dimensions"][1]["action"] == "KEEP"
    assert mock_llm.called
    print("✅ PASS: step_clustering_edit works with mocks")


def test_imports():
    """Test that all necessary components can be imported."""
    print("\nTesting imports...")

    from scholarqa.llms.edit_prompts import (
        SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT,
        SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT,
        PROMPT_ASSEMBLE_SUMMARY_EDIT
    )

    # Verify prompts are strings
    assert isinstance(SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT, str)
    assert isinstance(SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT, str)
    assert isinstance(PROMPT_ASSEMBLE_SUMMARY_EDIT, str)

    # Verify they contain edit-specific content
    assert "edit" in SYSTEM_PROMPT_QUOTE_PER_PAPER_EDIT.lower()
    assert "current_report" in SYSTEM_PROMPT_QUOTE_CLUSTER_EDIT
    assert "action" in PROMPT_ASSEMBLE_SUMMARY_EDIT.lower()

    print("✅ PASS: All imports successful and prompts valid")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Running Edit Workflow Structural Tests")
    print("=" * 60)

    tests = [
        test_edit_pipeline_init,
        test_mock_data_structures,
        test_tool_request_edit_fields,
        test_format_methods,
        test_step_select_quotes_edit_mock,
        test_step_clustering_edit_mock,
        test_imports,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test.__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
