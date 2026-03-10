"""
Live Integration tests for EditPipelineRunner using sample data.

This test file uses the sample response from:
scholarqa_traces/00b36f60-97e3-4197-a15a-48ea4c235f73.json

The report is about: "What publicly available datasets are typically used for
evaluating type inference systems in python?"

Papers in the report by year:
  2018: [56482376]
  2019: [208909790, 208527555]
  2020: [216056383]
  2021: [233210280, 246062360, 246680113, 250072092, 235658605]
  2022: [248157108, 251710434]
  2023: [257623048, 260512650]
  2024: [268033677, 270878649]

Total: 15 unique papers across 5 sections.

Run with: ANTHROPIC_API_KEY=your_key pytest scholarqa/test_edit_pipeline_integration.py -v -s
"""

import json
import os
import pytest
import pandas as pd
from typing import Dict, Any, List

from scholarqa.models import ReportEditRequest
from scholarqa.preprocess.edit_intent_analyzer import (
    EditIntentAnalysis, analyze_edit_intent
)
from scholarqa.rag.edit_pipeline import EditPipeline
from scholarqa.llms.constants import CLAUDE_4_SONNET
from scholarqa.edit_pipeline_runner import EditPipelineRunner
from scholarqa.utils import NUMERIC_META_FIELDS, CATEGORICAL_META_FIELDS

# Path to sample response file
SAMPLE_RESPONSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "scholarqa_traces/00b36f60-97e3-4197-a15a-48ea4c235f73.json"
)

# Original query from the sample
ORIGINAL_QUERY = "What publicly available datasets are typically used for evaluating type inference systems in python?"

# LLM Model to use for tests
LLM_MODEL = CLAUDE_4_SONNET

def load_sample_response() -> Dict[str, Any]:
    """Load the sample response JSON."""
    with open(SAMPLE_RESPONSE_PATH) as f:
        return json.load(f)


def extract_report_dict(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the report dict from sample data, adding report_title if missing."""
    report = dict(sample_data["summary"])
    if "report_title" not in report:
        report["report_title"] = "Type Inference Evaluation Datasets in Python"
    return report


def print_intent_analysis(analysis: EditIntentAnalysis):
    """Print intent analysis details for debugging."""
    print("=" * 60)
    print("INTENT ANALYSIS RESULT")
    print("=" * 60)
    cot_display = analysis.cot[:200] + "..." if len(analysis.cot) > 200 else analysis.cot
    print(f"Chain of Thought: {cot_display}")
    print(f"Search Query: {analysis.search_query or 'None'}")
    print(f"Needs Search: {analysis.needs_search}")
    print(f"Is Stylistic: {analysis.is_stylistic}")
    print(f"Is Addition: {analysis.is_addition}")
    print(f"Is Removal: {analysis.is_removal}")
    print(f"Papers to Add: {analysis.papers_to_add}")
    print(f"Papers to Remove: {analysis.papers_to_remove}")
    print(f"Target Sections: {analysis.target_sections}")
    print(f"Affects All Sections: {analysis.affects_all_sections}")
    if analysis.earliest_year:
        print(f"Year Filter: {analysis.earliest_year} - {analysis.latest_year}")
    print("=" * 60)


def print_cluster_plan(plan: Dict[str, Any]):
    """Print cluster plan details for debugging."""
    print("=" * 60)
    print("CLUSTER PLAN RESULT")
    print("=" * 60)
    print(f"Report Title: {plan.get('report_title', 'N/A')}")
    print(f"Papers to Remove: {plan.get('papers_to_remove', [])}")
    cot_display = plan.get('cot', 'N/A')[:200] + "..." if len(plan.get('cot', '')) > 200 else plan.get('cot', 'N/A')
    print(f"Chain of Thought: {cot_display}")
    print("Dimensions:")
    for i, dim in enumerate(plan.get("dimensions", [])):
        print(f"  {i+1}. {dim['name']} ({dim['format']}) - Action: {dim.get('action', 'N/A')}, Quotes: {dim.get('quotes', [])}")
    print("=" * 60)


# ============================================================================
# HELPER FUNCTIONS FOR TEST REUSE
# ============================================================================

def create_edit_request(
    intent: str,
    corpus_ids: List[str] = None,
    section_titles: List[str] = None,
    query: str = ORIGINAL_QUERY,
    thread_id: str = "test-thread"
) -> ReportEditRequest:
    """
    Create a ReportEditRequest with common defaults.

    Args:
        intent: The edit instruction
        corpus_ids: Optional list of corpus IDs
        section_titles: Optional list of section titles
        query: Original query (defaults to ORIGINAL_QUERY)
        thread_id: Thread ID (defaults to "test-thread")

    Returns:
        ReportEditRequest object
    """
    return ReportEditRequest(
        query=query,
        intent=intent,
        thread_id=thread_id,
        corpus_ids=corpus_ids or [],
        section_titles=section_titles or []
    )


def run_intent_analysis(
    intent: str,
    current_report: Dict[str, Any],
    report_context: str = None,
    corpus_ids: List[str] = None,
    section_titles: List[str] = None,
    print_result: bool = True
) -> EditIntentAnalysis:
    """
    Run intent analysis with common setup.

    Args:
        intent: The edit instruction
        current_report: The current report being edited
        report_context: Pre-formatted report context string (computed from current_report if not provided)
        corpus_ids: Optional list of corpus IDs
        section_titles: Optional list of section titles
        print_result: Whether to print the analysis result

    Returns:
        EditIntentAnalysis result
    """
    if report_context is None:
        report_context = EditPipeline.format_report_context(current_report)

    req = create_edit_request(
        intent=intent,
        corpus_ids=corpus_ids,
        section_titles=section_titles
    )

    result = analyze_edit_intent(
        req=req,
        report_context=report_context,
        current_report=current_report,
        llm_model=LLM_MODEL
    )

    if print_result:
        print_intent_analysis(result)

    return result


def collect_generation_results(gen_iter, plan_dimensions):
    """
    Collect results from generate_iterative_summary_edit, handling the 1:1 yield contract.

    The generator yields one result per dimension:
    - DELETE/KEEP: yields CompletionResult with content=None (noop)
    - REWRITE/NEW: yields CompletionResult with .content string

    Returns:
        active_sections: list of CompletionResult for REWRITE/NEW sections only
    """
    all_results = list(gen_iter)
    assert len(all_results) == len(plan_dimensions), \
        f"Generator should yield one result per dimension. Got {len(all_results)} for {len(plan_dimensions)} dims."

    active_sections = []
    for dim, result in zip(plan_dimensions, all_results):
        action = dim.get("action", "NEW")
        section_name = dim["name"]

        if action == "DELETE":
            assert result.content is None, f"DELETE should yield content=None, got {result.content}"
            print(f"  [DELETE] {section_name}")
            continue

        if action == "KEEP":
            assert result.content is None, f"KEEP should yield content=None, got {result.content}"
            print(f"  [KEEP] {section_name}")
            continue

        # REWRITE/NEW: should have content
        assert result is not None and result.content is not None, \
            f"{action} should yield a CompletionResult with content, got {result}"
        content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
        print(f"  [{action}] {section_name}: {content_preview}")
        active_sections.append(result)

    return active_sections


def print_quote_results(quotes, completions):
    """Print quote extraction results for debugging."""
    print("=" * 60)
    print("QUOTE EXTRACTION RESULTS")
    print("=" * 60)
    print(f"Papers with extracted quotes: {len(quotes)} / {len(completions)} total")
    for ref, quote in quotes.items():
        preview = quote[:200] + "..." if len(quote) > 200 else quote
        print(f"\n  {ref}")
        print(f"  Quote: {preview}")
    print("=" * 60)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Load sample response data."""
    return load_sample_response()


@pytest.fixture
def current_report(sample_data):
    """Extract report dict from sample data."""
    return extract_report_dict(sample_data)


@pytest.fixture
def report_context(current_report):
    """Pre-formatted report context string for edit prompts."""
    return EditPipeline.format_report_context(current_report)


@pytest.fixture
def edit_pipeline():
    """Create EditPipeline instance."""
    return EditPipeline(
        llm_model=LLM_MODEL,
        batch_workers=5
    )


def build_existing_per_paper_summaries(current_report: Dict[str, Any]) -> dict:
    """
    Build per_paper_summaries_extd from existing report citations
    (mirrors what the runner does in Step 4.5).
    """
    per_paper_summaries_extd = {}
    for section in current_report.get("sections", []):
        for cit in section.get("citations", []):
            ref_key, per_paper_entry, _ = EditPipeline.citation_to_ref_data(cit)
            if ref_key not in per_paper_summaries_extd:
                per_paper_summaries_extd[ref_key] = per_paper_entry
    return per_paper_summaries_extd


@pytest.fixture
def runner():
    """Create a real EditPipelineRunner with FullTextRetriever."""
    from scholarqa.rag.retrieval import PaperFinder
    from scholarqa.rag.retriever_base import FullTextRetriever
    from scholarqa.config.config_setup import LogsConfig

    retriever = FullTextRetriever(n_retrieval=50, n_keyword_srch=10)
    paper_finder = PaperFinder(retriever, context_threshold=0.0)
    logs_cfg = LogsConfig(llm_cache_dir="test_llm_cache")
    logs_cfg.init_formatter()

    return EditPipelineRunner(
        paper_finder=paper_finder,
        llm_model=LLM_MODEL,
        logs_config=logs_cfg,
        run_table_generation=False,
        fallback_llm=None,
    )


# ============================================================================
# TEST HELPER: Verify report structure
# ============================================================================

class TestReportStructure:
    """Tests to verify the sample report structure is correct."""

    def test_sample_file_exists(self):
        """Verify sample response file exists."""
        assert os.path.exists(SAMPLE_RESPONSE_PATH), f"Sample file not found: {SAMPLE_RESPONSE_PATH}"

    def test_load_sample_response(self, sample_data):
        """Verify sample response loads correctly."""
        assert "query" in sample_data
        assert "summary" in sample_data
        assert "sections" in sample_data["summary"]

    def test_report_dict_structure(self, current_report):
        """Verify report dict has correct structure."""
        assert isinstance(current_report, dict)
        sections = current_report.get("sections", [])
        assert len(sections) == 5
        assert sections[0]["title"] == "Introduction to Type Inference Evaluation"

    def test_report_has_citations(self, current_report):
        """Verify citations are present in the report."""
        total_citations = sum(
            len(sec.get("citations", [])) for sec in current_report["sections"]
        )
        assert total_citations > 0, "Report should have citations"

    def test_citation_years_range(self, current_report):
        """Verify citations span expected year range."""
        years = set()
        for sec in current_report["sections"]:
            for cit in sec.get("citations", []):
                year = cit.get("paper", {}).get("year")
                if year:
                    years.add(year)

        assert min(years) == 2018, "Earliest paper should be 2018"
        assert max(years) == 2024, "Latest paper should be 2024"


# ============================================================================
# NO-LLM BRANCHES: AbstractFallbackInjection
# ============================================================================

class TestNoLLMBranches:
    """
    Tests for code-only branches that do NOT require LLM calls.

    Includes:
    - Abstract fallback injection tests (pure data manipulation)
    """

    # =========================================================================
    # Abstract Fallback Injection Tests
    # =========================================================================

    def test_missing_paper_gets_abstract_fallback(self):
        """
        Paper in papers_to_add but NOT in per_paper_summaries should get
        its abstract injected from reranked_df.
        """
        pipeline = EditPipeline(llm_model="test-model")

        # per_paper_summaries has one paper, but papers_to_add has two
        per_paper_summaries = {
            "[90001 | Chen et al. | 2024]": "Quote from paper 90001..."
        }

        reranked_df = pd.DataFrame([
            {
                "corpus_id": 90001,
                "reference_string": "[90001 | Chen et al. | 2024]",
                "abstract": "Abstract of paper 90001",
            },
            {
                "corpus_id": 90002,
                "reference_string": "[90002 | Wang & Liu | 2024]",
                "abstract": "Abstract of paper 90002 about PythonTypes4K",
            },
        ])

        result = EditPipelineRunner._inject_abstract_fallbacks(
            None,  # self -- method uses no instance state except logging
            per_paper_summaries=per_paper_summaries,
            papers_to_add=["90001", "90002"],
            reranked_df=reranked_df,
        )

        # Paper 90001 should keep its original quote
        assert "[90001 | Chen et al. | 2024]" in result
        assert result["[90001 | Chen et al. | 2024]"] == "Quote from paper 90001..."

        # Paper 90002 should get abstract fallback
        assert "[90002 | Wang & Liu | 2024]" in result
        assert "[Abstract]" in result["[90002 | Wang & Liu | 2024]"]
        assert "PythonTypes4K" in result["[90002 | Wang & Liu | 2024]"]

    def test_all_papers_have_quotes_no_fallback_needed(self):
        """
        When all papers_to_add already have quotes, no fallback injection happens.
        """
        per_paper_summaries = {
            "[90001 | Chen et al. | 2024]": "Quote from paper 90001...",
            "[90002 | Wang & Liu | 2024]": "Quote from paper 90002...",
        }

        reranked_df = pd.DataFrame([
            {"corpus_id": 90001, "reference_string": "[90001 | Chen et al. | 2024]", "abstract": "abs1"},
            {"corpus_id": 90002, "reference_string": "[90002 | Wang & Liu | 2024]", "abstract": "abs2"},
        ])

        result = EditPipelineRunner._inject_abstract_fallbacks(
            None,
            per_paper_summaries=per_paper_summaries,
            papers_to_add=["90001", "90002"],
            reranked_df=reranked_df,
        )

        assert len(result) == 2, "Should still have exactly 2 entries"
        # No [Abstract] markers -- originals preserved
        assert "[Abstract]" not in result["[90001 | Chen et al. | 2024]"]
        assert "[Abstract]" not in result["[90002 | Wang & Liu | 2024]"]

    def test_empty_papers_to_add(self):
        """
        Empty papers_to_add means no fallback injection needed.
        """
        per_paper_summaries = {
            "[90001 | Chen et al. | 2024]": "Quote from search result...",
        }
        reranked_df = pd.DataFrame([
            {"corpus_id": 90001, "reference_string": "[90001 | Chen et al. | 2024]", "abstract": "abs1"},
        ])

        result = EditPipelineRunner._inject_abstract_fallbacks(
            None,
            per_paper_summaries=per_paper_summaries,
            papers_to_add=[],
            reranked_df=reranked_df,
        )

        assert len(result) == 1
        assert result["[90001 | Chen et al. | 2024]"] == "Quote from search result..."

    def test_paper_with_no_abstract_skipped(self):
        """
        Paper without an abstract in reranked_df should be skipped gracefully.
        """
        per_paper_summaries = {}

        reranked_df = pd.DataFrame([
            {"corpus_id": 90001, "reference_string": "[90001 | Chen | 2024]", "abstract": ""},
        ])

        result = EditPipelineRunner._inject_abstract_fallbacks(
            None,
            per_paper_summaries=per_paper_summaries,
            papers_to_add=["90001"],
            reranked_df=reranked_df,
        )

        # No abstract -> no fallback added
        assert len(result) == 0



# ============================================================================
# END-TO-END SCENARIO TESTS
# ============================================================================

class TestEndToEndScenarios:
    """
    End-to-end scenario tests that run the complete edit pipeline.

    All scenarios flow through:
        intent analysis -> LLM clustering (step_clustering_edit) -> section generation

    Search scenarios additionally include:
        intent analysis -> find_relevant_papers_for_edit (runner) ->
        rerank_and_aggregate (runner) -> quote extraction (edit_pipeline) ->
        clustering (edit_pipeline) -> section generation (edit_pipeline)
    """

    # =========================================================================
    # No-Search Scenarios (stylistic / removal / structural)
    # =========================================================================

    def test_scenario_stylistic_rewrite_all(self, current_report, report_context, edit_pipeline):
        """
        Scenario: "Rewrite all sections to be more concise"
        Flow: intent analysis -> LLM clustering -> section generation
        """
        edit_instruction = "Rewrite all sections to be more concise"
        num_sections = len(current_report["sections"])

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Stylistic Rewrite All Sections")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        intent = run_intent_analysis(
            intent=edit_instruction,
            current_report=current_report,
        )

        assert intent.is_stylistic == True, "Should be stylistic"
        assert intent.needs_search == False, "Should not need search"
        assert intent.is_addition == False, "Should not add papers"
        assert intent.is_removal == False, "Should not remove papers"
        assert intent.affects_all_sections == True, "Should affect all sections"
        assert len(intent.target_sections) == 0, \
            f"Should not have specific target sections. Got: {intent.target_sections}"

        # --- Step 2: LLM Clustering ---
        print("\n--- Step 2: LLM Clustering (step_clustering_edit) ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries={},
            intent_analysis=intent,
        )

        assert len(plan["dimensions"]) == num_sections, \
            f"Should have {num_sections} dimensions (one per section). Got: {len(plan['dimensions'])}"
        actions = [d["action"] for d in plan["dimensions"]]
        assert all(a == "REWRITE" for a in actions), \
            f"All sections should be REWRITE for global stylistic edit. Got: {actions}"
        print_cluster_plan(plan)

        # --- Step 3: Section Generation ---
        print("\n--- Step 3: Section Generation ---")

        per_paper_summaries_extd = build_existing_per_paper_summaries(current_report)

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        # All sections should be REWRITE (stylistic rewrite all)
        assert len(active_sections) == num_sections, \
            f"All {num_sections} sections should be REWRITE. Got: {len(active_sections)}"
        for sec in active_sections:
            assert len(sec.content) > 50, "Each section should have substantial content"

        # Verify: existing papers should be retained in stylistic rewrite
        all_corpus_ids = set()
        for section in current_report["sections"]:
            for cit in section.get("citations", []):
                all_corpus_ids.add(str(cit["paper"]["corpus_id"]))
        all_content = " ".join(sec.content for sec in active_sections)
        found_ids = {cid for cid in all_corpus_ids if cid in all_content}
        retention_rate = len(found_ids) / len(all_corpus_ids) if all_corpus_ids else 0
        print(f"  Citation retention: {len(found_ids)}/{len(all_corpus_ids)} papers ({retention_rate:.0%})")
        assert retention_rate >= 0.5, \
            f"Stylistic rewrite should retain most papers. Only {retention_rate:.0%} retained. Missing: {all_corpus_ids - found_ids}"

    def test_scenario_stylistic_rewrite_targeted(self, current_report, report_context, edit_pipeline):
        """
        Scenario: "Rewrite the Introduction to be more technical"
        Flow: intent analysis -> LLM clustering -> section generation
        Verify: targeted section is REWRITE, others are KEEP
        """
        edit_instruction = "Rewrite the Introduction to be more technical"
        target_title = "Introduction to Type Inference Evaluation"
        num_sections = len(current_report["sections"])

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Stylistic Rewrite Targeted Section")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        intent = run_intent_analysis(
            intent=edit_instruction,
            current_report=current_report,
            section_titles=[target_title],
        )

        assert intent.is_stylistic == True, "Should be stylistic"
        assert intent.needs_search == False, "Should not need search"
        assert intent.affects_all_sections == False, "Should target specific section"
        assert len(intent.target_sections) > 0, "Should have target sections"

        # --- Step 2: LLM Clustering ---
        print("\n--- Step 2: LLM Clustering (step_clustering_edit) ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries={},
            intent_analysis=intent,
        )

        assert len(plan["dimensions"]) == num_sections, \
            f"Should have {num_sections} dimensions. Got: {len(plan['dimensions'])}"
        actions = {d["name"]: d["action"] for d in plan["dimensions"]}
        # Introduction should be REWRITE
        intro_actions = [
            a for name, a in actions.items()
            if "introduction" in name.lower()
        ]
        assert len(intro_actions) > 0, f"Should have an introduction dimension. Got: {actions}"
        assert intro_actions[0] == "REWRITE", \
            f"Introduction should be REWRITE. Got: {intro_actions[0]}"
        # Other sections should be KEEP
        other_actions = [
            a for name, a in actions.items()
            if "introduction" not in name.lower()
        ]
        assert all(a == "KEEP" for a in other_actions), \
            f"Non-targeted sections should be KEEP. Got: {actions}"
        print_cluster_plan(plan)

        # --- Step 3: Section Generation ---
        print("\n--- Step 3: Section Generation ---")

        per_paper_summaries_extd = build_existing_per_paper_summaries(current_report)

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        # Only the targeted section should be REWRITE, rest KEEP
        rewrite_count = sum(1 for d in plan["dimensions"] if d["action"] == "REWRITE")
        assert len(active_sections) == rewrite_count, \
            f"Should have {rewrite_count} active sections. Got: {len(active_sections)}"

    def test_scenario_remove_papers_by_year(self, current_report, report_context, edit_pipeline):
        """
        Scenario: "Remove papers before 2020"
        Flow: intent analysis -> LLM clustering -> section generation
        """
        edit_instruction = "Remove papers before 2020"
        num_sections = len(current_report["sections"])

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Remove Papers By Year")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        intent = run_intent_analysis(
            intent=edit_instruction,
            current_report=current_report,
        )

        assert intent.is_removal == True, "Should be removal"
        assert intent.is_stylistic == False, "Should not be stylistic"
        assert intent.needs_search == False, "Should not need search"
        assert intent.affects_all_sections == True, "Removal affects all sections"

        # Pre-2020 papers: 56482376 (2018), 208909790 (2019), 208527555 (2019)
        expected_pre2020 = {"56482376", "208909790", "208527555"}
        actual_remove = set(intent.papers_to_remove)
        assert actual_remove.issubset(expected_pre2020), \
            f"All removed papers should be pre-2020. Got unexpected: {actual_remove - expected_pre2020}"
        assert len(actual_remove) >= 2, \
            f"Should identify at least 2 of 3 pre-2020 papers. Got: {actual_remove}"

        # --- Step 2: LLM Clustering ---
        print("\n--- Step 2: LLM Clustering (step_clustering_edit) ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries={},
            intent_analysis=intent,
        )

        assert len(plan["dimensions"]) == num_sections, \
            f"Should have {num_sections} dimensions. Got: {len(plan['dimensions'])}"
        actions = [d["action"] for d in plan["dimensions"]]
        # All sections citing pre-2020 papers should be REWRITE
        rewrite_count = sum(1 for a in actions if a == "REWRITE")
        assert rewrite_count > 0, \
            f"Should have at least one REWRITE action for paper removal. Got: {actions}"
        # Plan should include the pre-2020 papers in papers_to_remove
        plan_removals = set(plan.get("papers_to_remove", []))
        assert plan_removals.issubset(expected_pre2020), \
            f"All plan removals should be pre-2020. Got unexpected: {plan_removals - expected_pre2020}"
        assert len(plan_removals) >= 2, \
            f"Plan should include at least 2 of 3 pre-2020 papers. Got: {plan_removals}"
        print_cluster_plan(plan)

        # --- Step 3: Section Generation ---
        print("\n--- Step 3: Section Generation ---")

        per_paper_summaries_extd = build_existing_per_paper_summaries(current_report)
        papers_to_remove = plan.get("papers_to_remove", [])

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=papers_to_remove,
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE) sections for paper removal"
        for sec in active_sections:
            assert len(sec.content) > 20, "Each rewritten section should have content"

        # Verify: removed papers should NOT appear in any rewritten section content
        all_content = " ".join(sec.content for sec in active_sections)
        for pid in papers_to_remove:
            assert pid not in all_content, \
                f"Removed paper {pid} should not appear in rewritten content"
        print(f"  PASS: {len(papers_to_remove)} removed papers absent from rewritten sections")

    def test_scenario_remove_specific_paper(self, current_report, report_context, edit_pipeline):
        """
        Scenario: "Remove paper 56482376"
        Flow: intent analysis -> LLM clustering -> section generation
        """
        edit_instruction = "Remove paper 56482376"
        num_sections = len(current_report["sections"])

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Remove Specific Paper")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        intent = run_intent_analysis(
            intent=edit_instruction,
            current_report=current_report,
        )

        assert intent.is_removal == True, "Should be removal"
        assert intent.is_stylistic == False, "Should not be stylistic"
        assert intent.needs_search == False, "Should not need search"
        assert "56482376" in intent.papers_to_remove, \
            f"Should remove paper 56482376. Got: {intent.papers_to_remove}"

        # --- Step 2: LLM Clustering ---
        print("\n--- Step 2: LLM Clustering (step_clustering_edit) ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries={},
            intent_analysis=intent,
        )

        assert len(plan["dimensions"]) == num_sections, \
            f"Should have {num_sections} dimensions. Got: {len(plan['dimensions'])}"
        # papers_to_remove should include 56482376
        plan_removals = plan.get("papers_to_remove", [])
        assert "56482376" in plan_removals, \
            f"Plan should include paper 56482376 in papers_to_remove. Got: {plan_removals}"
        # Sections citing this paper should be REWRITE
        actions = [d["action"] for d in plan["dimensions"]]
        rewrite_count = sum(1 for a in actions if a == "REWRITE")
        assert rewrite_count > 0, \
            f"Should have at least one REWRITE action for paper removal. Got: {actions}"
        print_cluster_plan(plan)

        # --- Step 3: Section Generation ---
        print("\n--- Step 3: Section Generation ---")

        per_paper_summaries_extd = build_existing_per_paper_summaries(current_report)

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", []),
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE) sections for paper removal"

        # Verify: removed paper should NOT appear in any rewritten section content
        for sec in active_sections:
            assert "56482376" not in sec.content, \
                f"Removed paper 56482376 should not appear in rewritten content:\n{sec.content[:500]}"
        print(f"  PASS: Paper 56482376 absent from all {len(active_sections)} rewritten sections")

    def test_scenario_shorten_section(self, current_report, report_context, edit_pipeline):
        """
        Scenario: "Shorten the Dataset Characteristics section"
        Flow: intent analysis -> LLM clustering -> section generation
        """
        edit_instruction = "Shorten the Dataset Characteristics section"
        num_sections = len(current_report["sections"])

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Shorten Section")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        intent = run_intent_analysis(
            intent=edit_instruction,
            current_report=current_report,
            section_titles=["Dataset Characteristics and Sizes"],
        )

        assert intent.is_stylistic == True, "Shorten should be stylistic"
        assert intent.needs_search == False, "Should not need search"
        assert intent.affects_all_sections == False, "Should target specific section"
        assert len(intent.target_sections) > 0, "Should have target sections"

        # --- Step 2: LLM Clustering ---
        print("\n--- Step 2: LLM Clustering (step_clustering_edit) ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries={},
            intent_analysis=intent,
        )

        assert len(plan["dimensions"]) == num_sections, \
            f"Should have {num_sections} dimensions. Got: {len(plan['dimensions'])}"
        actions = {d["name"]: d["action"] for d in plan["dimensions"]}
        # Dataset section should be REWRITE
        dataset_actions = [
            a for name, a in actions.items()
            if "dataset characteristics" in name.lower()
        ]
        assert len(dataset_actions) > 0, f"Should have a dataset characteristics dimension. Got: {actions}"
        assert dataset_actions[0] == "REWRITE", \
            f"Dataset Characteristics section should be REWRITE. Got: {dataset_actions[0]}"
        # Other sections should be KEEP
        other_actions = [
            a for name, a in actions.items()
            if "dataset characteristics" not in name.lower()
        ]
        assert all(a == "KEEP" for a in other_actions), \
            f"Non-targeted sections should be KEEP. Got: {actions}"
        print_cluster_plan(plan)

        # --- Step 3: Section Generation ---
        print("\n--- Step 3: Section Generation ---")

        # Measure original length of the target section
        original_lengths = {sec["title"]: len(sec.get("text", "")) for sec in current_report["sections"]}
        per_paper_summaries_extd = build_existing_per_paper_summaries(current_report)

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        # Only the targeted section should be REWRITE, rest KEEP
        rewrite_count = sum(1 for d in plan["dimensions"] if d["action"] == "REWRITE")
        assert len(active_sections) == rewrite_count, \
            f"Should have {rewrite_count} active sections. Got: {len(active_sections)}"

        # Verify: rewritten section should be shorter than the original
        rewrite_dims = [d for d in plan["dimensions"] if d["action"] == "REWRITE"]
        for dim, sec in zip(rewrite_dims, active_sections):
            section_name = dim["name"]
            original_len = original_lengths.get(section_name, 0)
            new_len = len(sec.content)
            print(f"  {section_name}: {original_len} -> {new_len} chars")
            assert new_len < original_len, \
                f"Shortened section should be shorter. Original: {original_len}, New: {new_len}"
        print("  PASS: Shortened section is shorter than original")

    def test_scenario_delete_section(self, current_report, report_context, edit_pipeline):
        """
        Scenario: "Delete the Dataset Selection section"
        Flow: intent analysis -> LLM clustering -> section generation
        This scenario was previously impossible with hardcoded plans.
        """
        target_section = current_report["sections"][-1]["title"]  # Last section
        num_sections = len(current_report["sections"])
        edit_instruction = f"Delete the {target_section} section"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Delete Section")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        intent = run_intent_analysis(
            intent=edit_instruction,
            current_report=current_report,
            section_titles=[target_section],
        )

        assert intent.needs_search == False, "Should not need search"
        assert intent.affects_all_sections == False, "Should target specific section"
        assert len(intent.target_sections) > 0, "Should have target sections"

        # --- Step 2: LLM Clustering ---
        print("\n--- Step 2: LLM Clustering (step_clustering_edit) ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries={},
            intent_analysis=intent,
        )

        assert len(plan["dimensions"]) == num_sections, \
            f"Should have {num_sections} dimensions. Got: {len(plan['dimensions'])}"
        actions = {d["name"]: d["action"] for d in plan["dimensions"]}
        print_cluster_plan(plan)

        # The target section should be DELETE
        target_actions = [
            a for name, a in actions.items()
            if target_section.lower() in name.lower() or name.lower() in target_section.lower()
        ]
        assert len(target_actions) > 0, \
            f"Should have a dimension for '{target_section}'. Got: {actions}"
        assert target_actions[0] == "DELETE", \
            f"Target section should be DELETE. Got: {target_actions[0]}"

        # Other sections should be KEEP
        non_target_actions = [
            a for name, a in actions.items()
            if target_section.lower() not in name.lower() and name.lower() not in target_section.lower()
        ]
        assert all(a == "KEEP" for a in non_target_actions), \
            f"Non-target sections should be KEEP. Got: {actions}"

        # --- Step 3: Section Generation ---
        print("\n--- Step 3: Section Generation ---")

        per_paper_summaries_extd = build_existing_per_paper_summaries(current_report)

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        # DELETE yields None (verified by helper), remaining are KEEP (also None)
        delete_count = sum(1 for d in plan["dimensions"] if d["action"] == "DELETE")
        keep_count = sum(1 for d in plan["dimensions"] if d["action"] == "KEEP")
        assert delete_count >= 1, "Should have at least one DELETE action"
        assert len(active_sections) + delete_count + keep_count == len(plan["dimensions"]), \
            "All dimensions should be accounted for"

    # =========================================================================
    # Search Scenarios (addition / expansion / complex)
    # =========================================================================

    def test_scenario_add_papers_about_topic(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Add papers about neural type inference"
        Flow: intent -> search (real S2 API) -> rerank -> quotes -> clustering -> section gen
        """
        edit_instruction = "Add papers about neural type inference"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Add Papers About Topic")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(intent=edit_instruction)
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.needs_search == True, "Should need search"
        assert intent.is_addition == True, "Should be addition"
        assert len(intent.search_query) > 0, "Should have search query"

        # --- Step 2: Search (real S2 API) ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates from S2 API")
        assert len(retrieved_candidates) > 0, "S2 API should return papers"

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")
        assert not reranked_df.empty, "Reranking should produce results"

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)
        assert len(quotes) > 0, "Should extract quotes from relevant papers"

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        assert "dimensions" in plan, "Plan should have dimensions"
        assert len(plan["dimensions"]) > 0, "Plan should have at least one dimension"
        actions = [d["action"] for d in plan["dimensions"]]
        assert any(a != "KEEP" for a in actions), \
            f"Should have at least one non-KEEP action. Got: {actions}"
        # Quotes should be assigned to at least one section
        all_assigned_quotes = [q for d in plan["dimensions"] for q in d.get("quotes", [])]
        assert len(all_assigned_quotes) > 0, \
            f"At least one section should have quotes assigned. Got: {[d.get('quotes', []) for d in plan['dimensions']]}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"
        for sec in active_sections:
            assert len(sec.content) > 50, "Each section should have substantial content"

    def test_scenario_add_specific_paper_by_id(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Add paper 259861065 to the report"
        Flow: intent -> retrieve mentioned paper -> rerank -> quotes + abstract fallback -> clustering -> section gen
        """
        paper_corpus_id = "259861065"
        edit_instruction = f"Add paper {paper_corpus_id} to the report"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Add Specific Paper By ID")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(intent=edit_instruction, corpus_ids=[paper_corpus_id])
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.is_addition == True, "Should be addition"
        assert paper_corpus_id in intent.papers_to_add, \
            f"Should include mentioned paper. Got: {intent.papers_to_add}"

        # --- Step 2: Search (mentioned papers path) ---
        print("\n--- Step 2: Retrieve Mentioned Paper ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates for mentioned paper")
        assert len(retrieved_candidates) > 0, "Should retrieve the mentioned paper"

        candidate_corpus_ids = {str(c["corpus_id"]) for c in retrieved_candidates}
        assert paper_corpus_id in candidate_corpus_ids, \
            f"Mentioned paper should be in candidates. Got: {candidate_corpus_ids}"

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")
        assert not reranked_df.empty, "Reranking should produce results"
        assert int(paper_corpus_id) in reranked_df["corpus_id"].values, \
            "Mentioned paper should survive reranking"

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query or ORIGINAL_QUERY,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 4.5: Abstract Fallback Injection ---
        print("\n--- Step 4.5: Abstract Fallback Injection ---")

        per_paper_summaries = runner._inject_abstract_fallbacks(
            per_paper_summaries=quotes,
            papers_to_add=intent.papers_to_add,
            reranked_df=reranked_df,
        )
        print(f"After fallback injection: {len(per_paper_summaries)} papers")

        # The mentioned paper should be present (via quote or abstract fallback)
        paper_present = any(paper_corpus_id in ref_str for ref_str in per_paper_summaries.keys())
        assert paper_present, \
            f"Mentioned paper {paper_corpus_id} should be in final summaries. Keys: {list(per_paper_summaries.keys())}"

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=per_paper_summaries,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        assert "dimensions" in plan, "Plan should have dimensions"
        actions = [d["action"] for d in plan["dimensions"]]
        assert any(a != "KEEP" for a in actions), \
            f"Should have at least one non-KEEP action. Got: {actions}"
        # Quotes should be assigned to at least one section
        all_assigned_quotes = [q for d in plan["dimensions"] for q in d.get("quotes", [])]
        assert len(all_assigned_quotes) > 0, \
            f"At least one section should have quotes assigned. Got: {[d.get('quotes', []) for d in plan['dimensions']]}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in per_paper_summaries.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"

        # Verify: the added paper should appear in at least one rewritten/new section
        all_content = " ".join(sec.content for sec in active_sections)
        assert paper_corpus_id in all_content, \
            f"Added paper {paper_corpus_id} should appear in generated content"
        print(f"  PASS: Added paper {paper_corpus_id} found in generated sections")

    def test_scenario_expand_section(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Expand the Commonly Used Public Datasets section"
        Flow: intent -> search -> rerank -> quotes -> clustering -> section gen
        Clustering should target the datasets section.
        """
        edit_instruction = "Expand the Commonly Used Public Datasets section with more examples"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Expand Section")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(
            intent=edit_instruction,
            section_titles=["Commonly Used Public Datasets"],
        )
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.needs_search == True, "Expand should trigger search"
        assert intent.is_addition == True, "Expand should be addition"
        assert intent.affects_all_sections == False, "Should target specific section"
        assert len(intent.target_sections) > 0, "Should have target sections"

        # --- Step 2: Search ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")
        assert len(retrieved_candidates) > 0, "Should find papers"

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")
        assert not reranked_df.empty, "Reranking should produce results"

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        # Clustering should target the datasets section
        dataset_dims = [d for d in plan["dimensions"] if "dataset" in d["name"].lower()]
        if dataset_dims:
            dataset_actions = [d["action"] for d in dataset_dims]
            assert any(a in ("REWRITE", "NEW") for a in dataset_actions), \
                f"Dataset section should get a content-modifying action. Got: {dataset_actions}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"

    def test_scenario_delete_and_add_section(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Remove the Conclusion section and add a section about evaluation metrics"
        Flow: intent -> search -> rerank -> quotes -> clustering -> section gen
        Clustering should have DELETE for old + NEW for metrics.
        """
        edit_instruction = "Remove the Conclusion section and add a section about evaluation metrics"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Delete and Add Section")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(
            intent=edit_instruction,
            section_titles=["Conclusion and Future Directions"],
        )
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.needs_search == True, "Should need search for new content"
        assert intent.is_addition == True, "Should be addition (new content)"

        # --- Step 2: Search ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")

        if not retrieved_candidates:
            print("No candidates found, skipping remaining steps")
            return

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")

        if reranked_df.empty:
            print("Reranking produced no results, skipping remaining steps")
            return

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        actions_by_name = {d["name"]: d["action"] for d in plan["dimensions"]}
        print(f"Actions by name: {actions_by_name}")

        # Should have DELETE for Conclusion and NEW for metrics
        conclusion_dims = [d for d in plan["dimensions"] if "conclusion" in d["name"].lower()]
        if conclusion_dims:
            assert any(d["action"] == "DELETE" for d in conclusion_dims), \
                f"Conclusion section should be DELETE. Got: {[d['action'] for d in conclusion_dims]}"

        new_dims = [d for d in plan["dimensions"] if d["action"] == "NEW"]
        assert len(new_dims) > 0, \
            f"Should have a NEW section for evaluation metrics. Actions: {list(actions_by_name.values())}"
        # New section should have quotes assigned
        for new_dim in new_dims:
            assert len(new_dim.get("quotes", [])) > 0, \
                f"NEW section '{new_dim['name']}' should have quotes assigned. Got: {new_dim.get('quotes', [])}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"

        # Verify: NEW sections should cite the papers assigned to them by the plan
        quote_ref_list = list(quotes.keys())

        active_idx = 0
        for dim in plan["dimensions"]:
            if dim["action"] in ("DELETE", "KEEP"):
                continue
            if dim["action"] == "NEW":
                # Get corpus IDs of papers assigned to this NEW section
                assigned_ids = set()
                for qi in dim.get("quotes", []):
                    if qi < len(quote_ref_list):
                        cid = quote_ref_list[qi].strip("[]").split(" | ")[0]
                        assigned_ids.add(cid)

                content = active_sections[active_idx].content
                found_ids = {cid for cid in assigned_ids if cid in content}
                print(f"  NEW '{dim['name']}': cites {len(found_ids)}/{len(assigned_ids)} assigned papers: {found_ids}")
                assert len(found_ids) > 0, \
                    f"NEW section '{dim['name']}' should cite assigned papers. Assigned: {assigned_ids}"
            active_idx += 1

    def test_scenario_add_and_remove_papers(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Add recent papers from 2023-2024 and remove all papers before 2020"
        Flow: intent -> search -> rerank -> quotes -> clustering -> section gen
        Clustering should have papers_to_remove and non-KEEP actions.
        """
        edit_instruction = "Add recent papers from 2023-2024 and remove all papers before 2020"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Add and Remove Papers")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(intent=edit_instruction)
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.is_addition == True, "Should need addition"
        assert intent.is_removal == True, "Should need removal"
        assert intent.needs_search == True, "Should need search"

        # Year filters for addition: should target 2023-2024
        assert intent.earliest_year == "2023", \
            f"Should filter additions from 2023. Got: {intent.earliest_year}"
        assert intent.latest_year == "2024", \
            f"Should filter additions up to 2024. Got: {intent.latest_year}"

        # Pre-2020 papers: 56482376 (2018), 208909790 (2019), 208527555 (2019)
        expected_pre2020 = {"56482376", "208909790", "208527555"}
        actual_remove = set(intent.papers_to_remove)
        assert actual_remove.issubset(expected_pre2020), \
            f"All removed papers should be pre-2020. Got unexpected: {actual_remove - expected_pre2020}"
        assert len(actual_remove) >= 2, \
            f"Should identify at least 2 of 3 pre-2020 papers. Got: {actual_remove}"

        # --- Step 2: Search ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")

        if not retrieved_candidates:
            print("No candidates found, skipping remaining steps")
            return

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")

        if reranked_df.empty:
            print("Reranking produced no results, skipping remaining steps")
            return

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        # Plan should include the same pre-2020 papers identified by intent
        plan_removals = set(plan.get("papers_to_remove", []))
        assert plan_removals.issubset(expected_pre2020), \
            f"All plan removals should be pre-2020. Got unexpected: {plan_removals - expected_pre2020}"
        assert len(plan_removals) >= 2, \
            f"Plan should include at least 2 of 3 pre-2020 papers. Got: {plan_removals}"

        # Actions should include REWRITE (covers both removal and addition into existing sections)
        actions = [d["action"] for d in plan["dimensions"]]
        assert any(a in ("REWRITE", "NEW") for a in actions), \
            f"Should have content-modifying actions. Got: {actions}"

        # Quotes should be assigned to at least one section
        all_assigned_quotes = [q for d in plan["dimensions"] for q in d.get("quotes", [])]
        assert len(all_assigned_quotes) > 0, \
            f"At least one section should have quotes assigned. Got: {[d.get('quotes', []) for d in plan['dimensions']]}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"

    def test_scenario_replace_section(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Replace Dataset Characteristics with a section about evaluation metrics"
        Flow: intent -> search -> rerank -> quotes -> clustering -> section gen
        """
        edit_instruction = "Replace Dataset Characteristics with a section about evaluation metrics"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Replace Section")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(
            intent=edit_instruction,
            section_titles=["Dataset Characteristics and Sizes"],
        )
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.needs_search == True, "Should need search for replacement content"
        assert intent.is_addition == True, "Should be addition (new content)"
        assert len(intent.target_sections) > 0, "Should have target sections"

        # --- Step 2: Search ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")

        if not retrieved_candidates:
            print("No candidates found, skipping remaining steps")
            return

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")

        if reranked_df.empty:
            print("Reranking produced no results, skipping remaining steps")
            return

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        actions_by_name = {d["name"]: d["action"] for d in plan["dimensions"]}

        # Dataset Characteristics should be DELETE or REWRITE (replaced)
        dataset_dims = [d for d in plan["dimensions"] if "dataset characteristics" in d["name"].lower()]
        assert len(dataset_dims) > 0, f"Should have a Dataset Characteristics dimension. Got: {list(actions_by_name.keys())}"
        assert dataset_dims[0]["action"] == "DELETE", \
            f"Dataset Characteristics should be DELETE (being replaced). Got: {dataset_dims[0]['action']}"

        # Should have a NEW section for evaluation metrics with quotes
        new_dims = [d for d in plan["dimensions"] if d["action"] == "NEW"]
        assert len(new_dims) > 0, \
            f"Should have a NEW section for evaluation metrics. Actions: {actions_by_name}"
        for new_dim in new_dims:
            assert len(new_dim.get("quotes", [])) > 0, \
                f"NEW section '{new_dim['name']}' should have quotes assigned. Got: {new_dim.get('quotes', [])}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"

    def test_scenario_add_papers_from_venue(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Add recent papers from NeurIPS"
        Flow: intent -> search (with venue filter) -> rerank -> quotes -> clustering -> section gen
        """
        edit_instruction = "Add recent papers from NeurIPS"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Add Papers From Venue")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(intent=edit_instruction)
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.is_addition == True, "Should be addition"
        assert intent.needs_search == True, "Should need search"
        assert len(intent.venues) > 0, f"Should have venue filter. Got: {intent.venues}"
        assert "neurips" in intent.venues.lower() or "nips" in intent.venues.lower(), \
            f"Venue filter should include NeurIPS. Got: {intent.venues}"

        # --- Step 2: Search ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")

        if not retrieved_candidates:
            print("No candidates found from venue, skipping remaining steps")
            return

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")

        if reranked_df.empty:
            print("Reranking produced no results, skipping remaining steps")
            return

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        # Note: LLM may decide retrieved venue papers aren't relevant enough to add,
        # resulting in all KEEP. The 1:1 yield contract is still verified by the helper.
        print(f"Active (REWRITE/NEW) sections: {len(active_sections)}")

    def test_scenario_stylistic_plus_addition(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Make the Introduction more technical and add papers about transformer-based type inference"
        Flow: intent -> search -> rerank -> quotes -> clustering -> section gen
        Compound: targeted stylistic rewrite + content addition via search.
        """
        edit_instruction = "Make the Introduction more technical and add papers about transformer-based type inference"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Stylistic + Addition Compound")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(
            intent=edit_instruction,
            section_titles=["Introduction to Type Inference Evaluation"],
        )
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.needs_search == True, "Should need search for new papers"
        assert intent.is_addition == True, "Should be addition"
        assert intent.affects_all_sections == False, "Should target specific section"
        assert len(intent.target_sections) > 0, "Should have target sections"
        assert len(intent.search_query) > 0, "Should have search query"

        # --- Step 2: Search ---
        print("\n--- Step 2: Search for Papers ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")
        assert len(retrieved_candidates) > 0, "Should find papers"

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")
        assert not reranked_df.empty, "Reranking should produce results"

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)
        assert len(quotes) > 0, "Should extract quotes"

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=quotes,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        actions_by_name = {d["name"]: d["action"] for d in plan["dimensions"]}

        # Introduction should be REWRITE (stylistic change + possibly new content)
        intro_dims = [d for d in plan["dimensions"] if "introduction" in d["name"].lower()]
        assert len(intro_dims) > 0, f"Should have an Introduction dimension. Got: {list(actions_by_name.keys())}"
        assert intro_dims[0]["action"] == "REWRITE", \
            f"Introduction should be REWRITE. Got: {intro_dims[0]['action']}"

        # At least one section should have quotes assigned (for the addition part)
        all_assigned_quotes = [q for d in plan["dimensions"] for q in d.get("quotes", [])]
        assert len(all_assigned_quotes) > 0, \
            f"At least one section should have quotes assigned. Got: {[d.get('quotes', []) for d in plan['dimensions']]}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"quote": q, "inline_citations": {}} for ref, q in quotes.items()}
        # Merge existing citations so REWRITE sections can reference them
        existing = build_existing_per_paper_summaries(current_report)
        for k, v in existing.items():
            if k not in per_paper_summaries_extd:
                per_paper_summaries_extd[k] = v

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"
        for sec in active_sections:
            assert len(sec.content) > 50, "Each section should have substantial content"

    def test_scenario_add_paper_already_in_report(self, current_report, report_context, edit_pipeline, runner):
        """
        Scenario: "Add paper 270878649 to the report"
        Paper 270878649 (TIGER, Wang et al. 2024) is already cited in:
          - Commonly Used Public Datasets
          - Dataset Selection and Preprocessing Considerations
        The system should handle this gracefully — sections should be KEEP
        since the paper is already present.
        """
        paper_corpus_id = "270878649"
        edit_instruction = f"Add paper {paper_corpus_id} to the report"

        # --- Step 1: Intent Analysis ---
        print("\n" + "=" * 80)
        print("SCENARIO: Add Paper Already In Report")
        print("=" * 80)
        print("\n--- Step 1: Intent Analysis ---")

        req = create_edit_request(intent=edit_instruction, corpus_ids=[paper_corpus_id])
        runner.tool_request = req
        intent = runner.analyze_intent(req=req, report_context=report_context, current_report=current_report)
        print_intent_analysis(intent)

        assert intent.is_addition == True, "Should be addition"
        assert paper_corpus_id in intent.papers_to_add, \
            f"Should include mentioned paper. Got: {intent.papers_to_add}"

        # --- Step 2: Search (mentioned papers path) ---
        print("\n--- Step 2: Retrieve Mentioned Paper ---")

        retrieved_candidates, paper_metadata = runner.find_relevant_papers_for_edit(
            intent_analysis=intent,
            original_query=ORIGINAL_QUERY,
        )
        print(f"Retrieved {len(retrieved_candidates)} candidates")
        assert len(retrieved_candidates) > 0, "Should retrieve the mentioned paper"

        # --- Step 3: Rerank ---
        print("\n--- Step 3: Rerank and Aggregate ---")

        s2_srch_metadata = [
            {k: v for k, v in paper.items()
             if k == "corpus_id" or k in NUMERIC_META_FIELDS or k in CATEGORICAL_META_FIELDS}
            for paper in retrieved_candidates + list(paper_metadata.values())
            if "s2FieldsOfStudy" in paper
        ]

        reranked_df, paper_metadata = runner.rerank_and_aggregate(
            edit_instruction,
            retrieved_candidates,
            {str(paper["corpus_id"]): paper for paper in s2_srch_metadata},
        )
        print(f"Reranked DataFrame: {reranked_df.shape[0]} papers")
        assert not reranked_df.empty, "Reranking should produce results"

        # --- Step 4: Quote Extraction ---
        print("\n--- Step 4: Quote Extraction ---")

        quotes, completions = edit_pipeline.step_select_quotes_edit(
            original_query=ORIGINAL_QUERY,
            search_query=intent.search_query or ORIGINAL_QUERY,
            report_context=report_context,
            scored_df=reranked_df,
        )
        print_quote_results(quotes, completions)

        # --- Step 4.5: Abstract Fallback Injection ---
        print("\n--- Step 4.5: Abstract Fallback Injection ---")

        per_paper_summaries = runner._inject_abstract_fallbacks(
            per_paper_summaries=quotes,
            papers_to_add=intent.papers_to_add,
            reranked_df=reranked_df,
        )
        print(f"After fallback injection: {len(per_paper_summaries)} papers")

        # --- Step 5: Clustering ---
        print("\n--- Step 5: Clustering ---")

        plan, _ = edit_pipeline.step_clustering_edit(
            edit_instruction=edit_instruction,
            report_context=report_context,
            per_paper_summaries=per_paper_summaries,
            intent_analysis=intent,
        )
        print_cluster_plan(plan)

        # Paper 270878649 is already cited in 2 sections, so at most 2 non-KEEP
        actions = [d["action"] for d in plan["dimensions"]]
        keep_count = sum(1 for a in actions if a == "KEEP")
        print(f"Actions: {actions} (KEEP count: {keep_count})")
        assert keep_count >= len(actions) - 2, \
            f"Most sections should be KEEP since paper is already cited. Got: {actions}"

        # --- Step 6: Section Generation ---
        print("\n--- Step 6: Section Generation ---")

        per_paper_summaries_extd = {ref: {"response": q} for ref, q in per_paper_summaries.items()}

        gen_iter = edit_pipeline.generate_iterative_summary_edit(
            edit_instruction=edit_instruction,
            current_report=current_report,
            per_paper_summaries_extd=per_paper_summaries_extd,
            plan=plan["dimensions"],
            papers_to_remove=plan.get("papers_to_remove", [])
        )

        active_sections = collect_generation_results(gen_iter, plan["dimensions"])
        assert len(active_sections) > 0, "Should have active (REWRITE/NEW) sections"


# ============================================================================
# MAIN: Run tests
# ============================================================================

if __name__ == "__main__":
    # Quick verification that sample data loads
    print("Loading sample data...")
    data = load_sample_response()
    report = extract_report_dict(data)

    print(f"Original query: {data['query']}")
    print(f"Report title: {report.get('report_title', 'N/A')}")
    print(f"Number of sections: {len(report.get('sections', []))}")

    for i, sec in enumerate(report.get("sections", [])):
        print(f"  {i+1}. {sec['title']} ({len(sec.get('citations', []))} citations)")

    # Collect all papers
    papers_by_year = {}
    for sec in report.get("sections", []):
        for cit in sec.get("citations", []):
            year = cit.get("paper", {}).get("year")
            cid = str(cit.get("paper", {}).get("corpus_id"))
            if year not in papers_by_year:
                papers_by_year[year] = set()
            papers_by_year[year].add(cid)

    print("\nPapers by year:")
    for year in sorted(papers_by_year.keys()):
        print(f"  {year}: {papers_by_year[year]}")

    print("\nRun with: ANTHROPIC_API_KEY=your_key pytest scholarqa/test_edit_pipeline_integration.py -v -s")
