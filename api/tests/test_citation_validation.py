#!/usr/bin/env python3
"""
pytest-style test script for citation validation in ScholarQA responses.

Run with: pytest test_citation_validation.py -v
or: python -m pytest test_citation_validation.py -v

Tests check for invalid citations based on metadata conditions that indicate
model reasoning instead of evidence from cited sources.

Invalid citation condition:
shouldFilter = metadata.pdf_hash === '' &&
              metadata.is_body === true &&
              metadata.sentence_offsets?.length === 0 &&
              metadata.section_title === '';
"""

import pytest
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configuration constants
QUESTIONS_FILE = "questions_list.txt"

# Add the parent api directory to the Python path
api_path = Path(__file__).parent.parent
sys.path.insert(0, str(api_path))

from scholarqa import ScholarQA
from scholarqa.rag.reranker.modal_engine import ModalReranker
from scholarqa.rag.retrieval import PaperFinderWithReranker
from scholarqa.rag.retriever_base import FullTextRetriever
from scholarqa.config.config_setup import LogsConfig

# Configure logging to show on console during tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True  # Override any existing logging configuration
)

# Enable specific loggers that might be suppressed
for logger_name in ['scholarqa', 'scholarqa.scholar_qa', 'scholarqa.rag', 'scholarqa.llms']:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = True


class PostProcessedScholarQA(ScholarQA):
    def postprocess_json_output(self, json_summary: List[Dict[str, Any]], **kwargs) -> None:
        def get_def_snippet_metadata(quote: str) -> List[Dict[str, str]]:
            return [{"quote": quote, "section_title": "abstract", "pdf_hash": "", "is_body": False}]

        quotes_meta = kwargs.get("quotes_meta", dict())
        quotes_meta = {int(k[1:-1].split(" | ")[0]): v for k, v in quotes_meta.items()}
        for section in json_summary:
            for cdict in section["citations"]:
                snippets_set = set(cdict["snippets"])
                cdict["snippet_metadata"] = quotes_meta.get(cdict["paper"]["corpus_id"],
                                                            get_def_snippet_metadata(cdict["snippets"][0]))
                cdict["snippet_metadata"] = [sm for sm in cdict["snippet_metadata"] if
                                             sm.get("quote") in snippets_set]
                for smeta in cdict["snippet_metadata"]:
                    smeta["is_body"] = smeta.get("section_title", "") not in {"abstract", "title"}
                    if smeta.get("sentence_offsets"):
                        for soffset in smeta["sentence_offsets"]:
                            if "boundingBoxes" in soffset:
                                soffset["bounding_boxes"] = soffset["boundingBoxes"]
                                del soffset["boundingBoxes"]


def should_filter_citation(citation: Dict[str, Any]) -> Tuple[bool, int]:
    """
    Check if a citation should be filtered based on metadata conditions.
    Returns True if citation is invalid (contains model reasoning instead of evidence).
    """
    metadata_list = citation.get('snippet_metadata', [])
    for i, metadata in enumerate(metadata_list):
        pdf_hash_empty = metadata.get('pdf_hash', '') == ''
        is_body_true = metadata.get('is_body', False) is True
        sentence_offsets_empty = len(metadata.get('sentence_offsets', [])) == 0
        section_title_empty = metadata.get('section_title', '') == ''
        if pdf_hash_empty and is_body_true and sentence_offsets_empty and section_title_empty:
            return True, i
    return False, -1

@pytest.fixture(scope="session")
def scholar_qa():
    """Initialize ScholarQA system for testing."""
    # Initialize retriever and reranker as in lib_example.py
    retriever = FullTextRetriever(n_retrieval=256, n_keyword_srch=20)
    reranker = ModalReranker(
        app_name="ai2-scholar-qa",
        api_name="inference_api",
        batch_size=256,
        gen_options=dict()
    )
    paper_finder = PaperFinderWithReranker(
        retriever,
        reranker,
        n_rerank=50,
        context_threshold=0.0
    )
    #delete existing log directory if exists
    log_dir = Path("test_citation_logs")
    if log_dir.exists() and log_dir.is_dir():
        import shutil
        shutil.rmtree(log_dir)
    # Initialize logs config
    logs_cfg = LogsConfig(log_dir="test_citation_logs")
    logs_cfg.init_formatter()

    # Initialize ScholarQA
    return PostProcessedScholarQA(
        paper_finder=paper_finder,
        run_table_generation=False,
        logs_config=logs_cfg,
        llm_model="anthropic/claude-sonnet-4-20250514",
        fallback_llm=None
    )


@pytest.fixture(scope="session")
def questions_list():
    """Load test questions from file."""
    try:
        with open(QUESTIONS_FILE, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        return questions
    except FileNotFoundError:
        pytest.skip(f"Questions file {QUESTIONS_FILE} not found")
    except Exception as e:
        pytest.fail(f"Failed to load questions: {e}")


def validate_citations_for_question(scholar_qa: ScholarQA, question: str) -> tuple[int, List[Dict]]:
    """
    Process a question and return citation validation results.
    Returns (invalid_count, invalid_details_list)
    """
    response = scholar_qa.answer_query(question, inline_tags=True)

    invalid_citations = []
    sections = response.get('sections', [])

    for section in sections:
        section_title = section.get('title', 'Unknown Section')
        citations = section.get('citations', [])

        for citation in citations:
            should_filter, idx = should_filter_citation(citation)
            if should_filter:
                invalid_citations.append({
                    'section': section_title,
                    'text': citation["snippets"][idx][:200],
                    'paper_title': citation.get('paper', {}).get('title', 'N/A'),
                    'corpus_id': citation.get('paper', {}).get('corpus_id', 'N/A'),
                    'metadata': citation.get('metadata', {})
                })

    return len(invalid_citations), invalid_citations


@pytest.mark.parametrize("question_idx", range(25))  # Adjust range based on your questions file
def test_citation_validation(scholar_qa, questions_list, question_idx):
    """Test that citations are valid for each question."""
    # Skip if question index exceeds available questions

    if question_idx >= len(questions_list):
        pytest.skip(f"Question index {question_idx} exceeds available questions ({len(questions_list)})")

    question = questions_list[question_idx]

    try:
        invalid_count, invalid_details = validate_citations_for_question(scholar_qa, question)

        # Create detailed assertion message
        if invalid_count > 0:
            details_msg = "\nInvalid citations found:\n"
            for detail in invalid_details:
                details_msg += f"  • Section '{detail['section']}': {detail['corpus_id']}: {detail['text'][:100]}...\n"
            details_msg += f"\nQuestion was: {question}"
        else:
            details_msg = f"All citations are valid for question: {question}"

        assert invalid_count == 0, f"Found {invalid_count} invalid citations.{details_msg}"

    except Exception as e:
        pytest.fail(f"Error processing question '{question}': {e}")


def test_questions_file_exists():
    """Test that the questions file exists and is readable."""
    questions_path = Path(QUESTIONS_FILE)
    assert questions_path.exists(), f"Questions file {QUESTIONS_FILE} does not exist"
    assert questions_path.is_file(), f"{QUESTIONS_FILE} is not a file"

    # Test that file is not empty
    with open(questions_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    assert content, f"Questions file {QUESTIONS_FILE} is empty"


def test_scholar_qa_initialization(scholar_qa):
    """Test that ScholarQA system initializes correctly."""
    assert scholar_qa is not None, "ScholarQA should be initialized"
    assert hasattr(scholar_qa, 'answer_query'), "ScholarQA should have answer_query method"


def test_citation_filter_logic():
    """Test the citation filtering logic with known cases."""
    # Test case 1: Valid citation (should not be filtered)
    valid_citation = {
        'metadata': {
            'pdf_hash': 'abc123',
            'is_body': True,
            'sentence_offsets': [{'start': 0, 'end': 100}],
            'section_title': 'Introduction'
        }
    }
    assert not should_filter_citation(valid_citation), "Valid citation should not be filtered"

    # Test case 2: Invalid citation (should be filtered)
    invalid_citation = {
        'metadata': {
            'pdf_hash': '',
            'is_body': True,
            'sentence_offsets': [],
            'section_title': ''
        }
    }
    assert should_filter_citation(invalid_citation), "Invalid citation should be filtered"

    # Test case 3: Partial invalid (should not be filtered)
    partial_invalid = {
        'metadata': {
            'pdf_hash': '',
            'is_body': False,  # This makes it not match the filter condition
            'sentence_offsets': [],
            'section_title': ''
        }
    }
    assert not should_filter_citation(partial_invalid), "Partially matching citation should not be filtered"


if __name__ == "__main__":
    """Allow running as a script with pytest."""
    pytest.main([__file__, "-v"])
