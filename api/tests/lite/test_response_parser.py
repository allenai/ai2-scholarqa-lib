from pathlib import Path

import pytest

from scholarqa.lite.response_parser import (
    parse_sections,
    filter_per_paper_summaries,
    _clean_tldr,
    _normalize_paragraph_breaks,
)
from scholarqa.lite.prompt_utils import prepare_references_data

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_response():
    return (FIXTURES_DIR / "response.txt").read_text()


@pytest.fixture
def sample_reranked_df():
    import json
    import pandas as pd
    data = json.loads((FIXTURES_DIR / "reranked_df.json").read_text())
    return pd.DataFrame(data)


class TestParseSections:

    def test_section_count(self, sample_response):
        sections, titles = parse_sections(sample_response)
        assert len(sections) == 6
        assert len(titles) == 6

    def test_section_titles(self, sample_response):
        _, titles = parse_sections(sample_response)
        expected_titles = [
            "Architectural innovations and theoretical insights",
            "Vision-specific advances",
            "NLP-specific advances",
            "Multi-modal and cross-domain advances",
            "Hardware acceleration and efficiency",
            "Theoretical foundations and future directions",
        ]
        assert titles == expected_titles


class TestCleanTldr:

    def test_removes_model_generated(self):
        assert _clean_tldr("TLDR; Summary. (Model-Generated)") == "TLDR; Summary."

    def test_removes_llm_memory(self):
        assert _clean_tldr("TLDR; Summary. (LLM Memory)") == "TLDR; Summary."

    def test_removes_source_count_singular(self):
        assert _clean_tldr("TLDR; Summary. (1 source)") == "TLDR; Summary."

    def test_removes_source_count_plural(self):
        assert _clean_tldr("TLDR; Summary. (9 sources)") == "TLDR; Summary."

    def test_removes_source_count_large(self):
        assert _clean_tldr("TLDR; Summary. (19 sources)") == "TLDR; Summary."

    def test_removes_single_inline_citation(self):
        tldr = "TLDR; Models learn patterns [258999746 | Bui et al. | 2023 | Citations: 30]."
        assert _clean_tldr(tldr) == "TLDR; Models learn patterns."

    def test_removes_multiple_inline_citations(self):
        tldr = (
            "TLDR; Models learn patterns [258999746 | Bui et al. | 2023 | Citations: 30] "
            "and data selection matters [234482526 | Zhou et al. | 2021 | Citations: 17]."
        )
        assert _clean_tldr(tldr) == "TLDR; Models learn patterns and data selection matters."

    def test_removes_citation_with_extra_whitespace(self):
        tldr = "TLDR; Summary [123 |  Smith et al.  |  2024  |  Citations:  5]."
        assert _clean_tldr(tldr) == "TLDR; Summary."

    def test_preserves_plain_tldr(self):
        tldr = "TLDR; A plain summary with no citations."
        assert _clean_tldr(tldr) == "TLDR; A plain summary with no citations."


class TestNormalizeParagraphBreaks:

    def test_converts_single_newline_to_double(self):
        text = "First sentence ends here.\nSecond paragraph starts here."
        result = _normalize_paragraph_breaks(text)
        assert result == "First sentence ends here.\n\nSecond paragraph starts here."

    def test_preserves_existing_double_newlines(self):
        text = "First paragraph.\n\nSecond paragraph."
        result = _normalize_paragraph_breaks(text)
        assert result == "First paragraph.\n\nSecond paragraph."

    def test_handles_multiple_paragraphs(self):
        text = "First paragraph!\nSecond paragraph?\nThird paragraph."
        result = _normalize_paragraph_breaks(text)
        assert result == "First paragraph!\n\nSecond paragraph?\n\nThird paragraph."

    def test_preserves_lowercase_continuations(self):
        text = "This is a sentence\nthat continues on the next line."
        result = _normalize_paragraph_breaks(text)
        # Should not add double newline since next line starts with lowercase
        assert result == "This is a sentence\nthat continues on the next line."

    def test_preserves_list_items(self):
        text = "Introduction.\n- Item one\n- Item two"
        result = _normalize_paragraph_breaks(text)
        # Should not add double newline before list items (they start with -)
        assert result == "Introduction.\n- Item one\n- Item two"


class TestFilterPerPaperSummaries:

    def test_extracts_citations(self, sample_response, sample_reranked_df):
        sections, _ = parse_sections(sample_response)
        # Get pre-computed data from prepare_references_data
        _, per_paper_data, all_quotes_metadata = prepare_references_data(sample_reranked_df)
        per_paper_summaries, quotes_metadata = filter_per_paper_summaries(
            sections, per_paper_data, all_quotes_metadata
        )
        # Should find citations that match papers in reranked_df
        assert len(per_paper_summaries) > 0
        assert len(quotes_metadata) > 0
        # Keys should match between the two dicts
        assert per_paper_summaries.keys() == quotes_metadata.keys()
