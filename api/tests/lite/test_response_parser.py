from pathlib import Path

import pytest

from scholarqa.lite.response_parser import (
    parse_sections,
    build_per_paper_summaries,
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
        sections = parse_sections(sample_response)
        assert len(sections) == 6

    def test_section_titles(self, sample_response):
        sections = parse_sections(sample_response)
        expected_titles = [
            "Architectural innovations and theoretical insights",
            "Vision-specific advances",
            "NLP-specific advances",
            "Multi-modal and cross-domain advances",
            "Hardware acceleration and efficiency",
            "Theoretical foundations and future directions",
        ]
        actual_titles = [s.split('\n')[0] for s in sections]
        assert actual_titles == expected_titles


class TestBuildPerPaperSummaries:

    def test_extracts_citations(self, sample_response, sample_reranked_df):
        sections = parse_sections(sample_response)
        # Get pre-computed data from prepare_references_data
        _, per_paper_data, all_quotes_metadata = prepare_references_data(sample_reranked_df)
        per_paper_summaries, quotes_metadata = build_per_paper_summaries(
            sections, per_paper_data, all_quotes_metadata
        )
        # Should find citations that match papers in reranked_df
        assert len(per_paper_summaries) > 0
        assert len(quotes_metadata) > 0
        # Keys should match between the two dicts
        assert per_paper_summaries.keys() == quotes_metadata.keys()
