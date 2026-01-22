from pathlib import Path

import pytest

from scholarqa.unified.response_parser import (
    parse_report_title,
    parse_sections,
    build_per_paper_summaries,
)

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


class TestParseReportTitle:

    def test_extracts_title(self, sample_response):
        title = parse_report_title(sample_response)
        assert title == "Latest Advances in Transformer Architectures"


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
        per_paper_summaries, quotes_metadata = build_per_paper_summaries(
            sections, sample_reranked_df
        )
        # Should find citations that match papers in reranked_df
        assert len(per_paper_summaries) > 0
        assert len(quotes_metadata) > 0
        # Keys should match between the two dicts
        assert per_paper_summaries.keys() == quotes_metadata.keys()
