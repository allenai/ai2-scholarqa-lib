import json
from pathlib import Path

import pandas as pd
import pytest

from scholarqa.lite.prompt_utils import (
    prepare_references_data,
    build_prompt,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename) as f:
        return f.read()


def load_json_fixture(filename: str):
    with open(FIXTURES_DIR / filename) as f:
        return json.load(f)


@pytest.fixture
def sample_query():
    return load_fixture("query.txt").strip()


@pytest.fixture
def sample_reranked_df():
    data = load_json_fixture("reranked_df.json")
    return pd.DataFrame(data)


@pytest.fixture
def expected_prompt():
    return load_fixture("expected_prompt.txt")


class TestPrepareReferencesData:

    def test_produces_expected_reference_keys(self, sample_reranked_df):
        """Verify prepare_references_data produces reference keys matching fixture data."""
        references, per_paper_data, quotes_metadata = prepare_references_data(sample_reranked_df)

        # Check that each paper's reference_string from fixture appears in output
        for _, row in sample_reranked_df.iterrows():
            expected_ref = row["reference_string"]
            assert expected_ref in references
            assert expected_ref in per_paper_data
            assert expected_ref in quotes_metadata

    def test_per_paper_data_structure(self, sample_reranked_df):
        """Verify per_paper_data has the expected structure."""
        _, per_paper_data, _ = prepare_references_data(sample_reranked_df)

        for ref_str, data in per_paper_data.items():
            assert "quote" in data
            assert "inline_citations" in data
            assert isinstance(data["inline_citations"], dict)


class TestBuildPrompt:

    def test_exact_match(self, sample_query, sample_reranked_df, expected_prompt):
        """Verify build_prompt produces the exact expected output."""
        references, _, _ = prepare_references_data(sample_reranked_df)
        actual_prompt = build_prompt(sample_query, references)
        assert actual_prompt == expected_prompt.strip()
