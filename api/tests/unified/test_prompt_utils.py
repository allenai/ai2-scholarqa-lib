import json
from pathlib import Path

import pandas as pd
import pytest

from scholarqa.unified.prompt_utils import (
    format_reference_string,
    format_df_as_references,
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
def sample_paper_metadata():
    return load_json_fixture("paper_metadata.json")


@pytest.fixture
def expected_prompt():
    return load_fixture("expected_prompt.txt")


class TestFormatReferenceString:

    def test_matches_fixture_reference_strings(self, sample_paper_metadata):
        """Verify format_reference_string produces the expected reference strings from fixtures."""
        expected = {
            "252788259": "[252788259 | Chitty-Venkata et al. | 2022 | Citations: 93]",
            "281315524": "[281315524 | Li et al. | 2025 | Citations: 0]",
            "274598165": "[274598165 | Chen et al. | 2024 | Citations: 0]",
        }
        for corpus_id, expected_ref in expected.items():
            result = format_reference_string(sample_paper_metadata[corpus_id])
            assert result == expected_ref


class TestFormatDfAsReferences:

    def test_produces_expected_reference_keys(self, sample_reranked_df, sample_paper_metadata):
        """Verify format_df_as_references produces reference keys matching fixture data."""
        references = format_df_as_references(sample_reranked_df, sample_paper_metadata)

        # Check that each paper's reference_string from fixture appears in output
        for _, row in sample_reranked_df.iterrows():
            expected_ref = row["reference_string"]
            assert expected_ref in references


class TestBuildPrompt:

    def test_exact_match(self, sample_query, sample_reranked_df, sample_paper_metadata, expected_prompt):
        """Verify build_prompt produces the exact expected output."""
        references = format_df_as_references(sample_reranked_df, sample_paper_metadata)
        actual_prompt = build_prompt(sample_query, references)
        assert actual_prompt == expected_prompt
