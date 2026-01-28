"""Tests for scholarqa.lite.prompt_utils module."""

from scholarqa.lite.prompt_utils import normalize_snippet_quote


class TestNormalizeSnippetQuote:

    def test_strips_whitespace(self):
        assert normalize_snippet_quote("  hello world  ") == "hello world"
        assert normalize_snippet_quote("\n\ttext\n") == "text"

    def test_strips_trailing_periods(self):
        assert normalize_snippet_quote("sentence.") == "sentence"
        assert normalize_snippet_quote("sentence..") == "sentence"
        assert normalize_snippet_quote("sentence...") == "sentence"

    def test_preserves_internal_periods(self):
        assert normalize_snippet_quote("Dr. Smith et al.") == "Dr. Smith et al"
        assert normalize_snippet_quote("Fig. 1 shows results.") == "Fig. 1 shows results"

    def test_converts_curly_quotes_to_straight(self):
        assert normalize_snippet_quote("He said \u201chello\u201d") == 'He said "hello"'
        assert normalize_snippet_quote("\u201cquoted\u201d") == '"quoted"'

    def test_converts_unicode_to_ascii(self):
        assert normalize_snippet_quote("text\u2026") == "text"
        assert normalize_snippet_quote("word\u2014another") == "word-another"

    def test_join_split_roundtrip(self):
        quotes = ["First sentence.", "Second sentence.", "Third sentence."]
        normalized = [normalize_snippet_quote(q) for q in quotes]
        combined = "...".join(normalized)
        split_back = [s.strip() for s in combined.split("...")]
        assert split_back == normalized
