from scholarqa.lite.response_parser import filter_per_paper_summaries
from scholarqa.utils import build_corpus_id_lookup, build_unique_author_lookup


AGARWAL_KEY = "[88521252 | Agarwal et al. | 2016 | Citations: 0]"
BUI_KEY = "[258999746 | Bui et al. | 2023 | Citations: 31]"

AGARWAL_WRONG_AUTHOR = "[88521252 | Agarwal | 2016 | Citations: 0]"  # dropped "et al."
AGARWAL_WRONG_COUNT = "[88521252 | Agarwal et al. | 2016 | Citations: 999]"  # stale citation count

def _make_paper_data(*keys):
    ppd = {k: {"quote": f"quote for {k}", "inline_citations": {}} for k in keys}
    qmd = {k: [{"quote": f"quote for {k}", "section_title": "abstract"}] for k in keys}
    return ppd, qmd


PER_PAPER_DATA, QUOTES_METADATA = _make_paper_data(AGARWAL_KEY, BUI_KEY)


MOLIN_KEY = "[12345678 | Molin et al. | 2020 | Citations: 15]"


class TestBuildUniqueAuthorLookup:
    def test_maps_unique_author_to_key(self):
        lookup = build_unique_author_lookup(PER_PAPER_DATA)
        assert lookup["Agarwal"] == AGARWAL_KEY
        assert lookup["Bui"] == BUI_KEY

    def test_excludes_ambiguous_authors(self):
        """Two papers by 'Agarwal' — neither should appear."""
        keys = [AGARWAL_KEY, "[99999999 | Agarwal et al. | 2020 | Citations: 5]"]
        lookup = build_unique_author_lookup(keys)
        assert "Agarwal" not in lookup


class TestBuildCorpusIdLookup:
    def test_maps_corpus_id_to_canonical_key(self):
        lookup = build_corpus_id_lookup(PER_PAPER_DATA)
        assert lookup["88521252"] == AGARWAL_KEY
        assert lookup["258999746"] == BUI_KEY

    def test_year_not_treated_as_corpus_id(self):
        """A 4-digit year like '2016' should not be indexed as a corpus ID."""
        lookup = build_corpus_id_lookup(["[2016 | Someone | 2016 | Citations: 0]"])
        assert "2016" not in lookup


class TestFilterPerPaperSummariesRelaxed:

    def test_exact_match(self):
        """Citation string matches canonical key exactly — no fallback needed."""
        sections = [
            "Agarwal et al. proposed a g-prior extension "
            "[88521252 | Agarwal et al. | 2016 | Citations: 0]."
        ]
        _, result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_KEY in result_ppd

    def test_wrong_author_falls_back_to_corpus_id(self):
        """The actual production bug: model wrote 'Agarwal' instead of 'Agarwal et al.'"""
        sections = [f"Agarwal proposed a g-prior extension {AGARWAL_WRONG_AUTHOR}."]
        _, result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_KEY in result_ppd

    def test_wrong_citation_count_falls_back(self):
        """Model outputs stale citation count — corpus ID fallback resolves it."""
        sections = [f"Agarwal et al. extended the g-prior setup {AGARWAL_WRONG_COUNT}."]
        _, result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_KEY in result_ppd

    def test_no_match(self):
        """Unknown corpus ID doesn't match anything."""
        sections = [
            "Unknown paper reference "
            "[11111111 | Nobody | 2024 | Citations: 0]."
        ]
        _, result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert len(result_ppd) == 0

    def test_same_corpus_id_twice_deduplicates(self):
        """Same paper cited with exact match and wrong count — appears once."""
        sections = [
            f"Agarwal et al. extended the g-prior setup {AGARWAL_WRONG_COUNT} "
            f"and later confirmed results {AGARWAL_KEY}."
        ]
        _, result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert len(result_ppd) == 1
        assert AGARWAL_KEY in result_ppd

    def test_malformed_bracket_rewritten_in_text(self):
        """Malformed bracket citation is replaced with canonical key in section text."""
        sections = [f"Agarwal proposed a g-prior extension {AGARWAL_WRONG_AUTHOR}."]
        result_texts, _, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_WRONG_AUTHOR not in result_texts[0]
        assert AGARWAL_KEY in result_texts[0]

    def test_prose_author_mention_without_bracket(self):
        """Model mentions author in prose but never emits a bracket citation."""
        ppd, qmd = _make_paper_data(AGARWAL_KEY, MOLIN_KEY)
        sections = ['As Molin et al. explain, "Software systems evolve."']
        _, result_ppd, _ = filter_per_paper_summaries(sections, ppd, qmd)
        assert MOLIN_KEY in result_ppd

    def test_prose_author_bracket_injected_in_text(self):
        """Prose-only author mention gets a bracket citation injected."""
        ppd, qmd = _make_paper_data(AGARWAL_KEY, MOLIN_KEY)
        sections = ['As Molin et al. explain, "Software systems evolve."']
        result_texts, _, _ = filter_per_paper_summaries(sections, ppd, qmd)
        assert MOLIN_KEY in result_texts[0]

    def test_prose_author_skipped_when_ambiguous(self):
        """Two papers by same first author — prose mention should not resolve."""
        keys = [MOLIN_KEY, "[99999999 | Molin et al. | 2022 | Citations: 3]"]
        ppd, qmd = _make_paper_data(*keys)
        sections = ['As Molin et al. explain, "Software systems evolve."']
        _, result_ppd, _ = filter_per_paper_summaries(sections, ppd, qmd)
        assert len(result_ppd) == 0

    def test_prose_author_not_double_counted_with_bracket(self):
        """Paper already resolved via bracket — prose scan should not duplicate it."""
        sections = [
            f"Agarwal et al. proposed a g-prior extension {AGARWAL_KEY}."
        ]
        _, result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert len(result_ppd) == 1

    def test_prose_repeated_author_gets_one_bracket(self):
        """Author mentioned twice in prose — bracket injected only on first occurrence."""
        ppd, qmd = _make_paper_data(MOLIN_KEY)
        sections = ['Molin et al. found X. Later, Molin et al. confirmed Y.']
        result_texts, _, _ = filter_per_paper_summaries(sections, ppd, qmd)
        assert result_texts[0].count(MOLIN_KEY) == 1

    def test_prose_single_author_bracket_injected(self):
        """Single-author paper (no 'et al.') gets bracket injected after bare last name."""
        smith_key = "[55555555 | Smith | 2021 | Citations: 10]"
        ppd, qmd = _make_paper_data(smith_key)
        sections = ['Smith proposed a novel framework for testing.']
        result_texts, result_ppd, _ = filter_per_paper_summaries(sections, ppd, qmd)
        assert smith_key in result_ppd
        assert smith_key in result_texts[0]
