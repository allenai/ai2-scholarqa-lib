from scholarqa.lite.response_parser import filter_per_paper_summaries
from scholarqa.utils import build_corpus_id_lookup


AGARWAL_KEY = "[88521252 | Agarwal et al. | 2016 | Citations: 0]"
BUI_KEY = "[258999746 | Bui et al. | 2023 | Citations: 31]"

AGARWAL_WRONG_AUTHOR = "[88521252 | Agarwal | 2016 | Citations: 0]"  # dropped "et al."
AGARWAL_WRONG_COUNT = "[88521252 | Agarwal et al. | 2016 | Citations: 999]"  # stale citation count

def _make_paper_data(*keys):
    ppd = {k: {"quote": f"quote for {k}", "inline_citations": {}} for k in keys}
    qmd = {k: [{"quote": f"quote for {k}", "section_title": "abstract"}] for k in keys}
    return ppd, qmd


PER_PAPER_DATA, QUOTES_METADATA = _make_paper_data(AGARWAL_KEY, BUI_KEY)


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
        result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_KEY in result_ppd

    def test_wrong_author_falls_back_to_corpus_id(self):
        """The actual production bug: model wrote 'Agarwal' instead of 'Agarwal et al.'"""
        sections = [f"Agarwal proposed a g-prior extension {AGARWAL_WRONG_AUTHOR}."]
        result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_KEY in result_ppd

    def test_wrong_citation_count_falls_back(self):
        """Model outputs stale citation count — corpus ID fallback resolves it."""
        sections = [f"Agarwal et al. extended the g-prior setup {AGARWAL_WRONG_COUNT}."]
        result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert AGARWAL_KEY in result_ppd

    def test_no_match(self):
        """Unknown corpus ID doesn't match anything."""
        sections = [
            "Unknown paper reference "
            "[11111111 | Nobody | 2024 | Citations: 0]."
        ]
        result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert len(result_ppd) == 0

    def test_same_corpus_id_twice_deduplicates(self):
        """Same paper cited with exact match and wrong count — appears once."""
        sections = [
            f"Agarwal et al. extended the g-prior setup {AGARWAL_WRONG_COUNT} "
            f"and later confirmed results {AGARWAL_KEY}."
        ]
        result_ppd, _ = filter_per_paper_summaries(sections, PER_PAPER_DATA, QUOTES_METADATA)
        assert len(result_ppd) == 1
        assert AGARWAL_KEY in result_ppd
