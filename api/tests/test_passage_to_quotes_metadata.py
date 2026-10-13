import pandas as pd

from scholarqa.scholar_qa import ScholarQA


REF_STR = "[12345678 | Doe et al. | 2024 | Citations: 3]"
UNRELATED_SENTENCE = {
    "text": "Completely unrelated passage about protein folding in yeast cells.",
    "section_title": "Introduction",
    "pdf_hash": "abc123",
}


def _run(quote, title, abstract):
    # The paper has passages, but none match the quote, so the title/abstract fallback is used
    retrieval_df = pd.DataFrame([{
        "reference_string": REF_STR,
        "sentences": [UNRELATED_SENTENCE],
        "title": title,
        "abstract": abstract,
    }])
    return ScholarQA.passage_to_quotes_metadata(retrieval_df, {REF_STR: quote}, {"Section 1": [0]})


class TestMissingPaperFields:
    def test_none_abstract_does_not_crash(self):
        quotes_metadata = _run("Large language models improve citation recall in literature review.",
                               title="A Study of Something Else", abstract=None)
        assert "section_title" not in quotes_metadata[REF_STR][0]

    def test_none_abstract_still_matches_title(self):
        title = "Large language models improve citation recall"
        quotes_metadata = _run(title, title=title, abstract=None)
        assert quotes_metadata[REF_STR][0]["section_title"] == "title"

    def test_none_title_still_matches_abstract(self):
        abstract = "We show that large language models improve citation recall in literature review."
        quotes_metadata = _run("large language models improve citation recall", title=None, abstract=abstract)
        assert quotes_metadata[REF_STR][0]["section_title"] == "abstract"
