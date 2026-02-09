from scholarqa.postprocess.json_output_utils import get_section_text, _CITATION_RE


class TestGetSectionTextTldrCitations:
    """Test that get_section_text extracts TLDRs correctly, paired with
    _CITATION_RE which strips raw citations in get_json_summary."""

    def test_tldr_with_citations_is_strippable(self):
        """Simulate what get_json_summary does: extract TLDR, then strip citations."""
        gen_text = (
            "Introduction/Background\n"
            "**TLDR:** Models learn patterns [258999746 | Bui et al. | 2023 | Citations: 30] "
            "and selection matters [234482526 | Zhou et al. | 2021 | Citations: 17].\n"
            "Body text with [258999746 | Bui et al. | 2023 | Citations: 30] inline."
        )
        section = get_section_text(gen_text)
        assert "tldr" in section
        # Raw TLDR has citations
        assert "258999746" in section["tldr"]
        # After stripping with _CITATION_RE (as get_json_summary does):
        cleaned = _CITATION_RE.sub("", section["tldr"]).strip()
        assert "258999746" not in cleaned
        assert "Citations:" not in cleaned
        assert "Models learn patterns" in cleaned

    def test_tldr_without_citations_unchanged(self):
        gen_text = (
            "Methods\n"
            "**TLDR:** A plain summary with no citations.\n"
            "Body text here."
        )
        section = get_section_text(gen_text)
        original = section["tldr"]
        cleaned = _CITATION_RE.sub("", original).strip()
        assert cleaned == original

    def test_citation_re_handles_extra_whitespace(self):
        tldr = "Summary [123 |  Smith et al.  |  2024  |  Citations:  5] end."
        cleaned = _CITATION_RE.sub("", tldr).strip()
        assert "123" not in cleaned
        assert "end." in cleaned

    def test_citation_re_handles_single_author(self):
        tldr = "Summary [999 | Zhang | 2025 | Citations: 0] end."
        cleaned = _CITATION_RE.sub("", tldr).strip()
        assert "999" not in cleaned
