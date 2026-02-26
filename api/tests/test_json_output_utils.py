from scholarqa.postprocess.json_output_utils import get_section_text, get_json_summary, _CITATION_RE


class TestGetSectionTextTldrCitations:
    """Test that get_section_text extracts TLDRs correctly, paired with
    _CITATION_RE which strips raw citations in get_json_summary."""

    def test_tldr_citations_are_stripped(self):
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

    def test_duplicate_tldr_tokens(self):
        """Duplicate TLDR lines should not crash."""
        gen_text = (
            "Introduction/Background\n"
            "TLDR; Large language models learn to predict the next token in huge text "
            "collections and are then adapted to specific uses. They rely on the Transformer "
            "architecture, scaling laws, and multi-stage training\u2014often using parameter-efficient "
            "methods when fine-tuning large models.\n"
            "TLDR; Large language models (LLMs) learn to predict the next token in huge text "
            "collections and are then adapted to specific uses. They rely on the Transformer "
            "architecture, scaling laws, and multi-stage training\u2014often using parameter-efficient "
            "methods when fine-turing large models.\n\n"
            "Training an LLM starts with understanding its purpose and basic setup. "
            "LLMs are deep neural networks trained for language modeling\u2014that is, predicting "
            "the next token in a sequence given its preceding context "
            "[277994124 | Wei et al. | 2025 | Citations: 12] "
            "[272827311 | Xu et al. | 2024 | Citations: 4]."
        )
        section = get_section_text(gen_text)
        assert section["title"] == "Introduction/Background"
        assert "predict the next token" in section["tldr"]
        assert "TLDR" not in section["text"]
        assert "Training an LLM" in section["text"]

    def test_tldr_with_no_body_text(self):
        """TLDR as last line with no body text should not crash."""
        gen_text = (
            "Regulatory & implementation milestones\n"
            "TLDR; RTS,S entered Phase-III trials in 2016 and is now in country roll-outs, "
            "while new dosing schedules and adjuvants are expected to raise protection levels.\n "
        )
        section = get_section_text(gen_text)
        assert section["title"] == "Regulatory & implementation milestones"
        assert "RTS,S" in section["tldr"]
        assert section["text"].strip() == ""



class TestGetJsonSummaryErrorHandling:
    """Test that get_json_summary skips sections that fail to parse
    instead of crashing the entire report."""

    def test_unparseable_section_is_skipped(self):
        """A single-line section with no TLDR and no newline is unparseable.
        get_json_summary should skip it and still return the good sections."""
        good_section = (
            "Introduction\n"
            "**TLDR:** A good summary.\n"
            "Body text with details."
        )
        bad_section = "Just a single line with no structure"
        summary_sections = [good_section, bad_section]
        # Minimal valid inputs for get_json_summary
        summary_quotes = {}
        paper_metadata = {}
        citation_ids = {}
        result = get_json_summary(
            llm_model="openai/gpt-4",
            summary_sections=summary_sections,
            summary_quotes=summary_quotes,
            paper_metadata=paper_metadata,
            citation_ids=citation_ids,
        )
        assert len(result) == 1
        assert result[0]["title"] == "Introduction"

    def test_all_good_sections_preserved(self):
        """When all sections are parseable, none are dropped."""
        sections = [
            "Section A\n**TLDR:** Summary A.\nBody A.",
            "Section B\n**TLDR:** Summary B.\nBody B.",
        ]
        result = get_json_summary(
            llm_model="openai/gpt-4",
            summary_sections=sections,
            summary_quotes={},
            paper_metadata={},
            citation_ids={},
        )
        assert len(result) == 2
        assert result[0]["title"] == "Section A"
        assert result[1]["title"] == "Section B"
