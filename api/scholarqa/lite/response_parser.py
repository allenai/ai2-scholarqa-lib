"""
Parser for one-shot generation model response output.

The model outputs structured text with SECTION; and TLDR; markers.
Citations are inline in format: [corpus_id | Author et al. | year | Citations: N]
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from anyascii import anyascii

logger = logging.getLogger(__name__)

# Regex pattern for inline citations: [corpus_id | Author et al. | year | Citations: N]
CITATION_PATTERN = re.compile(
    r"\[(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*Citations:\s*(\d+)\]"
)


def _strip_think_block(response: str) -> str:
    """Remove <think>...</think> block from response."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def parse_title(response: str) -> str:
    """Parse title from model response, stripping think blocks."""
    return _strip_think_block(response)


def _extract_sections_raw(response: str) -> List[str]:
    """Split response into raw section strings on SECTION; markers."""
    # Example input: "SECTION; Intro\nTLDR; ...\nBody...\nSECTION; Methods\n..."
    first_section = response.find("SECTION;")
    if first_section == -1:
        return []

    # Strip everything before first SECTION;
    response = response[first_section:]

    # Split on "SECTION;" markers
    # Example parts: ["", "Intro\nTLDR; ...\nBody...", "Methods\n..."]
    parts = re.split(r"SECTION;\s*", response)

    # Filter empty strings, return list of raw section content
    # Example output: ["Intro\nTLDR; ...\nBody...", "Methods\n..."]
    return [p.strip() for p in parts if p.strip()]


def _clean_tldr(tldr: str) -> str:
    """Remove LLM/Model citations and source counts from TLDR text."""
    # Remove patterns like (LLM Memory), (Model-Generated), [LLM Memory], etc.
    cleaned = re.sub(r"\s*[\(\[][^)\]]*(?:LLM|Model)[^)\]]*[\)\]]", "", tldr)
    # Remove patterns like (N sources), (1 source)
    cleaned = re.sub(r"\s*\(\d+\s+sources?\)", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _normalize_llm_memory(text: str) -> str:
    """Convert (LLM Memory) to citation format to match multi-step pipeline."""
    return re.sub(r"\(LLM Memory\)", "[LLM MEMORY | 2024]", text, flags=re.IGNORECASE)


def _normalize_paragraph_breaks(text: str) -> str:
    """Ensure proper paragraph breaks to match the multi-step pipeline format."""
    # Pattern: end of sentence (. ! ?) followed by single newline and capital letter
    # Convert to double newline for markdown paragraph break
    text = re.sub(r"([.!?])\n([A-Z])", r"\1\n\n\2", text)
    return text


def _convert_section_format(raw_section: str) -> Tuple[str, str]:
    """Normalize section to format: Title, TLDR line, body text.

    Returns:
        Tuple of (title, formatted_section_text)
    """
    # Example input: "Introduction to Transformers\nTLDR; Overview of attention.\nTransformers use..."

    # Split into title and everything else
    # Example: lines = ["Introduction to Transformers", "TLDR; Overview of attention.\nTransformers use..."]
    lines = raw_section.split("\n", 1)
    title = lines[0].strip()
    remaining = lines[1].strip() if len(lines) > 1 else ""

    # Split remaining into TLDR line and body
    # Example: remaining_lines = ["TLDR; Overview of attention.", "Transformers use..."]
    remaining_lines = remaining.split("\n", 1)
    tldr_line = _clean_tldr(remaining_lines[0])
    body = remaining_lines[1].strip() if len(remaining_lines) > 1 else ""
    body = _normalize_llm_memory(body)
    body = _normalize_paragraph_breaks(body)

    return title, f"{title}\n{tldr_line}\n{body}"


def parse_sections(response: str) -> Tuple[List[str], List[str]]:
    """Parse response into section strings and titles.

    Returns:
        Tuple of (section_texts, section_titles)
    """
    cleaned = _strip_think_block(response)
    raw_sections = _extract_sections_raw(cleaned)
    results = [_convert_section_format(s) for s in raw_sections]
    titles = [r[0] for r in results]
    texts = [r[1] for r in results]
    return texts, titles


def filter_per_paper_summaries(
    section_texts: List[str],
    per_paper_data: Dict[str, Dict[str, Any]],
    all_quotes_metadata: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Filter pre-computed per-paper data to only include citations found in section texts.

    Args:
        section_texts: List of section text strings from the model response
        per_paper_data: Pre-computed {ref_str: {"quote": ..., "inline_citations": {}}} from prepare_references_data
        all_quotes_metadata: Pre-computed {ref_str: [snippet_metadata]} from prepare_references_data

    Returns:
        per_paper_summaries_extd: Filtered per_paper_data for citations found in response
        quotes_metadata: Filtered quotes_metadata for citations found in response
    """
    per_paper_summaries_extd = {}
    quotes_metadata = {}

    all_text = "\n".join(section_texts)
    citations = CITATION_PATTERN.findall(all_text)

    seen_corpus_ids = set()
    for corpus_id, author_str, year, citation_count in citations:
        if corpus_id in seen_corpus_ids:
            continue
        seen_corpus_ids.add(corpus_id)

        citation_key = anyascii(f"[{corpus_id} | {author_str.strip()} | {year} | Citations: {citation_count}]")

        if citation_key in per_paper_data:
            per_paper_summaries_extd[citation_key] = per_paper_data[citation_key]
            quotes_metadata[citation_key] = all_quotes_metadata[citation_key]

    logger.info(f"Built per_paper_summaries with {len(per_paper_summaries_extd)} citations")
    return per_paper_summaries_extd, quotes_metadata
