"""
Parser for unified generation model response output.

The model outputs structured text with TITLE;, SECTION;, and TLDR; markers.
Citations are inline in format: [corpus_id | Author et al. | year | Citations: N]
"""

import logging
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Regex pattern for inline citations: [corpus_id | Author et al. | year | Citations: N]
CITATION_PATTERN = re.compile(
    r"\[(\d+)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*Citations:\s*(\d+)\]"
)


def _strip_think_block(response: str) -> str:
    """Remove <think>...</think> block from response."""
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


def _extract_title(response: str) -> str:
    """Extract title from TITLE; marker in response."""
    match = re.search(r"TITLE;\s*(.+?)(?:\n|SECTION;)", response, re.DOTALL)
    if match:
        title = match.group(1).strip()
        logger.info(f"Extracted report title: '{title}'")
        return title
    logger.warning("No TITLE; marker found in response")
    return ""


def _extract_sections_raw(response: str) -> List[str]:
    """Split response into raw section strings on SECTION; markers."""
    # Example input: "TITLE; ...\nSECTION; Intro\nTLDR; ...\nBody...\nSECTION; Methods\n..."
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


def _convert_section_format(raw_section: str) -> str:
    """Normalize section to format: Title, TLDR line, body text."""
    # Example input: "Introduction to Transformers\nTLDR; Overview of attention.\nTransformers use..."

    # Split into title and everything else
    # Example: lines = ["Introduction to Transformers", "TLDR; Overview of attention.\nTransformers use..."]
    lines = raw_section.split("\n", 1)
    title = lines[0].strip()  # "Introduction to Transformers"
    remaining = lines[1].strip() if len(lines) > 1 else ""

    # Split remaining into TLDR line and body
    # Example: remaining_lines = ["TLDR; Overview of attention.", "Transformers use..."]
    remaining_lines = remaining.split("\n", 1)
    tldr_line = remaining_lines[0]  # "TLDR; Overview of attention."
    body = remaining_lines[1].strip() if len(remaining_lines) > 1 else ""

    # Return normalized format: "Title\nTLDR line\nBody"
    return f"{title}\n{tldr_line}\n{body}"


def _get_snippets_for_paper(
    corpus_id: str, scored_df: pd.DataFrame
) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract snippet text and metadata for a corpus_id from scored_df."""
    snippets = []
    snippet_metadata = []

    for _, row in scored_df.iterrows():
        if str(row["corpus_id"]) == corpus_id:
            sentences = row["sentences"]
            if sentences:
                for sent in sentences:
                    snippets.append(sent["text"])
                    snippet_metadata.append({
                        "quote": sent["text"],
                        "section_title": sent.get("section_title", "abstract"),
                        "pdf_hash": sent.get("pdf_hash", ""),
                        "sentence_offsets": sent.get("sentence_offsets", []),
                    })
            else:
                # Fall back to abstract if no sentences
                abstract = row.get("abstract", "")
                if abstract:
                    snippets.append(abstract)
                    snippet_metadata.append({
                        "quote": abstract,
                        "section_title": "abstract",
                        "pdf_hash": "",
                    })
            break

    combined_quote = "...".join(snippets) if snippets else ""
    return combined_quote, snippet_metadata


def parse_report_title(response: str) -> str:
    """Extract the report title from the TITLE; marker in the response."""
    cleaned = _strip_think_block(response)
    return _extract_title(cleaned)


def parse_sections(response: str) -> List[str]:
    """Parse response into section strings."""
    cleaned = _strip_think_block(response)
    raw_sections = _extract_sections_raw(cleaned)
    return [_convert_section_format(s) for s in raw_sections]


def build_per_paper_summaries(
    section_texts: List[str], scored_df: pd.DataFrame
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Extract citations from section texts and build per_paper_summaries_extd and quotes_metadata.
    Citations are matched back to the original snippets from scored_df.
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

        citation_key = f"[{corpus_id} | {author_str} | {year} | Citations: {citation_count}]"
        quote_text, snippet_meta = _get_snippets_for_paper(corpus_id, scored_df)

        if quote_text:
            per_paper_summaries_extd[citation_key] = {
                "quote": quote_text,
                "inline_citations": {},
            }
            quotes_metadata[citation_key] = snippet_meta

    logger.info(f"Built per_paper_summaries with {len(per_paper_summaries_extd)} citations")
    return per_paper_summaries_extd, quotes_metadata
