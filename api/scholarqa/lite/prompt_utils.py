"""
Prompt building and reference formatting for lite generation.
"""

import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from anyascii import anyascii
from scholarqa.llms.prompts import UNIFIED_GENERATION_PROMPT, REPORT_TITLE_DIRECTIVE


TITLE_GENERATION_PROMPT = f"""Generate a concise report title based on the user query and section titles provided.

{REPORT_TITLE_DIRECTIVE}

User query: {{query}}

Section titles:
{{section_titles}}

Output ONLY the title text, nothing else."""


def normalize_snippet_quote(text: str) -> str:
    """
    Normalize quote for matching after "...".join() and split("...").

    Strips trailing periods to avoid "s1." + "..." + "s2" = "s1....s2" splitting to ".s2".
    """
    return anyascii(text.strip()).rstrip(".")


def prepare_references_data(
    scored_df: pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    references = {}
    per_paper_data = {}
    quotes_metadata = {}

    for _, row in scored_df.iterrows():
        # Use the reference_string already computed in retrieval.py
        # Example: ref_str = "[12345678 | Smith and Doe | 2023 | Citations: 150]"
        ref_str = row["reference_string"]

        # Skip if we've already seen this paper
        if ref_str in references:
            continue

        # Get retrieved sentences (snippets from full-text search)
        # Example: sentences = [{"text": "Transformers use attention.", ...}, {"text": "This improves...", ...}]
        sentences = row["sentences"]

        if sentences:
            # Sort sentences by char_offset to maintain document order
            sentences = sorted(sentences, key=lambda s: s.get("char_offset", 0))

            # Concatenate all snippet texts into one passage for the prompt
            text = " ".join(sent["text"] for sent in sentences)

            snippet_metadata = []
            normalized_quotes = []
            for sent in sentences:
                quote = normalize_snippet_quote(sent["text"])
                normalized_quotes.append(quote)
                snippet_metadata.append({
                    "quote": quote,
                    "section_title": sent.get("section_title", "abstract"),
                    "pdf_hash": sent.get("pdf_hash", ""),
                    "sentence_offsets": sent.get("sentence_offsets", []),
                })

            # Combined quote uses same normalized texts so splitting produces matching quotes
            combined_quote = "...".join(normalized_quotes)
        else:
            # Fall back to abstract if no sentences (e.g., abstract-only retrieval)
            text = row.get("abstract", "")
            snippet_metadata = []
            if text:
                combined_quote = normalize_snippet_quote(text)
                snippet_metadata.append({
                    "quote": combined_quote,
                    "section_title": "abstract",
                    "pdf_hash": "",
                })
            else:
                combined_quote = ""

        if text:
            references[ref_str] = text
            per_paper_data[ref_str] = {
                "quote": combined_quote,
                "inline_citations": {},
            }
            quotes_metadata[ref_str] = snippet_metadata

    return references, per_paper_data, quotes_metadata


def build_prompt(query: str, section_references: Dict[str, str]) -> str:
    """Build the prompt in the format expected by the lite generation model."""
    refs_json = json.dumps(section_references, indent=2)
    return UNIFIED_GENERATION_PROMPT.format(query=query, refs_json=refs_json)


def build_title_prompt(query: str, section_titles: List[str]) -> str:
    """Build the prompt for title generation from query and section titles."""
    titles_str = "\n".join(f"- {title}" for title in section_titles)
    return TITLE_GENERATION_PROMPT.format(query=query, section_titles=titles_str)
