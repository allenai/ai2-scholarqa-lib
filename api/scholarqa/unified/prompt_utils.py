"""
Prompt building and reference formatting for unified generation.
"""

import json
from typing import Any, Dict

import pandas as pd
from scholarqa.llms.prompts import UNIFIED_GENERATION_PROMPT
from scholarqa.utils import get_ref_author_str, make_int


def format_reference_string(paper_metadata: Dict[str, Any]) -> str:
    """Format paper metadata into reference string like [corpus_id | Author et al. | year | Citations: N]"""
    corpus_id = paper_metadata.get("corpusId", paper_metadata.get("corpus_id", ""))
    authors = paper_metadata.get("authors", [])
    year = paper_metadata.get("year", "")
    citations = paper_metadata.get(
        "citationCount", paper_metadata.get("citation_count", 0)
    )

    return f"[{corpus_id} | {get_ref_author_str(authors)} | {make_int(year)} | Citations: {make_int(citations)}]"


def format_df_as_references(
    scored_df: pd.DataFrame, paper_metadata: Dict[str, Any]
) -> Dict[str, str]:
    """Format the reranked DataFrame into section_references for the model prompt."""
    references = {}

    for _, row in scored_df.iterrows():
        # Example: corpus_id = "12345678"
        corpus_id = str(row["corpus_id"])

        # Look up full metadata; fall back to row data if not found
        # Example: meta = {"corpusId": 12345678, "authors": [...], "year": 2023, "citationCount": 150}
        meta = paper_metadata.get(corpus_id, row.to_dict())

        # Build the reference key the model will use for citations
        # Example: ref_str = "[12345678 | Smith and Doe | 2023 | Citations: 150]"
        ref_str = format_reference_string(meta)

        # Get retrieved sentences (snippets from full-text search)
        # Example: sentences = [{"text": "Transformers use attention.", ...}, {"text": "This improves...", ...}]
        sentences = row["sentences"]

        if sentences:
            # Concatenate all snippet texts into one passage
            # Example: text = "Transformers use attention. This improves performance."
            text = " ".join(sent["text"] for sent in sentences)
        else:
            # Fall back to abstract if no sentences (e.g., abstract-only retrieval)
            text = row.get("abstract", "")

        # Only add if we have text and haven't seen this paper yet
        # (same paper may appear multiple times in scored_df with different snippets)
        if text and ref_str not in references:
            references[ref_str] = text

    return references


def build_prompt(query: str, section_references: Dict[str, str]) -> str:
    """Build the prompt in the format expected by the unified generation model."""
    refs_json = json.dumps(section_references, indent=2)
    return UNIFIED_GENERATION_PROMPT.format(query=query, refs_json=refs_json)
