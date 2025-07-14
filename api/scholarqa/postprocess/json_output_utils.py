import re
from typing import Optional, Dict, Any, List
from scholarqa.utils import make_int
from langsmith import traceable
import logging
from anyascii import anyascii

logger = logging.getLogger(__name__)


def find_tldr_super_token(text: str) -> Optional[str]:
    # First, find the first instance of any token that has text "tldr" or "TLDR" in it, considering word boundaries
    tldr_token = re.search(r"\b\w*tldr\w*\b", text, re.IGNORECASE)

    if tldr_token:
        tldr_token = tldr_token.group(0)
        # Now find the word or token that the tldr_token is a subtoken in. This includes punctuation and markdown symbols
        tldr_super_token_pattern = re.compile(rf"[^\s]*{re.escape(tldr_token)}[^\s]*", re.IGNORECASE)
        tldr_super_token = re.search(tldr_super_token_pattern, text)

        if tldr_super_token:
            return tldr_super_token.group(0)
        else:
            return None
    else:
        return None


def get_section_text(gen_text: str) -> Dict[str, Any]:
    # Assume each section starts with 'Section X' followed by 'TLDR:'
    # find the first instance of any token surrounded by spaces or newlines that has text "tldr" or "TLDR" in it
    tldr_token = find_tldr_super_token(gen_text)
    curr_section = dict()
    if tldr_token is not None:
        parts = gen_text.split(tldr_token)
    else:
        parts = [gen_text]
    try:
        if len(parts) > 1:
            title = parts[0].strip()
            title = re.sub(r"\s*\(list\)", "", title)
            title = re.sub(r"\s*\(synthesis\)", "", title)
            curr_section["title"] = title.strip('#').strip()
            if tldr_token is not None:
                text_parts = parts[1].strip().split("\n", 1)
                tldr = text_parts[0]  # Assume TLDR is a single line
                text = text_parts[1]
                curr_section["tldr"] = tldr.strip('#').strip()
            else:
                text = parts[1].strip()
            curr_section["text"] = text
        else:
            raise Exception("Invalid content generated for the query by the LLM")
    except Exception as e:
        logger.exception(f"Error while parsing llm gen text: {gen_text} - {e}")
        raise e

    return curr_section


def resolve_ref_id(ref_str, ref_corpus_id, citation_ids):
    # in case of multiple papers from same author in the same year, add a count suffix
    if ref_str not in citation_ids:
        citation_ids[ref_str] = dict()
    if ref_corpus_id not in citation_ids[ref_str]:
        if citation_ids[ref_str]:
            rfsplits = ref_str.split(",")
            # in case of 2 (Doe et al., 2024), the one found later becomes (Doe et al._1, 2024) and so on...
            if len(rfsplits) > 1:
                ref_str_id = f"{rfsplits[0]}_{len(citation_ids[ref_str])},{rfsplits[1]}"
            else:
                ref_str_id = f"{ref_str}_{len(citation_ids[ref_str])}"
        else:
            ref_str_id = ref_str
        citation_ids[ref_str][ref_corpus_id] = ref_str_id
    else:
        ref_str_id = citation_ids[ref_str][ref_corpus_id]
    return ref_str_id


def pop_ref_data(ref_str_id, ref_corpus_id, fixed_quote, curr_paper_metadata) -> Dict[str, Any]:
    curr_ref = dict()
    curr_ref["id"] = ref_str_id
    curr_ref["snippets"] = [fq.strip() for fq in fixed_quote.split("...")]
    curr_ref["paper"] = dict()
    curr_ref["paper"]["corpus_id"] = make_int(ref_corpus_id)
    if curr_paper_metadata:
        #Commenting out the open access check as we switch to s2 api for the open access logic

        # if not (curr_paper_metadata.get("isOpenAccess") and curr_paper_metadata.get("openAccessPdf")):
        #     if curr_paper_metadata.get("abstract"):
        #         curr_ref["snippets"] = [s for s in curr_ref["snippets"] if
        #                                 s[:100] not in curr_paper_metadata["abstract"]]
        #     if not curr_ref["snippets"]:
        #         curr_ref["snippets"] = ["Please click on the paper title to read the abstract on Semantic Scholar."]

        curr_ref["score"] = curr_paper_metadata.get("relevance_judgement", 0)
        curr_ref["paper"]["title"] = curr_paper_metadata["title"]
        curr_ref["paper"]["authors"] = curr_paper_metadata["authors"]
        curr_ref["paper"]["year"] = make_int(curr_paper_metadata.get("year", 0))
        curr_ref["paper"]["venue"] = curr_paper_metadata["venue"]
        curr_ref["paper"]["n_citations"] = curr_paper_metadata["citationCount"]
    return curr_ref


@traceable(name="Postprocessing: Converted LLM generated output to json summary")
def get_json_summary(llm_model: str, summary_sections: List[str], summary_quotes: Dict[str, Any],
                     paper_metadata: Dict[str, Any], citation_ids: Dict[str, Dict[int, str]],
                     inline_tags=False) -> List[Dict[str, Any]]:
    text_ref_format = '<Paper corpusId="{corpus_id}" paperTitle="{ref_str}" isShortName></Paper>'
    sections = []
    llm_name_parts = llm_model.split("/", maxsplit=1)
    llm_ref_format = f'<Model name="{llm_name_parts[0].capitalize()}" version="{llm_name_parts[1]}">'
    summary_quotes = {anyascii(k): v for k, v in summary_quotes.items()}
    inline_citation_quotes = {anyascii(k): v for incite in summary_quotes.values() for k, v in
                              incite["inline_citations"].items()}
    for sec in summary_sections:
        curr_section = get_section_text(sec)
        text = curr_section["text"]
        if curr_section:
            pattern = r"(?:; )?(\d+ \| [A-Za-z. ]+ \| \d+ \| Citations: \d+)"
            replacement = r"] [\1"
            text = re.sub(pattern, replacement, text)
            text = re.sub(r"\[\]", "", text)
            curr_section["text"] = text.replace("[LLM MEMORY | 2024]", llm_ref_format)
            refs_list = []
            # tool tips inserted via span tags
            references = re.findall(r"\[.*?\]", text)
            refs_done = set()

            for ref in references:
                ref = anyascii(ref)
                if ref in summary_quotes or ref in inline_citation_quotes:
                    ref_parts = ref[1:-1].split(" | ")
                    ref_corpus_id, ref_str = ref_parts[0], f"({ref_parts[1]}, {make_int(ref_parts[2])})".replace(
                        "NULL, ", "")
                    if ref_corpus_id not in refs_done:
                        if ref in summary_quotes:
                            fixed_quote = summary_quotes[ref]["quote"]
                        else:
                            # abstract for inline citation
                            fixed_quote = inline_citation_quotes[ref]
                        fixed_quote = fixed_quote.strip().replace("“", '"').replace("”", '"')
                        if fixed_quote.startswith("..."):
                            fixed_quote = fixed_quote[3:]
                        if fixed_quote.endswith("..."):
                            fixed_quote = fixed_quote[:-3]
                        # dict to save reference strings as there is a possibility of having multiple papers in the same year from the same author
                        refs_done.add(ref_corpus_id)
                        ref_str_id = resolve_ref_id(ref_str, ref_corpus_id, citation_ids)
                        ref_data = pop_ref_data(ref_str_id, ref_corpus_id, fixed_quote,
                                                paper_metadata.get(ref_corpus_id))
                        if inline_tags:
                            curr_section["text"] = curr_section["text"].replace(ref, text_ref_format.format(
                                corpus_id=ref_data["paper"]["corpus_id"], ref_str=ref_data["id"]))
                        else:
                            curr_section["text"] = curr_section["text"].replace(ref, ref_data["id"])
                        refs_list.append(ref_data)
                else:
                    curr_section["text"] = curr_section["text"].replace(ref, "")
                    logger.warning(f"Reference not found in the summary quotes: {ref}")
            curr_section["text"] = re.sub(r"[ ]+", " ", curr_section["text"])
            # curr_section["text"] = curr_section["text"].replace(") ; (", "]; [")
            curr_section["citations"] = refs_list
            # add number of unique citations to section tldr
            if curr_section["tldr"]:
                if refs_list:
                    curr_section["tldr"] += (
                        f" ({len(refs_list)} sources)" if len(refs_list) > 1 else " (1 source)")
                else:
                    curr_section["tldr"] += " (LLM Memory)"
            sections.append(curr_section)
    return sections
