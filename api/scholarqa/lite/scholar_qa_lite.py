import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
from anyascii import anyascii
from langsmith import traceable

from scholarqa.llms.litellm_helper import llm_completion, CostAwareLLMResult, TokenUsage, register_model
from scholarqa.models import GeneratedReportData
from scholarqa.postprocess.json_output_utils import get_json_summary
from scholarqa.scholar_qa import ScholarQA
from scholarqa.lite.prompt_utils import prepare_references_data, build_prompt, build_title_prompt
from scholarqa.lite.response_parser import (
    parse_sections,
    parse_title,
    filter_per_paper_summaries,
    CITATION_PATTERN,
)

logger = logging.getLogger(__name__)


@traceable(name="Generation: Generate report title from query and sections")
def _generate_title(query: str, section_titles: List[str], model: str, llm_kwargs: dict,
                    fallback: str = None) -> str:
    """Generate a report title from query and section titles. Returns empty string on failure."""
    if not section_titles:
        return ""
    try:
        prompt = build_title_prompt(query, section_titles)
        result = llm_completion(user_prompt=prompt, model=model, fallback=fallback, **llm_kwargs)
        return parse_title(result.content)
    except Exception as e:
        logger.warning(f"Failed to generate report title: {e}")
        return ""


class ScholarQALite(ScholarQA):
    """
    ScholarQA using one-shot generation instead of quote extraction + clustering.

    Works with any litellm-compatible model. The model identifier and any extra
    completion kwargs (temperature, max_tokens, reasoning_effort, ...) are passed
    through ``lite_pipeline_args``. An optional ``fallback`` key (comma-separated
    model list, same convention as litellm_helper) is forwarded to
    ``llm_completion`` for both report and title generation.

    Mirrors the multi-step pipeline by extracting inline citations from the
    retrieved passages BEFORE the generation call so the model can cite the
    referenced papers (with their abstracts) directly alongside the quoted ones.
    """

    def __init__(self, *args, lite_pipeline_args: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not lite_pipeline_args or "model" not in lite_pipeline_args:
            raise ValueError("ScholarQALite requires 'model' in lite_pipeline_args")
        self.lite_pipeline_args = lite_pipeline_args

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:
        _, per_paper_data, all_quotes_metadata = prepare_references_data(reranked_df)
        logger.info(f"Built lite generation context with {len(per_paper_data)} references")

        # Step 2.1 (mirroring the multi-step pipeline): extract inline citations
        # so the LLM sees them in the prompt and can cite them directly.
        per_paper_data, all_quotes_metadata = self._extract_inline_citations(
            per_paper_data, all_quotes_metadata, reranked_df, paper_metadata
        )

        # Feed the LLM the same rich context the default pipeline uses
        # (title + venue + authors + abstract + markdown snippets), preserving
        # the inline_citations harvested from per_paper_data. per_paper_data and
        # all_quotes_metadata stay untouched for downstream postprocessing.
        rich_context = dict(zip(
            reranked_df["reference_string"],
            reranked_df["relevance_judgment_input_expanded"],
        ))
        prompt_view = {
            ref_str: {
                "quote": rich_context.get(ref_str, data["quote"]),
                "inline_citations": data.get("inline_citations", {}),
            }
            for ref_str, data in per_paper_data.items()
        }
        prompt = build_prompt(query, prompt_view)

        llm_kwargs = {**self.lite_pipeline_args}
        register_model(llm_kwargs)
        model = llm_kwargs.pop("model")
        fallback = llm_kwargs.pop("fallback", None)

        completion_result = llm_completion(user_prompt=prompt, model=model, fallback=fallback, **llm_kwargs)
        response = completion_result.content

        section_texts, section_titles = parse_sections(response)
        logger.info(f"Parsed {len(section_texts)} sections from response")

        self.report_title = _generate_title(query, section_titles, model, llm_kwargs, fallback=fallback)

        # Rewrite malformed / prose-only citations in the section texts. We deliberately
        # discard the filter's pruned dicts: any ref the LLM cited indirectly via another
        # ref's inline_citations would otherwise be dropped here. get_json_summary already
        # ignores unreferenced entries in per_paper_data, so passing the full map through
        # is both safe and required for inline-citation lookup.
        section_texts, _, _ = filter_per_paper_summaries(
            section_texts, per_paper_data, all_quotes_metadata
        )

        citation_ids: Dict[str, Any] = {}
        json_summary = get_json_summary(
            model, section_texts, per_paper_data,
            paper_metadata, citation_ids, inline_tags
        )
        generated_sections = [self.get_gen_sections_from_json(s) for s in json_summary]

        cost_result = CostAwareLLMResult(
            result=section_texts,
            tot_cost=completion_result.cost,
            models=[model] * len(section_texts),
            tokens=TokenUsage(
                input=completion_result.input_tokens,
                output=completion_result.output_tokens,
                total=completion_result.total_tokens,
                reasoning=completion_result.reasoning_tokens,
            )
        )

        return GeneratedReportData(
            report_title=self.report_title,
            sections=generated_sections,
            json_summary=json_summary,
            cost_result=cost_result,
            tcosts=[],
            quotes_metadata=all_quotes_metadata
        )

    def _extract_inline_citations(
        self,
        per_paper_data: Dict[str, Dict[str, Any]],
        all_quotes_metadata: Dict[str, List[Dict[str, Any]]],
        reranked_df: pd.DataFrame,
        paper_metadata: Dict[str, Any],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Populate ``inline_citations`` for every retrieved reference by delegating to
        the parent's ``extract_quote_citations``. Called before the LLM completion so
        the enriched dict can be serialized into the prompt.

        Falls back to the original (empty inline_citations) data on failure so a flaky
        S2 fetch never breaks generation.
        """
        if not per_paper_data:
            return per_paper_data, all_quotes_metadata

        try:
            per_paper_quotes = {k: v["quote"] for k, v in per_paper_data.items()}
            plan_json = {"all_refs": list(range(len(per_paper_quotes)))}
            extd, quotes_metadata = self.extract_quote_citations(
                reranked_df, per_paper_quotes, plan_json, paper_metadata
            )
            return extd, quotes_metadata
        except Exception as e:
            logger.warning(
                f"Failed to extract inline citations, continuing with references only: {e}"
            )
            return per_paper_data, all_quotes_metadata

    @staticmethod
    def build_plan_json(
        section_texts: List[str],
        section_titles: List[str],
        ref_keys: List[str],
    ) -> Dict[str, List[int]]:
        """
        Map each section title to the indices of ``ref_keys`` it cites, matching on the
        ``[corpus_id | Author | year | Citations: N]`` bracket pattern. Any ref that
        survived filtering but isn't bracket-cited (e.g. introduced via prose-author
        injection) gets bucketed under ``__unassigned__`` so it still flows through
        ``passage_to_quotes_metadata``.
        """
        ref_str_to_idx = {k: i for i, k in enumerate(ref_keys)}

        plan_json: Dict[str, List[int]] = {}
        for title, text in zip(section_titles, section_texts):
            indices = []
            for corpus_id, author_str, year, citation_count in CITATION_PATTERN.findall(text):
                citation_key = anyascii(
                    f"[{corpus_id} | {author_str.strip()} | {year} | Citations: {citation_count}]"
                )
                idx = ref_str_to_idx.get(citation_key)
                if idx is not None:
                    indices.append(idx)
            if indices:
                plan_json[title] = sorted(set(indices))

        covered = {i for idxs in plan_json.values() for i in idxs}
        missing = [i for i in range(len(ref_keys)) if i not in covered]
        if missing:
            plan_json["__unassigned__"] = missing

        return plan_json
