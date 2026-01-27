import logging
import os
from typing import Any, Dict

from scholarqa.llms.litellm_helper import llm_completion, CostAwareLLMResult, TokenUsage, register_model
from scholarqa.models import GeneratedReportData
from scholarqa.postprocess.json_output_utils import get_json_summary
from scholarqa.scholar_qa import ScholarQA
from scholarqa.lite.prompt_utils import prepare_references_data, build_prompt
from scholarqa.lite.response_parser import (
    parse_report_title,
    parse_sections,
    build_per_paper_summaries,
)

logger = logging.getLogger(__name__)


class ScholarQALite(ScholarQA):
    """
    ScholarQA using one-shot generation instead of quote extraction + clustering.
    """

    def __init__(self, *args, report_generation_args: Dict[str, Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not report_generation_args or "model" not in report_generation_args:
            raise ValueError(
                "ScholarQALite requires 'report_generation_args' with at least a 'model' key."
            )
        self.report_generation_args = report_generation_args

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:
        section_references, per_paper_data, all_quotes_metadata = prepare_references_data(reranked_df)
        prompt = build_prompt(query, section_references)
        logger.info(f"Built lite generation prompt with {len(section_references)} references")

        model = self.report_generation_args["model"]
        llm_kwargs = {k: v for k, v in self.report_generation_args.items() if k != "model"}
        if "api_key" not in llm_kwargs:
            llm_kwargs["api_key"] = os.environ.get("REPORT_GENERATION_API_KEY")
        register_model(model, llm_kwargs)
        logger.info(f"Using model {model} for report generation")
        completion_result = llm_completion(user_prompt=prompt, model=model, fallback=None, **llm_kwargs)
        response = completion_result.content

        self.report_title = parse_report_title(response)
        section_texts = parse_sections(response)
        logger.info(f"Parsed {len(section_texts)} sections from response")

        per_paper_summaries_extd, quotes_metadata = build_per_paper_summaries(
            section_texts, per_paper_data, all_quotes_metadata
        )

        citation_ids = {}
        json_summary = get_json_summary(
            model, section_texts, per_paper_summaries_extd,
            paper_metadata, citation_ids, inline_tags
        )
        generated_sections = [self.get_gen_sections_from_json(s) for s in json_summary]

        cost_result = CostAwareLLMResult(
            result=section_texts,
            tot_cost=completion_result.cost,
            # trace_summary_event expects one model per section, so repeat for our single call
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
            quotes_metadata=quotes_metadata
        )
