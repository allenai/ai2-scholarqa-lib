import logging

from scholarqa.llms.litellm_helper import llm_completion, CostAwareLLMResult, TokenUsage
from scholarqa.models import GeneratedReportData
from scholarqa.postprocess.json_output_utils import get_json_summary
from scholarqa.scholar_qa import ScholarQA
from scholarqa.unified.prompt_utils import format_df_as_references, build_prompt
from scholarqa.unified.response_parser import (
    parse_report_title,
    parse_sections,
    build_per_paper_summaries,
)

logger = logging.getLogger(__name__)


class UnifiedScholarQA(ScholarQA):
    """
    ScholarQA using one-shot generation instead of quote extraction + clustering.
    """

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:
        section_references = format_df_as_references(reranked_df, paper_metadata)
        prompt = build_prompt(query, section_references)
        logger.info(f"Built unified generation prompt with {len(section_references)} references")

        model = self.multi_step_pipeline.llm_model
        completion_result = llm_completion(user_prompt=prompt, model=model, **self.llm_kwargs)
        response = completion_result.content

        self.report_title = parse_report_title(response)
        section_texts = parse_sections(response)
        logger.info(f"Parsed {len(section_texts)} sections from response")

        per_paper_summaries_extd, quotes_metadata = build_per_paper_summaries(section_texts, reranked_df)

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
