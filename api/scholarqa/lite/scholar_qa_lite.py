import logging
import os

from scholarqa.llms.litellm_helper import llm_completion, CostAwareLLMResult, TokenUsage
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
    ScholarQA using one-shot generation. Integrated with CostAwareLLMCaller.
    """
    def __init__(self, *args, modal_config: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.modal_config = modal_config

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:

        # Prepare data and prompt
        section_references, per_paper_data, all_quotes_metadata = prepare_references_data(reranked_df)
        prompt = build_prompt(query, section_references)

        if self.modal_config:
            llm_params = {
                "model": f"openai/{self.modal_config['model']}",
                "api_base": self.modal_config['endpoint'],
                "api_key": self.modal_config.get("api_key") or os.environ.get("MODAL_PLAYGROUND_API_KEY"),
                "temperature": 0.1,
                "max_tokens": 4096,
                "fallback": None
            }
            logger.info(f"Routing to Modal: {self.modal_config['endpoint']} with model {llm_params['model']}")
        else:
            # Standard Path
            llm_params = {
                "model": self.multi_step_pipeline.llm_model,
                **self.llm_kwargs
            }
        print(f"AHAH DEBUG: api_base is {llm_params['api_base']}")

        llm_result: CostAwareLLMResult = self.llm_caller.call_method(
            cost_args=cost_args,
            method=llm_completion,
            user_prompt=prompt,
            **llm_params
        )

        response = llm_result.result.content

        self.report_title = parse_report_title(response)
        section_texts = parse_sections(response)
        logger.info(f"Parsed {len(section_texts)} sections from response")

        per_paper_summaries_extd, quotes_metadata = build_per_paper_summaries(
            section_texts, per_paper_data, all_quotes_metadata
        )

        citation_ids = {}
        actual_model = llm_result.models[0]

        json_summary = get_json_summary(
            actual_model, section_texts, per_paper_summaries_extd,
            paper_metadata, citation_ids, inline_tags
        )
        generated_sections = [self.get_gen_sections_from_json(s) for s in json_summary]

        cost_result = CostAwareLLMResult(
            result=section_texts,
            tot_cost=llm_result.tot_cost,
            models=[actual_model] * len(section_texts),
            tokens=llm_result.tokens
        )

        return GeneratedReportData(
            report_title=self.report_title,
            sections=generated_sections,
            json_summary=json_summary,
            cost_result=cost_result,
            tcosts=[],
            quotes_metadata=quotes_metadata
        )