import logging
import os

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
from scholarqa.unified.response_generator import ResponseGenerator, ModalResponseGenerator

logger = logging.getLogger(__name__)


class UnifiedScholarQA(ScholarQA):
    """
    ScholarQA using one-shot generation instead of quote extraction + clustering.
    """

    def __init__(self, *args, response_generator: ResponseGenerator = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_generator = response_generator

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:
        # Setup prompt
        section_references = format_df_as_references(reranked_df, paper_metadata)
        prompt = build_prompt(query, section_references)
        logger.info(f"Built unified generation prompt with {len(section_references)} references")

        model = self.multi_step_pipeline.llm_model or "allenai/sqa_basicsftdpo"
        completion_result = None
        response = ""

        # Check if we should use Modal or standard LiteLLM
        if self.response_generator:
            logger.info(f"Invoking custom ResponseGenerator (Modal)")
            try:
                response = self.response_generator.generate(prompt)
            except Exception as e:
                logger.error(f"Custom generator failed: {e}. Falling back to default.")
        # If custom generator wasn't used or failed, use standard path
        if not response:
            logger.info(f"Calling standard llm_completion model: {model}")
            completion_result = llm_completion(
                user_prompt=prompt,
                model=model,
                **self.llm_kwargs
            )
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

        # Cost Tracking
        if completion_result:
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
        else:
            # Fallback for custom generator where we might not have token data
            cost_result = CostAwareLLMResult(
                result=section_texts,
                tot_cost=0.0,
                models=[model] * len(section_texts),
                tokens=TokenUsage(input=0, output=0, total=0)
            )

        return GeneratedReportData(
            report_title=self.report_title,
            sections=generated_sections,
            json_summary=json_summary,
            cost_result=cost_result,
            tcosts=[],
            quotes_metadata=quotes_metadata
        )

# --- THE SCRIPT ---
if __name__ == "__main__":
    """
    This script demonstrates how to invoke the model using the correct configuration.
    """
    logging.basicConfig(level=logging.INFO)

    api_key = os.environ.get("MODAL_PLAYGROUND_API_KEY")

    if not api_key:
        print("❌ ERROR: MODAL_PLAYGROUND_API_KEY environment variable not found.")
        print("Please run: export MODAL_PLAYGROUND_API_KEY='your-actual-key-here' or save it to your .env")
    else:
        modal_gen = ModalResponseGenerator(
            endpoint="https://ai2-reviz--allenai-sqa-basicsftdpo-serve.modal.run",
            api_key=api_key,
            model="allenai/sqa_basicsftdpo"
        )

        # Simple Connectivity Test
        test_prompt = "Explain the importance of peer review in 2 sentences."

        print(f"\n🚀 Testing Modal connection...")
        print(f"URL: {modal_gen.endpoint}")
        print(f"Prompt: '{test_prompt}'\n")

        try:
            # This directly invokes the model using the credentials
            response = modal_gen.generate(test_prompt)

            print("✅ CONNECTION SUCCESSFUL!")
            print("--- Model Response ---")
            print(response)
            print("----------------------\n")

        except Exception as e:
            print(f"❌ CONNECTION FAILED!")
            print(f"Error details: {e}")

    # 4. Ready for full integration
    print("To use this in the full pipeline, pass 'modal_gen' to UnifiedScholarQA.")