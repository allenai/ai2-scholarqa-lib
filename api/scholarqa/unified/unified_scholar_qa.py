import logging
import os

from scholarqa.llms.litellm_helper import llm_completion
from scholarqa.models import GeneratedReportData
from scholarqa.scholar_qa import ScholarQA
from scholarqa.unified.prompt_utils import format_df_as_references, build_prompt
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
        """Override to use one-shot generation."""
        # Build the unified generation prompt
        section_references = format_df_as_references(reranked_df, paper_metadata)
        prompt = build_prompt(query, section_references)

        logger.info(f"Built unified generation prompt with {len(section_references)} references")

        # Check if we should use Modal or standard LiteLLM
        if self.response_generator:
            logger.info(f"Invoking custom ResponseGenerator (Modal)")
            try:
                response = self.response_generator.generate(prompt)
            except Exception as e:
                logger.error(f"Custom generator failed: {e}. Falling back to default.")
                # Fallback to standard litellm if generator fails
                completion_result = llm_completion(
                    user_prompt=prompt,
                    model=self.multi_step_pipeline.llm_model,
                    **self.llm_kwargs
                )
                response = completion_result.content
        else:
            # Standard path
            model = self.multi_step_pipeline.llm_model
            logger.info(f"Calling standard llm_completion model: {model}")
            completion_result = llm_completion(
                user_prompt=prompt,
                model=model,
                **self.llm_kwargs
            )
            response = completion_result.content

        logger.info(f"Unified generation response:\n{response}")

        # TODO: Parse response into sections and return proper GeneratedReportData?
        # For now, fall back to the parent's multi-step generation
        return super().generate_report(
            query, reranked_df, paper_metadata, cost_args,
            event_trace, user_id, inline_tags
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