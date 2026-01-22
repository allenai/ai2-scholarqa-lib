import logging

from scholarqa.models import GeneratedReportData
from scholarqa.scholar_qa import ScholarQA
from scholarqa.unified.prompt_utils import format_df_as_references, build_prompt

logger = logging.getLogger(__name__)


class UnifiedScholarQA(ScholarQA):
    """
    ScholarQA using one-shot generation instead of quote extraction + clustering.
    """

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:
        """Override to use one-shot generation."""
        # Build the unified generation prompt
        section_references = format_df_as_references(reranked_df, paper_metadata)
        prompt = build_prompt(query, section_references)

        logger.info(f"Built unified generation prompt with {len(section_references)} references")
        logger.info(f"Unified generation prompt:\n{prompt}")

        # TODO: Replace with actual unified generation model call
        # For now, fall back to the parent's multi-step generation
        return super().generate_report(
            query, reranked_df, paper_metadata, cost_args,
            event_trace, user_id, inline_tags
        )
