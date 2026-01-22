from scholarqa.models import GeneratedReportData
from scholarqa.scholar_qa import ScholarQA


class UnifiedScholarQA(ScholarQA):
    """
    ScholarQA using one-shot generation instead of quote extraction + clustering.
    """

    def generate_report(self, query, reranked_df, paper_metadata, cost_args,
                        event_trace, user_id, inline_tags=False) -> GeneratedReportData:
        """Override to use one-shot generation."""
        raise NotImplementedError(
            "UnifiedScholarQA.generate_report is not yet implemented."
        )
