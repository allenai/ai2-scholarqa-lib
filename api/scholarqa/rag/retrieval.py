import logging
from abc import abstractmethod
from typing import List, Dict, Any, Optional

import pandas as pd

from scholarqa.rag.reranker.reranker_base import AbstractReranker
from scholarqa.rag.retriever_base import AbstractRetriever
from scholarqa.utils import make_int, get_ref_author_str
from anyascii import anyascii

logger = logging.getLogger(__name__)


class AbsPaperFinder(AbstractRetriever):

    @abstractmethod
    def rerank(self, query: str, retrieved_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass


class PaperFinder(AbsPaperFinder):
    snippet_srch_fields = ["text", "snippetKind", "snippetOffset", "section", "annotations.refMentions",
                           "annotations.sentences.start","annotations.sentences.end"]
    def __init__(self, retriever: AbstractRetriever, context_threshold: float = 0.0, n_rerank: int = -1, max_date: Optional[str] = None):
        self.retriever = retriever
        self.context_threshold = context_threshold
        self.n_rerank = n_rerank
        self.max_date = max_date

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant passages along with scores from an index for the given query"""
        filter_kwargs.update({
            "fields": ",".join([f"snippet.{f}" for f in self.snippet_srch_fields])
        })
        if self.max_date:
            filter_kwargs.update({"insertedBefore": '{}'.format(self.max_date)})
        return self.retriever.retrieve_passages(query, **filter_kwargs)

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        if self.max_date:
            # S2 API doesnt have insertedBefore for this endpoint, so use publicationDateOrYear:
            filter_kwargs.update({"publicationDateOrYear": ':{}'.format(self.max_date)})
        return self.retriever.retrieve_additional_papers(query, **filter_kwargs)

    def rerank(self, query: str, retrieved_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return retrieved_ctxs

    def aggregate_into_dataframe(self, snippets_list: List[Dict[str, Any]], paper_metadata: Dict[str, Any]) -> \
            pd.DataFrame:
        """The reranked snippets is passage level. This function aggregates the passages to the paper level,
        The Dataframe also consists of aggregated passages stitched together with the paper title and abstract in the markdown format."""
        snippets_list = [snippet for snippet in snippets_list if snippet["corpus_id"] in paper_metadata
                         and snippet["text"] is not None]
        aggregated_candidates = self.aggregate_snippets_to_papers(snippets_list, paper_metadata)
        aggregated_candidates = [acand for acand in aggregated_candidates if
                                 acand["relevance_judgement"] >= self.context_threshold]

        return self.format_retrieval_response(aggregated_candidates)

    @staticmethod
    def aggregate_snippets_to_papers(snippets_list: List[Dict[str, Any]], paper_metadata: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        logging.info(f"Aggregating {len(snippets_list)} passages at paper level with metadata")
        paper_snippets = dict()
        for snippet in snippets_list:
            corpus_id = snippet["corpus_id"]
            if corpus_id not in paper_snippets:
                paper_snippets[corpus_id] = paper_metadata[corpus_id]
                paper_snippets[corpus_id]["corpus_id"] = corpus_id
                paper_snippets[corpus_id]["sentences"] = []
            if snippet["stype"] != "public_api":
                paper_snippets[corpus_id]["sentences"].append(snippet)
            if "paperId" in paper_snippets[corpus_id]:
                del paper_snippets[corpus_id]["paperId"]
            paper_snippets[corpus_id]["relevance_judgement"] = max(
                paper_snippets[corpus_id].get("relevance_judgement", -1),
                snippet.get("rerank_score", snippet["score"]))
            if not paper_snippets[corpus_id]["abstract"] and snippet["section_title"] == "abstract":
                paper_snippets[corpus_id]["abstract"] = snippet["text"]
        sorted_ctxs = sorted(paper_snippets.values(), key=lambda x: x["relevance_judgement"], reverse=True)
        logger.info(f"Scores after aggregation: {[s['relevance_judgement'] for s in sorted_ctxs]}")
        return sorted_ctxs

    def format_retrieval_response(self, agg_reranked_candidates: List[Dict[str, Any]]) -> pd.DataFrame:
        def format_sections_to_markdown(row: List[Dict[str, Any]]) -> str:
            # convenience function to format the sections of a paper into markdown for function below
            # Convert the list of dictionaries to a DataFrame
            sentences_df = pd.DataFrame(row)
            if sentences_df.empty:
                return ""
            # Sort by 'char_offset' to ensure sentences are in the correct order
            sentences_df.sort_values(by="char_start_offset", inplace=True)

            # Group by 'section_title', concatenate sentences, and maintain overall order by the first 'char_offset'
            grouped = sentences_df.groupby("section_title", sort=False)["text"].apply("\n...\n".join)

            # Exclude sections titled 'Abstract' or 'Title'
            grouped = grouped[(grouped.index != "abstract") & (grouped.index != "title")]

            # Format as Markdown
            markdown_output = "\n\n".join(f"## {title}\n{text}" for title, text in grouped.items())
            return markdown_output

        df = pd.DataFrame(agg_reranked_candidates)
        try:
            df = df.drop(["text", "section_title", "ref_mentions", "score", "stype", "rerank_score"], axis=1)
        except Exception as e:
            logger.info(e)
        df = df[~df.sentences.isna() & ~df.year.isna()] if not df.empty else df
        if df.empty:
            return df
        df["corpus_id"] = df["corpus_id"].astype(int)

        # there are multiple relevance judgments in ['relevance_judgements'] for each paper
        # we will keep rows where ANY of the relevance judgments are 2 or 3
        # df = df[df["relevance_judgement"] >= self.context_threshold]

        if df.empty:
            return df

        # authors are lists of jsons. process with "name" key inside

        df["year"] = df["year"].apply(make_int)

        df["authors"] = df["authors"].fillna(value="")

        df.rename(
            columns={
                "citationCount": "citation_count",
                "referenceCount": "reference_count",
                "influentialCitationCount": "influential_citation_count",
            },
            inplace=True,
        )

        # drop corpusId
        df = df.drop(columns=["corpusId"])

        # now we need the big relevance_judgment_input_expanded
        # top of it
        # \n## Abstract\n{row['abstract']} --> Not using abstracts OR could use and not show
        prepend_text = df.apply(
            lambda
                row: f"# Title: {row['title']}\n# Venue: {row['venue']}\n"
                     f"# Authors: {', '.join([a['name'] for a in row['authors']])}\n## Abstract\n{row['abstract']}\n",
            axis=1,
        )
        section_text = df["sentences"].apply(format_sections_to_markdown)
        # update relevance_judgment_input
        df.loc[:, "relevance_judgment_input_expanded"] = prepend_text + section_text
        df["reference_string"] = df.apply(
            lambda
                row: anyascii(f"[{make_int(row.corpus_id)} | {get_ref_author_str(row.authors)} | "
                              f"{make_int(row['year'])} | Citations: {make_int(row['citation_count'])}]"),
            axis=1,
        )
        return df


class PaperFinderWithReranker(PaperFinder):
    def __init__(self, retriever: AbstractRetriever, reranker: AbstractReranker, n_rerank: int = -1,
                 context_threshold: float = 0.5, max_date: Optional[str] = None):
        super().__init__(retriever, context_threshold, n_rerank, max_date=max_date)
        if reranker:
            self.reranker_engine = reranker
        else:
            raise Exception(f"Reranker not initialized: {reranker}")

    def rerank(
            self, query: str, retrieved_ctxs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank the retrieved passages using a cross-encoder model and return the top n passages."""
        passages = [doc["title"] + " " + doc["text"] if "title" in doc else doc["text"] for doc in retrieved_ctxs]
        rerank_scores = self.reranker_engine.get_scores(
            query, passages
        )
        logger.info(f"Reranker scores: {rerank_scores}")

        for doc, rerank_score in zip(retrieved_ctxs, rerank_scores):
            doc["rerank_score"] = rerank_score
        sorted_ctxs = sorted(
            retrieved_ctxs, key=lambda x: x["rerank_score"], reverse=True
        )
        sorted_ctxs = super().rerank(query, sorted_ctxs)
        sorted_ctxs = sorted_ctxs[:self.n_rerank] if self.n_rerank > 0 else sorted_ctxs
        logging.info(f"Done reranking: {len(sorted_ctxs)} passages remain")
        return sorted_ctxs
