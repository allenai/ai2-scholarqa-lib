from abc import ABC, abstractmethod
from typing import List, Any, Dict
import logging

from scholarqa.utils import query_s2_api, METADATA_FIELDS, make_int, NUMERIC_META_FIELDS

logger = logging.getLogger(__name__)


class AbstractRetriever(ABC):
    @abstractmethod
    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        pass


class FullTextRetriever(AbstractRetriever):
    def __init__(self, n_retrieval: int = 256, n_keyword_srch: int = 20):
        self.n_retrieval = n_retrieval
        self.n_keyword_srch = n_keyword_srch

    def retrieve_passages(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Query the Semantic Scholar API full text search end point to retrieve papers based on the query.
        The full text search end point does not return the required metadata fields, so we will fetch metadata for
        these later."""
        snippets_list = self.snippet_search(query, **filter_kwargs)
        snippets_list = [
            snippet for snippet in snippets_list if len(snippet["text"].split(" ")) > 20
        ]
        return snippets_list

    def snippet_search(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        if not self.n_retrieval:
            return []
        query_params = {fkey: fval for fkey, fval in filter_kwargs.items() if fval}
        query_params.update({"query": query, "limit": self.n_retrieval})
        print(query_params)
        snippets = query_s2_api(
            end_pt="snippet/search",
            params=query_params,
            method="get",
        )
        snippets_list = []
        res_data = snippets["data"]
        if res_data:
            for fields in res_data:
                res_map = dict()
                snippet, paper = fields["snippet"], fields["paper"]
                res_map["corpus_id"] = paper["corpusId"]
                res_map["title"] = paper["title"]
                res_map["text"] = snippet["text"]
                res_map["score"] = fields["score"]
                res_map["section_title"] = snippet["snippetKind"]
                if snippet["snippetKind"] == "body":
                    if section := snippet.get("section"):
                        res_map["section_title"] = section
                if "snippetOffset" in snippet and snippet["snippetOffset"].get("start"):
                    res_map["char_start_offset"] = snippet["snippetOffset"]["start"]
                else:
                    res_map["char_start_offset"] = 0
                if "annotations" in snippet and "sentences" in snippet["annotations"] and snippet["annotations"][
                    "sentences"]:
                    res_map["sentence_offsets"] = snippet["annotations"]["sentences"]
                else:
                    res_map["sentence_offsets"] = []

                if snippet.get("annotations") and snippet["annotations"].get("refMentions"):
                    res_map["ref_mentions"] = [rmen for rmen in
                                               snippet["annotations"]["refMentions"] if rmen.get("matchedPaperCorpusId")
                                               and rmen.get("start") and rmen.get("end")]
                else:
                    res_map["ref_mentions"] = []
                res_map["pdf_hash"] = snippet.get("extractionPdfHash","")
                res_map["stype"] = "vespa"
                if res_map:
                    snippets_list.append(res_map)
        return snippets_list

    def retrieve_additional_papers(self, query: str, **filter_kwargs) -> List[Dict[str, Any]]:
        return self.keyword_search(query, **filter_kwargs) if self.n_keyword_srch else []

    def keyword_search(self, kquery: str, **filter_kwargs) -> List[Dict[str, Any]]:
        """Query the Semantic Scholar API keyword search end point and return top n papers.
        The keyword search api also accepts filters for fields like year, venue, etc. which we obtain after decomposing
        the initial user query. This end point returns the required metadata fields as well, so we will skip fetching
        metadata for these later."""

        paper_data = []
        query_params = {fkey: fval for fkey, fval in filter_kwargs.items() if fval}
        query_params.update({"query": kquery, "limit": self.n_keyword_srch, "fields": METADATA_FIELDS})
        res = query_s2_api(
            end_pt="paper/search",
            params=query_params,
            method="get",
        )
        if "data" in res:
            paper_data = res["data"]
            paper_data = [pd for pd in paper_data if pd.get("corpusId") and pd.get("title") and pd.get("abstract")]
            paper_data = [{k: make_int(v) if k in NUMERIC_META_FIELDS else pd.get(k) for k, v in pd.items()}
                          for pd in paper_data]
            for pd in paper_data:
                pd["corpus_id"] = str(pd["corpusId"])
                pd["text"] = pd["abstract"]
                pd["section_title"] = "abstract"
                pd["char_start_offset"] = 0
                pd["sentence_offsets"] = []
                pd["ref_mentions"] = []
                pd["score"] = 0.0
                pd["stype"] = "public_api"
                pd["pdf_hash"] = ""
                pd["s2FieldsOfStudy"] = [f["category"] for f in pd["s2FieldsOfStudy"] if
                                                    f["source"] == "s2-fos-model"]
        return paper_data
