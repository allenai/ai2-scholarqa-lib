from datetime import datetime
from typing import List, Dict, Any

from scholarqa.config.config_setup import LogsConfig
from scholarqa.llms.constants import CostAwareLLMResult
from scholarqa.models import ToolRequest
from scholarqa.trace.trace_writer import GCSWriter, LocalWriter


class EventTrace:
    def __init__(self, task_id: str, n_retrieval: int, n_rerank: int, req: ToolRequest, user_id: str = None):
        self.query = req.query
        if not user_id:
            try:
                self.user_id = req.user_id
            except Exception as e:
                self.user_id = None
        else:
            self.user_id = user_id
        self.task_id = task_id
        self.timestamp = datetime.now().isoformat()
        self.n_retrieval = n_retrieval
        self.n_retrieved = 0
        self.n_candidates = 0
        self.n_rerank = n_rerank
        self.opt_in = req.opt_in
        self.total_cost = 0.0

        self.decomposed_query = dict()
        self.candidates = []
        self.retrieved = []
        self.quotes = {"cost": 0.0, "quotes": []}
        self.cluster = dict()
        self.summary = dict()
        self.total_cost = 0.0
        self.tokens = {"input": 0, "output": 0, "total": 0, "reasoning": 0}

    def trace_decomposition_event(self, decomposed_query: CostAwareLLMResult):
        self.decomposed_query = decomposed_query.result._asdict()
        self.decomposed_query["cost"] = decomposed_query.tot_cost
        self.decomposed_query["model"] = decomposed_query.models[0]
        self.decomposed_query["tokens"] = decomposed_query.tokens._asdict()
        self.total_cost += decomposed_query.tot_cost
        for k,v in self.tokens.items():
            self.tokens[k] += self.decomposed_query["tokens"][k]

    def trace_retrieval_event(self, retrieved: List[Dict[str, Any]]):
        self.n_retrieved = len(retrieved)
        self.retrieved = retrieved

    def trace_rerank_event(self, candidates: List[Dict[str, Any]]):
        self.n_candidates = len(candidates)
        self.candidates = candidates

    def trace_quote_event(self, paper_summaries: CostAwareLLMResult):
        topk = [{"idx": i, "key": k, "snippets": v} for
                i, (k, v) in enumerate(paper_summaries.result.items())]
        topk_models = paper_summaries.models
        for idx, tk in enumerate(topk):
            tk["model"] = topk_models[idx]

        self.quotes["cost"] = paper_summaries.tot_cost
        self.quotes["tokens"] = paper_summaries.tokens._asdict()
        self.quotes["quotes"] = topk
        self.total_cost += paper_summaries.tot_cost
        for k, v in self.tokens.items():
            self.tokens[k] += self.quotes["tokens"][k]

    def trace_clustering_event(self, cluster_json: CostAwareLLMResult, plan_str: Dict[str, Any]):
        self.cluster["cost"] = cluster_json.tot_cost
        self.cluster["tokens"] = cluster_json.tokens._asdict()
        self.cluster["cot"] = cluster_json.result["cot"]
        self.cluster["plan"] = plan_str
        self.cluster["model"] = cluster_json.models[0]
        self.total_cost += cluster_json.tot_cost
        for k, v in self.tokens.items():
            self.tokens[k] += self.cluster["tokens"][k]

    def trace_inline_citation_following_event(self, paper_summaries_extd: Dict[str, Any], quotes_metadata: Dict[str, List[Dict[str, Any]]]):
        for quote_obj in self.quotes["quotes"]:
            quote_obj["snippets"] = paper_summaries_extd[quote_obj["key"]].get("quote", quote_obj["snippets"])
            quote_obj["inline_citations"] = paper_summaries_extd[quote_obj["key"]].get("inline_citations", dict())
            quote_obj["metadata"] = quotes_metadata.get(quote_obj["key"], [])

    def trace_summary_event(self, json_summary: List[Dict[str, Any]], cost_result: CostAwareLLMResult, tab_costs: List[Dict] = None):
        self.summary = {"sections": json_summary, "cost": cost_result.tot_cost, "tokens": cost_result.tokens._asdict(), "table_costs": tab_costs}
        for idx, section in enumerate(self.summary["sections"]):
            section["model"] = cost_result.models[idx]
        self.total_cost += cost_result.tot_cost
        for k, v in self.tokens.items():
            self.tokens[k] += self.summary["tokens"][k]

        if tab_costs:
            for tcost in tab_costs:
                column_cost = tcost.get("column_cost", 0.0)
                if column_cost:
                    self.total_cost += column_cost["cost_value"]
                    self.tokens["input"] += column_cost["tokens"]["prompt"]
                    self.tokens["output"] += column_cost["tokens"]["completion"]
                    self.tokens["total"] += column_cost["tokens"]["total"]
                    self.tokens["reasoning"] += column_cost["tokens"].get("reasoning", 0)

                cell_cost = tcost.get("cell_cost", 0.0)
                if cell_cost:
                    for ccost in cell_cost:
                        if not isinstance(ccost, dict):
                            continue
                        for v in ccost.values():
                            self.total_cost += v["cost_value"]
                            self.tokens["input"] += v["tokens"]["prompt"]
                            self.tokens["output"] += v["tokens"]["completion"]
                            self.tokens["total"] += v["tokens"]["total"]
                            self.tokens["reasoning"] += v["tokens"].get("reasoning", 0)


    def persist_trace(self, logs_config: LogsConfig):
        trace_writer = GCSWriter(bucket_name=logs_config.event_trace_loc) if logs_config.tracing_mode == "gcs" \
            else LocalWriter(local_dir=f"{logs_config.log_dir}/{logs_config.event_trace_loc}")
        trace_writer.write(trace_json=self, file_name=self.task_id)
