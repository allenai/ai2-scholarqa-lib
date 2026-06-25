"""
Evaluate 4 ScholarQA configurations on asta-bench/sqa rubrics_v2_recomputed.

Configurations:
  * lite_low    -- ScholarQALite, reasoning_effort=low
  * lite_medium -- ScholarQALite, reasoning_effort=medium
  * lite_high   -- ScholarQALite, reasoning_effort=high
  * default     -- ScholarQA multi-step flow, tables disabled,
                   step_select_quotes overridden to pass relevance_judgment_input_expanded through.

The shared paper_finder + litellm cache ensures the retrieval context fed into
generation is consistent across configurations.

Each config writes its results to a separate JSON file as a list of
``{"question": ..., "response": ...}`` records, persisted after every question
so the run is resumable.

Usage:
    python run_eval.py [--config NAME] [--limit N]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from scholarqa import (  # noqa: E402
    FullTextRetriever,
    PaperFinderWithReranker,
    ScholarQA,
    ScholarQALite,
)
from scholarqa.config.config_setup import LogsConfig  # noqa: E402
from scholarqa.llms.constants import (  # noqa: E402
    CostAwareLLMResult,
    CostReportingArgs,
    TokenUsage,
)
from scholarqa.llms.prompts import (  # noqa: E402
    PROMPT_ASSEMBLE_SUMMARY,
    SYSTEM_PROMPT_QUOTE_CLUSTER,
)
from scholarqa.models import ToolRequest  # noqa: E402
from scholarqa.rag.reranker.modal_engine import ModalReranker  # noqa: E402

logger = logging.getLogger("run_eval")

MODEL = "openai/gpt-5.4-2026-03-05"
RUBRICS_PATH = Path(__file__).resolve().parent / "asta_bench_rubrics.json"
RESULT_DIR = Path(__file__).resolve().parent / "eval_results"
DEFAULT_TOKEN_USAGE = TokenUsage(0, 0, 0, 0)


class _ToolRequestWithThread(ToolRequest):
    """ToolRequest base lacks thread_id; EventTrace expects it."""
    thread_id: str = ""


class PassthroughScholarQA(ScholarQA):
    """
    Default ScholarQA where Step 1 (LLM quote extraction) is short-circuited.

    Canonical ``per_paper_summaries`` is ``"...".join(sentences)`` per paper (or
    the paper's abstract if it has no retrieved sentences) — the same shape
    ScholarQALite uses, so ``passage_to_quotes_metadata`` matches cleanly,
    inline-citation extraction works, and ``get_json_summary`` builds
    ``citation.snippets`` correctly without any post-fix.

    Rich ``relevance_judgment_input_expanded`` is swapped into the Step 2
    (clustering) and Step 3 (iterative summary) LLM prompts via overrides on
    ``step_clustering`` and ``step_gen_iterative_summary``, mirroring the
    ``prompt_view`` pattern in ``ScholarQALite.generate_report``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rich_text_map: Dict[str, str] = {}

    def step_select_quotes(
        self,
        query: str,
        scored_df: pd.DataFrame,
        cost_args: CostReportingArgs = None,
        sys_prompt: str = None,
    ) -> CostAwareLLMResult:
        logger.info("Running Step 1 - quote extraction (passthrough)")
        start = time.time()
        canonical: Dict[str, str] = {}
        rich: Dict[str, str] = {}
        for _, row in scored_df.iterrows():
            ref_str = row["reference_string"]
            sentences = row.get("sentences")
            if isinstance(sentences, list) and sentences:
                snippet_texts = [s["text"] for s in sentences if s.get("text")]
                canonical[ref_str] = "...".join(snippet_texts) if snippet_texts else (row.get("abstract") or "")
            else:
                canonical[ref_str] = row.get("abstract") or ""
            rich[ref_str] = row.get("relevance_judgment_input_expanded") or canonical[ref_str]
        self._rich_text_map = rich
        per_paper_summaries = CostAwareLLMResult(
            canonical, 0.0, ["paper finder"] * len(canonical), DEFAULT_TOKEN_USAGE
        )
        logger.info(
            f"Step 1 done - {len(canonical)} papers with quotes extracted, "
            f"cost: 0.0, time: {time.time() - start:.2f}"
        )
        return per_paper_summaries

    def step_clustering(
        self,
        query: str,
        per_paper_summaries: Dict[str, str],
        cost_args: CostReportingArgs = None,
        sys_prompt: str = SYSTEM_PROMPT_QUOTE_CLUSTER,
    ):
        # Swap in rich context for the clustering prompt; canonical
        # per_paper_summaries (sentence-joined) is preserved for downstream
        # passage matching and citation building.
        rich = {
            ref: self._rich_text_map.get(ref, snippet)
            for ref, snippet in per_paper_summaries.items()
        }
        return super().step_clustering(query, rich, cost_args, sys_prompt)

    def step_gen_iterative_summary(
        self,
        query: str,
        per_paper_summaries: Dict[str, Dict[str, Any]],
        plan_json: Dict[str, Any],
        cost_args: CostReportingArgs = None,
        sys_prompt: str = PROMPT_ASSEMBLE_SUMMARY,
    ):
        # per_paper_summaries here is per_paper_summaries_extd (with inline_citations).
        # Swap "quote" to rich text for the LLM; keep inline_citations intact.
        rich = {
            ref: {**v, "quote": self._rich_text_map.get(ref, v.get("quote", ""))}
            for ref, v in per_paper_summaries.items()
        }
        return_val = yield from super().step_gen_iterative_summary(
            query, rich, plan_json, cost_args, sys_prompt
        )
        return return_val




def load_questions() -> List[str]:
    with open(RUBRICS_PATH) as f:
        rubrics = json.load(f)
    questions: List[str] = []
    seen = set()
    for entry in rubrics:
        q = entry.get("question")
        if q and q not in seen:
            questions.append(q)
            seen.add(q)
    return questions


def build_paper_finder() -> PaperFinderWithReranker:
    retriever = FullTextRetriever(n_retrieval=256, n_keyword_srch=20)
    reranker = ModalReranker(
        app_name="ai2-scholar-qa",
        api_name="inference_api",
        batch_size=256,
        gen_options={},
    )
    return PaperFinderWithReranker(retriever, reranker, n_rerank=50, context_threshold=0.0)


# Output budget large enough that reasoning_effort=high does not starve
# the final content. With reasoning_effort=high we observed >8k reasoning
# tokens for a single section; the multi-section reports need substantially
# more room for content on top of reasoning.
LITE_MAX_TOKENS = 32768
DEFAULT_MAX_TOKENS = 16384


def build_lite(paper_finder, logs_cfg, reasoning_effort: str) -> ScholarQALite:
    return ScholarQALite(
        paper_finder=paper_finder,
        run_table_generation=False,
        logs_config=logs_cfg,
        llm_model=MODEL,
        fallback_llm=None,
        lite_pipeline_args={
            "model": MODEL,
            "max_tokens": LITE_MAX_TOKENS,
            "reasoning_effort": reasoning_effort,
        },
    )


def build_default(paper_finder, logs_cfg) -> PassthroughScholarQA:
    return PassthroughScholarQA(
        paper_finder=paper_finder,
        run_table_generation=False,
        logs_config=logs_cfg,
        llm_model=MODEL,
        fallback_llm=None,
        llm_kwargs={"max_tokens": DEFAULT_MAX_TOKENS},
    )


def _coerce_response(task_result: Any, model: str) -> Dict[str, Any]:
    """Shape the TaskResult dict to match the format requested by the user:
    drops report_title, adds table_costs=[], and adds format/model per section."""
    data = task_result.model_dump() if hasattr(task_result, "model_dump") else dict(task_result)
    response: Dict[str, Any] = {
        "sections": [],
        "cost": data.get("cost", 0.0),
        "tokens": data.get("tokens"),
        "table_costs": [],
    }
    for sec in data.get("sections") or []:
        sec = dict(sec)
        sec.setdefault("format", "synthesis" if not sec.get("table") else "list")
        sec.setdefault("model", model)
        response["sections"].append(sec)
    return response


def run_one(sqa: ScholarQA, query: str) -> Dict[str, Any]:
    task_id = str(uuid4())
    sqa.logs_config.task_id = task_id
    tool_req = _ToolRequestWithThread(
        task_id=task_id, query=query, user_id="eval", thread_id=task_id
    )
    task_result = sqa.run_qa_pipeline(tool_req, inline_tags=True)
    return _coerce_response(task_result, model=MODEL)


def _classify_error(e: BaseException) -> str:
    msg = str(e).lower()
    if "context_length" in msg or "maximum context length" in msg or "too long" in msg or "token limit" in msg:
        return "context_length_exceeded"
    if "rate" in msg and "limit" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    return type(e).__name__


def run_config(
    config_name: str,
    sqa_factory: Callable[[], ScholarQA],
    questions: List[str],
    out_path: Path,
) -> None:
    results: List[Dict[str, Any]] = []
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        answered = {r["question"] for r in results}
        questions = [q for q in questions if q not in answered]
        logger.info(
            "[%s] resuming - %d done, %d remaining", config_name, len(answered), len(questions)
        )
    else:
        logger.info("[%s] starting fresh - %d questions", config_name, len(questions))

    if not questions:
        return

    sqa = sqa_factory()
    for i, q in enumerate(questions, 1):
        logger.info("[%s] %d/%d  %s", config_name, i, len(questions), q[:100])
        t0 = time.time()
        try:
            response = run_one(sqa, q)
        except Exception as e:
            kind = _classify_error(e)
            logger.exception("[%s] error (%s) on question: %s", config_name, kind, q[:80])
            response = {"error": repr(e), "error_kind": kind}
        elapsed = time.time() - t0
        results.append({"question": q, "response": response, "elapsed_sec": round(elapsed, 2)})
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if isinstance(response, dict):
            cost = response.get("cost")
            tokens = response.get("tokens") or {}
            input_tokens = tokens.get("input")
            warning = " [LARGE]" if isinstance(input_tokens, int) and input_tokens > 200_000 else ""
            logger.info(
                "[%s] saved (%.1fs, cost=%s, in=%s, out=%s%s) -> %s",
                config_name,
                elapsed,
                cost,
                input_tokens,
                tokens.get("output"),
                warning,
                out_path.name,
            )
        else:
            logger.info("[%s] saved (%.1fs) -> %s", config_name, elapsed, out_path.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["lite_low", "lite_medium", "lite_high", "default", "all"],
        default="all",
        help="Run a single configuration or all (default: all).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only run the first N questions (default: all)."
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(str(Path(__file__).resolve().parent / "run_eval.log")),
        ],
    )

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set in environment.")
        return 2

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    logger.info("Loaded %d questions", len(questions))

    paper_finder = build_paper_finder()
    logs_cfg = LogsConfig(
        log_dir="./logs",
        llm_cache_dir="./eval_llm_cache",
        event_trace_loc="./scholarqa_traces",
    )
    logs_cfg.init_formatter()

    configs = {
        "lite_low": (lambda: build_lite(paper_finder, logs_cfg, "low"), RESULT_DIR / "lite_low.json"),
        "lite_medium": (lambda: build_lite(paper_finder, logs_cfg, "medium"), RESULT_DIR / "lite_medium.json"),
        "lite_high": (lambda: build_lite(paper_finder, logs_cfg, "high"), RESULT_DIR / "lite_high.json"),
        "default": (lambda: build_default(paper_finder, logs_cfg), RESULT_DIR / "default.json"),
    }

    targets = configs.keys() if args.config == "all" else [args.config]
    for name in targets:
        factory, out_path = configs[name]
        run_config(name, factory, questions, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
