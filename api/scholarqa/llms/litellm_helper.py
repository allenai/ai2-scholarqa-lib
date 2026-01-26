import logging
from scholarqa.llms.constants import *
from typing import List, Any, Callable, Tuple, Iterator, Union, Generator, Optional

import litellm
from litellm.caching import Cache
from litellm.utils import trim_messages
from langsmith import traceable

from scholarqa.state_mgmt.local_state_mgr import AbsStateMgrClient
from time import sleep

logger = logging.getLogger(__name__)

def register_custom_models():
    """
    Tells LiteLLM how to handle the model we're using on Modal.
    """
    litellm.register_model({
        "allenai/sqa_basicsftdpo": {
            "max_tokens": 4096,      # adjust these values as needed
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "lite_llm_model_name": "allenai/sqa_basicsftdpo"
        }
    })
register_custom_models()

class CostAwareLLMCaller:
    def __init__(self, state_mgr: 'AbsStateMgrClient'):
        self.state_mgr = state_mgr

    def _calculate_cost(self, response: Any, model_identifier: str) -> float:
        # For now default to 0.0 for hosted_vllm
        if model_identifier.startswith("hosted_vllm"):
            return 0.0

        # Cache hits are always free
        if response.get("cache_hit"):
            return 0.0
        # otherwise use the real data
        return round(litellm.completion_cost(response), 6)

    @staticmethod
    def parse_litellm_response(response: Any, requested_model: Optional[str] = None) -> 'CompletionResult':
        model_id = response.get("model") or requested_model

        usage = response.usage
        prompt_t = getattr(usage, 'prompt_tokens', 0) or 0
        completion_t = getattr(usage, 'completion_tokens', 0) or 0
        total_t = getattr(usage, 'total_tokens', 0) or 0

        reasoning_t = 0
        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            reasoning_t = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0) or 0

        # Content Extraction
        choice = response["choices"][0]["message"]
        content = choice.get("content") or ""
        if not content and "tool_calls" in choice:
            content = choice["tool_calls"][0].function.arguments

        # Note: We use an instance if available, otherwise static logic
        caller = CostAwareLLMCaller(None) # Temporary for static context if needed
        cost = caller._calculate_cost(response, str(model_id))

        return CompletionResult(
            content=content.strip(),
            model=str(model_id),
            cost=cost,
            input_tokens=prompt_t,
            output_tokens=completion_t,
            total_tokens=total_t,
            reasoning_tokens=reasoning_t
        )

    def _unpack_result(self, result: Union[Tuple[Any, 'CompletionResult'], 'CompletionResult']) -> Tuple[Any, List['CompletionResult']]:
        """Internal helper to handle tuple vs single result outputs."""
        if type(result) is tuple:
            main_res, completion_costs = result
        else:
            main_res, completion_costs = result, result

        costs_list = [completion_costs] if not isinstance(completion_costs, list) else completion_costs
        return main_res, costs_list

    def call_method(self, cost_args: 'CostReportingArgs', method: Callable, **kwargs) -> 'CostAwareLLMResult':
        """Orchestrates the method call and reports usage to the state manager."""
        method_result = method(**kwargs)

        main_res, costs_list = self._unpack_result(method_result)
        completion_models = [cost.model for cost in costs_list]

        llm_usage = self.state_mgr.report_llm_usage(completion_costs=costs_list, cost_args=cost_args)

        # Parse state manager response (cost, TokenUsage)
        total_cost = llm_usage[0] if isinstance(llm_usage, tuple) else llm_usage
        tokens = llm_usage[1] if isinstance(llm_usage, tuple) else TokenUsage(input=0, output=0, total=0, reasoning=0)

        return CostAwareLLMResult(
            result=main_res,
            tot_cost=total_cost,
            models=completion_models,
            tokens=tokens
        )

    def call_iter_method(self, cost_args: 'CostReportingArgs', gen_method: Callable, **kwargs) -> Generator[Any, None, 'CostAwareLLMResult']:
        """Orchestrates streaming/generator calls and aggregates usage for the state manager."""
        all_results, all_completion_costs, all_completion_models = [], [], []

        for method_result in gen_method(**kwargs):
            main_res, costs_list = self._unpack_result(method_result)
            all_completion_costs.extend(costs_list)
            all_completion_models.extend([cost.model for cost in costs_list])
            all_results.append(main_res)
            yield main_res

        llm_usage = self.state_mgr.report_llm_usage(completion_costs=all_completion_costs, cost_args=cost_args)

        # Parse state manager response
        total_cost = llm_usage[0] if isinstance(llm_usage, tuple) else llm_usage
        tokens = llm_usage[1] if isinstance(llm_usage, tuple) else TokenUsage(input=0, output=0, total=0, reasoning=0)

        return CostAwareLLMResult(
            result=all_results,
            tot_cost=total_cost,
            models=all_completion_models,
            tokens=tokens
        )


def success_callback(kwargs, completion_response):
    """Callback method to update the response object with cache hit/miss info"""
    completion_response.cache_hit = kwargs.get("cache_hit", False)

NUM_RETRIES = 3
litellm.success_callback = [success_callback]

def setup_llm_cache(cache_type: str = "s3", **cache_args):
    logger.info("Setting up LLM cache...")
    litellm.cache = Cache(type=cache_type, **cache_args)
    litellm.enable_cache()


@traceable(run_type="llm", name="completion")
def llm_completion(user_prompt: str, system_prompt: str = None, fallback=GPT_5_CHAT, **llm_lite_params) -> CompletionResult:
    messages = []
    fallbacks = [f.strip() for f in fallback.split(",")] if fallback else []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = litellm.completion_with_retries(
        messages=messages,
        num_retries=NUM_RETRIES,
        fallbacks=fallbacks,
        **llm_lite_params
    )

    return CostAwareLLMCaller.parse_litellm_response(response, requested_model=llm_lite_params.get("model"))

@traceable(run_type="llm", name="batch completion")
def batch_llm_completion(model: str, messages: List[str], system_prompt: str = None, fallback: Optional[str] = GPT_5_CHAT,
                         **llm_lite_params) -> List[Optional[CompletionResult]]:
    fallbacks = [f.strip() for f in fallback.split(",")] if fallback else []
    formatted_messages = [
        trim_messages([{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}], model)
        for msg in messages
    ]

    results, pending = [None] * len(messages), list(range(len(messages)))
    curr_retry = 0

    while pending and curr_retry <= NUM_RETRIES:
        pending_messages = [formatted_messages[idx] for idx in pending]
        responses = litellm.completion_with_retries(
            messages=pending_messages,
            model=model,
            fallbacks=fallbacks,
            num_retries=NUM_RETRIES,
            original_function=litellm.batch_completion,
            **llm_lite_params
        )

        for i, res in enumerate(responses):
            try:
                # Use the centralized parser
                results[pending[i]] = CostAwareLLMCaller.parse_litellm_response(res, requested_model=model)
            except Exception as e:
                if curr_retry == NUM_RETRIES:
                    logger.error(f"Error in batch instance {i}: {e}")
                    raise e

        pending = [i for i, r in enumerate(results) if r is None]
        curr_retry += 1
        if pending:
            sleep(2 ** curr_retry)

    return results
