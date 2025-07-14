import logging
import os
from scholarqa.llms.constants import *
from typing import List, Any, Callable, Tuple, Iterator, Union, Generator

import litellm
from litellm.caching import Cache
from litellm.utils import trim_messages
from langsmith import traceable

from scholarqa.state_mgmt.local_state_mgr import AbsStateMgrClient

logger = logging.getLogger(__name__)


class CostAwareLLMCaller:
    def __init__(self, state_mgr: AbsStateMgrClient):
        self.state_mgr = state_mgr

    @staticmethod
    def parse_result_args(method_result: Union[Tuple[Any, CompletionResult], CompletionResult]) -> Tuple[
        Any, List[CompletionResult], List[str]]:
        if type(method_result) == tuple:
            result, completion_costs = method_result
        else:
            result, completion_costs = method_result, method_result
        completion_costs = [completion_costs] if type(completion_costs) != list else completion_costs
        completion_models = [cost.model for cost in completion_costs]
        return result, completion_costs, completion_models

    def parse_usage_args(self, method_result: Union[Tuple[float, TokenUsage], float]) -> Tuple[float, TokenUsage]:
        if isinstance(method_result, tuple):
            total_cost, tokens = method_result
        else:
            total_cost, tokens = method_result, TokenUsage(input=0, output=0, total=0, reasoning=0)
        return total_cost, tokens

    def call_method(self, cost_args: CostReportingArgs, method: Callable, **kwargs) -> CostAwareLLMResult:
        method_result = method(**kwargs)
        result, completion_costs, completion_models = self.parse_result_args(method_result)
        llm_usage = self.state_mgr.report_llm_usage(completion_costs=completion_costs, cost_args=cost_args)
        total_cost, tokens = self.parse_usage_args(llm_usage)
        return CostAwareLLMResult(result=result, tot_cost=total_cost, models=completion_models, tokens=tokens)

    def call_iter_method(self, cost_args: CostReportingArgs, gen_method: Callable, **kwargs) -> Generator[
        Any, None, CostAwareLLMResult]:
        all_results, all_completion_costs, all_completion_models = [], [], []
        for method_result in gen_method(**kwargs):
            result, completion_costs, completion_models = self.parse_result_args(method_result)
            all_completion_costs.extend(completion_costs)
            all_completion_models.extend(completion_models)
            all_results.append(result)
            yield result
        llm_usage = self.state_mgr.report_llm_usage(completion_costs=all_completion_costs, cost_args=cost_args)
        total_cost, tokens = self.parse_usage_args(llm_usage)
        return CostAwareLLMResult(result=all_results, tot_cost=total_cost, models=all_completion_models, tokens=tokens)


def success_callback(kwargs, completion_response, start_time, end_time):
    """required callback method to update the response object with cache hit/miss info"""
    completion_response.cache_hit = kwargs["cache_hit"] if kwargs["cache_hit"] is not None else False


litellm.success_callback = [success_callback]


def setup_llm_cache(cache_type: str = "s3", **cache_args):
    logger.info("Setting up LLM cache...")
    litellm.cache = Cache(type=cache_type, **cache_args)
    litellm.enable_cache()


@traceable(run_type="llm", name="batch completion")
def batch_llm_completion(model: str, messages: List[str], system_prompt: str = None, fallback=GPT_4o,
                         **llm_lite_params) -> List[
    CompletionResult]:
    """returns the result from the llm chat completion api with cost and tokens used"""
    fallbacks = [
        fallback] if fallback else []  # Disable for now in lieu of https://github.com/BerriAI/litellm/issues/10517
    messages = [trim_messages([{"role": "system", "content": system_prompt}, {"role": "user", "content": msg}], model)
                for msg in messages]
    responses = litellm.completion_with_retries(messages=messages, model=model,
                                                original_function=litellm.batch_completion, **llm_lite_params)
    results = []
    for i, res in enumerate(responses):
        try:
            res_cost = round(litellm.completion_cost(res), 6)
        except Exception as e:
            logger.warning(f"Error calculating cost: {e}")
            res_cost = 0.0

        res_usage = res.usage
        reasoning_tokens = 0 if not (res_usage.completion_tokens_details and
                                     res_usage.completion_tokens_details.reasoning_tokens) else \
            res_usage.completion_tokens_details.reasoning_tokens
        res_str = res["choices"][0]["message"]["content"].strip()
        cost_tuple = CompletionResult(content=res_str, model=res["model"],
                                      cost=res_cost if not res.get("cache_hit") else 0.0,
                                      input_tokens=res_usage.prompt_tokens,
                                      output_tokens=res_usage.completion_tokens, total_tokens=res_usage.total_tokens,
                                      reasoning_tokens=reasoning_tokens)
        results.append(cost_tuple)
    return results


@traceable(run_type="llm", name="completion")
def llm_completion(user_prompt: str, system_prompt: str = None, fallback=GPT_4o, **llm_lite_params) -> CompletionResult:
    """returns the result from the llm chat completion api with cost and tokens used"""
    messages = []
    fallbacks = [
        fallback] if fallback else []  # Disable for now in lieu of https://github.com/BerriAI/litellm/issues/10517
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = litellm.completion_with_retries(messages=messages, **llm_lite_params)
    try:
        res_cost = round(litellm.completion_cost(response), 6)
    except Exception as e:
        logger.warning(f"Error calculating cost: {e}")
        res_cost = 0.0

    res_usage = response.usage
    reasoning_tokens = 0 if not (res_usage.completion_tokens_details and
                                 res_usage.completion_tokens_details.reasoning_tokens) else \
        res_usage.completion_tokens_details.reasoning_tokens
    res_str = response["choices"][0]["message"]["content"]
    if res_str is None:
        logger.warning("Content returned as None, checking for response in tool_calls...")
        res_str = response["choices"][0]["message"]["tool_calls"][0].function.arguments
    cost_tuple = CompletionResult(content=res_str.strip(), model=response.model,
                                  cost=res_cost if not response.get("cache_hit") else 0.0,
                                  input_tokens=res_usage.prompt_tokens,
                                  output_tokens=res_usage.completion_tokens, total_tokens=res_usage.total_tokens,
                                  reasoning_tokens=reasoning_tokens)
    return cost_tuple
