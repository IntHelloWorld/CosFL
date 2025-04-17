import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import responses
from typing import Any, Dict, List

from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from tenacity import retry, stop_after_attempt, wait_fixed

from Diagnose.prompt import (
    DIAGNOSE_END_TEMPLATE,
    DIAGNOSE_TEMPLATE,
    FAULTY_FUNCTIONALITY_EXAMPLE,
    REQUEST_EXAMPLE,
)
from functions.my_types import TestCase, TestFailure
from Retrieve.index import get_context_index
from Storage.store import HybridStore
from Utils.model import calculate_in_cost, calculate_out_cost, parse_llm_output
from Utils.path_manager import PathManager

DEFAULT_MAX_WORKERS = 8


class DiagnoseAgent:
    def __init__(self, path_manager: PathManager, store: HybridStore):
        self.llm: LLM = path_manager.reasoning_llm
        self.path_manager = path_manager
        self.logger = path_manager.logger
        self.dialogue_dir = os.path.join(path_manager.res_path, "diagnose")
        self.use_context = path_manager.config.use_context
        if self.use_context:
            context_index = get_context_index(store)
            self.context_retriever = context_index.as_retriever(
                similarity_top_k=path_manager.retrieve_top_n
            )

    def diagnose(self, test_failure: TestFailure) -> List[Dict[str, str]]:
        self.logger.info(f"Diagnosing faulty functionality...")
        with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
            futures = []
            for test_class in test_failure.test_classes:
                for test_case in test_class.test_cases:
                    future = executor.submit(
                        self._diagnose_test_case, test_case
                    )
                    futures.append(future)

            result = []
            for future in as_completed(futures):
                result.append(future.result())

        faulty_func = [res["response"] for res in result]
        tokens = sum([res["tokens"] for res in result])
        cost = sum([res["cost"] for res in result])
        self.logger.info(
            f"Diagnose finished, total cost: {tokens} tokens, {cost} USD"
        )
        return faulty_func

    def calculate_cost(self, dialog: Dict[str, str]) -> Dict[str, str]:
        in_costs = [calculate_in_cost(str(dialog[k]["user"])) for k in dialog]
        out_costs = [calculate_out_cost(str(dialog[k]["llm"])) for k in dialog]
        in_tokens, in_money = zip(*in_costs)
        out_tokens, out_money = zip(*out_costs)
        tokens = sum(in_tokens) + sum(out_tokens)
        money = sum(in_money) + sum(out_money)
        return tokens, money

    def _diagnose_test_case(self, test_case: TestCase) -> Dict[str, str]:
        max_rounds = self.path_manager.config.hyper.max_diagnose_rounds

        dialog_dir = os.path.join(self.dialogue_dir, test_case.name)
        os.makedirs(dialog_dir, exist_ok=True)
        if self.use_context:
            dialog_file = os.path.join(dialog_dir, "dialog.json")
        else:
            dialog_file = os.path.join(dialog_dir, "dialog_NC.json")

        if os.path.exists(dialog_file):
            with open(dialog_file, "r") as f:
                dialog = json.load(f)
                tokens, money = self.calculate_cost(dialog)
                return {
                    "response": dialog["end"]["llm"],
                    "tokens": tokens,
                    "cost": money,
                }

        dialog = {}
        cur_round = 0
        component_details = {}

        while True:
            llm_input = {
                "test_code": test_case.test_method.get_lined_code(),
                "stack_trace": test_case.stack_trace,
                "test_output": test_case.test_output,
                "component_details": "\n\n".join(component_details.values()),
            }

            if cur_round < max_rounds - 1 and self.use_context:
                messages = DIAGNOSE_TEMPLATE.format_messages(**llm_input)
            else:
                messages = DIAGNOSE_END_TEMPLATE.format_messages(**llm_input)

            response = self.llm.chat(messages)
            result = parse_llm_output(response.message.content)

            if "request" in result:
                assert (
                    cur_round < max_rounds - 1
                ), "LLM should not request more information in the last round"
                context_node = self.get_context(result["request"])
                if context_node.id_ not in component_details:
                    component_details[context_node.id_] = context_node.text
                dialog[cur_round] = {
                    "user": messages[0].content,
                    "llm": result,
                }
                cur_round += 1
            elif (
                "context" in result
                and "functionality" in result
                and "logic" in result
            ):
                dialog["end"] = {"user": messages[0].content, "llm": result}
                with open(dialog_file, "w") as f:
                    json.dump(dialog, f, indent=4)
                tokens, money = self.calculate_cost(dialog)
                return {"response": result, "tokens": tokens, "cost": money}
            else:
                raise ValueError("Unexpected response format from LLM")

    def get_context(self, request: str) -> NodeWithScore:
        context_nodes = self.context_retriever.retrieve(request)
        return context_nodes[0]
