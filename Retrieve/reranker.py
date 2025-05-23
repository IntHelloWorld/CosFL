import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from llama_index.core.schema import NodeWithScore
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from Retrieve.hack import postprocess_nodes
from Retrieve.prompt import (
    CHAT_RERANK_TEMPLATE,
    EXAMPLE_CHAT_RERANK_PROMPT,
    EXAMPLE_RERANK_RESPONSE,
)
from Utils.model import (
    calculate_in_cost,
    calculate_out_cost,
    get_embedding_reranker,
    parse_llm_output,
)
from Utils.path_manager import PathManager


class ChatReranker:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.rerank_cache_dir = os.path.join(
            self.path_manager.res_path, "rerank"
        )
        self.diagnose_text = self._get_diagnose_text()
        if not os.path.exists(self.rerank_cache_dir):
            os.mkdir(self.rerank_cache_dir)

    def _get_diagnose_text(self) -> str:
        diagnose_path = os.path.join(self.path_manager.res_path, "diagnose")
        diagnose_text = ""

        # tranverse the files in diagnose cache path
        diagnose_files = Path(diagnose_path).rglob("dialog.json")
        for i, diagnose_file in enumerate(diagnose_files):
            with open(os.path.join(diagnose_path, diagnose_file), "r") as f:
                diagnose = json.load(f)
            diagnose_text += f"hypothesis {i+1}:\n"
            for k, v in diagnose["end"]["llm"].items():
                diagnose_text += f"  - {k}: {v}\n"
            diagnose_text += "\n"
        return diagnose_text

    def rerank(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        reranked_nodes = self._get_score_for_nodes(nodes)
        reranked_nodes.sort(
            key=lambda x: x.metadata["llm_score"], reverse=True
        )
        return reranked_nodes

    # @retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
    def _get_score_for_node(self, node: NodeWithScore) -> dict:
        rerank_file = os.path.join(self.rerank_cache_dir, f"{node.id_}.json")
        if os.path.exists(rerank_file):
            with open(rerank_file, "r") as f:
                result = json.load(f)
                in_tokens, in_cost = calculate_in_cost(
                    EXAMPLE_CHAT_RERANK_PROMPT
                )
                out_tokens, out_cost = calculate_out_cost(str(result))
            return {
                "response": result,
                "tokens": in_tokens + out_tokens,
                "cost": in_cost + out_cost,
            }

        messages = CHAT_RERANK_TEMPLATE.format_messages(
            causes_ph=self.diagnose_text, method_code_ph=node.text
        )
        in_tokens, in_cost = calculate_in_cost(str(messages))

        if self.path_manager.config.mimic:
            result = EXAMPLE_RERANK_RESPONSE
        else:
            response = self.path_manager.reasoning_llm.chat(messages)
            result = parse_llm_output(response.message.content)
            with open(rerank_file, "w") as f:
                json.dump(result, f, indent=4)
        out_tokens, out_cost = calculate_out_cost(str(result))
        return {
            "response": result,
            "tokens": in_tokens + out_tokens,
            "cost": in_cost + out_cost,
        }

    def _run_jobs_with_thread_pool(self, jobs, limit, desc=""):
        results = []

        with ThreadPoolExecutor(max_workers=limit) as executor:
            futures = []
            for node in jobs:
                futures.append(executor.submit(self._get_score_for_node, node))

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.path_manager.logger.exception(f"Error in {desc}: {e}")
                    raise Exception(f"Error in {desc}: {e}") from e

        return results

    def _get_score_for_nodes(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        results = self._run_jobs_with_thread_pool(
            nodes,
            limit=self.path_manager.config.models.reason.rate_limit,
            desc="Chat Reranking",
        )

        responses = [result["response"] for result in results]
        tokens = [result["tokens"] for result in results]
        costs = [result["cost"] for result in results]
        self.path_manager.logger.info(
            f"Chat rerank cost: {sum(tokens)} tokens, {sum(costs)} money"
        )

        for i, node in enumerate(nodes):
            node.metadata["llm_score"] = responses[i]["Score"]
            node.metadata["llm_reason"] = responses[i]["Reason"]
        return nodes


class EmbeddingReranker:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.reranker = get_embedding_reranker(path_manager)

    def rerank(
        self,
        retrieved_nodes_list: List[List[NodeWithScore]],
        queries: List[str],
    ) -> List[List[NodeWithScore]]:
        self.path_manager.logger.info(
            f"embedding rerank based on {self.path_manager.config.models.rerank.model}..."
        )
        reranked_nodes_list = []
        for i, nodes in tqdm(
            enumerate(retrieved_nodes_list), total=len(retrieved_nodes_list)
        ):
            if len(nodes) == 0:
                reranked_nodes_list.append([])
                continue
            reranked_nodes = postprocess_nodes(
                self.reranker,
                nodes,
                query_str=queries[i],
                use_context=self.path_manager.config.use_context,
            )
            reranked_nodes_list.append(reranked_nodes)

        return reranked_nodes_list
