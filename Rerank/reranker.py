import json
import os
import re
import sys
from pathlib import Path
from typing import List

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, asyncio_run, run_jobs
from llama_index.core.schema import NodeWithScore

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from functions.my_types import TestFailure
from Rerank.prompt import rerank_template
from Utils.path_manager import PathManager


class ChatReranker:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager

        query_file = os.path.join(
            self.path_manager.res_path,
            "query.txt"
        )
        with open(query_file, "r") as f:
            self.queries_text = f.read()
        
        self.rerank_cache_dir = os.path.join(self.path_manager.res_path, "rerank")
        if not os.path.exists(self.rerank_cache_dir):
            os.mkdir(self.rerank_cache_dir)

    def rerank(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        reranked_nodes = self._get_score_for_nodes(nodes)
        reranked_nodes.sort(key=lambda x: x.metadata["llm_score"], reverse=True)
        return reranked_nodes
    
    async def _aget_score_for_node(self, node: NodeWithScore) -> float:
        rerank_file = os.path.join(self.rerank_cache_dir,f"{node.id_}.json")
        if os.path.exists(rerank_file):
            with open(rerank_file, "r") as f:
                score = json.load(f)["Score"]
            return score
        
        messages = rerank_template.format_messages(
            causes_ph=self.queries_text["Causes"],
            method_code_ph=node.node.text
        )
        
        response = await self.path_manager.reasoning_llm.achat(messages)
        result = self._parse_json_response(response.message.content)
        with open(rerank_file, "w") as f:
            json.dump(result, f, indent=4)
        score = result["Score"]
        return score
    
    async def _aget_score_for_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        node_score_jobs = []
        for node in nodes:
            node_score_jobs.append(self._aget_score_for_node(node))

        scores = await run_jobs(
            node_score_jobs,
            show_progress=True,
            workers=DEFAULT_NUM_WORKERS,
            desc="Chat-based Reranking"
        )
        
        for i, node in enumerate(nodes):
            node.metadata["llm_score"] = scores[i]
        return nodes
    
    def _get_score_for_nodes(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        return asyncio_run(self._aget_score_for_nodes(nodes))
    
    def _parse_json_response(self, text):
        pattern = r'```json(.+)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return json.loads(match.group(1))
        else:
            return json.loads(text)
