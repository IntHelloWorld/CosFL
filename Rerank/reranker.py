import json
import os
import re
import sys
from pathlib import Path
from typing import List

from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, asyncio_run, run_jobs
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.postprocessor.voyageai_rerank import VoyageAIRerank

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from functions.my_types import TestFailure
from Rerank.hack import postprocess_nodes
from Rerank.prompt import rerank_template
from Utils.path_manager import PathManager


class ChatReranker:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager

        self.causes_text = self._get_causes_text()
        
        self.rerank_cache_dir = os.path.join(self.path_manager.res_path, "rerank")
        if not os.path.exists(self.rerank_cache_dir):
            os.mkdir(self.rerank_cache_dir)

    def _get_causes_text(self) -> str:
        query_path = os.path.join(
            self.path_manager.bug_path,
            self.path_manager.reasoning_model
        )
        
        if self.path_manager.query_type == "one_query":
            query_file = os.path.join(query_path, "one_query.json")
            with open(query_file, "r") as f:
                json_obj = json.load(f)
            return json_obj["Causes"]
        elif self.path_manager.query_type == "normal":
            counter = 0
            causes_text = ""

            # tranverse the files in query_path
            query_files = os.listdir(query_path)
            for query_file in query_files:
                if query_file in ["merged_queries.json", "one_query.json"]:
                    continue
                with open(os.path.join(query_path, query_file), "r") as f:
                    query = json.load(f)
                causes_text += f"{counter+1} :{query['Causes']}\n"
                counter += 1
            return causes_text
        else:
            raise ValueError(f"Invalid query type {self.path_manager.query_type}")
    
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
            causes_ph=self.causes_text,
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
    
    def _parse_json_response(self, text: str):
        pattern = r'\{\s*"\w+":.+\}'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            print(f"Failed to parse json from:\n{text}")
            raise ValueError("Failed to match string in json format")
        
        text_to_parse = match.group(0).replace("\n", "")
        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError as e:
            print(f"Failed to parse json from:\n{text_to_parse}")
            # try to escape backslashes
            err_char = text_to_parse[e.pos - 1]
            print(f"Error char: {err_char}")
            if err_char == "\\":
                new_text = text_to_parse[:e.pos - 1] + "\\\\" + text_to_parse[e.pos:]
                return self._parse_json_response(new_text)
            elif err_char == "\"":
                new_text = text_to_parse[:e.pos - 1] + "\\\"" + text_to_parse[e.pos:]
                return self._parse_json_response(new_text)
            elif e.msg.startswith("Invalid \\escape"):
                new_text = text_to_parse[:e.pos - 1] + "\\" + text_to_parse[e.pos - 1:]
                return self._parse_json_response(new_text)
            else:
                raise ValueError("Failed to parse json")

class Reranker():
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        
        self.chat_reranker = None
        if path_manager.config["rerank"]["use_chat_rerank"]:
            self.chat_reranker = ChatReranker(path_manager)
        
        self.embedding_reranker = None
        if path_manager.rerank_model:
            if path_manager.config["rerank"]["rerank_series"] == "jina":
                self.embedding_reranker = JinaRerank(
                    api_key=path_manager.config["rerank"]["rerank_api_key"],
                    model=path_manager.rerank_model,
                    top_n=path_manager.rerank_top_n
                )
            elif path_manager.config["rerank"]["rerank_series"] == "cohere":
                self.embedding_reranker = CohereRerank(
                    model=path_manager.config["rerank"]["rerank_model"],
                    api_key=path_manager.config["rerank"]["rerank_api_key"],
                    top_n=path_manager.rerank_top_n
                )
            elif path_manager.config["rerank"]["rerank_series"] == "voyage":
                self.embedding_reranker = VoyageAIRerank(
                    model=path_manager.config["rerank"]["rerank_model"],
                    api_key=path_manager.config["rerank"]["rerank_api_key"],
                    top_k=path_manager.rerank_top_n,
                    truncation=False
                )
    
    def rerank(self, retrieved_nodes_list: List[List[NodeWithScore]], queries: List[str]):
        self.path_manager.logger.info(f"[Rerank] rerank model {self.path_manager.config['rerank']['rerank_model']}...")
        reranked_nodes_list = []
        for i, nodes in enumerate(retrieved_nodes_list):
            if len(nodes) == 0:
                reranked_nodes_list.append([])
                continue
            if self.embedding_reranker is None:
                reranked_nodes = nodes
            else:
                reranked_nodes = postprocess_nodes(
                    self.embedding_reranker,
                    nodes,
                    query_str=queries[i]
                )
            reranked_nodes_list.append(reranked_nodes)
        
        self.path_manager.logger.info(f"[Rerank] use chat rerank? {self.path_manager.config['rerank']['use_chat_rerank']}")
        if self.chat_reranker is None:
            # merge all reranked nodes, keep the highest score for duplicate nodes
            all_node_dict = {}
            for reranked_nodes in reranked_nodes_list:
                for n in reranked_nodes:
                    if n.id_ not in all_node_dict:
                        all_node_dict[n.id_] = n
                    else:
                        if n.score > all_node_dict[n.id_].score:
                            all_node_dict[n.id_] = n

            result_nodes = list(all_node_dict.values())
            result_nodes.sort(key=lambda x: x.score, reverse=True)
        else:
            chat_rerank_dict = {}
            extra_dict = {}
            for reranked_nodes in reranked_nodes_list:
                for n in reranked_nodes[:self.path_manager.chat_rerank_top_n]:
                    if n.id_ not in chat_rerank_dict:
                        chat_rerank_dict[n.id_] = n
                for n in reranked_nodes[self.path_manager.chat_rerank_top_n:]:
                    if n.id_ not in chat_rerank_dict:
                        if n.id_ not in extra_dict:
                            extra_dict[n.id_] = n

            chat_reranked_nodes = self.chat_reranker.rerank(list(chat_rerank_dict.values()))
            extra_nodes = list(extra_dict.values())
            extra_nodes.sort(key=lambda x: x.score, reverse=True)
            result_nodes = chat_reranked_nodes + extra_nodes
        return result_nodes, reranked_nodes_list