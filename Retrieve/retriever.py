import os
import pickle
from typing import Dict, List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore
from tqdm import tqdm

from Retrieve.index import (
    get_context_index,
    get_method_description_index,
    get_method_functionality_index,
)
from Retrieve.reranker import ChatReranker, EmbeddingReranker
from Storage.store import HybridStore
from Utils.async_utils import asyncio_run, run_jobs_with_worker_limit
from Utils.path_manager import PathManager


class MethodRetriever:
    def __init__(self, path_manager: PathManager, store: HybridStore):
        self.path_manager = path_manager
        self.logger = path_manager.logger
        self.store = store
        retrieve_dir = os.path.join(path_manager.res_path, "retrieve")
        os.makedirs(retrieve_dir, exist_ok=True)
        self.context_nodes_file = os.path.join(retrieve_dir, "context_nodes.pkl")
        self.method_nodes_file = os.path.join(retrieve_dir, "method_nodes.pkl")
        self.desc_nodes_file = os.path.join(retrieve_dir, "desc_nodes.pkl")
    
    
    def retrieve(self, queries: List[str], retriever: BaseRetriever) -> List[List[NodeWithScore]]:
        nodes_list = asyncio_run(self._aretrieve_nodes(queries, retriever))
        return nodes_list
    
    
    async def _aretrieve_nodes(self, queries: List[str], retriever: BaseRetriever) -> List[List[NodeWithScore]]:
        jobs = []
        for query in queries:
            jobs.append(retriever.aretrieve(query))
        nodes = await run_jobs_with_worker_limit(
            jobs,
            desc="Retrieve Nodes",
            show_progress=True)
        return nodes
    
    
    def get_context_score(self, context_queries: List[str], embedding_reranker: EmbeddingReranker):
        if os.path.exists(self.context_nodes_file):
            self.logger.info(f"Loading reranked context nodes from {self.context_nodes_file}")
            with open(self.context_nodes_file, "rb") as f:
                reranked_context_list = pickle.load(f)
        else:
            self.logger.info(f"Retrieving and reranking context nodes...")
            context_index = get_context_index(self.store)
            context_retriever = context_index.as_retriever(similarity_top_k=self.path_manager.retrieve_top_n)
            context_nodes_list = self.retrieve(context_queries, context_retriever)
            reranked_context_list = embedding_reranker.rerank(context_nodes_list, context_queries)
            with open(self.context_nodes_file, "wb") as f:
                pickle.dump(reranked_context_list, f)
        
        context_score_dict = {}
        for context_nodes in reranked_context_list:
            for node in context_nodes:
                if node.id_ not in context_score_dict:
                    context_score_dict[node.id_] = [node.score]
                else:
                    context_score_dict[node.id_].append(node.score)
        return context_score_dict
    
    
    def get_description_score(self,
            desc_queries: List[str],
            reranked_methods_list: List[List[NodeWithScore]],
            embedding_reranker: EmbeddingReranker
        ):
        if os.path.exists(self.desc_nodes_file):
            self.logger.info(f"Loading reranked description nodes from {self.desc_nodes_file}")
            with open(self.desc_nodes_file, "rb") as f:
                reranked_desc_list = pickle.load(f)
        else:
            self.logger.info(f"Retrieving and reranking description nodes...")
            method_nodes_ids = list(set([node.id_ for nodes in reranked_methods_list for node in nodes]))
            all_method_nodes = self.store.doc_store.get_nodes(method_nodes_ids)
            desc_index = get_method_description_index(
                all_method_nodes,
                self.store,
                use_context=self.path_manager.config.use_context
            )
            desc_retriever = desc_index.as_retriever(similarity_top_k=self.path_manager.retrieve_top_n)
            desc_nodes_list = self.retrieve(desc_queries, desc_retriever)
            reranked_desc_list = embedding_reranker.rerank(desc_nodes_list, desc_queries)
            with open(self.desc_nodes_file, "wb") as f:
                pickle.dump(reranked_desc_list, f)
        
        desc_score_dict = {}
        for desc_nodes in reranked_desc_list:
            for node in desc_nodes:
                if node.id_ not in desc_score_dict:
                    desc_score_dict[node.id_] = [node.score]
                else:
                    desc_score_dict[node.id_].append(node.score)
        return desc_score_dict
    
    
    def retrieve_methods(self, faulty_funcs: List[Dict[str, str]]) -> List[NodeWithScore]:
        context_queries = [func["context"] for func in faulty_funcs]
        method_queries = [func["functionality"] for func in faulty_funcs]
        desc_queries = [func["logic"] for func in faulty_funcs]
        embedding_reranker = EmbeddingReranker(self.path_manager)
        
        if os.path.exists(self.method_nodes_file):
            self.logger.info(f"Loading reranked method nodes from {self.method_nodes_file}")
            with open(self.method_nodes_file, "rb") as f:
                reranked_methods_list = pickle.load(f)
        else:
            self.logger.info(f"Retrieving and reranking method nodes...")
            method_index = get_method_functionality_index(self.store)
            method_retriever = method_index.as_retriever(similarity_top_k=self.path_manager.retrieve_top_n)
            method_nodes_list = self.retrieve(method_queries, method_retriever)
            reranked_methods_list = embedding_reranker.rerank(method_nodes_list, method_queries)
            with open(self.method_nodes_file, "wb") as f:
                pickle.dump(reranked_methods_list, f)
            
        method_score_dict = {}
        for method_nodes in reranked_methods_list:
            for node in method_nodes:
                if node.id_ not in method_score_dict:
                    method_score_dict[node.id_] = [node.score]
                else:
                    method_score_dict[node.id_].append(node.score)
        
        if self.path_manager.config.use_context_retrieval:
            if self.path_manager.config.use_context:
                context_score_dict = self.get_context_score(context_queries, embedding_reranker)
        
        if self.path_manager.config.use_description_retrieval:
            desc_score_dict = self.get_description_score(
                desc_queries,
                reranked_methods_list,
                embedding_reranker
            )
        
        self.logger.info(f"Combining multi-level retrieval results...")
        result_nodes = []
        for id in tqdm(method_score_dict):
            method_node = self.store.doc_store.get_node(id)
            node_with_score = NodeWithScore(node=method_node, score=sum(method_score_dict[id]))
            
            if self.path_manager.config.use_context_retrieval:
                context_node_id = method_node.metadata.get("context_node_id", None)
                if context_node_id:
                    if context_node_id in context_score_dict:
                        node_with_score.score += sum(context_score_dict[context_node_id])
                
            if self.path_manager.config.use_description_retrieval:
                if self.path_manager.config.use_context:
                    desc_node_ids = method_node.metadata.get("desc_node_ids", None)
                else:
                    desc_node_ids = method_node.metadata.get("desc_node_ids_NC", None)
                assert desc_node_ids is not None, f"Method node {id} has no description node"
                for desc_node_id in desc_node_ids:
                    if desc_node_id in desc_score_dict:
                        node_with_score.score += sum(desc_score_dict[desc_node_id])
            
            result_nodes.append(node_with_score)
        result_nodes.sort(key=lambda x: x.score, reverse=True)
        
        if self.path_manager.config.use_chat_rerank:
            n_chat_rerank = self.path_manager.config.hyper.chat_rerank_top_n
            if len(result_nodes) <= n_chat_rerank:
                todo_nodes = result_nodes
                undo_nodes = []
            else:
                todo_nodes = result_nodes[:n_chat_rerank]
                undo_nodes = result_nodes[n_chat_rerank:]
            self.logger.info(f"Chat rerank {len(todo_nodes)} method nodes")
            chat_reranker = ChatReranker(self.path_manager)
            todo_nodes = chat_reranker.rerank(todo_nodes)
            result_nodes = todo_nodes + undo_nodes
        
        # Filter out test methods
        result_nodes = [node for node in result_nodes if node.metadata["is_test_method"] == False]
        return result_nodes