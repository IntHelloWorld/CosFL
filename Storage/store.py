import hashlib
import json
import os
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import chromadb
import more_itertools
from code_node import CodeNode
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from networkx import DiGraph
from tenacity import retry, stop_after_attempt, wait_fixed

from CallGraph.cg import (
    CGMethodNode,
    callstack_to_graph,
    cg_summary_to_text,
    cluster_graph,
    subgraph_to_text,
)
from CallGraph.prompt import METHOD_CALL_SUBGRAPH_SUMMARIZATION_TEMPLATE
from functions.sbfl import get_all_sbfl_res
from Storage.code_extractors import CodeSummaryExtractor
from Storage.node_parser import JavaNodeParser
from Utils.async_utils import asyncio_run, run_jobs_with_rate_limit
from Utils.path_manager import PathManager


class HybridStore:
    def __init__(self, path_manager: PathManager) -> None:
        self.path_manager = path_manager
        self.logger = path_manager.logger
        self.src_path = os.path.join(path_manager.buggy_path, path_manager.src_prefix)
        self.test_path = os.path.join(path_manager.buggy_path, path_manager.test_prefix)
        self.init_stores()

    def init_stores(self):
        if os.path.exists(self.path_manager.doc_store_file):
            doc_store = SimpleDocumentStore.from_persist_dir(self.path_manager.stores_dir)
        else:
            doc_store = SimpleDocumentStore()

        db = chromadb.PersistentClient(path=self.path_manager.vector_store_dir)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        self.doc_store = doc_store
        self.vector_store = vector_store

        method_nodes = self._get_method_nodes()
        call_graph, binded_method_nodes, cg_to_mn_map = self.bind_call_graph(method_nodes)

        sub_graphs = self.cluster_call_graph(call_graph)

        context_nodes = self.subgraphs_summarization(sub_graphs, binded_method_nodes, cg_to_mn_map)
        print("over")

    def cluster_call_graph(self, call_graph):
        """
        cluster the call graph
        """
        self.logger.info("clustering call graph...")
        sub_graphs = cluster_graph(call_graph)
        self.logger.info(
            f"call graph clustered into {len(sub_graphs)} subgraphs: "
            f"{[len(subgraph.nodes) for subgraph in sub_graphs]}"
        )
        return sub_graphs


    def build_context_nodes(
        self,
        subgraphs: List[DiGraph],
        responses: List[Dict],
        binded_method_nodes: List[CodeNode],
        cg_to_mn_map: Dict[CGMethodNode, str]
    ) -> List[CodeNode]:
        """Build nodes from LLM call graph summarization."""

        def ctxt_id_func(text: str) -> str:
            sha256 = hashlib.sha256()
            sha256.update(text.encode("utf-8"))
            unique_id = sha256.hexdigest()
            return unique_id

        mn_dict = {mn.id_: mn for mn in binded_method_nodes}
        nodes: List[CodeNode] = []
        for i in range(len(subgraphs)):
            ctxt_text = cg_summary_to_text(responses[i])
            node_id = ctxt_id_func(responses[i]["summary"])
            node = CodeNode(id_= node_id, text=ctxt_text)
            node.metadata.update(responses[i])
            node.metadata.update(
                {
                    "node_type": "context_node",
                }
            )
            nodes.append(node)

            # update method node metadata
            for nx_node in subgraphs[i].nodes:
                if nx_node in cg_to_mn_map:
                    mn = mn_dict[cg_to_mn_map[nx_node]]
                    mn.metadata.update(
                        {
                            "ctxt_node_id": node_id,
                        }
                    )

        return nodes

    def subgraphs_summarization(
        self,
        subgraphs: List[DiGraph],
        binded_method_nodes: List[CodeNode],
        cg_to_mn_map: Dict[CGMethodNode, str]
    ):
        """
        summarize the subgraphs to context nodes
        """
        responses = asyncio_run(self._asubgraphs_summarization(subgraphs))
        context_nodes = self.build_context_nodes(
            subgraphs,
            responses,
            binded_method_nodes,
            cg_to_mn_map
        )
        return context_nodes

    async def _asubgraphs_summarization(self, subgraphs: List[DiGraph]):
        jobs = []

        for subgraph in subgraphs:
            jobs.append(self._asubgraph_summarization(subgraph))

        responses = await run_jobs_with_rate_limit(
            jobs,
            limit=self.path_manager.config["reason"]["reasoning_rate_limit"],
            desc="Subgraph Summarization",
            show_progress=True
        )
        return responses

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    async def _asubgraph_summarization(self, subgraph: DiGraph):
        input_text = subgraph_to_text(subgraph)
        messages = METHOD_CALL_SUBGRAPH_SUMMARIZATION_TEMPLATE.format_messages(
            input_text=input_text
        )
        response = self.path_manager.reasoning_llm.chat(messages)
        json_res = json.loads(response.message.content)
        return json_res

    def bind_call_graph(self, method_nodes):
        """
        bind call graph with source code
        """

        self.logger.info("binding call graph with source code...")
        callstack_files = [
            f for f in Path(self.path_manager.bug_path).rglob("*.txt") if f.name == "calltrace.txt"
        ]
        call_graph = callstack_to_graph(callstack_files)
        self.logger.info(f"call graph loaded with {call_graph.number_of_nodes()} nodes and {call_graph.number_of_edges()} edges")

        method_node_map = {}
        for method_node in method_nodes:
            file_path = Path(method_node.metadata["file_path"])
            method_name = method_node.metadata["method_name"]
            class_name = file_path.name.split(".")[0]
            key = f"{class_name}:{method_name}"
            if key not in method_node_map:
                method_node_map[key] = []
            method_node_map[key].append(method_node)

        cg_to_mn_map = {}
        binded_method_nodes = []

        n_found = 0
        for nx_node, _ in call_graph.nodes(data=True):
            if "$" in nx_node.class_name: # may be a inner class
                last_idx = nx_node.class_name.rfind("$")
                while last_idx != -1:
                    query_key = f"{nx_node.class_name[:last_idx]}:{nx_node.method_name}"
                    if query_key in method_node_map:
                        for mn in method_node_map[query_key]:
                            if nx_node.start_line >= mn.metadata["start_line"] and nx_node.end_line <= mn.metadata["end_line"]:
                                nx_node.source = mn.text
                                nx_node.comment = mn.metadata["comment"]
                                cg_to_mn_map[nx_node] = mn.id_
                                binded_method_nodes.append(mn)
                                n_found += 1
                                break
                    last_idx = nx_node.class_name.rfind("$", 0, last_idx)
            else:
                query_key = f"{nx_node.class_name}:{nx_node.method_name}"
                if query_key in method_node_map:
                    for mn in method_node_map[query_key]:
                        if nx_node.start_line >= mn.metadata["start_line"] and nx_node.end_line <= mn.metadata["end_line"]:
                            nx_node.source = mn.text
                            nx_node.comment = mn.metadata["comment"]
                            cg_to_mn_map[nx_node] = mn.id_
                            binded_method_nodes.append(mn)
                            n_found += 1
                            break
        self.logger.info(f"{n_found} out of {len(call_graph.nodes)} nodes bounded with source code")
        return call_graph, binded_method_nodes,cg_to_mn_map

    def _get_loaded_classes(self):
        loaded_classes = set()
        for path in Path(self.path_manager.bug_path).rglob("*"):
            if path.name == "loaded_classes.txt":
                with open(path, "r") as f:
                    for line in f.readlines():
                        loaded_classes.add(line.strip().split(".")[-1])
        return loaded_classes

    def _get_method_nodes(self):
        # load method nodes from cache
        if os.path.exists(self.path_manager.method_nodes_file):
            with open(self.path_manager.method_nodes_file, "rb") as f:
                method_nodes = pickle.load(f)
            self.logger.info(f"Load {len(method_nodes)} method nodes from cache {self.path_manager.method_nodes_file}")
            return method_nodes

        # load file nodes from java files
        self.logger.info(f"Loading src java files from {self.src_path}")
        reader = SimpleDirectoryReader(
            input_dir=self.src_path,
            recursive=True,
            required_exts=[".java"],
            encoding="utf-8"
        )
        src_documents = reader.load_data(show_progress=True)

        self.logger.info(f"Loading test java files from {self.test_path}")
        test_reader = SimpleDirectoryReader(
            input_dir=self.test_path,
            recursive=True,
            required_exts=[".java"],
            encoding="utf-8"
        )
        test_documents = test_reader.load_data(show_progress=True)
        documents = src_documents + test_documents
        for doc in documents:
            doc.text = doc.text.replace("\r", "")
        self.logger.info(f"{len(documents)} java files loaded")

        # parse file nodes to method nodes
        self.logger.info(f"Loading method nodes")
        java_node_parser = JavaNodeParser.from_defaults()
        loaded_classes = self._get_loaded_classes()
        nodes = java_node_parser.get_nodes_from_documents(
            documents,
            loaded_classes,
            show_progress=True
        )
        self.logger.info(f"all {len(nodes)} nodes loaded")

        # filter method nodes
        method_nodes = [node for node in nodes if node.metadata["node_type"] == "method_node"]
        self.logger.info(f"{len(method_nodes)} method nodes loaded")

        # save method nodes to cache
        with open(self.path_manager.method_nodes_file, "wb") as f:
            pickle.dump(method_nodes, f)
        return method_nodes
