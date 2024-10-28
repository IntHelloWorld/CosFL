import copy
import os
import pickle
from pathlib import Path
from typing import Dict, List

import chromadb
import more_itertools
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore
from networkx import DiGraph
from tenacity import retry, stop_after_attempt, wait_fixed

from CallGraph.cg import (
    CGMethodNode,
    cg_summary_to_text,
    cluster_graph,
    load_callgraph,
    prepare_method_summarization_input,
    subgraph_to_text,
)
from CallGraph.prompt import (
    METHOD_CALL_SUBGRAPH_SUMMARIZATION_TEMPLATE,
    METHOD_SUMMARIZATION_EXAMPLE,
    METHOD_SUMMARIZATION_TEMPLATE,
    OUTPUT_EXAMPLE,
)
from Storage.node_parser import JavaNodeParser
from Storage.node_utils import default_id_func, get_node_text_for_embedding
from Utils.async_utils import asyncio_run, run_jobs_with_rate_limit
from Utils.model import parse_llm_output
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
            doc_store = SimpleDocumentStore.from_persist_dir(self.path_manager.doc_stores_dir)
        else:
            doc_store = SimpleDocumentStore()
        db = chromadb.PersistentClient(path=self.path_manager.vector_stores_dir)
        chroma_collection = db.get_or_create_collection(self.path_manager.bug_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        self.doc_store = doc_store
        self.vector_store = vector_store
        self.use_context = self.path_manager.config.use_context

        raw_method_nodes = self.get_raw_method_nodes()
        call_graph, binded_method_nodes, cg_to_mn_map = self.bind_call_graph(raw_method_nodes)
        method_nodes = self.get_method_nodes_from_docstore(binded_method_nodes)

        # these steps also update the method nodes
        if self.use_context:
            sub_graphs = self.cluster_call_graph(call_graph)
            context_nodes = self.subgraphs_summarization(sub_graphs, method_nodes, cg_to_mn_map)
            desc_nodes = self.methods_summarization(method_nodes)
        else:
            context_nodes = []
            desc_nodes = self.methods_summarization_no_context(method_nodes)

        self.embedded_nodes = self.get_node_embeddings(context_nodes, method_nodes, desc_nodes)

    def cluster_call_graph(self, call_graph):
        """
        cluster the call graph
        """
        self.logger.info("clustering call graph...")
        sub_graphs = cluster_graph(
            call_graph,
            min_size=self.path_manager.config.hyper.min_module_size,
            max_size=self.path_manager.config.hyper.max_module_size)
        self.logger.info(
            f"call graph clustered into {len(sub_graphs)} subgraphs: "
            f"{[len(subgraph.nodes) for subgraph in sub_graphs]}"
        )
        return sub_graphs

    def build_context_nodes(
        self,
        subgraphs: List[DiGraph],
        responses: List[Dict],
        binded_method_nodes: List[TextNode],
        cg_to_mn_map: Dict[CGMethodNode, str]
    ) -> List[TextNode]:
        """Build nodes from LLM call graph summarization."""

        mn_dict = {mn.id_: mn for mn in binded_method_nodes}
        context_nodes_dict: Dict[str, TextNode] = {}
        updated_mn_nodes: List[TextNode] = []
        for i in range(len(subgraphs)):
            ctxt_text = cg_summary_to_text(responses[i])
            node_id = default_id_func(responses[i]["title"])
            node = TextNode(id_= node_id, text=ctxt_text)
            node.metadata.update(responses[i])
            node.metadata.update(
                {
                    "node_type": "context_node",
                }
            )
            context_nodes_dict[node_id] = node # this prevent the duplicated context nodes

            # update method node metadata
            for nx_node in subgraphs[i].nodes:
                if nx_node in cg_to_mn_map:
                    mn = mn_dict[cg_to_mn_map[nx_node]]
                    mn.metadata.update(
                        {
                            "ctxt_node_id": node_id,
                        }
                    )
                    updated_mn_nodes.append(mn)

        # the add_documents will override the existing nodes with the same id
        context_nodes = list(context_nodes_dict.values())
        self.doc_store.add_documents(updated_mn_nodes)
        self.doc_store.add_documents(context_nodes)
        self.doc_store.persist(self.path_manager.doc_store_file)
        return context_nodes

    def subgraphs_summarization(
        self,
        subgraphs: List[DiGraph],
        binded_method_nodes: List[TextNode],
        cg_to_mn_map: Dict[CGMethodNode, str]
    ) -> List[TextNode]:
        """
        summarize the subgraphs to context nodes
        """
        context_nodes_dict = {}
        todo_subgraphs = []
        unbinded_subgraphs = []
        for subgraph in subgraphs:
            subgraph_ok = True
            ctxt_node_id = None
            for node in subgraph.nodes:
                node_ok = False
                if node not in cg_to_mn_map:  # unbinded call graph node
                    continue
                mn_id = cg_to_mn_map[node]
                if self.doc_store.document_exists(mn_id):  # check method node
                    mn = self.doc_store.get_node(mn_id)
                    if "ctxt_node_id" in mn.metadata:
                        ctxt_node_id = mn.metadata["ctxt_node_id"]
                        if self.doc_store.document_exists(ctxt_node_id):  # check context node
                            node_ok = True
                if not node_ok:
                    subgraph_ok = False
                    break
            if subgraph_ok is False:
                todo_subgraphs.append(subgraph)
            elif subgraph_ok is True and ctxt_node_id is None:
                # this means all nodes in the subgraph are not binded with method nodes
                # we skip these subgraphs since no source code available
                unbinded_subgraphs.append(subgraph)
            else:
                context_nodes_dict[ctxt_node_id] = self.doc_store.get_node(ctxt_node_id)
        context_nodes = list(context_nodes_dict.values())
        self.logger.info(f"found {len(context_nodes)} subgraphs already summarized")
        self.logger.info(f"found {len(unbinded_subgraphs)} subgraphs unbinded with source code")
        
        if len(todo_subgraphs) == 0:
            return context_nodes

        self.logger.info(f"summarizing {len(todo_subgraphs)} contexts...")
        responses = asyncio_run(self._asubgraphs_summarization(todo_subgraphs))
        new_context_nodes = self.build_context_nodes(
            todo_subgraphs,
            responses,
            binded_method_nodes,
            cg_to_mn_map
        )
        context_nodes.extend(new_context_nodes)
        return context_nodes

    async def _asubgraphs_summarization(self, subgraphs: List[DiGraph]):
        jobs = []

        for subgraph in subgraphs:
            jobs.append(self._asubgraph_summarization(subgraph))

        responses = await run_jobs_with_rate_limit(
            jobs,
            limit=self.path_manager.config.models.summary.rate_limit,
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
        response = await Settings.llm.achat(messages)
        json_res = parse_llm_output(response.message.content)
        # FIXME: for testing
        # json_res = OUTPUT_EXAMPLE
        # json_res["title"] = input_text
        return json_res

    def bind_call_graph(self, method_nodes):
        """
        bind call graph with source code
        """

        self.logger.info("binding call graph with source code...")
        callstack_files = list(Path(self.path_manager.bug_path).rglob("callgraph.graphml"))
        call_graph = load_callgraph(callstack_files)
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
        binded_method_nodes_dict = {}  # use dict to prevent duplicated nodes

        n_found = 0
        unbinded_nodes = []
        for nx_node, _ in call_graph.nodes(data=True):
            found = False
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
                                binded_method_nodes_dict[mn.id_] = mn
                                n_found += 1
                                found = True
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
                            binded_method_nodes_dict[mn.id_] = mn
                            n_found += 1
                            found = True
                            break
            if not found:
                unbinded_nodes.append(nx_node)

        binded_method_nodes = list(binded_method_nodes_dict.values())
        # save method nodes to cache
        with open(self.path_manager.method_nodes_file, "wb") as f:
            pickle.dump(binded_method_nodes, f)

        self.logger.info(f"{n_found} out of {len(call_graph.nodes)} nodes binded with source code")
        return call_graph, binded_method_nodes, cg_to_mn_map

    def _get_loaded_classes(self):
        loaded_classes = set()
        for path in Path(self.path_manager.bug_path).rglob("*"):
            if path.name == "loaded_classes.txt":
                with open(path, "r") as f:
                    for line in f.readlines():
                        loaded_classes.add(line.strip().split(".")[-1])
        return loaded_classes

    def get_method_nodes_from_docstore(self, raw_method_nodes):
        """
        get method nodes from docstore
        """
        method_nodes = []
        n_found = 0
        for raw_method_node in raw_method_nodes:
            if self.doc_store.document_exists(raw_method_node.id_):
                method_nodes.append(self.doc_store.get_node(raw_method_node.id_))
                n_found += 1
            else:
                method_nodes.append(raw_method_node)
        self.logger.info(f"{n_found} out of {len(raw_method_nodes)} method nodes found in docstore")
        
        return method_nodes

    def get_raw_method_nodes(self):
        """
        load raw method nodes
        """
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
        java_node_parser = JavaNodeParser.from_defaults()
        loaded_classes = self._get_loaded_classes()
        
        self.logger.info(f"Loading src method nodes...")
        src_method_nodes = java_node_parser.get_nodes_from_documents(
            src_documents,
            loaded_classes,
            show_progress=True
        )
        src_method_nodes = [node for node in src_method_nodes if node.metadata["node_type"] == "method_node"]
        for node in src_method_nodes:
            node.metadata["is_test_method"] = False
        self.logger.info(f"{len(src_method_nodes)} src method nodes loaded")
        
        self.logger.info(f"Loading test method nodes...")
        test_method_nodes = java_node_parser.get_nodes_from_documents(
            test_documents,
            loaded_classes,
            show_progress=True
        )
        test_method_nodes = [node for node in test_method_nodes if node.metadata["node_type"] == "method_node"]
        for node in test_method_nodes:
            node.metadata["is_test_method"] = True
        self.logger.info(f"{len(test_method_nodes)} test method nodes loaded")

        method_nodes = src_method_nodes + test_method_nodes
        return method_nodes

    def methods_summarization(self, method_nodes: List[TextNode]):
        """
        use the sub-call graph report to summarize the method nodes
        """
        desc_nodes_dict = {}
        todo_methods = []
        for node in method_nodes:
            if "functionality" not in node.metadata or "desc_node_ids" not in node.metadata:
                todo_methods.append(node)
            else:
                cached_desc_nodes = self.doc_store.get_nodes(node.metadata["desc_node_ids"])
                for desc_node in cached_desc_nodes:  # this prevent the duplicated desc nodes
                    desc_nodes_dict[desc_node.id_] = desc_node
        self.logger.info(f"found {len(method_nodes) - len(todo_methods)} methods already summarized")
        desc_nodes = list(desc_nodes_dict.values())

        if len(todo_methods) == 0:
            return desc_nodes

        self.logger.info(f"summarizing {len(todo_methods)} methods...")
        method_contexts = []
        for mn in todo_methods:
            if "ctxt_node_id" in mn.metadata:
                method_contexts.append(self.doc_store.get_node(mn.metadata["ctxt_node_id"]))
            else:
                method_contexts.append(None)
        responses = asyncio_run(self._amethods_summarization(todo_methods, method_contexts))
        new_desc_nodes = self.build_description_nodes(
            todo_methods,
            responses
        )
        desc_nodes.extend(new_desc_nodes)
        self.logger.info(f"get {len(desc_nodes)} description nodes")
        return desc_nodes
    
    
    def methods_summarization_no_context(self, method_nodes: List[TextNode]):
        """
        NOT use the sub-call graph report to summarize the method nodes
        """
        desc_nodes_dict = {}
        todo_methods = []
        for node in method_nodes:
            if "functionality_NC" not in node.metadata or "desc_node_ids_NC" not in node.metadata:
                todo_methods.append(node)
            else:
                cached_desc_nodes = self.doc_store.get_nodes(node.metadata["desc_node_ids_NC"])
                for desc_node in cached_desc_nodes:  # this prevent the duplicated desc nodes
                    desc_nodes_dict[desc_node.id_] = desc_node
        self.logger.info(f"found {len(method_nodes) - len(todo_methods)} methods already summarized")
        desc_nodes = list(desc_nodes_dict.values())

        if len(todo_methods) == 0:
            return desc_nodes

        self.logger.info(f"summarizing {len(todo_methods)} methods without context...")
        method_contexts = [None] * len(todo_methods)
        responses = asyncio_run(self._amethods_summarization(todo_methods, method_contexts))
        new_desc_nodes = self.build_description_nodes(
            todo_methods,
            responses,
            no_context=True
        )
        desc_nodes.extend(new_desc_nodes)
        self.logger.info(f"get {len(desc_nodes)} description nodes")
        return desc_nodes

    async def _amethods_summarization(self, method_nodes, method_contexts):
        jobs = []
        for i in range(len(method_nodes)):
            jobs.append(self._amethod_summarization(method_nodes[i], method_contexts[i]))
        responses = await run_jobs_with_rate_limit(
            jobs,
            limit=self.path_manager.config.models.summary.rate_limit,
            desc="Method Summarization",
            show_progress=True
        )
        return responses

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    async def _amethod_summarization(self, method_node, method_context):
        input_text = prepare_method_summarization_input(method_node, method_context)
        messages = METHOD_SUMMARIZATION_TEMPLATE.format_messages(
            input_text=input_text
        )
        response = await Settings.llm.achat(messages)
        json_res = parse_llm_output(response.message.content)
        # print(json_res)
        # FIXME: for testing
        # json_res = METHOD_SUMMARIZATION_EXAMPLE
        return json_res

    def build_description_nodes(self, method_nodes, responses, no_context=False):
        """
        build description nodes from method summarization responses
        """

        desc_nodes_dict: Dict[str, TextNode] = {}
        for i in range(len(responses)):
            desc_node_ids = []
            for desc in responses[i]["description"]:
                node_id = default_id_func(desc)
                node = TextNode(id_=node_id, text=desc)
                node.metadata.update(
                    {
                        "node_type": "desc_node"
                    }
                )
                desc_nodes_dict[node_id] = node # this prevent the duplicated desc nodes
                desc_node_ids.append(node_id)
            # update method node metadata
            if no_context:
                method_nodes[i].metadata.update(
                    {
                        "functionality_NC": responses[i]["functionality"],
                        "desc_node_ids_NC": desc_node_ids
                    }
                )
            else:
                method_nodes[i].metadata.update(
                    {
                        "functionality": responses[i]["functionality"],
                        "desc_node_ids": desc_node_ids
                    }
                )

        # the add_documents will override the existing nodes with the same id
        desc_nodes = list(desc_nodes_dict.values())
        self.doc_store.add_documents(desc_nodes)
        self.doc_store.add_documents(method_nodes)
        self.doc_store.persist(self.path_manager.doc_store_file)
        return desc_nodes


    def get_node_embeddings(self, context_nodes, method_nodes, desc_nodes):
        """
        get node embeddings
        """
        self.logger.info(f"get node embeddings...")
        all_nodes = context_nodes + method_nodes + desc_nodes
        
        embedded_results = self.vector_store._collection.get(
            ids=[node.id_ for node in all_nodes],
            include=["embeddings"]
        )
        if embedded_results["ids"]:
            no_embeded_nodes = []
            embedding_dict = dict(zip(embedded_results["ids"], embedded_results["embeddings"]))
            for node in all_nodes:
                if node.id_ in embedding_dict:
                    node.embedding = embedding_dict[node.id_]
                else:
                    no_embeded_nodes.append(node)
        else:
            no_embeded_nodes = all_nodes
        self.logger.info(f"found {len(all_nodes) - len(no_embeded_nodes)} nodes already embedded")

        if len(no_embeded_nodes) == 0:
            return all_nodes

        batches = list(
            # Save embeddings periodically to avoid losing them in case of a crash
            more_itertools.chunked(
                no_embeded_nodes, self.path_manager.config.models.embed.batch_size * 10
            )
        )
        self.logger.info(f"Generating Embedding for {len(no_embeded_nodes)} nodes in {len(batches)} batches")
        for batch in batches:
            texts_to_embed = [get_node_text_for_embedding(node, self.use_context) for node in batch]
            new_embeddings = Settings.embed_model.get_text_embedding_batch(
                texts_to_embed,
                show_progress=True
            )

            for i, node in enumerate(batch):
                node.embedding = new_embeddings[i]
            # chromadb only support flat metadata
            copied_batch = copy.deepcopy(batch)
            for n in copied_batch:
                keys_to_remove = []
                for k, v in n.metadata.items():
                    if isinstance(v, list):
                        keys_to_remove.append(k)
                for k in keys_to_remove:
                    del n.metadata[k]
            self.vector_store.add(copied_batch)
        return all_nodes
