import os
import shutil
import sys
from pathlib import Path
from tabnanny import verbose
from typing import Dict, List

import chromadb
import more_itertools
from attr import has
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.indices.utils import embed_nodes
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME
from llama_index.vector_stores.chroma import ChromaVectorStore
from sympy import true

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from preprocess.code_extractors import CodeSummaryExtractor
from preprocess.node_parser import JavaNodeParser
from Utils.path_manager import PathManager


class ProjectIndexBuilder():
    
    def __init__(self, path_manager: PathManager) -> None:
        self.path_manager = path_manager
        self.doc_store_file = os.path.join(path_manager.stores_dir, DEFAULT_PERSIST_FNAME)
        self.vector_store_dir = os.path.join(path_manager.stores_dir, "chroma")
        self.bug_vector_store_dir = os.path.join(path_manager.cache_path, "chroma")
        
        src_path = os.path.join(path_manager.buggy_path, path_manager.src_prefix)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"{src_path} does not exist!")
        self.src_path = src_path
        self.class_names = self._get_all_loaded_classes()

    def _get_all_loaded_classes(self):
        class_names = set()
        for path in Path(self.path_manager.cache_path).rglob("*"):
            if path.name == "load.log":
                with open(path, "r") as f:
                    for line in f.readlines():
                        class_names.add(line.strip().split(".")[-1])
        return class_names
    
    def _load_documents(self):
        # load java files as documents
        self.path_manager.logger.info(f"[loading] Loading java files from {self.src_path}")
        reader = SimpleDirectoryReader(
            input_dir=self.src_path,
            recursive=True,
            required_exts=[".java"],
            encoding="utf-8"
        )
        documents = reader.load_data(show_progress=True)
        for doc in documents:
            doc.text = doc.text.replace("\r", "")
        self.path_manager.logger.info(f"[loading] {len(documents)} java files loaded")
        return documents
    
    def _load_nodes(self, documents, class_names):
        # parse documents to code nodes according to the AST
        self.path_manager.logger.info(f"[loading] Loading method nodes")
        java_node_parser = JavaNodeParser.from_defaults()
        nodes = java_node_parser.get_nodes_from_documents(documents, class_names, show_progress=True)
        self.path_manager.logger.info(f"[loading] {len(nodes)} nodes loaded")
        return nodes
    
    def _extract_summaries(self, nodes):
        # extract summaries for each code node
        extractor = CodeSummaryExtractor(
            language="java",
            num_workers=self.path_manager.summary_workers
        )
        nodes = extractor.process_nodes(nodes, show_progress=True)
        return nodes
    
    def _is_covered(self, node, sbfl_res: Dict[str, List[int]]):
        file_path = node.metadata["file_path"]
        start_line = node.metadata["start_line"]
        end_line = node.metadata["end_line"]
        prefix = self.path_manager.buggy_path + '/' + self.path_manager.src_prefix + '/'
        file_path = file_path.replace(prefix, "")
        class_name = file_path.split("/")[-1].split(".")[0]
        pkg_name = ".".join(file_path.split("/")[:-1])
        full_name = pkg_name + "$" + class_name
        
        if full_name in sbfl_res:
            for line_num in sbfl_res[full_name]:
                if start_line <= line_num <= end_line:
                    return True
        return False
    
    def _get_all_method_nodes(self, sbfl_res: Dict[str, List[int]]):
        # get documents and nodes
        documents = self._load_documents()
        all_nodes = self._load_nodes(documents, self.class_names)
        method_nodes_dict = {}
        num_methods = 0
        num_covered = 0
        for node in all_nodes:
            if node.metadata["node_type"] == "method_node":
                num_methods += 1
                if self._is_covered(node, sbfl_res):
                    num_covered += 1
                    if node.id_ not in method_nodes_dict:
                        method_nodes_dict[node.id_] = node
        self.path_manager.logger.info(f"[loading] {num_covered}/{num_methods} method nodes are covered")
        return method_nodes_dict
    
    def _summarize_nodes(self, method_nodes_dict):
        # init with cached doc store
        if os.path.exists(self.doc_store_file):
            self.path_manager.logger.info(f"[loading] Loading nodes from cache {self.doc_store_file}")
            doc_store = SimpleDocumentStore.from_persist_dir(self.path_manager.stores_dir)
        else:
            doc_store = SimpleDocumentStore()
        
        # nodes = nodes[:5]  # for testing

        # extract summaries for method nodes
        summarized_nodes = []
        no_summary_nodes = []
        for node_id in method_nodes_dict:
            if doc_store.document_exists(node_id):
                old_doc = doc_store.get_document(node_id)
                new_doc = method_nodes_dict[node_id]
                new_doc.metadata["summary"] = old_doc.metadata["summary"]
                doc_store.delete_document(node_id)
                summarized_nodes.append(new_doc)
            else:
                no_summary_nodes.append(method_nodes_dict[node_id])
        
        if no_summary_nodes:
            batches = list(more_itertools.chunked(no_summary_nodes, 50))
            for i, batch in enumerate(batches):
                self.path_manager.logger.info(f"[loading] Extracting Summaries for {len(no_summary_nodes)} code, chunk {i+1}/{len(batches)}")
                nodes = self._extract_summaries(batch)
                summarized_nodes.extend(nodes)

        doc_store.add_documents(summarized_nodes)
        doc_store.persist(self.doc_store_file)
        return summarized_nodes
    
    def _embed_nodes(self, summarized_nodes):
        db = chromadb.PersistentClient(path=self.vector_store_dir)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embedded_results = vector_store._collection.get(
            ids=[node.id_ for node in summarized_nodes],
            include=["embeddings"]
        )
        if embedded_results["ids"]:
            no_embeded_nodes = []
            embedding_dict = dict(zip(embedded_results["ids"], embedded_results["embeddings"]))
            for node in summarized_nodes:
                if node.id_ in embedding_dict:
                    node.embedding = embedding_dict[node.id_]
                else:
                    no_embeded_nodes.append(node)
        else:
            no_embeded_nodes = summarized_nodes
        
        if no_embeded_nodes:
            id_to_embed_map = embed_nodes(
                nodes=no_embeded_nodes,
                embed_model=Settings.embed_model,
                show_progress=True,
            )
            for node in no_embeded_nodes:
                node.embedding = id_to_embed_map[node.id_]
            vector_store.add(no_embeded_nodes)
        return summarized_nodes

    def build_index(self, sbfl_res: Dict[str, List[int]]):
        # load from cached index
        if os.path.exists(self.bug_vector_store_dir):
            db = chromadb.PersistentClient(path=self.bug_vector_store_dir)
            chroma_collection = db.get_or_create_collection("quickstart")
            bug_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(bug_vector_store)
            return index
        
        method_nodes_dict = self._get_all_method_nodes(sbfl_res)
        
        nodes = self._summarize_nodes(method_nodes_dict)
        
        nodes = self._embed_nodes(nodes)
        
        # build bug specific index
        db_2 = chromadb.PersistentClient(path=self.bug_vector_store_dir)
        chroma_collection_2 = db_2.get_or_create_collection("quickstart")
        bug_vector_store = ChromaVectorStore(chroma_collection=chroma_collection_2)
        storage_context = StorageContext.from_defaults(vector_store=bug_vector_store)
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True
        )
        return index