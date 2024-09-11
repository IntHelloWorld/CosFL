import os
import pickle
from pathlib import Path
from typing import Dict, List

import chromadb
import more_itertools
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.chroma import ChromaVectorStore

from functions.sbfl import get_all_sbfl_res
from preprocess.code_extractors import CodeSummaryExtractor
from preprocess.node_parser import JavaNodeParser
from Utils.path_manager import PathManager


class ProjectIndexBuilder():
    
    def __init__(self, path_manager: PathManager) -> None:
        self.path_manager = path_manager
        
        self.src_path = os.path.join(path_manager.buggy_path, path_manager.src_prefix)
        self.class_names = self._get_all_loaded_classes()

    def _get_all_loaded_classes(self):
        class_names = set()
        for path in Path(self.path_manager.bug_path).rglob("*"):
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
    
    def _load_nodes(self, documents, all_methods=False):
        # parse documents to code nodes according to the AST
        self.path_manager.logger.info(f"[loading] Loading method nodes")
        java_node_parser = JavaNodeParser.from_defaults()
        nodes = java_node_parser.get_nodes_from_documents(
            documents,
            self.class_names,
            show_progress=True,
            all_methods=all_methods)
        self.path_manager.logger.info(f"[loading] all {len(nodes)} nodes loaded")
        
        nodes = [node for node in nodes if node.metadata["node_type"] == "method_node"]
        self.path_manager.logger.info(f"[loading] {len(nodes)} method nodes loaded")
        return nodes
    
    def _get_method_nodes(self, all_methods=False):
        # get method nodes from java files
        if not all_methods:
            if os.path.exists(self.path_manager.method_nodes_file):
                self.path_manager.logger.info(f"[loading] Loading method nodes from cache {self.path_manager.method_nodes_file}")
                with open(self.path_manager.method_nodes_file, "rb") as f:
                    method_nodes = pickle.load(f)
                return method_nodes

        documents = self._load_documents()
        method_nodes = self._load_nodes(documents, all_methods)
        if not all_methods:
            with open(self.path_manager.method_nodes_file, "wb") as f:
                pickle.dump(method_nodes, f)
        return method_nodes
    
    def _extract_summaries(self, nodes):
        # extract summaries for each code node
        extractor = CodeSummaryExtractor(
            language="java",
            num_workers=self.path_manager.config["summary"]["summary_workers"]
        )
        nodes = extractor.process_nodes(nodes, show_progress=True)
        return nodes
    
    def _any_covered(self, node, sbfl_reses):
        for res in sbfl_reses:
            if self._is_covered(node, res):
                return True
        return False
    
    def _is_covered(self, node, sbfl_res: Dict[str, List[int]]):
        file_path = node.metadata["file_path"]
        start_line = node.metadata["start_line"]
        end_line = node.metadata["end_line"]
        file_name = file_path.split("/")[-1]
        class_name = file_name.split(".")[0]
        
        if class_name in sbfl_res:
            for line_num in sbfl_res[class_name]:
                if start_line <= line_num <= end_line:
                    return True
        return False
    
    def _filter_nodes(self, nodes, sbfl_res_list, all_methods):
        method_nodes_dict = {}
        num_methods = 0
        num_covered = 0
        for node in nodes:
            num_methods += 1
            
            if not all_methods:
                if not self._any_covered(node, sbfl_res_list):
                    continue
            num_covered += 1
            
            if node.id_ not in method_nodes_dict:
                method_nodes_dict[node.id_] = node
        self.path_manager.logger.info(f"[loading] {num_covered}/{num_methods} method nodes are covered")
        return method_nodes_dict
    
    def _summarize_nodes(self, method_nodes_dict):
        # init with cached doc store
        if os.path.exists(self.path_manager.doc_store_file):
            self.path_manager.logger.info(f"[loading] Loading nodes from cache {self.path_manager.doc_store_file}")
            doc_store = SimpleDocumentStore.from_persist_dir(self.path_manager.stores_dir)
        else:
            doc_store = SimpleDocumentStore()
        
        # for testing
        # method_nodes_dict = dict(list(method_nodes_dict.items())[:5])

        # keep the already summarized nodes
        already_summarized_nodes = []
        no_summary_nodes = []
        for node_id in method_nodes_dict:
            this_node = method_nodes_dict[node_id]
            if doc_store.document_exists(node_id):
                cached_node = doc_store.get_node(node_id)
                this_node.metadata["summary"] = cached_node.metadata["summary"]
                already_summarized_nodes.append(this_node)
            else:
                no_summary_nodes.append(this_node)
        
        # extract summaries for the rest nodes
        new_summarized_nodes = []
        if no_summary_nodes:
            batches = list(more_itertools.chunked(no_summary_nodes, 50))
            for i, batch in enumerate(batches):
                self.path_manager.logger.info(f"[loading] Extracting Summaries for {len(no_summary_nodes)} code, chunk {i+1}/{len(batches)}")
                batch_summarized_nodes = self._extract_summaries(batch)
                doc_store.add_documents(batch_summarized_nodes)
                doc_store.persist(self.path_manager.doc_store_file)
                new_summarized_nodes.extend(batch_summarized_nodes)

        return already_summarized_nodes + new_summarized_nodes
    
    def _read_vector_store(self, nodes):
        """This function only read the embeddings from the vector store"""
        self.path_manager.logger.info(f"[loading] get node embeddings......")
        db = chromadb.PersistentClient(path=self.path_manager.vector_store_dir)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embedded_results = vector_store._collection.get(
            ids=[node.id_ for node in nodes],
            include=["embeddings"]
        )
        if embedded_results["ids"]:
            no_embeded_nodes = []
            embedding_dict = dict(zip(embedded_results["ids"], embedded_results["embeddings"]))
            for node in nodes:
                if node.id_ in embedding_dict:
                    node.embedding = embedding_dict[node.id_]
                else:
                    no_embeded_nodes.append(node)
        else:
            no_embeded_nodes = nodes
        return nodes, no_embeded_nodes
    
    def _embed_nodes(self, summarized_nodes):
        self.path_manager.logger.info(f"[loading] get node embeddings......")
        db = chromadb.PersistentClient(path=self.path_manager.vector_store_dir)
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
            batches = list(more_itertools.chunked(
                no_embeded_nodes,
                self.path_manager.config["embed"]["embed_batch_size"] * 10)
            )
            print(f"[loading] Generating Embedding for {len(no_embeded_nodes)} nodes in {len(batches)} batches")
            for batch in batches:
                texts_to_embed = [self._get_node_text(node) for node in batch]
                new_embeddings = Settings.embed_model.get_text_embedding_batch(
                    texts_to_embed, show_progress=True
                )

                for i, node in enumerate(batch):
                    node.embedding = new_embeddings[i]
                vector_store.add(batch)
        return summarized_nodes
    
    def _get_node_text(self, node):
        node_text = None
        if self.path_manager.embed_text == "summary":
            node_text = node.metadata["summary"]
        elif self.path_manager.embed_text == "code":
            node_text = node.text
        else:
            raise ValueError(f"Unknown embed text type: {self.path_manager.embed_text}")
        
        estimate_tokens = len(node_text) / 4
        if estimate_tokens > 8000:
            print(f"Node text unexpected long: {node_text}")
            node_text = node_text[:8000]
        return node_text
    
    def build_nodes(self, sbfl_res_list, all_methods=False):
        all_nodes = self._get_method_nodes(all_methods)
        method_nodes_dict = self._filter_nodes(all_nodes, sbfl_res_list, all_methods)
        return list(method_nodes_dict.values())

    def build_index(self, sbfl_res_list):
        """This method only read from document store and vector store"""
        self.path_manager.logger.info(f"[loading] Loading nodes from cache {self.path_manager.doc_store_file}")
        if not os.path.exists(self.path_manager.doc_store_file):
            raise FileNotFoundError(f"Document store {self.path_manager.doc_store_file} not found")
        doc_store = SimpleDocumentStore.from_persist_dir(self.path_manager.stores_dir)
        
        # Integrity Check
        summarized_nodes = []
        project_nodes = self._get_method_nodes(self.path_manager.all_methods)
        for project_node in project_nodes:
            summarized_nodes.append(doc_store.get_node(project_node.id_))
        
        # filter nodes based on coverage
        method_nodes_dict = self._filter_nodes(summarized_nodes, sbfl_res_list, self.path_manager.all_methods)
        nodes = list(method_nodes_dict.values())

        # load embeddings
        self.path_manager.logger.info(f"[loading] Loading embeddings from cache {self.path_manager.vector_store_dir}")
        nodes, no_embedded_nodes = self._read_vector_store(nodes)
        assert len(no_embedded_nodes) == 0, f"Nodes without embeddings: {len(no_embedded_nodes)}"
        
        index = VectorStoreIndex(nodes, show_progress=True)
        return index
    
    def build_summary(self, all_methods=False):
        if not all_methods:
            # get documents and nodes based on coverage
            sbfl_res_list = get_all_sbfl_res(self.path_manager)
        else:
            sbfl_res_list = []
        all_nodes = self._get_method_nodes(all_methods)
        method_nodes_dict = self._filter_nodes(all_nodes, sbfl_res_list, all_methods)
        nodes = self._summarize_nodes(method_nodes_dict)
    
    def build_embeddings(self, sbfl_res_list):
        """Build embeddings for all nodes, make sure the nodes are already summarized"""
        # init with cached doc store
        assert os.path.exists(self.path_manager.doc_store_file)
        self.path_manager.logger.info(f"[loading] Loading nodes from document store {self.path_manager.doc_store_file}")
        doc_store = SimpleDocumentStore.from_persist_dir(self.path_manager.stores_dir)
        nodes_dict = doc_store.docs
        nodes = list(nodes_dict.values())
        
        # filter nodes based on coverage
        method_nodes_dict = self._filter_nodes(nodes, sbfl_res_list, self.path_manager.all_methods)
        nodes = list(method_nodes_dict.values())
        
        # embed nodes and save to central vector store
        _ = self._embed_nodes(nodes)
