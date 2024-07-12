import argparse
import os
import pickle
import shutil
import sys

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.postprocessor.jinaai_rerank import JinaRerank

from Evaluation.evaluate import evaluate
from functions.d4j import (
    check_out,
    get_failed_tests,
    get_properties,
    parse_sbfl,
    run_all_tests,
)
from preprocess.index_builder import ProjectIndexBuilder
from Rerank.reranker import ChatReranker
from TestAnalysis.analyzer import TestAnalyzer
from Utils.model import set_models
from Utils.path_manager import PathManager

root = os.path.dirname(__file__)
sys.path.append(root)

def main():
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--config', type=str, default="LMStudio+jina+openai",
                        help="Name of config, which is used to load configuration under Config/")
    parser.add_argument('--version', type=str, default="2.0.1",
                        help="Version of defects4j")
    parser.add_argument('--project', type=str, default="Closure",
                        help="Name of project, your debug result will be generated in DebugResult/d4jversion_project_bugID")
    parser.add_argument('--bugID', type=int, default=4,
                        help="Prompt of software")
    args = parser.parse_args()

    # ----------------------------------------
    #          Init Test Failure
    # ----------------------------------------

    path_manager = PathManager(args)
    path_manager.logger.info("*" * 100)
    path_manager.logger.info(f"Start debugging bug d4j{args.version}-{args.project}-{args.bugID}")
    
    if os.path.exists(path_manager.res_file):
        path_manager.logger.info(f"d4j{args.version}-{args.project}-{args.bugID} already finished, skip!")
        return

    # check out the d4j project
    path_manager.logger.info("[checkout] start...")
    check_out(path_manager)
    
    # get bug specific information
    path_manager.logger.info("[get bug properties] start...")
    get_properties(path_manager)
    
    # get test failure object
    path_manager.logger.info("[get test failure object] start...")
    test_failure_obj = get_failed_tests(path_manager)
    
    # run all tests
    path_manager.logger.info("[run all tests] start...")
    run_all_tests(path_manager, test_failure_obj)


    # ----------------------------------------
    #          Set Models
    # ----------------------------------------
    
    set_models(path_manager)

    # ----------------------------------------
    #          Test Analysis
    # ----------------------------------------

    path_manager.logger.info("[Test Analysis] start...")
    test_analyzer = TestAnalyzer(path_manager)
    test_analyzer.analyze(test_failure_obj)

    nodes_file = os.path.join(path_manager.cache_path, "nodes.pkl")
    if os.path.exists(nodes_file):
        # ----------------------------------------
        #      Load Cached Retrieval Result
        # ----------------------------------------
        
        with open(nodes_file, "rb") as f:
            nodes_list = pickle.load(f)
    else:
        # ----------------------------------------
        #          SBFL results
        # ----------------------------------------

        sbfl_res = parse_sbfl(path_manager)

        # ----------------------------------------
        #          Load Index
        # ----------------------------------------

        path_manager.logger.info("[load data] start...")
        index_builder = ProjectIndexBuilder(path_manager)
        index = index_builder.build_index(sbfl_res)
        
        # doc_store = SimpleDocumentStore.from_persist_dir(path_manager.stores_dir)
        # import hashlib
        # sha256 = hashlib.sha256()
        # buggy_text = test_failure_obj.buggy_methods[0].text
        # sha256.update(buggy_text.encode("utf-8"))
        # unique_id = sha256.hexdigest()
        # node = doc_store.get_document(unique_id)
        
        # ----------------------------------------
        #          Retrieve
        # ----------------------------------------
        
        path_manager.logger.info("[Retrieve] start...")
        retriever = index.as_retriever(similarity_top_k=path_manager.retrieve_top_n)
        nodes_list = []
        for query in test_failure_obj.queries:
            nodes = retriever.retrieve(query)
            nodes_list.append(nodes)
        with open(nodes_file, 'wb') as f:
            pickle.dump(nodes_list, f)

    # ----------------------------------------
    #          Rerank
    # ----------------------------------------
    
    path_manager.logger.info("[Rerank] start...")

    path_manager.logger.info("[Rerank] run embedding-based rerank...")
    jina_rerank = JinaRerank(
        api_key=path_manager.rerank_api_key,
        model=path_manager.rerank_model,
        top_n=path_manager.rerank_top_n)
    
    chat_rerank_dict = {}
    extra_dict = {}
    for i, nodes in enumerate(nodes_list):
        reranked_nodes = jina_rerank.postprocess_nodes(
            nodes,
            query_str=test_failure_obj.queries[i]
        )
        for n in reranked_nodes[:path_manager.chat_rerank_top_n]:
            if n.id_ not in chat_rerank_dict:
                chat_rerank_dict[n.id_] = n
        for n in reranked_nodes[path_manager.chat_rerank_top_n:]:
            if n.id_ not in chat_rerank_dict:
                if n.id_ not in extra_dict:
                    extra_dict[n.id_] = n
    
    path_manager.logger.info("[Rerank] run chat-based rerank...")
    reranker = ChatReranker(path_manager)
    reranked_nodes = reranker.rerank(list(chat_rerank_dict.values()))
    extra_nodes = list(extra_dict.values())
    extra_nodes.sort(key=lambda x: x.score, reverse=True)
    
    result_nodes = reranked_nodes + extra_nodes

    # ----------------------------------------
    #          Evaluate
    # ----------------------------------------
    
    evaluate(path_manager, result_nodes, test_failure_obj)
    
    shutil.rmtree(path_manager.buggy_path)
    shutil.rmtree(path_manager.fixed_path)

if __name__ == "__main__":
    main()
