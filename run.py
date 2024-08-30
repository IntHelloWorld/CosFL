import argparse
import os
import pickle
import shutil
import sys
from typing import List

from Evaluation.evaluate import evaluate
from functions.d4j import check_out, get_failed_tests, get_properties, run_all_tests
from functions.sbfl import parse_sbfl
from preprocess.index_builder import ProjectIndexBuilder
from Query.query import QueryGenerator
from Rerank.reranker import Reranker
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
    parser.add_argument('--subproj', type=str, required=False, default="",
                        help="The subproject of the project")
    
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


    # ----------------------------------------
    #          Set Models
    # ----------------------------------------
    
    set_models(path_manager)
    
    # ----------------------------------------
    #          Check Mode
    # ----------------------------------------
    
    if path_manager.mode == "debug":
        run_debug(path_manager)
    elif path_manager.mode == "embed":
        run_embed(path_manager)
    elif path_manager.mode == "summary":
        run_summary(path_manager)
    elif path_manager.mode == "query":
        run_query(path_manager)
    else:
        raise ValueError("Invalid mode")
    
    if path_manager.clear:
        shutil.rmtree(path_manager.proj_tmp_path, ignore_errors=True)


def run_query(path_manager: PathManager):
    if path_manager.query_type != "normal":
        raise ValueError("Invalid mode")
    
    # get test failure object
    path_manager.logger.info("[get test failure object] start...")
    test_failure_obj = get_failed_tests(path_manager)
    
    # run all tests
    path_manager.logger.info("[run all tests] start...")
    run_all_tests(path_manager, test_failure_obj)
    
    path_manager.logger.info("[Query Generation] start...")
    query_generator = QueryGenerator(path_manager)
    _ = query_generator.generate(test_failure_obj)


def run_embed(path_manager: PathManager):
    path_manager.logger.info("[load data] start...")
    
    # ----------------------------------------
    #          SBFL results
    # ----------------------------------------

    sbfl_res = None
    if not path_manager.all_methods:
        sbfl_res = parse_sbfl(path_manager.sbfl_file)
    
    index_builder = ProjectIndexBuilder(path_manager)
    _ = index_builder.build_embeddings([sbfl_res])


def run_summary(path_manager: PathManager):
    path_manager.logger.info("[load data] start...")
    index_builder = ProjectIndexBuilder(path_manager)
    _ = index_builder.build_summary(all_methods=True)


def run_debug(path_manager: PathManager):
    
    # get test failure object
    path_manager.logger.info("[get test failure object] start...")
    test_failure_obj = get_failed_tests(path_manager)
    
    # run all tests
    path_manager.logger.info("[run all tests] start...")
    run_all_tests(path_manager, test_failure_obj)
    
    # ----------------------------------------
    #          Query Generation
    # ----------------------------------------

    path_manager.logger.info("[Query Generation] start...")
    query_generator = QueryGenerator(path_manager)
    if path_manager.query_type == "no_query":
        queries: List[str] = query_generator.generate_no_query(test_failure_obj)
    elif path_manager.query_type == "one_query":
        queries: List[str] = query_generator.generate_one_query(test_failure_obj)
    elif path_manager.query_type == "normal":
        queries: List[str] = query_generator.generate(test_failure_obj)
    elif path_manager.query_type == "causes":
        queries: List[str] = query_generator.generate_causes(test_failure_obj)
    else:
        raise ValueError(f"Invalid query type {path_manager.query_type}")
    
    # ----------------------------------------
    #             Retrieval
    # ----------------------------------------

    try:
        with open(path_manager.retrieved_nodes_file, "rb") as f:
            retrieved_nodes_list = pickle.load(f)
    except FileNotFoundError:
        # ----------------------------------------
        #          SBFL results
        # ----------------------------------------

        sbfl_res = None
        if path_manager.sbfl_formula:
            sbfl_res = parse_sbfl(path_manager.sbfl_file)

        # ----------------------------------------
        #          Load Index
        # ----------------------------------------

        path_manager.logger.info("[load data] start...")
        index_builder = ProjectIndexBuilder(path_manager)
        index = index_builder.build_index([sbfl_res])
        
        # ----------------------------------------
        #          Retrieve
        # ----------------------------------------
        
        path_manager.logger.info("[Retrieve] start...")
        retriever = index.as_retriever(similarity_top_k=path_manager.retrieve_top_n)
        retrieved_nodes_list = []
        for query in queries:
            nodes = retriever.retrieve(query)
            retrieved_nodes_list.append(nodes)
        with open(path_manager.retrieved_nodes_file, 'wb') as f:
            pickle.dump(retrieved_nodes_list, f)

    # ----------------------------------------
    #          Rerank
    # ----------------------------------------
    
    path_manager.logger.info("[Rerank] start...")
    reranker = Reranker(path_manager)
    result_nodes, reranked_nodes_list = reranker.rerank(retrieved_nodes_list, queries)

    # ----------------------------------------
    #          Evaluate
    # ----------------------------------------
    
    evaluate(path_manager, result_nodes, reranked_nodes_list, test_failure_obj)

if __name__ == "__main__":
    main()
