import argparse
import os
import pickle
import shutil
import sys
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from Evaluation.evaluate import evaluate_others
from functions.d4j import check_out, get_failed_tests, get_properties, run_all_tests
from functions.sbfl import parse_sbfl
from preprocess.index_builder import ProjectIndexBuilder
from Query.query import QueryGenerator
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
    parser.add_argument('--bugID', type=int, default=1,
                        help="Prompt of software")
    parser.add_argument('--subproj', type=str, required=False, default="",
                        help="The subproject of the project")
    parser.add_argument('--clear', type=bool, default=True,
                        help="If clear the checkout project")
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
    #          Query Generation
    # ----------------------------------------

    path_manager.logger.info("[Query Generation] start...")
    query_generator = QueryGenerator(path_manager)
    queries: List[str] = query_generator.generate(test_failure_obj)

    # ----------------------------------------
    #          SBFL results
    # ----------------------------------------

    sbfl_res = parse_sbfl(path_manager.sbfl_file)

    # ----------------------------------------
    #          Load Index
    # ----------------------------------------

    if os.path.exists(path_manager.retrieved_nodes_file):
        with open(path_manager.retrieved_nodes_file, 'rb') as f:
            result_nodes = pickle.load(f)
    else:
        path_manager.logger.info("[load data] start...")
        index_builder = ProjectIndexBuilder(path_manager)
        nodes = index_builder.build_nodes([sbfl_res])
        documents = [Document(page_content=node.text) for node in nodes]
        if documents == []:
            result_nodes = []
        else:
            retriever = BM25Retriever.from_documents(documents, k=path_manager.retrieve_top_n)
            
            # ----------------------------------------
            #          Retrieve
            # ----------------------------------------
            
            path_manager.logger.info("[Retrieve] start...")
            all_queries = " ".join(queries)
            result_nodes = retriever._get_relevant_documents(all_queries, run_manager=None)
            with open(path_manager.retrieved_nodes_file, 'wb') as f:
                pickle.dump(result_nodes, f)

    # ----------------------------------------
    #          Evaluate
    # ----------------------------------------
    
    evaluate_others(path_manager, result_nodes, test_failure_obj)
    
    if args.clear:
        if os.path.exists(path_manager.buggy_path):
            shutil.rmtree(path_manager.buggy_path)
        if os.path.exists(path_manager.fixed_path):
            shutil.rmtree(path_manager.fixed_path)

if __name__ == "__main__":
    main()