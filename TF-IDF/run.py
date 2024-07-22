import argparse
import os
import pickle
import shutil
import sys

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document

from Evaluation.evaluate import evaluate_others
from functions.d4j import (
    check_out,
    get_failed_tests,
    get_properties,
    parse_sbfl,
    run_all_tests,
)
from preprocess.index_builder import ProjectIndexBuilder
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
    parser.add_argument('--bugID', type=int, default=1,
                        help="Prompt of software")
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
    #          Test Analysis
    # ----------------------------------------

    path_manager.logger.info("[Test Analysis] start...")
    test_analyzer = TestAnalyzer(path_manager)
    test_analyzer.analyze(test_failure_obj)

    # ----------------------------------------
    #          SBFL results
    # ----------------------------------------

    sbfl_res = parse_sbfl(path_manager)

    # ----------------------------------------
    #          Load Index
    # ----------------------------------------

    path_manager.logger.info("[load data] start...")
    index_builder = ProjectIndexBuilder(path_manager)
    nodes = index_builder.build_nodes(sbfl_res)
    documents = [Document(page_content=node.text) for node in nodes]
    retriever = TFIDFRetriever.from_documents(documents, k=path_manager.retrieve_top_n)
    
    # ----------------------------------------
    #          Retrieve
    # ----------------------------------------
    
    path_manager.logger.info("[Retrieve] start...")
    results = []
    for test_class in test_failure_obj.test_classes:
        for test_case in test_class.test_cases:
            result = retriever.invoke(
                "\n".join([
                    test_case.test_output,
                    test_case.stack_trace,
                    test_case.test_method.text
                ])
            )
            results.append(result)

    # ----------------------------------------
    #          Evaluate
    # ----------------------------------------
    
    evaluate_others(path_manager, results, test_failure_obj)
    
    if args.clear:
        shutil.rmtree(path_manager.buggy_path)
        shutil.rmtree(path_manager.fixed_path)

if __name__ == "__main__":
    main()
