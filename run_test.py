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
from Storage.store import HybridStore
from Utils.model import set_models
from Utils.path_manager import PathManager

root = os.path.dirname(__file__)
sys.path.append(root)

def main():
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--config', type=str, default="TEST",
                        help="Name of config, which is used to load configuration under Config/")
    parser.add_argument('--version', type=str, default="d4j1.4.0",
                        help="Version of defects4j")
    parser.add_argument('--project', type=str, default="Chart",
                        help="Name of project, your debug result will be generated in DebugResult/d4jversion_project_bugID")
    parser.add_argument('--bugID', type=int, default=1,
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
    
    # if os.path.exists(path_manager.res_file):
    #     path_manager.logger.info(f"d4j{args.version}-{args.project}-{args.bugID} already finished, skip!")
    #     return

    # check out the d4j project
    path_manager.logger.info("checkout ...")
    check_out(path_manager)
    
    # get bug specific information
    path_manager.logger.info("get bug properties...")
    get_properties(path_manager)
    
    # run all tests
    path_manager.logger.info("run all tests...")
    test_failure_obj = get_failed_tests(path_manager)
    run_all_tests(path_manager, test_failure_obj)

    store = HybridStore(path_manager)

if __name__ == "__main__":
    main()
