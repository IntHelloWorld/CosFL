import os
import sys
from argparse import Namespace
from typing import List

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from functions.d4j import get_failed_tests, get_properties
from projects import ALL_BUGS
from Query.query import QueryGenerator
from Utils.model import set_models
from Utils.path_manager import PathManager


def run_one_bug(config: str, version: str, project: str, bugID: int, subproj: str = ""):
    args = Namespace(
        config=config,
        version=version,
        project=project,
        bugID=str(bugID),
        subproj=subproj
    )

    # ----------------------------------------
    #          Init
    # ----------------------------------------

    path_manager = PathManager(args)
    path_manager.logger.info("*" * 100)
    path_manager.logger.info(f"Start generate embeddings for bug {args.version}-{args.project}-{args.bugID}")
    
    # get bug specific information
    path_manager.logger.info("[get bug properties] start...")
    get_properties(path_manager)
    
    # get test failure object
    path_manager.logger.info("[get test failure object] start...")
    test_failure_obj = get_failed_tests(path_manager)
    
    # ----------------------------------------
    #          Set Model
    # ----------------------------------------
    
    set_models(path_manager)

    # ----------------------------------------
    #          Query Generation
    # ----------------------------------------

    path_manager.logger.info("[Query Generation] start...")
    query_generator = QueryGenerator(path_manager)
    queries: List[str] = query_generator.generate(test_failure_obj)


def run_all_bugs(config_name: str):
    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            bugIDs = ALL_BUGS[version][proj][0]
            deprecatedIDs = ALL_BUGS[version][proj][1]
            subproj = ALL_BUGS[version][proj][2] if version == "GrowingBugs" else ""
            for bug_id in bugIDs:
                if bug_id in deprecatedIDs:
                    continue
                run_one_bug(config_name, version, proj, bug_id, subproj)

if __name__ == "__main__":
    # config = "QUERY_Q(qwen2-72b)"
    config = "QUERY_Q(llama3.1-405b)"
    # config = "QUERY_Q(gpt-4o)"
    # config = "QUERY_Q(gpt-3.5-turbo)"
    # config = "QUERY_Q(claude-3-5-sonnet)"
    # config = "QUERY_Q(claude-3-sonnet)"
    
    run_all_bugs(config)