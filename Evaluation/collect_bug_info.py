import json
import os
import shutil
from argparse import Namespace

from functions.d4j import check_out, get_failed_tests, get_properties, run_all_tests
from functions.my_types import TestFailure
from functions.sbfl import get_all_sbfl_res
from projects import ALL_BUGS
from Utils.path_manager import PathManager

root = os.path.dirname(__file__)

def compare(test_failure_obj: TestFailure, all_sbfl_res):
    for method in test_failure_obj.buggy_methods:
        class_full_name = method.class_full_name
        idx = class_full_name.rfind('.')
        assert idx != -1
        class_full_name = class_full_name[:idx] + "$" + class_full_name[idx+1:]
        start_line = method.loc[0][0]
        end_line = method.loc[1][0]
        
        for sbfl_res in all_sbfl_res:
            if class_full_name in sbfl_res:
                for line_num in sbfl_res[class_full_name]:
                    if start_line <= line_num <= end_line:
                        return True
        return False

def run_one_bug(config: str, version: str, project: str, bugID: int, clear: bool = True, sub_proj: str = ""):
    args = Namespace(
        config=config,
        version=version,
        project=project,
        bugID=str(bugID),
        sub_proj=sub_proj
    )

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
    #          SBFL results
    # ----------------------------------------

    all_sbfl_res = get_all_sbfl_res(path_manager)

    # ----------------------------------------
    #           Compare
    # ----------------------------------------
    if_covered = compare(test_failure_obj, all_sbfl_res)
    
    if clear:
        shutil.rmtree(os.path.join(path_manager.bug_path, "buggy"))
        shutil.rmtree(os.path.join(path_manager.bug_path, "fixed"))

    return if_covered

def run_all_bugs(config_name: str):
    res_file = os.path.join(root, "cover_res.json")
    if os.path.exists(res_file):
        with open(res_file, "r") as f:
            cover_res = json.load(f)
    else:
        cover_res = {}

    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            if proj not in cover_res:
                cover_res[proj] = {"covered": [], "uncovered": []}
            bugIDs = ALL_BUGS[version][proj][0]
            deprecatedIDs = ALL_BUGS[version][proj][1]
            sub_proj = ALL_BUGS[version][proj][2] if version == "GrowingBugs" else ""
            for bug_id in bugIDs:
                if bug_id in deprecatedIDs:
                    continue
                
                if bug_id in cover_res[proj]["covered"] or bug_id in cover_res[proj]["uncovered"]:
                    continue
                
                try:
                    if_covered = run_one_bug(config_name, version, proj, bug_id, True, sub_proj)
                    if if_covered:
                        cover_res[proj]["covered"].append(bug_id)
                    else:
                        cover_res[proj]["uncovered"].append(bug_id)
                except Exception:
                    raise Exception(f"Error in running d4j{version}-{proj}-{bug_id}")
                finally:
                    with open(res_file, "w") as f:
                        json.dump(cover_res, f, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    config_name = "BugInfoCollection"
    run_all_bugs(config_name)
