from argparse import Namespace
import os
import shutil
import sys

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from functions.d4j import check_out, get_properties
from preprocess.index_builder import ProjectIndexBuilder
from Utils.model import set_summary_model
from Utils.path_manager import PathManager
from projects import ALL_BUGS

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
    
    # ----------------------------------------
    #          Set Model
    # ----------------------------------------
    
    set_summary_model(path_manager)

    # ----------------------------------------
    #          Build Summary
    # ----------------------------------------

    path_manager.logger.info("[load data] start...")
    index_builder = ProjectIndexBuilder(path_manager)
    _ = index_builder.build_summary(all_methods=True)

    shutil.rmtree(path_manager.buggy_path)
    shutil.rmtree(path_manager.fixed_path)
    
    if clear:
        shutil.rmtree(os.path.join(path_manager.bug_path, "buggy"))
        shutil.rmtree(os.path.join(path_manager.bug_path, "fixed"))


def run_all_bugs(config_name: str):
    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            bugIDs = ALL_BUGS[version][proj][0]
            deprecatedIDs = ALL_BUGS[version][proj][1]
            sub_proj = ALL_BUGS[version][proj][2] if version == "GrowingBugs" else ""
            for bug_id in bugIDs:
                res_path = f"DebugResult/Summarization/{version}/{proj}/{proj}-{bug_id}"
                res_path = os.path.join(root, res_path)
                if bug_id in deprecatedIDs:
                    continue
                
                if os.path.exists(res_path):
                    continue
                
                try:
                    run_one_bug(config_name, version, proj, bug_id, True, sub_proj)
                except Exception as e:
                    shutil.rmtree(res_path)
                    raise e

if __name__ == "__main__":
    config = "Summarization"
    run_all_bugs(config)
