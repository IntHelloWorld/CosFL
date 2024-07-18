import argparse
import os
import shutil
import sys


from functions.d4j import (
    check_out, get_properties
)
from preprocess.index_builder import ProjectIndexBuilder
from Utils.model import set_models, set_summary_model
from Utils.path_manager import PathManager

root = os.path.dirname(__file__)
sys.path.append(root)

def main():
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--config', type=str, default="openai_codellama-7B_jina",
                        help="Name of config, which is used to load configuration under Config/")
    parser.add_argument('--version', type=str, default="2.0.1",
                        help="Version of defects4j")
    parser.add_argument('--project', type=str, default="Chart",
                        help="Name of project, your debug result will be generated in DebugResult/d4jversion_project_bugID")
    parser.add_argument('--bugID', type=int, default=1,
                        help="Prompt of software")
    args = parser.parse_args()

    # ----------------------------------------
    #          Init
    # ----------------------------------------

    path_manager = PathManager(args)
    path_manager.logger.info("*" * 100)
    path_manager.logger.info(f"Build vector database for bug d4j{args.version}-{args.project}-{args.bugID}")
    
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
    shutil.rmtree(path_manager.cache_path)

if __name__ == "__main__":
    main()
