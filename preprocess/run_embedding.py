import os
import shutil
import sys
from argparse import Namespace

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

from functions.d4j import get_properties
from preprocess.index_builder import ProjectIndexBuilder
from projects import ALL_BUGS
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
    
    
    # ----------------------------------------
    #          Set Model
    # ----------------------------------------
    
    set_models(path_manager)

    # ----------------------------------------
    #          Build Embedding
    # ----------------------------------------

    path_manager.logger.info("[load data] start...")
    index_builder = ProjectIndexBuilder(path_manager)
    try:
        _ = index_builder.build_embeddings()
    except Exception as e:
        raise e


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
                
                # only run once for a project
                break

if __name__ == "__main__":
    # config = "EMBED_S(codellama-7B)_E(Jina)"
    # config = "EMBED_S(codellama-13B)_E(Jina)"
    # config = "EMBED_S(codellama-34B)_E(Jina)"
    # config = "EMBED_S(codegemma-7B)_E(Jina)"
    # config = "EMBED_S(codeqwen-1_5-7B)_E(Jina)"
    # config = "EMBED_S(Codestral-22B)_E(Jina)"
    # config = "EMBED_S(DeepSeek-Coder-V2-Lite)_E(Jina)"
    # config = "EMBED_S(starcoder2-15B)_E(Jina)"
    
    # config = "EMBED_S(codeqwen-1_5-7B)_E(Voyage)"
    
    config = "EMBED_S(codeqwen-1_5-7B)_E(Cohere)"
    
    # config = "EMBED_S(codeqwen-1_5-7B)_E(OpenAI)"
    
    run_all_bugs(config)
