import json
import logging
import logging.config
import os
import sys
from pathlib import Path
from time import time

from fastapi import dependencies

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

log_config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(levelname)s - %(asctime)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'DEBUG',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'level': 'DEBUG',
            'filename': '',
            'mode': 'w',
            'encoding': 'utf-8',
        }
    },
    'loggers': {
        'default': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG'
        },
    }
}

class PathManager():
    
    def __init__(self, args):
        self.root_path = Path(__file__).resolve().parents[1].as_posix()
        self.config_file = os.path.join(self.root_path, "Config", args.config, "config.json")
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # LLMs
        if "llms" in self.config:
            llms = self.config["llms"]
            self.rerank_series = llms["rerank_series"]
            self.rerank_model = llms["rerank_model"]
            self.rerank_api_key = llms["rerank_api_key"]
            self.rerank_base_url = llms["rerank_base_url"]
            
            self.embed_series = llms["embed_series"]
            self.embed_model = llms["embed_model"]
            self.embed_api_key = llms["embed_api_key"]
            self.embed_base_url = llms["embed_base_url"]
            
            self.reasoning_series = llms["reasoning_series"]
            self.reasoning_model = llms["reasoning_model"]
            self.reasoning_api_key = llms["reasoning_api_key"]
            self.reasoning_base_url = llms["reasoning_base_url"]
        
        if "summary" in self.config:
            summary = self.config["summary"]
            self.summary_series = summary["summary_series"]
            self.summary_model = summary["summary_model"]
            self.summary_api_key = summary["summary_api_key"]
            self.summary_base_url = summary["summary_base_url"]
            self.summary_workers = summary["summary_workers"]
            
            # stores_dir
            self.index_id = f"{args.project}-{args.bugID}"
            self.stores_dir = os.path.join(self.root_path, "Stores", self.summary_model, args.project)
            if not os.path.exists(self.stores_dir):
                os.makedirs(self.stores_dir, exist_ok=True)
        
        if "dependencies" in self.config:
            dependencies = self.config["dependencies"]
            # dependencies
            self.agent_lib = dependencies["agent_lib"]
            self.tree_sitter_lib = dependencies["tree_sitter_lib"]
            self.D4J_exec = dependencies["D4J_exec"]
            self.GB_exec = dependencies["GB_exec"]
        
        if "hyper" in self.config:
            hyper = self.config["hyper"]
            # retrieve configurations
            self.retrieve_top_n = hyper["retrieve_top_n"]
            self.rerank_top_n = hyper["rerank_top_n"]
            self.chat_rerank_top_n = hyper["chat_rerank_top_n"]
            # sbfl
            self.sbfl_formula = hyper["sbfl_formula"]
            self.sbfl_file = os.path.join(
                self.root_path,
                "SBFL",
                "results",
                args.project,
                args.bug_id,
                f"{self.sbfl_formula}.ranking.csv"
            )
        
        # bug info
        self.version = args.version
        if self.version == "GrowingBugs":
            self.bug_exec = self.GB_exec
        else:
            self.bug_exec = self.D4J_exec
        self.project = args.project
        self.sub_proj = args.sub_proj
        self.bug_id = args.bugID
        
        # global paths/files
        self.output_path = os.path.join(self.root_path, "DebugResult")
        self.res_path = os.path.join(
            self.output_path,
            args.config,
            f"{args.version}",
            args.project,
            f"{args.project}-{args.bugID}")
        self.cache_path = os.path.join(self.res_path, "cache")
        self.res_file = os.path.join(self.res_path, "result.json")
        self.projects_path = os.path.join(self.root_path, "Projects")
        self.bug_path = os.path.join(self.projects_path, args.project, args.bugID)
        self.test_failure_file = os.path.join(self.bug_path, "test_failure.pkl")
        self.buggy_path = os.path.join(self.bug_path, "buggy")
        self.fixed_path = os.path.join(self.bug_path, "fixed")
        if self.sub_proj:
            self.buggy_path = os.path.join(self.buggy_path, self.sub_proj)
            self.fixed_path = os.path.join(self.fixed_path, self.sub_proj)
        
        # temp paths for each test case
        self.test_cache_dir = None
        self.failed_test_names = []
        self.modified_classes = []
        self.src_prefix = None
        self.test_prefix = None
        self.src_class_prefix = None
        self.test_class_prefix = None

        for path in [
            self.output_path,
            self.res_path,
            self.bug_path,
            self.cache_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # init logger with time
        log_config['handlers']['file']['filename'] = os.path.join(self.res_path, f"{int(time())}.log")
        logging.config.dictConfig(log_config)
        self.logger = logging.getLogger("default")
    
    def get_call_graph_file(self, test_name):
        return os.path.join(self.cache_path,
                            test_name.split("::")[0],
                            test_name,
                            "call_graph.jsonl")
    
    def get_class_file(self, class_name):
        class_file = os.path.join(self.buggy_path,
                                  self.src_prefix,
                                  class_name.split("$")[0].replace(".", "/") + ".java")
        if not os.path.exists(class_file):
            class_file = os.path.join(self.buggy_path,
                                      self.test_prefix,
                                      class_name.split("$")[0].replace(".", "/") + ".java")
        if not os.path.exists(class_file):
            self.logger.warning(f"{class_file} not exists!")
            return None
        return class_file
