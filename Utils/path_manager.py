import json
import logging
import logging.config
import os
import sys
from pathlib import Path
from time import time

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
        
        # retrieve configurations
        self.retrieve_top_n = self.config["retrieve_top_n"]
        self.rerank_top_n = self.config["rerank_top_n"]
        self.chat_rerank_top_n = self.config["chat_rerank_top_n"]

        # LLMs
        self.rerank_series = self.config["rerank_series"]
        self.rerank_model = self.config["rerank_model"]
        self.rerank_api_key = self.config["rerank_api_key"]
        self.rerank_base_url = self.config["rerank_base_url"]
        
        self.embed_series = self.config["embed_series"]
        self.embed_model = self.config["embed_model"]
        self.embed_api_key = self.config["embed_api_key"]
        self.embed_base_url = self.config["embed_base_url"]
        
        self.summary_series = self.config["summary_series"]
        self.summary_model = self.config["summary_model"]
        self.summary_api_key = self.config["summary_api_key"]
        self.summary_base_url = self.config["summary_base_url"]
        self.summary_workers = self.config["summary_workers"]
        
        self.reasoning_series = self.config["reasoning_series"]
        self.reasoning_model = self.config["reasoning_model"]
        self.reasoning_api_key = self.config["reasoning_api_key"]
        self.reasoning_base_url = self.config["reasoning_base_url"]
        
        # dependencies
        self.agent_lib = self.config["agent_lib"]
        self.tree_sitter_lib = self.config["tree_sitter_lib"]
        
        # defects4j
        self.version = args.version
        self.project = args.project
        self.bug_id = args.bugID
        
        # sbfl
        self.sbfl_formula = self.config["sbfl_formula"]
        self.sbfl_file = os.path.join(
            self.root_path,
            "SBFL",
            "results",
            self.project,
            str(self.bug_id),
            f"{self.sbfl_formula}.ranking.csv"
        )
        
        # global paths/files
        self.output_path = os.path.join(self.root_path, "DebugResult")
        self.agent_world_path = os.path.join(self.root_path, "AgentWorld")
        self.res_path = os.path.join(self.output_path, f"d4j{args.version}-{args.project}-{args.bugID}")
        self.buggy_path = os.path.join(self.res_path, "buggy")
        self.fixed_path = os.path.join(self.res_path, "fixed")
        self.cache_path = os.path.join(self.res_path, "cache")
        
        # stores_dir
        self.index_id = f"{args.project}-{args.bugID}"
        self.stores_dir = os.path.join(self.root_path, "Stores", args.project)
        
        self.res_file = os.path.join(self.res_path, "result.json")
        self.test_failure_file = os.path.join(self.cache_path, "test_failure.pkl")
        
        # a temp paths for each test case
        self.test_cache_dir = None
        
        self.failed_test_names = []
        self.modified_classes = []
        self.src_prefix = None
        self.test_prefix = None
        self.src_class_prefix = None
        self.test_class_prefix = None

        for path in [self.output_path, self.res_path, self.cache_path, self.stores_dir]:
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
