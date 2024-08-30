import hashlib
import json
import logging
import logging.config
import os
import sys
from pathlib import Path
from time import time

from fastapi import dependencies
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

DEFAULT_VECTOR_STORE_NAME = "chroma"

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
        
        # bug info
        self.version = args.version
        self.project = args.project
        self.subproj = args.subproj
        self.bug_id = args.bugID
        self.config_hash = self.get_md5_hash(args.config)
        
        # global paths/files
        self.output_path = os.path.join(self.root_path, "DebugResult")
        self.res_path = os.path.join(
            self.output_path,
            args.config,
            f"{args.version}",
            args.project,
            f"{args.project}-{args.bugID}")
        self.retrieved_nodes_file = os.path.join(self.res_path, "retrieved_nodes.pkl")
        self.res_file = os.path.join(self.res_path, "result.json")
        self.projects_path = os.path.join(self.root_path, "Projects")
        self.bug_path = os.path.join(self.projects_path, args.project, str(args.bugID))
        self.test_failure_file = os.path.join(self.bug_path, "test_failure.pkl")
        self.method_nodes_file = os.path.join(self.bug_path, "method_nodes.pkl")
        self.proj_tmp_path = os.path.join(
            self.output_path,
            self.config_hash,
            f"{args.version}-{args.project}-{args.bugID}"
        )
        self.buggy_path = os.path.join(self.proj_tmp_path, "buggy")
        self.fixed_path = os.path.join(self.proj_tmp_path, "fixed")
        if self.subproj:
            self.buggy_path = os.path.join(self.buggy_path, self.subproj)
            self.fixed_path = os.path.join(self.fixed_path, self.subproj)
        
        # temp paths for each test case
        self.test_cache_dir = None
        self.failed_test_names = []
        self.modified_classes = []
        self.src_prefix = None
        self.test_prefix = None
        self.src_class_prefix = None
        self.test_class_prefix = None

        for path in [
            self.res_path,
            self.bug_path]:
            if not os.path.exists(path):
                os.makedirs(path)
        
        # read config file
        self.config_file = os.path.join(self.root_path, "Config", args.config, "config.json")
        with open(self.config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.mode = self.config["mode"]
        self.clear = self.config["clear"]
        
        if "reason" in self.config:
            self.reasoning_model = self.config["reason"]["reasoning_model"]
            if "reasoning_cache_name" in self.config["reason"]:
                self.reasoning_cache_name = self.config["reason"]["reasoning_cache_name"]
            else:
                self.reasoning_cache_name = self.reasoning_model
            self.query_type = self.config["query_type"]
        
        if "summary" in self.config:
            self.summary_model = self.config["summary"]["summary_model"]
        
            # stores_dir
            self.stores_dir = os.path.join(self.root_path, "Stores", self.summary_model, args.project)
            self.doc_store_file = os.path.join(self.stores_dir, DEFAULT_PERSIST_FNAME)
            if not os.path.exists(self.stores_dir):
                os.makedirs(self.stores_dir, exist_ok=True)
        
        # for embeddings generation
        if "embed" in self.config:
            assert "summary" in self.config, "summary model is required for embeddings generation"
            self.embed_model = self.config["embed"]["embed_model"]
            self.summary_model = self.config["summary"]["summary_model"]
            self.embed_text = self.config["embed"]["embed_text"]
            if "embed_cache_name" in self.config["embed"]:
                self.embed_cache_name = self.config["embed"]["embed_cache_name"]
            else:
                self.embed_cache_name = self.embed_model
            
            self.vector_store_dir = os.path.join(self.stores_dir, self.embed_cache_name, DEFAULT_VECTOR_STORE_NAME)
            if not os.path.exists(self.vector_store_dir):
                os.makedirs(path, exist_ok=True)
        
        # for fault localization
        if "rerank" in self.config:
            self.rerank_model = self.config["rerank"]["rerank_model"]
        
        if "dependencies" in self.config:
            dependencies = self.config["dependencies"]
            # dependencies
            self.agent_lib = dependencies["agent_lib"]
            self.D4J_exec = dependencies["D4J_exec"]
            self.GB_exec = dependencies["GB_exec"]
            
            if self.version == "GrowingBugs":
                self.bug_exec = self.GB_exec
            else:
                self.bug_exec = self.D4J_exec
        
        if "hyper" in self.config:
            hyper = self.config["hyper"]
            # retrieve configurations
            self.retrieve_top_n = hyper["retrieve_top_n"]
            self.rerank_top_n = hyper["rerank_top_n"]
            self.chat_rerank_top_n = hyper["chat_rerank_top_n"]
            # sbfl
            self.sbfl_formula = hyper["sbfl_formula"]
            if self.sbfl_formula:
                self.sbfl_file = os.path.join(
                    self.root_path,
                    "SBFL",
                    "results",
                    args.project,
                    str(args.bugID),
                    f"{self.sbfl_formula}.ranking.csv"
                )
                self.all_methods = False
            else:
                self.all_methods = True
        
        # init logger with time
        log_config['handlers']['file']['filename'] = os.path.join(self.res_path, f"{int(time())}.log")
        logging.config.dictConfig(log_config)
        self.logger = logging.getLogger("default")
    
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
    
    def get_md5_hash(self, input_string):
        md5_hash = hashlib.md5()
        md5_hash.update(input_string.encode('utf-8'))
        return md5_hash.hexdigest()
