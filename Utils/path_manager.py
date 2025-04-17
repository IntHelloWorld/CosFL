import hashlib
import logging
import logging.config
import os
import sys
from pathlib import Path
from time import time

import yaml

DEFAULT_VECTOR_STORE_NAME = "chroma"
DEFAULT_PERSIST_FNAME = "docstore.json"

log_config = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "%(levelname)s - %(asctime)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "filename": "",
            "mode": "w",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "default": {"handlers": ["file", "console"], "level": "DEBUG"},
    },
}


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


class PathManager:

    def __init__(self, args):
        self.root_path = Path(__file__).resolve().parents[1].as_posix()

        self.verbose = args.verbose

        # bug info
        self.version = args.version
        self.project = args.project
        self.subproj = args.subproj
        self.bug_id = args.bugID
        self.bug_name = f"{args.project}-{args.bugID}"
        self.config_name = Path(args.config).stem
        self.config_hash = self.get_md5_hash(args.config)

        # global paths/files
        self.output_path = os.path.join(self.root_path, "DebugResult")
        self.res_path = os.path.join(
            self.output_path,
            self.config_name,
            f"{args.version}",
            args.project,
            f"{args.project}-{args.bugID}",
        )
        self.retrieved_nodes_file = os.path.join(
            self.res_path, "retrieved_nodes.pkl"
        )
        self.res_file = os.path.join(self.res_path, "result.json")
        self.projects_path = os.path.join(self.root_path, "Projects")
        self.bug_path = os.path.join(
            self.projects_path, args.project, str(args.bugID)
        )
        self.test_failure_file = os.path.join(
            self.bug_path, "test_failure.pkl"
        )
        self.method_nodes_file = os.path.join(self.bug_path, "nodes.pkl")
        self.proj_tmp_path = os.path.join(
            self.output_path,
            self.config_hash,
            f"{args.version}-{args.project}-{args.bugID}",
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

        for path in [self.res_path, self.bug_path]:
            if not os.path.exists(path):
                os.makedirs(path)

        # read config file
        self.config_file = os.path.join(self.root_path, "Config", args.config)
        with open(self.config_file, "r") as f:
            self.config = Config(yaml.safe_load(f))

        # doc stores dir
        self.doc_stores_dir = os.path.join(
            self.root_path,
            "DocStores",
            f"{self.config.models.summary.cache_name}(module_size_{self.config.hyper.min_module_size}-{self.config.hyper.max_module_size})",
            args.project,
            self.bug_name,
        )
        self.doc_store_file = os.path.join(
            self.doc_stores_dir, DEFAULT_PERSIST_FNAME
        )
        if not os.path.exists(self.doc_stores_dir):
            os.makedirs(self.doc_stores_dir, exist_ok=True)

        # vector stores dir
        self.vector_stores_dir = os.path.join(
            self.root_path,
            "VectorStores",
            f"{self.config.models.embed.cache_name}(module_size_{self.config.hyper.min_module_size}-{self.config.hyper.max_module_size})",
            args.project,
            DEFAULT_VECTOR_STORE_NAME,
        )
        if not os.path.exists(self.vector_stores_dir):
            os.makedirs(self.vector_stores_dir, exist_ok=True)

        # dependencies
        self.agent_lib = self.config.dependencies.agent_lib
        self.D4J_exec = self.config.dependencies.D4J_exec
        self.GB_exec = self.config.dependencies.GB_exec
        if self.version == "GrowingBugs":
            self.bug_exec = self.GB_exec
        else:
            self.bug_exec = self.D4J_exec

        # hyper-parameters
        self.retrieve_top_n = self.config.hyper.retrieve_top_n
        self.rerank_top_n = self.config.hyper.rerank_top_n
        self.chat_rerank_top_n = self.config.hyper.chat_rerank_top_n
        # sbfl
        self.sbfl_formula = self.config.hyper.sbfl_formula
        if self.sbfl_formula:
            self.sbfl_file = os.path.join(
                self.root_path,
                "SBFL",
                "results",
                self.project,
                str(self.bug_id),
                f"{self.sbfl_formula}.ranking.csv",
            )

        # init logger with time
        log_config["handlers"]["file"]["filename"] = os.path.join(
            self.res_path, f"{int(time())}.log"
        )
        logging.config.dictConfig(log_config)
        self.logger = logging.getLogger("default")

    def get_class_file(self, class_name):
        class_file = os.path.join(
            self.buggy_path,
            self.src_prefix,
            class_name.split("$")[0].replace(".", "/") + ".java",
        )
        if not os.path.exists(class_file):
            class_file = os.path.join(
                self.buggy_path,
                self.test_prefix,
                class_name.split("$")[0].replace(".", "/") + ".java",
            )
        if not os.path.exists(class_file):
            self.logger.warning(f"{class_file} not exists!")
            return None
        return class_file

    def get_md5_hash(self, input_string):
        md5_hash = hashlib.md5()
        md5_hash.update(input_string.encode("utf-8"))
        return md5_hash.hexdigest()
