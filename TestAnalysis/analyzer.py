import enum
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from functions.my_types import TestFailure
from TestAnalysis.prompt import query_merge_template, single_test_analysis_template
from Utils.path_manager import PathManager


class TestAnalyzer:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager

    def analyze(self, test_failure: TestFailure):
        self._analyze_single_test(test_failure)
        self._analyze_all_test(test_failure)
    
    def _parse_json_response(self, text):
        pattern = r'```json(.+)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return json.loads(match.group(1))
        else:
            return json.loads(text)
    
    def _analyze_single_test(self, test_failure: TestFailure):
        for test_class in test_failure.test_classes:
            for test_case in test_class.test_cases:
                self.path_manager.logger.info(f"[Analysis] Analyzing test {self.path_manager.project}-{self.path_manager.bug_id} {test_case.name}")
                query_file = os.path.join(
                    self.path_manager.cache_path,
                    test_case.test_class_name,
                    test_case.name,
                    "query.txt"
                )
                
                if os.path.exists(query_file):
                    with open(query_file, "r") as f:
                        json_res = json.load(f)
                        test_case.queries = json_res["Queries"]
                    continue
                
                messages = single_test_analysis_template.format_messages(
                    test_code_ph=test_case.test_method.text,
                    test_output_ph=test_case.test_output,
                    stack_trace_ph=test_case.stack_trace
                )
                response = self.path_manager.reasoning_llm.chat(messages)
                json_res = self._parse_json_response(response.message.content)
                test_case.queries = json_res["Queries"]
                with open(query_file, "w") as f:
                    json.dump(json_res, f, indent=4)
    
    def _analyze_all_test(self, test_failure: TestFailure):
        n_test_cases = sum([len(test_class.test_cases) for test_class in test_failure.test_classes])
        self.path_manager.logger.info(f"[Analysis] Analyzing all {n_test_cases} tests for bug {self.path_manager.project}-{self.path_manager.bug_id}")
        
        final_query_file = os.path.join(
            self.path_manager.cache_path,
            "query.txt")
        
        if os.path.exists(final_query_file):
            with open(final_query_file, "r") as f:
                json_res = json.load(f)
                test_failure.queries = json_res["Queries"]
            return
        
        if n_test_cases == 1:
            test_dir = os.path.join(
                self.path_manager.cache_path,
                test_failure.test_classes[0].name,
                test_failure.test_classes[0].test_cases[0].name)
            query_file = os.path.join(test_dir, "query.txt")
            with open(query_file, "r") as f:
                json_res = json.load(f)
            
            test_failure.queries = json_res["Queries"]
            with open(final_query_file, "w") as f:
                json.dump({"Queries": json_res["Queries"]}, f, indent=4)
            
            return
        
        all_queries = ""
        for test_class in test_failure.test_classes:
            for i, test_case in enumerate(test_class.test_cases):
                all_queries += f"query for test case \"{test_case.name}\":\n"
                for query in test_case.queries:
                    all_queries += "{0}: {1}\n".format(i+1, query.replace("\n", ""))
                all_queries += "\n"
                
                
        messages = query_merge_template.format_messages(queries_ph=all_queries)
        response = self.path_manager.reasoning_llm.chat(messages)
        json_res = self._parse_json_response(response.message.content)
        test_failure.queries = json_res["Queries"]
        
        with open(final_query_file, "w") as f:
            json.dump(json_res, f, indent=4)