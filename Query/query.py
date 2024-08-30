import asyncio
import json
import os
import re
import sys
import time
from multiprocessing import pool
from pathlib import Path
from time import sleep
from typing import Coroutine, List

from aiolimiter import AsyncLimiter
from llama_index.core.async_utils import DEFAULT_NUM_WORKERS, asyncio_run
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.asyncio import tqdm_asyncio

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from functions.my_types import TestCase, TestFailure
from Query.prompt import (
    one_query_template,
    query_merge_template,
    single_test_query_template,
)
from Utils.path_manager import PathManager

DEFAULT_RATELIMIT = 50

async def run_jobs(jobs, limit=DEFAULT_RATELIMIT, desc=""):
    limiter = AsyncLimiter(limit)
    async def worker(job: Coroutine):
        async with limiter:
            return await job
    
    pool_jobs = [worker(job) for job in jobs]

    results = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    return results

class QueryGenerator:
    def __init__(self, path_manager: PathManager):
        self.path_manager = path_manager
        self.query_path = os.path.join(
            self.path_manager.bug_path,
            self.path_manager.reasoning_cache_name
        )
        if not os.path.exists(self.query_path):
            os.makedirs(self.query_path, exist_ok=True)
        
        self.merged_query_file = os.path.join(self.query_path, "merged_queries.json")
        self.one_query_file = os.path.join(self.query_path, "one_query.json")

    def generate(self, test_failure: TestFailure) -> List[str]:
        # read from cache
        if os.path.exists(self.merged_query_file):
            with open(self.merged_query_file, "r") as f:
                json_res = json.load(f)
                merged_queries = json_res["Queries"]
            return merged_queries
        
        test_cases: List[TestCase] = []
        for test_class in test_failure.test_classes:
            test_cases.extend(test_class.test_cases)

        queries_list = self._queries_generation(test_cases)
        queries = self._queries_merge(queries_list)
        
        return queries

    def generate_no_query(self, test_failure: TestFailure) -> List[str]:
        test_cases = [case for clazz in test_failure.test_classes for case in clazz.test_cases]
        queries = []
        for test_case in test_cases:
            query = f"{test_case.test_method.text}\n{test_case.stack_trace}\n{test_case.test_output}"
            queries.append(query)
        return queries
    
    def generate_one_query(self, test_failure: TestFailure) -> List[str]:
        if os.path.exists(self.one_query_file):
            with open(self.one_query_file, "r") as f:
                json_res = json.load(f)
                return [json_res["Query"]]
        
        test_cases = [case for clazz in test_failure.test_classes for case in clazz.test_cases]
        if len(test_cases) > 5:
            import random
            test_cases = random.sample(test_cases, 5)
        
        self.path_manager.logger.info(f"[Query Generation] generate one query for {len(test_cases)} tests")
        info = ""
        for i, test_case in enumerate(test_cases):
            info += (
                f"{i+1} Test Name: {test_case.name}\n" +
                f"Test Code: {test_case.test_method.text}\n" +
                f"Test Output: {test_case.test_output}\n" +
                f"Stack Trace: {test_case.stack_trace}\n\n"
            )

        messages = one_query_template.format_messages(info_ph=info)
        response = self.path_manager.reasoning_llm.chat(messages)
        json_res = self._parse_json_response(response.message.content)
        with open(self.one_query_file, "w") as f:
            json.dump(json_res, f, indent=4)
        
        return [json_res["Query"]]
    
    
    def generate_causes(self, test_failure: TestFailure) -> List[str]:
        queries = []
        test_cases = [case for clazz in test_failure.test_classes for case in clazz.test_cases]
        if len(test_cases) > 5:
            import random
            test_cases = random.sample(test_cases, 5)
        
        self.path_manager.logger.info(f"[Query Generation] generate root causes query for {len(test_cases)} tests")
        for test_case in test_cases:
            cause_query_file = os.path.join(self.query_path, f"{test_case.name}.json")
            with open(cause_query_file, "r") as f:
                json_res = json.load(f)
                queries.append(json_res["Causes"])
        
        return queries
    
    
    def _parse_json_response(self, text: str):
        pattern = r'\{\s*"\w+":.+\}'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            print(f"Failed to parse json from:\n{text}")
            raise ValueError("Failed to match string in json format")
        
        text_to_parse = match.group(0).replace("\n", "")
        try:
            return json.loads(text_to_parse)
        except json.JSONDecodeError:
            # try to escape backslashes
            print(f"Failed to parse json from:\n{text_to_parse}")
            text_to_parse = re.sub(r'\\\\', r'\\\\\\', text_to_parse)
            try:
                return json.loads(text_to_parse)
            except json.JSONDecodeError:
                print(f"Failed to parse json from:\n{text_to_parse}")
                raise ValueError("Failed to parse json")
    
    def _queries_generation(self, test_cases: List[TestCase]) -> List[List[str]]:
        return asyncio_run(self._aqueries_generation(test_cases))
    
    async def _aqueries_generation(self, test_cases: List[TestCase]) -> List[List[str]]:
        jobs = []
        
        for test_case in test_cases:
            jobs.append(self._aquery_generation(test_case))
        
        queries_list = await run_jobs(
            jobs,
            limit=self.path_manager.config["reason"]["reasoning_rate_limit"],
            desc="Query Generation"
        )
        return queries_list
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    async def _aquery_generation(self, test_case: TestCase) -> List[str]:
        query_file = os.path.join(self.query_path, f"{test_case.name}.json")
        
        if os.path.exists(query_file):
            with open(query_file, "r") as f:
                json_res = json.load(f)
                queries = json_res["Queries"]
                return queries
        
        messages = single_test_query_template.format_messages(
            test_code_ph=test_case.test_method.text,
            test_output_ph=test_case.test_output,
            stack_trace_ph=test_case.stack_trace
        )
        response = self.path_manager.reasoning_llm.chat(messages)
        json_res = self._parse_json_response(response.message.content)
        queries = json_res["Queries"]
        with open(query_file, "w") as f:
            json.dump(json_res, f, indent=4)
        return queries
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
    def _queries_merge(self, queries_list: List[List[str]]) -> List[str]:
        
        # skip merge if only one test case
        if len(queries_list) == 1:
            merged_queries = queries_list[0]
        else:
            # optimize queries
            n_queries = sum([len(q) for q in queries_list])
            self.path_manager.logger.info(f"[Query Generation] Merge all {n_queries} for {len(queries_list)} tests")
            all_queries = ""
            for i, queries in enumerate(queries_list):
                all_queries += f"queries for test case {i+1}:\n"
                for j, query in enumerate(queries):
                    all_queries += "{0}: {1}\n".format(j+1, query.replace("\n", ""))
                all_queries += "\n"
                
            messages = query_merge_template.format_messages(queries_ph=all_queries)
            response = self.path_manager.reasoning_llm.chat(messages)
            json_res = self._parse_json_response(response.message.content)
            merged_queries = json_res["Queries"]
        
        with open(self.merged_query_file, "w") as f:
            json.dump({"Queries": merged_queries}, f, indent=4)
        return merged_queries