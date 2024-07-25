import json
import os
import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from llama_index.core.schema import NodeWithScore

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from functions.my_types import TestFailure
from run_all import D4J
from Utils.path_manager import PathManager


def evaluate(
    path_manager: PathManager,
    nodes: List[NodeWithScore],
    test_failure_obj: TestFailure
):
    results = {"matches":[], "methods":[]}
    for i, node in enumerate(nodes):
        localized_method_text = node.text
        results["methods"].append(localized_method_text)
        for buggy_method in test_failure_obj.buggy_methods:
            if buggy_method.text == localized_method_text:
                results["matches"].append(i + 1)

    with open(path_manager.res_file, 'w') as f:
        json.dump(results, f, indent=4)


def evaluate_others(
    path_manager: PathManager,
    docs_list: List[List[Document]],
    test_failure_obj: TestFailure
):
    match = [0 * len(docs_list[0])]
    for docs in docs_list:
        for i, doc in enumerate(docs):
            localized_method_text = doc.page_content
            for buggy_method in test_failure_obj.buggy_methods:
                if buggy_method.text == localized_method_text:
                    match[i] = 1
    
    results = {"matches":[]}
    for i in range(len(match)):
        if match[i] == 1:
            results["matches"].append(i + 1)

    with open(path_manager.res_file, 'w') as f:
        json.dump(results, f, indent=4)


def evaluate_all(res_path: str):
    all_bugs = D4J
    top_n = {}
    mfr = {}
    mar = {}
    
    for version in all_bugs:
        for proj in all_bugs[version]:
            if proj not in top_n:
                top_n[proj] = {"top_1": 0, "top_3": 0, "top_5": 0, "top_10": 0}
            mfr_tmp = []
            mar_tmp = []
            
            for bug_id in range(all_bugs[version][proj]["begin"], all_bugs[version][proj]["end"] + 1):
                if bug_id in all_bugs[version][proj]["deprecate"]:
                    continue
                res_file = os.path.join(
                    res_path,
                    f"d4j{version}-{proj}-{bug_id}",
                    "result.json"
                )
                with open(res_file, 'r') as f:
                    matched_indexes = json.load(f)["matches"]
                if matched_indexes:
                    if matched_indexes[0] == 1:
                        top_n[proj]["top_1"] += 1
                    if matched_indexes[0] <= 3:
                        top_n[proj]["top_3"] += 1
                    if matched_indexes[0] <= 5:
                        top_n[proj]["top_5"] += 1
                    if matched_indexes[0] <= 10:
                        top_n[proj]["top_10"] += 1
                    mfr_tmp.append(matched_indexes[0])
                    mar_tmp.extend(matched_indexes)
                else:
                    print(f"Warning: no matched indexes found for {version}-{proj}-{bug_id}")
            
            if mfr_tmp:
                mfr[proj] = sum(mfr_tmp) / len(mfr_tmp)
            else:
                mfr[proj] = None
            
            if mar_tmp:
                mar[proj] = sum(mar_tmp) / len(mar_tmp)
            else:
                mar[proj] = None
    
    print("Top N:")
    print(top_n)
    print("MFR:")
    print(mfr)
    print("MAR:")
    print(mar)

if __name__ == "__main__":
    res_path = "/home/qyh/projects/GarFL/DebugResult"
    evaluate_all(res_path)