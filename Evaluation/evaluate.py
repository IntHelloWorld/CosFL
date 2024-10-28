import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from typing import List

from langchain_core.documents import Document
from llama_index.core.schema import NodeWithScore

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
from functions.my_types import TestFailure
from projects import ALL_BUGS
from Utils.path_manager import PathManager


def evaluate(
    path_manager: PathManager,
    nodes: List[NodeWithScore],
    reranked_nodes_list: List[List[NodeWithScore]],
    test_failure_obj: TestFailure
):
    results = {"matches":[], "query_matches":[], "methods":[], "reasons":[]}
    for i, node in enumerate(nodes):
        localized_method_text = node.text
        results["methods"].append(localized_method_text)
        if "llm_reason" in node.metadata:
            results["reasons"].append(node.metadata["llm_reason"])
        for buggy_method in test_failure_obj.buggy_methods:
            if buggy_method.code == localized_method_text:
                results["matches"].append(i + 1)
    
    for reranked_nodes in reranked_nodes_list:
        query_matches = []
        for i, node in enumerate(reranked_nodes):
            localized_method_text = node.text
            for buggy_method in test_failure_obj.buggy_methods:
                if buggy_method.code == localized_method_text:
                    query_matches.append(i + 1)
        results["query_matches"].append(query_matches)

    with open(path_manager.res_file, 'w') as f:
        json.dump(results, f, indent=4)


def evaluate_others(
    path_manager: PathManager,
    docs: List[Document],
    test_failure_obj: TestFailure
):
    
    results = {"matches":[]}

    for i, doc in enumerate(docs):
        localized_method_text = doc.page_content
        for buggy_method in test_failure_obj.buggy_methods:
            if buggy_method.code == localized_method_text:
                results["matches"].append(i + 1)

    with open(path_manager.res_file, 'w') as f:
        json.dump(results, f, indent=4)


def evaluate_all(res_path: str, if_size: bool = False):
    all_bugs = ALL_BUGS
    top_n = OrderedDict()
    mfr = OrderedDict()
    mar = OrderedDict()
    mrr = OrderedDict()
    
    top_1_bugs = []
    top_5_bugs = []
    
    mfr_all, mar_all, mrr_all = [], [], []
    
    for version in all_bugs:
        for proj in all_bugs[version]:
            if proj == "Spoon":
                print(1)
            if proj not in top_n:
                top_n[proj] = {"top_1": 0, "top_3": 0, "top_5": 0, "top_10": 0}
            mfr_tmp = []
            mar_tmp = []
            mrr_tmp = []
            
            for bug_id in all_bugs[version][proj][0]:
                if bug_id in all_bugs[version][proj][1]:
                    continue
                res_file = os.path.join(
                    res_path,
                    version,
                    proj,
                    f"{proj}-{bug_id}",
                    "result.json"
                )
                with open(res_file, 'r') as f:
                    results = json.load(f)
                    matched_indexes = results["matches"]
                    matched_indexes = [m for m in matched_indexes if m <= 50]
                    if matched_indexes == []:
                        matched_indexes = [51]
                    try:
                        query_matches = results["query_matches"]
                        query_matches = [[m for m in q if m <= 50] for q in query_matches]
                    except KeyError:
                        query_matches = [[]]

                if matched_indexes:
                    if matched_indexes[0] == 1:
                        top_n[proj]["top_1"] += 1
                        top_1_bugs.append(f"{proj}-{bug_id}")
                    if matched_indexes[0] <= 3:
                        top_n[proj]["top_3"] += 1
                    if matched_indexes[0] <= 5:
                        top_n[proj]["top_5"] += 1
                        top_5_bugs.append(f"{proj}-{bug_id}")
                    if matched_indexes[0] <= 10:
                        top_n[proj]["top_10"] += 1
                    mfr_tmp.append(matched_indexes[0])
                    mar_tmp.append(sum(matched_indexes) / len(matched_indexes))
                else:
                    print(f"Warning: no matched indexes found for {version}-{proj}-{bug_id}")
                
                if query_matches:
                    query_tmp = []
                    for query_match in query_matches:
                        if len(query_match) == 0:
                            query_tmp.append(0)
                        else:
                            reciprocal = [1 / i for i in query_match]
                            query_tmp.append(sum(reciprocal) / len(reciprocal))
                    mrr_tmp.append(sum(query_tmp) / len(query_tmp))
                else:
                    print(f"Warning: no query matches found for {version}-{proj}-{bug_id}")
            
            if mfr_tmp:
                mfr[proj] = sum(mfr_tmp) / len(mfr_tmp)
                mfr_all.extend(mfr_tmp)
            else:
                mfr[proj] = None
            
            if mar_tmp:
                mar_all.extend(mar_tmp)
                mar[proj] = sum(mar_tmp) / len(mar_tmp)
            else:
                mar[proj] = None
            
            if mrr_tmp:
                mrr_all.extend(mrr_tmp)
                mrr[proj] = sum(mrr_tmp) / len(mrr_tmp)
            else:
                mrr[proj] = None
    
    top_5_file = "Evaluation/CosFL_top5.txt"
    with open(top_5_file, 'w') as f:
        f.write("\n".join(top_5_bugs))
    
    top_1_file = "Evaluation/CosFL_top1.txt"
    with open(top_1_file, 'w') as f:
        f.write("\n".join(top_1_bugs))
    
    print("Top N:")
    pprint(top_n)
    print("MFR:")
    pprint(mfr)
    print("MAR:")
    pprint(mar)
    print("MRR:")
    pprint(mrr)
    
    print(f"Total Top 1: {sum([v['top_1'] for v in top_n.values()])}")
    print(f"Total Top 5: {sum([v['top_5'] for v in top_n.values()])}")
    print(f"Total Top 10: {sum([v['top_10'] for v in top_n.values()])}")
    
    print(f"Total MFR: {sum(mfr_all) / len(mfr_all)}")
    print(f"Total MAR: {sum(mar_all) / len(mar_all)}")
    print(f"Total MRR: {sum(mrr_all) / len(mrr_all)}")


def evaluate_size(config_name: str, autofl_file: str, agentfl_file: str, cosfl_file: str):
    with open(autofl_file, 'r') as f:
        autofl_bugs = f.readlines()
        autofl_bugs = [b.strip() for b in autofl_bugs]
    with open(agentfl_file, 'r') as f:
        agentfl_bugs = f.readlines()
        agentfl_bugs = [b.strip() for b in agentfl_bugs]
    with open(cosfl_file, 'r') as f:
        cosfl_bugs = f.readlines()
        cosfl_bugs = [b.strip() for b in cosfl_bugs]
    
    all_bugs = ALL_BUGS
    autofl_sizes, agentfl_sizes, cosfl_sizes = [], [], []
    all_sizes = []
    for version in all_bugs:
        for proj in all_bugs[version]:
            for bug_id in all_bugs[version][proj][0]:
                bug_name = f"{proj}-{bug_id}"
                if bug_id in all_bugs[version][proj][1]:
                    continue
                res_dir = os.path.join(
                    config_name,
                    version,
                    proj,
                    bug_name
                )
                log_files = Path(res_dir).glob("*.log")
                pattern = re.compile(r"call graph loaded with (\d+) nodes and (\d+) edges")
                n_nodes, n_edges = 0, 0
                for log_file in log_files:
                    log_text = log_file.read_text()
                    match = re.search(pattern, log_text)
                    if match:
                        n = int(match.group(1))
                        e = int(match.group(2))
                        n_nodes = n if n > n_nodes else n_nodes
                        n_edges = e if e > n_edges else n_edges
                    else:
                        raise ValueError(f"Error: no match found in {log_file}")
                all_sizes.append(n_nodes)
                if bug_name in autofl_bugs:
                    autofl_sizes.append(n_nodes)
                if bug_name in agentfl_bugs:
                    agentfl_sizes.append(n_nodes)
                if bug_name in cosfl_bugs:
                    cosfl_sizes.append(n_nodes)
    
    ranges = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 1500)]
    range_dict = {f"({r[0]},{r[1]}]": 0 for r in ranges}
    res_dict = {
        "All": range_dict.copy(),
        "AutoFL": range_dict.copy(),
        "AgentFL": range_dict.copy(),
        "CosFL": range_dict.copy()
    }
    for n in all_sizes:
        for r in ranges:
            if r[0] < n <= r[1]:
                res_dict["All"][f"({r[0]},{r[1]}]"] += 1
                break
    for n in autofl_sizes:
        for r in ranges:
            if r[0] < n <= r[1]:
                res_dict["AutoFL"][f"({r[0]},{r[1]}]"] += 1
                break
    for n in agentfl_sizes:
        for r in ranges:
            if r[0] < n <= r[1]:
                res_dict["AgentFL"][f"({r[0]},{r[1]}]"] += 1
                break
    for n in cosfl_sizes:
        for r in ranges:
            if r[0] < n <= r[1]:
                res_dict["CosFL"][f"({r[0]},{r[1]}]"] += 1
                break
    pprint(res_dict)
    csv_file = "Evaluation/Size.csv"
    fields = ["Tool", "Range", "Size"]
    with open(csv_file, 'w') as f:
        f.write(f"{'|'.join(fields)}\n")
        for tool in res_dict:
            for r in res_dict[tool]:
                f.write(f"{tool}|{r}|{res_dict[tool][r]}\n")


def find_example(garfl_res_dir, autofl_res_dir):
    candidates = {}
    all_bugs = ALL_BUGS
    for version in all_bugs:
        for proj in all_bugs[version]:
            for bug_id in all_bugs[version][proj][0]:
                bug_name = f"{proj}-{bug_id}"
                if bug_id in all_bugs[version][proj][1]:
                    continue
                garfl_res_file = os.path.join(garfl_res_dir, version, proj, bug_name, "result.json")
                garfl_res = json.load(open(garfl_res_file, 'r'))
                first_rank = 51
                if garfl_res["matches"]:
                    first_rank = min(garfl_res["matches"])
                if_garfl = True if first_rank <= 5 else False
                autofl_res_file = os.path.join(autofl_res_dir, f"XFL-{proj}_{bug_id}.json")
                autofl_res = json.load(open(autofl_res_file, 'r'))
                founds = []
                for method in autofl_res["buggy_methods"]:
                    try:
                        if autofl_res["buggy_methods"][method]["is_found"]:
                            founds.append(True)
                    except Exception:
                        founds.append(False)
                if_autofl = True if any(founds) else False
                if if_garfl and not if_autofl:
                    try:
                        candidates[proj].append(bug_id)
                    except KeyError:
                        candidates[proj] = [bug_id]
    pprint(candidates)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    res_path = os.path.join(root, "DebugResult")
    res_name = "default"
    # res_name = "no_context"
    # res_name = "no_context_retrieval"
    # res_name = "no_desc_retrieval"
    # res_name = "no_chat_rerank"
    
    # res_name = "retrieval_25"
    # res_name = "retrieval_75"
    # res_name = "embedding_openai"
    # res_name = "embedding_voyage"
    # res_name = "module_size_3-10"
    res_name = "module_size_8-20"
    # res_name = "gpt-3.5-turbo-16k"
    # res_name = "gpt-4o"
    
    res_path = os.path.join(res_path, res_name)
    evaluate_all(res_path)
    # evaluate_size(res_path, "Evaluation/AutoFL_top5.txt", "Evaluation/AgentFL_top5.txt", "Evaluation/CosFL_default_top5.txt")