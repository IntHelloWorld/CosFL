import json
import os
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
    results = {"matches":[], "query_matches":[], "methods":[]}
    for i, node in enumerate(nodes):
        localized_method_text = node.text
        results["methods"].append(localized_method_text)
        for buggy_method in test_failure_obj.buggy_methods:
            if buggy_method.text == localized_method_text:
                results["matches"].append(i + 1)
    
    for reranked_nodes in reranked_nodes_list:
        query_matches = []
        for i, node in enumerate(reranked_nodes):
            localized_method_text = node.text
            for buggy_method in test_failure_obj.buggy_methods:
                if buggy_method.text == localized_method_text:
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
            if buggy_method.text == localized_method_text:
                results["matches"].append(i + 1)

    with open(path_manager.res_file, 'w') as f:
        json.dump(results, f, indent=4)


def evaluate_all(res_path: str):
    all_bugs = ALL_BUGS
    top_n = OrderedDict()
    mfr = OrderedDict()
    mar = OrderedDict()
    mrr = OrderedDict()
    
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
                    if matched_indexes[0] <= 3:
                        top_n[proj]["top_3"] += 1
                    if matched_indexes[0] <= 5:
                        top_n[proj]["top_5"] += 1
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

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(__file__))
    res_path = os.path.join(root, "DebugResult")
    
    # res_path = "/home/qyh/projects/GarFL/DebugResult/ALL_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_C(false)"
    # res_path = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_C(true)"
    # res_path = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    # res_path = "/home/qyh/projects/GarFL/DebugResult/TF-IDF"
    # res_path = "/home/qyh/projects/GarFL/DebugResult/BM25"
    
    """Ablation Study"""
    # res_name = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o+one)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    # res_name = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(None)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    # res_name = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina-code)_R(Jina+chat)_Cov(true)"
    # res_name = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(None+chat)_Cov(true)"
    # res_name = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_Cov(false)"
    # res_name = "/home/qyh/projects/GarFL/DebugResult/DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    
    """Reasoning Model Study"""
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(claude-3-5-sonnet)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(claude-3-sonnet)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(gpt-3.5-turbo)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(qwen2-72b-instruct)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(llama3.1-405b-instruct)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    
    "Summarization Model Study"
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codegemma-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codellama-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codellama-13B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codellama-34B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(Codestral-22B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(DeepSeek-Coder-V2-Lite)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(starcoder2-15B)_E(Jina)_R(Jina)_Cov(true)"
    
    """Embedding Model Study"""
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(OpenAI)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Voyage)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Cohere)_R(Jina)_Cov(true)"
    
    """Rerank Model Study"""
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Cohere)_Cov(true)"
    # res_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Voyage)_Cov(true)"
    
    """Test use root causes queries"""
    res_name = "DEBUG_Q(GPT-4o+causes)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    
    res_path = os.path.join(res_path, res_name)
    evaluate_all(res_path)