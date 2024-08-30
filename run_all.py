import os
import shutil
import subprocess
import sys

from projects import ALL_BUGS

root = os.path.dirname(__file__)
sys.path.append(root)

def run_all_bugs(config_name: str):
    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            bugIDs = ALL_BUGS[version][proj][0]
            deprecatedIDs = ALL_BUGS[version][proj][1]
            subproj = ALL_BUGS[version][proj][2] if version == "GrowingBugs" else ""
            for bug_id in bugIDs:
                res_path = f"DebugResult/{config_name}/{version}/{proj}/{proj}-{bug_id}"
                res_path = os.path.join(root, res_path)
                if bug_id in deprecatedIDs:
                    continue
                if os.path.exists(res_path):
                    print(f"{version}-{proj}-{bug_id} already finished, skip!")
                    continue
                
                ret_code = run_one_bug(config_name, version, proj, bug_id, subproj)
                if ret_code != 0:
                    shutil.rmtree(res_path, ignore_errors=True)
                    raise Exception(f"Error in running {version}-{proj}-{bug_id}!")

def run_one_bug(config_name, version, proj, bug_id, subproj):
    cmd = f"python run.py --config {config_name} --version {version} --project {proj} --bugID {bug_id} --subproj {subproj}"
    result = subprocess.run(cmd.split(" "))
    return result.returncode

if __name__ == "__main__":
    # config_name = "ALL_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    # run_one_bug(config_name, "d4j1.4.0", "Chart", 1, True, "")
    
    """Embedding"""
    # config_name = "EMBED_S(codeqwen-1_5-7B)_E(Jina-code)"
    
    """Ablation Study"""
    # config_name = "DEBUG_Q(None)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    # config_name = "DEBUG_Q(GPT-4o+one)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_C(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina-code)_R(Jina+chat)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(None+chat)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina+chat)_Cov(false)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_C(true)"
    
    """Reasoning Model Study"""
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(claude-3-5-sonnet)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(claude-3-sonnet)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(gpt-3.5-turbo)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(llama3.1-405b-instruct)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(qwen2-72b-instruct)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    
    "Summarization Model Study"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codegemma-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codellama-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codellama-13B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codellama-34B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(Codestral-22B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(DeepSeek-Coder-V2-Lite)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(starcoder2-15B)_E(Jina)_R(Jina)_Cov(true)"
    
    """Rerank Model Study"""
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Cohere)_Cov(true)"
    # config_name = "DEBUG_Q(GPT-4o)_S(codeqwen-1_5-7B)_E(Jina)_R(Voyage)_Cov(true)"
    
    """Test best model"""
    # config_name = "DEBUG_Q(GPT-4o+causes)_S(codeqwen-1_5-7B)_E(Jina)_R(Jina)_Cov(true)"
    config_name = "BEST_Q(qwen2-72b+causes)_E(Jina-code)_R(Cohere)_Cov(true)"
    
    run_all_bugs(config_name)