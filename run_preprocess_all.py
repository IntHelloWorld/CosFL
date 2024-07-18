import subprocess
import os
from projects import D4J_ALL

def run_project():
    for version in D4J_ALL:
        for proj in D4J_ALL[version]:
            for bug_id in range(D4J_ALL[version][proj]["begin"], D4J_ALL[version][proj]["end"] + 1):
                if bug_id in D4J_ALL[version][proj]["deprecate"]:
                    continue
                if os.path.exists(f"/home/qinyh/GarFL/DebugResult/d4j{version}_{proj}_{bug_id}"):
                    continue
                cmd = f"/root/miniconda3/envs/llm-py39/bin/python run_preprocess.py --config openai_codellama-7B_jina --version {version} --project {proj} --bugID {bug_id}"
                result = subprocess.run(cmd.split(" "))
                if result.returncode != 0:
                    return

if __name__ == "__main__":
    run_project()