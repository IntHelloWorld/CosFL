import subprocess

from projects import D4J_ALL


def run_project(all_bugs):
    for version in all_bugs:
        for proj in all_bugs[version]:
            for bug_id in range(all_bugs[version][proj]["begin"], all_bugs[version][proj]["end"] + 1):
                if bug_id in all_bugs[version][proj]["deprecate"]:
                    continue
                cmd = f"/home/qyh/micromamba/envs/LLM-py39/bin/python3 run.py --config LMStudio+jina+openai --version {version} --project {proj} --bugID {bug_id}"
                result = subprocess.run(cmd.split(" "))
                if result.returncode != 0:
                    return

def run_single():
    cmd = f"/home/qyh/micromamba/envs/LLM-py39/bin/python3 run.py --config LMStudio+jina+openai --version 2.0.1 --project Chart --bugID 15"
    result = subprocess.run(cmd.split(" "))
    if result.returncode != 0:
        return

if __name__ == "__main__":
    run_project(D4J_ALL)
    # run_single()