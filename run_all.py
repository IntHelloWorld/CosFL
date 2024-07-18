import subprocess

# D4J = {
#     "1.4.0": {
#         "Time": {
#             "begin": 1,
#             "end": 27,
#             "deprecate": []
#         }
#     }
# }
D4J = {
    "2.0.1": {
        "Closure": {
            "begin": 53,
            "end": 133,
            "deprecate": [49, 63, 93]
        },
    }
}
# D4J = {
#     "1.4.0": {
#         "Lang": {
#             "begin": 21,
#             "end": 40,
#             "deprecate": []
#         }
#     }
# }
# D4J = {
#     "1.4.0": {
#         "Mockito": {
#             "begin": 1,
#             "end": 38,
#             "deprecate": []
#         }
#     }
# }
# D4J = {
#     "1.4.0": {
#         "Math": {
#             "begin": 91,
#             "end": 106,
#             "deprecate": []
#         }
#     }
# }
# D4J = {
#     "1.4.0": {
#         "Closure": {
#             "begin": 127,
#             "end": 133,
#             "deprecate": []
#         }
#     }
# }

def run_project():
    for version in D4J:
        for proj in D4J[version]:
            for bug_id in range(D4J[version][proj]["begin"], D4J[version][proj]["end"] + 1):
                if bug_id in D4J[version][proj]["deprecate"]:
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
    run_project()
    # run_single()