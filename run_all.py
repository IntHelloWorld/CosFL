import subprocess

D4J_ALL = {
    "1.4.0": {
        "Chart": {
            "begin": 1,
            "end": 26,
            "deprecate": [23]
        },
        "Closure": {
            "begin": 1,
            "end": 133,
            "deprecate": [63, 93]
        },
        "Lang": {
            "begin": 1,
            "end": 65,
            "deprecate": []
        },
        "Math": {
            "begin": 1,
            "end": 106,
            "deprecate": []
        },
        "Time": {
            "begin": 1,
            "end": 27,
            "deprecate": []
        },
        "Mockito": {
            "begin": 1,
            "end": 38,
            "deprecate": []
        },
    },
    "2.0.0": {
        "Cli": {
            "begin": 1,
            "end": 40,
            "deprecate": [6]
        },
        "Closure": {
            "begin": 134,
            "end": 176,
            "deprecate": []
        },
        "Codec": {
            "begin": 1,
            "end": 18,
            "deprecate": []
        },
        "Collections": {
            "begin": 25,
            "end": 28,
            "deprecate": []
        },
        "Compress": {
            "begin": 1,
            "end": 47,
            "deprecate": []
        },
        "Csv": {
            "begin": 1,
            "end": 16,
            "deprecate": []
        },
        "Gson": {
            "begin": 1,
            "end": 18,
            "deprecate": []
        },
        "JacksonCore": {
            "begin": 1,
            "end": 26,
            "deprecate": []
        },
        "JacksonDatabind": {
            "begin": 1,
            "end": 112,
            "deprecate": []
        },
        "JacksonXml": {
            "begin": 1,
            "end": 6,
            "deprecate": []
        },
        "Jsoup": {
            "begin": 1,
            "end": 93,
            "deprecate": []
        },
        "JxPath": {
            "begin": 1,
            "end": 22,
            "deprecate": []
        },
    },
}


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
            "begin": 15,
            "end": 133,
            "deprecate": [63, 93]
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