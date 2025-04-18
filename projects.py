import os
import shutil

ALL_BUGS = {
    # "d4j1.4.0": {
    #     'Chart': (list(range(1, 27)), [23]),
    #     'Closure': (list(range(1, 134)), [28, 49, 57, 63, 93, 104, 106, 125]),
    #     'Lang': (list(range(1, 66)), [2, 23, 25, 43, 56]),
    #     'Math': (list(range(1, 107)), [3, 12, 14, 35, 41, 67, 104]),
    #     'Mockito': (list(range(1, 39)), [5, 16, 26]),
    #     'Time': (list(range(1, 28)), [11, 21])
    # },
    # "d4j2.0.1": {
    #     'Cli': (list(range(1, 41)), [6, 7, 36]),
    #     'Closure': (list(range(134, 177)), [144, 169]),
    #     'Codec': (list(range(1, 19)), [12, 13, 16]),
    #     'Collections': (list(range(1, 29)), list(range(1, 25)) + [28]),
    #     'Compress': (list(range(1, 48)), [7, 42, 44]),
    #     'Csv': (list(range(1, 17)), [12]),
    #     'Gson': (list(range(1, 19)), [9, 14, 16, 18]),
    #     'JacksonCore': (list(range(1, 27)), list(range(21, 27)) + [9]),
    #     'JacksonDatabind': (list(range(1, 113)), [20, 21, 23, 26, 34, 36, 37, 38, 40, 41, 43, 44, 45, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112]),
    #     'JacksonXml': (list(range(1, 7)), []),
    #     'Jsoup': (list(range(1, 94)), [4, 9, 17, 25, 71]),
    #     # 'JxPath': (list(range(1, 23)), list(range(1, 21)) + [22]),
    # },
    "GrowingBugs": {
        'IO': (list(range(1, 32)), [4, 7, 19, 20, 21, 23, 24, 26, 28], ""),
        'Validator': (list(range(1, 26)), [3, 5, 6, 10, 11, 12, 15, 16, 18, 20, 21, 22, 23, 25], ""),
        'Javapoet': (list(range(1, 18)), [1], ""),
        'Zip4j': (list(range(1, 53)), [15, 25, 42, 43, 50], ""),
        'Spoon': (list(range(1, 18)), [5], ""),
        'Markedj': (list(range(1, 18)), [2, 8, 13, 17], ""),
        'Dagger_core': (list(range(1, 21)), [5], "core"),
    }
}

# Math-6

# ALL_BUGS = {
#     "d4j1.4.0": {
#         'Chart': (list(range(1, 27)), [23]),
#         'Closure': (list(range(1, 134)), [28, 49, 57, 63, 93, 104, 106, 125]),
#     },
# }

def delete_deprecated():
    projects_dir = "/home/qyh/projects/GarFL/Projects"
    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            for bug_id in ALL_BUGS[version][proj][1]:
                bug_dir = os.path.join(projects_dir, proj, str(bug_id))
                if os.path.exists(bug_dir):
                    print(f"Delete {bug_dir}")
                    shutil.rmtree(bug_dir)

def statistic_bugs():
    for version in ALL_BUGS:
        print(f"Version: {version}")
        print("PROJECT, ALL, Deprecated, Bugs")
        for proj in ALL_BUGS[version]:
            all = len(ALL_BUGS[version][proj][0])
            deprecated = len(ALL_BUGS[version][proj][1])
            print(f"{proj}, {all}, {deprecated}, {all-deprecated}")

def statistic_methods():
    import json
    from pathlib import Path
    
    results = []
    root_path = Path(__file__).resolve().parents[0].as_posix()
    stores_dir = os.path.join(root_path, "Stores", "codeqwen-1_5-7B")
    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            for bug_id in ALL_BUGS[version][proj][0]:
                if bug_id in ALL_BUGS[version][proj][1]:
                    continue
                doc_store = os.path.join(stores_dir, proj, "docstore.json")
                method_nodes = json.load(open(doc_store, "r"))
                results.append(f"{proj}: {len(method_nodes['docstore/data'])}")
                break
    print(results)

if __name__ == "__main__":
    # delete_deprecated()
    # statistic_bugs()
    statistic_methods()