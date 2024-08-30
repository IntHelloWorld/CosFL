import os
import subprocess
import sys

from projects import ALL_BUGS

root = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root)

def run_all_bugs(config_name: str, clear: bool = True):
    for version in ALL_BUGS:
        for proj in ALL_BUGS[version]:
            bugIDs = ALL_BUGS[version][proj][0]
            deprecatedIDs = ALL_BUGS[version][proj][1]
            subproj = ALL_BUGS[version][proj][2] if version == "GrowingBugs" else "\"\""
            for bug_id in bugIDs:
                res_path = f"DebugResult/{config_name}/{version}/{proj}/{proj}-{bug_id}"
                res_path = os.path.join(root, res_path)
                res_file = os.path.join(res_path, "result.json")
                if bug_id in deprecatedIDs:
                    continue
                
                if os.path.exists(res_file):
                    print(f"{version}-{proj}-{bug_id} already finished, skip!")
                    continue
                
                run_one_bug(config_name, version, proj, bug_id, clear, subproj)

def run_one_bug(config_name, version, proj, bug_id, clear, subproj):
    cmd = f"python TF-IDF/run.py --config {config_name} --version {version} --project {proj} --bugID {bug_id} --clear {clear} --subproj {subproj}"
    result = subprocess.run(cmd.split(" "))
    if result.returncode != 0:
        raise Exception(f"Failed to run {version}-{proj}-{bug_id}")

if __name__ == "__main__":
    config_name = "TF-IDF"
    # run_one_bug(config_name, "d4j1.4.0", "Chart", 1, True, "")
    run_all_bugs(config_name, True)