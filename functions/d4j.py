import copy
import json
import os
import pickle
import re
import shutil
import sys
from functools import reduce
from typing import Dict, List, Tuple

from numpy import full

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME

from functions.line_parser import (
    JavaClass,
    JavaMethod,
    parse_coverage,
    parse_stack_trace,
    parse_test_report,
    parse_test_run_log,
    read_text,
)
from functions.MethodExtractor.java_method_extractor import JavaMethodExtractor
from functions.my_types import JMethod, TestCase, TestClass, TestFailure
from functions.utils import auto_read, clean_doc, git_clean, run_cmd
from Utils.context_manager import WorkDir
from Utils.path_manager import PathManager

filepath = os.path.dirname(__file__)
TREE_SITTER_LIB_PATH = os.path.join(
    filepath, "methodExtractor/my-languages.so")
root = os.path.dirname(filepath)
directory = os.path.join(root, "DebugResult")


def check_out(path_manager: PathManager):
    if not path_manager.all_methods and os.path.exists(path_manager.method_nodes_file):
        return
    with WorkDir(path_manager.proj_tmp_path):
        if not os.path.exists(path_manager.buggy_path):
            if path_manager.subproj:
                run_cmd(f"{path_manager.bug_exec} checkout -p {path_manager.project} -v {path_manager.bug_id}b -w buggy -s {path_manager.subproj}")
            else:
                run_cmd(f"{path_manager.bug_exec} checkout -p {path_manager.project} -v {path_manager.bug_id}b -w buggy")
        if not os.path.exists(path_manager.fixed_path):
            if path_manager.subproj:
                run_cmd(f"{path_manager.bug_exec} checkout -p {path_manager.project} -v {path_manager.bug_id}f -w fixed -s {path_manager.subproj}")
            else:
                run_cmd(f"{path_manager.bug_exec} checkout -p {path_manager.project} -v {path_manager.bug_id}f -w fixed")


def run_single_test(test_case: TestCase, path_manager: PathManager):
    test_output_dir = os.path.join(path_manager.cache_path,
                                   test_case.test_class_name,
                                   test_case.name)
    os.makedirs(test_output_dir, exist_ok=True)
    test_output_file = os.path.join(test_output_dir, "test_output.txt")
    stack_trace_file = os.path.join(test_output_dir, "stack_trace.txt")
    if os.path.exists(test_output_file) and os.path.exists(stack_trace_file):
        with open(test_output_file, "r") as f:
            test_output = f.readlines()
        with open(stack_trace_file, "r") as f:
            stack_trace = f.readlines()
        return test_output, stack_trace
    
    git_clean(path_manager.buggy_path)
    out, err = run_cmd(f"{path_manager.bug_exec} compile -w {path_manager.buggy_path}")
    out, err = run_cmd(f"timeout 90 {path_manager.bug_exec} test -n -t {test_case.name} -w {path_manager.buggy_path}")
    with open(f"{path_manager.buggy_path}/failing_tests", "r") as f:
        test_res = f.readlines()
    test_output, stack_trace = parse_test_report(test_res)
    with open(test_output_file, "w") as f:
        f.writelines(test_output)
    with open(stack_trace_file, "w") as f:
        f.writelines(stack_trace)
    return test_output, stack_trace

def run_test_with_instrument(test_case: TestCase, path_manager: PathManager):
    loaded_classes_file = os.path.join(path_manager.test_cache_dir, "load.log")
    inst_methods_file = os.path.join(path_manager.test_cache_dir, "inst.log")
    run_methods_file = os.path.join(path_manager.test_cache_dir, "run.log")
    test_output_file = os.path.join(path_manager.test_cache_dir, "test_output.txt")
    stack_trace_file = os.path.join(path_manager.test_cache_dir, "stack_trace.txt")
    all_files = [loaded_classes_file, inst_methods_file, run_methods_file, test_output_file, stack_trace_file]
    class_path = os.path.join(path_manager.buggy_path, path_manager.src_class_prefix)

    if (all(os.path.exists(f) for f in all_files)):
        path_manager.logger.info("[run all tests]     instrumentation already done, skip!")
    else:
        shutil.rmtree(path_manager.test_cache_dir, ignore_errors=True)
        os.makedirs(path_manager.test_cache_dir, exist_ok=True)
        git_clean(path_manager.buggy_path)
        cmd = f"{path_manager.bug_exec} test -n -w {path_manager.buggy_path} "\
            f"-t {test_case.name} "\
            f"-a -Djvmargs=-javaagent:{path_manager.agent_lib}=outputDir={path_manager.test_cache_dir},classesPath={class_path}"
        run_cmd(cmd)
        with open(f"{path_manager.buggy_path}/failing_tests", "r") as f:
            test_res = f.readlines()
        test_output, stack_trace = parse_test_report(test_res)
        with open(test_output_file, "w") as f:
            f.writelines(test_output)
        with open(stack_trace_file, "w") as f:
            f.writelines(stack_trace)
        assert all(os.path.exists(f) for f in all_files)
    
    test_case.test_output = read_text(test_output_file, max_lines=100)
    test_case.stack_trace = read_text(stack_trace_file, max_lines=50)


def get_test_method(path_manager: PathManager,
                    test_class_name: str,
                    test_method_name: str):
    buggy_path = path_manager.buggy_path
    test_path = path_manager.test_prefix
    test_file = os.path.join(
        buggy_path,
        test_path,
        test_class_name.replace(".", "/") + ".java"
    )

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Error: {test_file} not exists.")
    
    code = auto_read(test_file)

    function_extractor = JavaMethodExtractor(TREE_SITTER_LIB_PATH)
    methods = function_extractor.get_java_methods(code)
    assert len(methods) > 0, f"Error: No method found in {test_file}."
    for method in methods:
        if method.name == test_method_name:
            return method
    else:
        # the test method may be in the father class
        try:
            dot_idx = test_class_name.rfind(".")
            pkg_name = test_class_name[:dot_idx]
            short_name = test_class_name[dot_idx + 1:]
            match_cls = re.search(rf"{short_name}\s+extends\s+(\w+)", code)
            f_class_name = match_cls.group(1)
            match_pkg = re.search(rf"import\s+([\w.]+).{f_class_name};", code)
            f_pkg_name = match_pkg.group(1) if match_pkg else pkg_name
            f_class_full_name = f_pkg_name + "." + f_class_name
            
            return get_test_method(path_manager, f_class_full_name, test_method_name)
        except Exception:
            raise ValueError(f"Error: No method named {test_method_name} in {test_file}.")


def get_modified_methods(path_manager: PathManager):
    buggy_path = path_manager.buggy_path
    fixed_path = path_manager.fixed_path
    src_path = path_manager.src_prefix
    modified_classes = path_manager.modified_classes
    buggy_methods = []

    for class_name in modified_classes:
        
        # fix errors in GrowingBugs
        if path_manager.project == "IO":
            extra_prefix = src_path.replace("/", ".") + "."
            class_name = class_name.replace(extra_prefix, "")
        elif path_manager.project == "Dagger_core":
            extra_prefix = "core."
            class_name = class_name.replace(extra_prefix, "")
        
        buggy_file = os.path.join(buggy_path,
                                  src_path,
                                  class_name.replace(".", "/") + ".java")

        fixed_file = os.path.join(fixed_path,
                                  src_path,
                                  class_name.replace(".", "/") + ".java")
        
        if not (os.path.exists(fixed_file) and os.path.exists(buggy_file)):
            raise FileNotFoundError(f"Warning: {fixed_file} or {buggy_file} not exists.")
        
        buggy_code = auto_read(buggy_file)
        
        fixed_code = auto_read(fixed_file)

        function_extractor = JavaMethodExtractor(TREE_SITTER_LIB_PATH)
        methods = function_extractor.get_buggy_methods(buggy_code, fixed_code)
        for method in methods:
            method.class_full_name = class_name
            method.text = method.text.replace("\r", "")
        buggy_methods.extend(methods)
    return buggy_methods


def get_properties(path_manager: PathManager):
    """
    Retrieves properties related to the project.
    """
    if os.path.exists(os.path.join(path_manager.bug_path, "properties.json")):
        with open(os.path.join(path_manager.bug_path, "properties.json"), "r") as f:
            properties = json.load(f)
    else:
        properties = {}
        
        # for some project such as Pool we have to compile first
        cmd = f"{path_manager.bug_exec} compile -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        
        cmd = f"{path_manager.bug_exec} export -p tests.trigger -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        properties["failed_test_names"] = out.split("\n")
        
        cmd = f"{path_manager.bug_exec} export -p dir.bin.classes -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        properties["src_class_prefix"] = out
        
        cmd = f"{path_manager.bug_exec} export -p dir.bin.tests -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        properties["test_class_prefix"] = out

        cmd = f"{path_manager.bug_exec} export -p dir.src.classes -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        properties["src_prefix"] = out

        cmd = f"{path_manager.bug_exec} export -p dir.src.tests -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        properties["test_prefix"] = out

        cmd = f"{path_manager.bug_exec} export -p classes.modified -w {path_manager.buggy_path}"
        out, err = run_cmd(cmd)
        properties["modified_classes"] = out.split("\n")
    
        with open(os.path.join(path_manager.bug_path, "properties.json"), "w") as f:
            json.dump(properties, f, indent=4)
    
    path_manager.failed_test_names = properties["failed_test_names"]
    path_manager.src_class_prefix = properties["src_class_prefix"]
    path_manager.test_class_prefix = properties["test_class_prefix"]
    path_manager.src_prefix = properties["src_prefix"]
    path_manager.test_prefix = properties["test_prefix"]
    path_manager.modified_classes = properties["modified_classes"]


def get_failed_tests(path_manager: PathManager) -> TestFailure:
    """Get the TestFailure object for a defect4j bug.
    """
    
    try:
        with open(path_manager.test_failure_file, "rb") as f:
            test_failure = pickle.load(f)
            print(f"Load cached TestFailure object from {path_manager.test_failure_file}")
            return test_failure
    except FileNotFoundError:
        pass

    # initialize test failure
    test_classes = {}
    for test_name in path_manager.failed_test_names:
        test_class_name, test_method_name = test_name.split("::")
        test_case = TestCase(test_name)
        test_case.test_method = get_test_method(
            path_manager,
            test_class_name,
            test_case.test_method_name,
        )
        if test_class_name not in test_classes:
            test_classes[test_class_name] = TestClass(test_class_name, [test_case])
        else:
            test_classes[test_class_name].test_cases.append(test_case)

    # get modified methods as the buggy methods for evaluation
    path_manager.logger.info("[get test failure object] get modified methods as the buggy methods for evaluation...")
    buggy_methods = get_modified_methods(path_manager)
    
    path_manager.logger.info("[get test failure object] construct the TestFailure object...")
    test_failure = TestFailure(path_manager.project,
                               path_manager.bug_id,
                               list(test_classes.values()),
                               buggy_methods)
    
    with open(path_manager.test_failure_file, "wb") as f:
        pickle.dump(test_failure, f)
        path_manager.logger.info(f"[get test failure object] Save failed tests to {path_manager.test_failure_file}")

    return test_failure


def merge_classes(class_name: str, covered_classes: List[Dict[str, JavaClass]]) -> JavaClass:
    merged_class = JavaClass(class_name)
    all_covered_methods = [[m for m in c[class_name].methods.values() if m._covered] for c in covered_classes]
    spc_methods = {}
    for covered_methods in all_covered_methods:
        for method in covered_methods:
            if method.inst_id not in spc_methods:
                spc_methods[method.inst_id] = method
    if len(spc_methods) == 0:  # no suspicious methods, which means nether of the methods in the class can be buggy
        return None
    merged_class.methods = spc_methods
    return merged_class

def filter_classes_Ochiai(project, bugID, extracted_classes: List[JavaClass]) -> List[JavaClass]:
    """
    Filter the classes according to the top 20 result of Ochiai (https://github.com/Instein98/D4jOchiai).
    """
    def parse_ochiai(path):
        """
        Parse the Ochiai result from line level to method level.
        """
        res = []
        with open(path, "r") as f:
            line = f.readline()
            line = f.readline()
            while line:
                name, _ = line.split(";")
                name = name.split(":")[0]
                if res == []:
                    res.append(name)
                else:
                    if name != res[-1]:
                        res.append(name)
                if len(res) == 20:
                    break
                line = f.readline()
        return res
    
    ochiai_res_path = os.path.join("functions/OchiaiResult", project, str(bugID), "ochiai.ranking.csv")
    if not os.path.exists(ochiai_res_path):
        print(f"Warning: No Ochiai result for {project}-{bugID}")
        return []
    ochiai_res = parse_ochiai(ochiai_res_path)
    filtered_classes = []
    bug_result_dict = {}
    for m in ochiai_res:
        class_name = m.split("#")[0].replace("$", ".")
        method_name = m.split("#")[1].split("(")[0]
        if class_name not in bug_result_dict:
            bug_result_dict[class_name] = [method_name]
        else:
            if method_name not in bug_result_dict[class_name]:
                bug_result_dict[class_name].append(method_name)
    
    # filter out useless classes and methods
    for javaclass in extracted_classes:
        if javaclass.class_name in bug_result_dict:
            new_javaclass = copy.deepcopy(javaclass)
            for inst_id in javaclass.methods:
                inst_method_name = inst_id.split("::")[1].split("(")[0]
                if inst_method_name not in bug_result_dict[javaclass.class_name]:
                    new_javaclass.methods.pop(inst_id)
            filtered_classes.append(new_javaclass)
    return filtered_classes


def filter_classes_Grace(project, bugID, extracted_classes: List[JavaClass]) -> List[JavaClass]:
    """
    Filter the classes according to the top 10 result of Grace (https://github.com/yilinglou/Grace/tree/master).
    """
    filtered_classes = []
    with open("functions/grace_result_dict.json", "r") as f:
        grace_result = json.load(f)
    if str(bugID) not in grace_result[project]:
        print(f"Warning: No Grace result for {project}-{bugID}")
        return filtered_classes
    bug_result = grace_result[project][str(bugID)]["top10_result"]
    bug_result_dict = {}
    for m in bug_result:
        class_name = m.split(":")[0].split("$")[0]
        method_name = m.split(":")[1].split("(")[0]
        if class_name not in bug_result_dict:
            bug_result_dict[class_name] = [method_name]
        else:
            if method_name not in bug_result_dict[class_name]:
                bug_result_dict[class_name].append(method_name)
    
    # filter out useless classes and methods
    for javaclass in extracted_classes:
        if javaclass.class_name in bug_result_dict:
            new_javaclass = copy.deepcopy(javaclass)
            for inst_id in javaclass.methods:
                inst_method_name = inst_id.split("::")[1].split("(")[0]
                if inst_method_name not in bug_result_dict[javaclass.class_name]:
                    new_javaclass.methods.pop(inst_id)
            filtered_classes.append(new_javaclass)
    return filtered_classes

def run_all_tests(path_manager: PathManager, test_failure: TestFailure):
    """
    Extract loaded java classes for a test suite (witch may contains multiple test methods)
    according to the method coverage information.
    """

    for test_class in test_failure.test_classes:
        path_manager.logger.info(f"[run all tests] test class: {path_manager.project}-{path_manager.bug_id} {test_class.name}")
        for test_case in test_class.test_cases:
            path_manager.logger.info(f"[run all tests]   \u14AA test case: {path_manager.project}-{path_manager.bug_id} {test_case.name}")
            test_cache_dir = os.path.join(path_manager.bug_path, test_class.name, test_case.name)
            os.makedirs(test_cache_dir, exist_ok=True)
            path_manager.test_cache_dir = test_cache_dir
            run_test_with_instrument(test_case, path_manager)

def get_class_name_from_msg(tmp_path, test_class):
    """
    Some buggy classes may have low method level coverage proportion rank because of the crash, 
    so we add these classes according to the error messages.
    """
    
    def get_target_classes(match):
        target_classes = []
        class_name = match.split(".")[-1]
        class_names = re.findall(r"[A-Z][a-zA-Z0-9]*", class_name)
        for class_name in class_names:
            if "Tests" in class_name:
                target_classes.append(class_name.replace("Tests", ""))
            elif "Test" in class_name:
                target_classes.append(class_name.replace("Test", ""))
            else:
                target_classes.append(class_name)
        return target_classes
    
    extra_class_names = set()
    for test_case in test_class.test_cases:
        test_name = test_case.name
        test_tmp_dir = os.path.join(tmp_path, test_class.name, test_name)
        stack_trace_file = os.path.join(test_tmp_dir, "stack_trace.txt")
        with open(stack_trace_file, "r") as f:
            stack_trace = f.read()
        matches = re.findall(r'\b(?:\w*\.)+[A-Z]\w*', stack_trace)
        matches = list(set(matches))
        candidates = []
        for match in matches:
            candidates.extend(get_target_classes(match))
        for candidate in candidates:
            extra_class_names.add(candidate)
    return list(extra_class_names)


def test():
    run_test_with_instrument(
        "com.google.javascript.jscomp.TypeCheckTest::testBadInterfaceExtendsNonExistentInterfaces",
        "/home/qyh/projects/LLM-Location/preprocess/Defects4J-1.2.0/Closure/2/buggy",
        "/home/qyh/projects/LLM-Location/AgentFL/DebugResult",
        "/home/qyh/projects/LLM-Location/preprocess/classtracer/target/classtracer-1.0.jar",
        "test"
    )


if __name__ == "__main__":
    test()
