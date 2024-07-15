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
from functions.line_parser import (
    JavaClass,
    JavaMethod,
    parse_coverage,
    parse_stack_trace,
    parse_test_report,
    parse_test_run_log,
)
from functions.MethodExtractor.java_method_extractor import JavaMethodExtractor
from functions.my_types import JMethod, TestCase, TestClass, TestFailure
from functions.utils import clean_doc, git_clean, run_cmd
from Utils.context_manager import WorkDir
from Utils.path_manager import PathManager

filepath = os.path.dirname(__file__)
TREE_SITTER_LIB_PATH = os.path.join(
    filepath, "methodExtractor/myparser/my-languages.so")
root = os.path.dirname(filepath)
directory = os.path.join(root, "DebugResult")
AGENT_JAR = os.path.join(root, "functions/classtracer/target/classtracer-1.0.jar")


def check_out(path_manager: PathManager):
    with WorkDir(path_manager.res_path):
        if not os.path.exists(path_manager.buggy_path):
            run_cmd(f"defects4j checkout -p {path_manager.project} -v {path_manager.bug_id}b -w buggy")
        if not os.path.exists(path_manager.fixed_path):
            run_cmd(f"defects4j checkout -p {path_manager.project} -v {path_manager.bug_id}f -w fixed")


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
    out, err = run_cmd(f"defects4j compile -w {path_manager.buggy_path}")
    out, err = run_cmd(f"timeout 90 defects4j test -n -t {test_case.name} -w {path_manager.buggy_path}")
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
        cmd = f"defects4j test -n -w {path_manager.buggy_path} "\
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
    
    with open(test_output_file, "r") as f:
        test_output = f.readlines()
    with open(stack_trace_file, "r") as f:
        stack_trace = f.readlines()
    test_case.test_output = test_output
    test_case.stack_trace = stack_trace


def get_test_method(path_manager: PathManager,
                    test_class_name: str,
                    test_method_name: str,
                    stack_trace: str):
    buggy_path = path_manager.buggy_path
    test_path = path_manager.test_prefix
    test_file = os.path.join(
        buggy_path,
        test_path,
        test_class_name.replace(".", "/") + ".java"
    )

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Error: {test_file} not exists.")
    try:
        code = open(test_file, "r").read()
    except UnicodeDecodeError:
        print(f"Warning: UnicodeDecodeError for {test_file}.")
        code = open(test_file, "r", errors="ignore").readlines()

    function_extractor = JavaMethodExtractor(path_manager.tree_sitter_lib)
    methods = function_extractor.get_java_methods(code)
    for method in methods:
        if method.name == test_method_name:
            # location = parse_stack_trace(stack_trace)
            # if location != -1:
            #     code_line = code.split("\n")[location-1].strip("\n")
            #     method.code = method.code.replace(code_line, code_line + " // error occurred here")
            return method
    else:
        raise ValueError(f"Error: No method named {test_method_name} in {test_file}.")


def get_modified_methods(path_manager: PathManager):
    buggy_path = path_manager.buggy_path
    fixed_path = path_manager.fixed_path
    src_path = path_manager.src_prefix
    modified_classes = path_manager.modified_classes
    buggy_methods = []

    for class_name in modified_classes:
        buggy_file = os.path.join(buggy_path,
                                  src_path,
                                  class_name.replace(".", "/") + ".java")

        fixed_file = os.path.join(fixed_path,
                                  src_path,
                                  class_name.replace(".", "/") + ".java")
        
        if not (os.path.exists(fixed_file) and os.path.exists(buggy_file)):
            raise FileNotFoundError(f"Warning: {fixed_file} or {buggy_file} not exists.")
        
        try:
            buggy_code = open(buggy_file, "r").readlines()
        except UnicodeDecodeError:
            print(f"Warning: UnicodeDecodeError for {buggy_file}.")
            buggy_code = open(buggy_file, "r", errors="ignore").readlines()
        try:
            fixed_code = open(fixed_file, "r").readlines()
        except UnicodeDecodeError:
            print(f"Warning: UnicodeDecodeError for {fixed_file}.")
            fixed_code = open(fixed_file, "r", errors="ignore").readlines()

        function_extractor = JavaMethodExtractor(path_manager.tree_sitter_lib)
        buggy_methods.extend(function_extractor.get_buggy_methods(buggy_code, fixed_code))
    return buggy_methods


def refine_failed_tests(version, project, bugID):
    project_path = os.path.join("/home/qyh/projects/LLM-Location/AgentFL/DebugResult_d4j140_GPT35", f"d4j{version}-{project}-{bugID}")
    pickle_file = os.path.join(project_path, "test_failure.pkl")
    with open(pickle_file, "rb") as f:
        test_failure = pickle.load(f)
    
    check_out_path = os.path.join(project_path, "refine_check_out")
    check_out(version, project, bugID, check_out_path)
    buggy_path = os.path.join(check_out_path, "buggy")
    
    cmd = f"defects4j export -p dir.src.classes -w {buggy_path}"
    src_path, err = run_cmd(cmd)
    
    cmd = f"defects4j export -p classes.modified -w {buggy_path}"
    out, err = run_cmd(cmd)
    modified_classes = out.split("\n")
    
    buggy_methods = get_modified_methods(buggy_path, src_path, modified_classes)  # for evaluation
    test_failure.buggy_methods = buggy_methods
    
    with open(pickle_file, "wb") as f:
        pickle.dump(test_failure, f)
    
    run_cmd(f"rm -rf {check_out_path}")


def get_properties(path_manager: PathManager):
    """
    Retrieves properties related to the project using Defects4J.
    """
    cmd = f"defects4j export -p tests.trigger -w {path_manager.buggy_path}"
    out, err = run_cmd(cmd)
    path_manager.failed_test_names = out.split("\n")
    
    cmd = f"defects4j export -p dir.bin.classes -w {path_manager.buggy_path}"
    out, err = run_cmd(cmd)
    path_manager.src_class_prefix = out
    
    cmd = f"defects4j export -p dir.bin.tests -w {path_manager.buggy_path}"
    out, err = run_cmd(cmd)
    path_manager.test_class_prefix = out

    cmd = f"defects4j export -p dir.src.classes -w {path_manager.buggy_path}"
    out, err = run_cmd(cmd)
    path_manager.src_prefix = out

    cmd = f"defects4j export -p dir.src.tests -w {path_manager.buggy_path}"
    out, err = run_cmd(cmd)
    path_manager.test_prefix = out

    cmd = f"defects4j export -p classes.modified -w {path_manager.buggy_path}"
    out, err = run_cmd(cmd)
    path_manager.modified_classes = out.split("\n")


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
        if test_class_name not in test_classes:
            test_classes[test_class_name] = TestClass(test_class_name, [TestCase(test_name)])
        else:
            test_classes[test_class_name].test_cases.append(TestCase(test_name))

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

def parse_sbfl(path_manager: PathManager):
    """
    Parse the SBFL result from line level to method level.
    e.g.:
        org.jfree.chart.plot$CategoryPlot#CategoryPlot():567;0.2581988897471611
        org.jfree.chart.plot$CategoryPlot#CategoryPlot():568;0.2581988897471611
        
        ==>
        
        {
            "org.jfree.chart.plot$CategoryPlot": (567, 568)
        }
    """
    res = {}
    with open(path_manager.sbfl_file, "r") as f:
        line = f.readline() # skip the first line
        line = f.readline().strip("\n")
        while line:
            full_name, method_name, line_num, score = re.split(r"[#:;]", line)
            temp = full_name.split("$")
            if len(temp) > 2: # inner class
                full_name = temp[0] + "$" + temp[1]
            
            if score == "0.0":
                break

            try:
                res[full_name].append(int(line_num))
            except:
                res[full_name] = [int(line_num)]
            line = f.readline().strip("\n")
    return res

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
            test_cache_dir = os.path.join(path_manager.cache_path, test_class.name, test_case.name)
            os.makedirs(test_cache_dir, exist_ok=True)
            path_manager.test_cache_dir = test_cache_dir
            run_test_with_instrument(test_case, path_manager)
            test_case.test_method = get_test_method(
                path_manager,
                test_class.name,
                test_case.test_method_name,
                test_case.stack_trace
            )

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
    # test()
    refine_failed_tests("1.4.0", "Mockito", "30")