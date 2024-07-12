import json
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions.MethodExtractor.java_method_extractor import JavaMethodExtractor
from functions.my_types import JMethod, TestCase, TestClass, TestFailure
from Utils.path_manager import PathManager


class World:
    def __init__(self, path_manager: PathManager, test_case: TestCase):
        
        self.world_map = defaultdict(dict)
        
        # load call graph
        call_dict = defaultdict(dict)  # store the raw call graph
        class_dict = defaultdict(set)  # store the methods need to be extracted in each class
        method_dict = defaultdict(JMethod)  # store the extracted methods

        call_graph_file = path_manager.get_call_graph_file(test_case.name)
        assert os.path.exists(call_graph_file), f"{call_graph_file} not exists!"
        with open(call_graph_file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                data = json.loads(line)
                method = data["method"]
                signature = World.make_signature(method["returnType"],
                                                 method["className"],
                                                 method["methodName"],
                                                 method["paramTypes"])
                call_dict[data["callSite"]][data["calledMethodName"]] = signature
                class_dict[method["className"]].add(signature)
        
        # extract methods
        for class_name in class_dict:
            short_class_name = class_name.split(".")[-1]
            package_name = ".".join(class_name.split(".")[:-1])
            class_file = path_manager.get_class_file(class_name)
            if class_file is None:
                continue
            try:
                code = open(class_file, "r").read()
            except UnicodeDecodeError as e:
                path_manager.logger.warning(f"UnicodeDecodeError: {class_file}")
                code = open(class_file, "r", errors="ignore").read()
            function_extractor = JavaMethodExtractor(path_manager.tree_sitter_lib)
            java_methods = function_extractor.get_java_methods(code, short_class_name)
            src_dict = {}
            for method in java_methods:
                method.class_name = package_name + "." + method.class_name
                signature = method.get_signature()
                src_dict[signature] = method
            
            
            for signature in class_dict[class_name]:
                if signature not in src_dict:
                    method_name = signature.split("(")[0].split("::")[-1]
                    if re.match(r"access\$\d+", method_name):
                        continue
                    if method_name == "values":
                        continue
                    
                    class_name = signature.split("::")[0].split(".")[1]
                    pattern = r"{}::\w+".format(class_name)
                    
                    ok = False
                    inst_params = signature.split("(")[1][:-1].split(",")
                    inst_return_type = signature.split(" ")[0]
                    if "Object" in inst_params or "Object" == inst_return_type:
                        pattern = r"{}".format(signature.replace("$", "\$")
                                               .replace(".", "\.")
                                               .replace("(", "\(")
                                               .replace("[", "\[")
                                               .replace("]", "\]")
                                               .replace(")", "\)")
                                               .replace("Object", "\w+"))
                        for src_sig in src_dict:
                            if re.match(pattern, src_sig):
                                method_dict[signature] = src_dict[src_sig]
                                ok = True
                                break
                    if ok:
                        continue
                    
                    # solve generics, e.g., "A" -> "Object"
                    ok = False
                    for sig in src_dict:
                        if "processResults" in sig:
                            print(1)
                        re_string = src_dict[sig].get_generics_re()
                        if re.match(r"{re}".format(re=re_string), signature):
                            method_dict[signature] = src_dict[sig]
                            ok = True
                            break
                    if ok:
                        continue
                    
                    path_manager.logger.warning(f"Method <{signature}> not found in {class_file}")
                    continue
                method_dict[signature] = src_dict[signature]
        
        for callSite in call_dict:
            for called_method in call_dict[callSite]:
                signature = call_dict[callSite][called_method]
                if signature in method_dict:
                    self.world_map[callSite][called_method] = method_dict[signature]
        
        print(1)
    
    @staticmethod
    def make_signature(return_type: str, class_name: str, method_name: str, param_types: list):
        return f"{return_type.split('.')[-1].split('$')[-1]} "\
               f"{class_name}::{method_name}"\
               f"({','.join([t.split('.')[-1].split('$')[-1] for t in param_types])})"