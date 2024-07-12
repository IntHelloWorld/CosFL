import os
import sys
from difflib import unified_diff
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter
from tree_sitter import Language, Parser
from unidiff import PatchSet

sys.path.append(Path(__file__).resolve().parents[2].as_posix())
from functions.my_types import JMethod

CLASS_DECLARATION_TYPES = ["class_declaration", "interface_declaration", "enum_declaration", "enum_body_declaration"]
CLASS_BODY_TYPES = ["class_body", "interface_body", "enum_body"]
METHOD_DECLARATION_TYPES = ["method_declaration", "constructor_declaration"]

class JavaMethodExtractor:
    def __init__(self, lib_path: str) -> None:
        self.parser = Parser()
        self.parser.set_language(Language(Path(lib_path), "java"))

    def get_java_methods(self, code: str, only_class: str = None) -> List[JMethod]:
        """
        find all method declarations, including methods in inner classes.
        
        ignore the nested relationship between methods, which means when there are 
        inner-methods (such as methods of anonymous class or else) in a outer-method, 
        only keep the outer-method.
        
        args:
            code: str, java code
            only_class: str, if not None, only return methods in this class
        """

        def get_param_types(method_declaration):
            type_list = []
            c = method_declaration.child_by_field_name("parameters").named_children
            for param in c:
                if param.type == "spread_parameter":  # solve spread parameter, e.g., "final String[]..." -> "String[][]"
                    for child in param.children:
                        if child.type != "modifiers":
                            arg = bytes.decode(child.text) + "[]"
                            type_list.append(arg)
                            break
                else:
                    type_identifier = param.child_by_field_name("type")
                    if type_identifier.type == "scoped_type_identifier":  # e.g., "Node.Type" -> "Type"
                        arg = bytes.decode(type_identifier.named_children[-1].text)
                    else:
                        arg = bytes.decode(type_identifier.text)
                    
                    # solve array parameter, e.g., "String" -> "String[]"
                    dimension = param.child_by_field_name("dimensions")
                    if dimension is not None:
                        arg += "[]"

                    # remove type parameters. e.g., "List<String>" -> "List"
                    if "<" in arg:
                        arg = arg.split("<")[0]
                    type_list.append(arg)
            return type_list
        
        def get_return_type(method_declaration):
            c = method_declaration.child_by_field_name("type")
            if c is None:  # constructor
                return ""
            elif c.type == "generic_type":
                c = c.named_children[0]
            elif c.type == "scoped_type_identifier":  # e.g., "Node.Type" -> "Type"
                c = c.named_children[-1]
            return bytes.decode(c.text)

        def get_method_name(method_declaration):
            for child in method_declaration.children:
                if child.type == "identifier":
                    return bytes.decode(child.text)

        def get_class_name_for_method(node):
            nonlocal loc2cname
            while True:
                if node.type in CLASS_BODY_TYPES:
                    break
                node = node.parent
            return loc2cname[node.byte_range]

        def get_method_object(node: tree_sitter.Node) -> JMethod:
            class_name = get_class_name_for_method(node)
            if only_class is not None:
                if class_name != only_class:
                    return None
            method_code = "\n".join(code_list[node.start_point[0]: node.end_point[0] + 1])
            method_name = get_method_name(node)
            param_types = get_param_types(node)
            return_type = get_return_type(node)
            method_location = (node.start_point, node.end_point)
            if "comment" in node.prev_sibling.type:
                method_comment = bytes.decode(node.prev_sibling.text)
                method_text = "\n".join(code_list[node.prev_sibling.start_point[0]: node.end_point[0] + 1])
            else:
                method_comment = ""
                method_text = method_code
            method = JMethod(method_name,
                             class_name,
                             param_types,
                             return_type,
                             method_code,
                             method_comment,
                             method_text,
                             method_location)
            return method
        
        def get_ancestor_class_body(class_body_node):
            node = class_body_node.parent
            while True:
                if node.type in CLASS_BODY_TYPES:
                    return node
                node = node.parent
                if node is None:
                    break
            return None
        
        def is_declared_class(class_body):
            if class_body.parent.type in CLASS_DECLARATION_TYPES:
                return True
            return False
        
        def get_class_name_for_class_body(class_body):
            if is_declared_class(class_body):  # declared class
                classes = []
                tmp_node = class_body.parent
                while True:
                    if tmp_node is None:
                        break
                    if tmp_node.type in CLASS_DECLARATION_TYPES:
                        for child in tmp_node.children:
                            if child.type == "identifier":
                                classes.insert(0, bytes.decode(child.text))
                    tmp_node = tmp_node.parent
                assert len(classes) > 0, "class name not found"
                return "$".join(classes)
            else:  # anonymous class
                nonlocal loc2cname, counter
                ancestor_body = get_ancestor_class_body(class_body)
                ancestor_cname = loc2cname[ancestor_body.byte_range]
                counter[ancestor_cname] += 1
                return ancestor_cname + "$" + str(counter[ancestor_cname])

        def dfs(node):
            node_childs = node.children
            if len(node_childs) == 0:
                return
            for child in node_childs:
                if child.type in METHOD_DECLARATION_TYPES:
                    method = get_method_object(child)
                    if method is not None:
                        nonlocal methods
                        methods.append(method)
                    dfs(child)
                elif child.type in CLASS_BODY_TYPES:
                    class_name = get_class_name_for_class_body(child)
                    loc = child.byte_range
                    nonlocal loc2cname, counter
                    counter[class_name] = 0
                    loc2cname[loc] = class_name
                    dfs(child)
                else:
                    dfs(child)

        loc2cname = {}
        methods = []
        counter = {}
        code_list = code.split("\n")
        tree = self.parser.parse(bytes(code, "utf8"))
        dfs(tree.root_node)
        return methods

    def get_buggy_methods(self, buggy_file: List[str], fixed_file: List[str]):
        methods = self.get_java_methods("".join(buggy_file))
        diff = list(unified_diff(buggy_file, fixed_file,
                    fromfile='text1',
                    tofile='text2',
                    n=0))
        assert len(diff) != 0, "buggy file and fixed file are the same"
        hunks = PatchSet("".join(diff))[0]
        changed_points_b = set()
        for hunk in hunks:
            changed_points_b.add(hunk.source_start)
            changed_points_b.add(hunk.source_start + hunk.source_length - 1)
        changed_buggy_methods = []
        for method in methods:
            loc = method.loc
            start = loc[0][0] + 1
            end = loc[1][0] + 1
            for point in changed_points_b:
                if start <= point <= end:
                    changed_buggy_methods.append(method)
                    break
        return changed_buggy_methods