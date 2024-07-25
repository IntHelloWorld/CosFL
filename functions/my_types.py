from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class JMethod():
    name: str
    class_name: str
    param_types: List[str]
    return_type: str
    code: str
    comment: str
    text: str
    loc: Tuple[Tuple[int, int], Tuple[int, int]]
    class_full_name: Optional[str] = None
    
    def get_signature(self) -> str:
        return f"{self.return_type} {self.class_name}::{self.name}({','.join(self.param_types)})"
    
    def get_generics_re(self) -> str:
        """Return a regular expression to match the method signature with generics.
        """
        def is_generics(t: str) -> bool:
            if t.isupper() and len(t) == 1:
                return True
            elif t == "Object":
                return True
            return False
        
        rt = self.return_type
        if is_generics(rt):
            rt = "\w+"
        pts = [t if not is_generics(t) else "\w+" for t in self.param_types]
        re_string = f"{rt} {self.class_name}::{self.name}\({','.join(pts)}\)"
        re_string = (re_string.replace("$", "\$")
                              .replace(".", "\.")
                              .replace("[", "\[")
                              .replace("]", "\]"))
        return re_string
    
    def get_lined_code(self) -> str:
        return "\n".join([f"{i+1+self.loc[0][0]:4d} {line}" for i, line in enumerate(self.code.split("\n"))])
        

@dataclass
class TestCase():
    name: str
    test_method: Optional[JMethod] = None
    test_output: Optional[str] = None
    stack_trace: Optional[str] = None
    queries: Optional[List[str]] = None
    
    def __str__(self) -> str:
        return self.name
    
    def __post_init__(self):
        self.test_class_name, self.test_method_name = self.name.split("::")

@dataclass
class TestClass():
    name: str
    test_cases: List[TestCase]
    
    def __str__(self) -> str:
        return f"{self.name}: {str(self.test_cases)}"


@dataclass
class TestFailure():
    project: str
    bug_ID: int
    test_classes: List[TestClass]
    buggy_methods: Optional[List[JMethod]] = None
    queries: Optional[List[str]] = None





