import os
from pathlib import Path

from java_method_extractor import JavaMethodExtractor


def test_nested_anoymous_class():
    os.chdir(Path(__file__).resolve().parents[0])
    extractor = JavaMethodExtractor("./my-languages.so")
    with open("./Example/NestedAnonymousClass.java", "r") as f:
        code = f.read()
    # tree = extractor.parser.parse(bytes(code, "utf8"))
    methods = extractor.get_java_methods(code)
    pass

def test_anoymous_class():
    os.chdir(Path(__file__).resolve().parents[0])
    extractor = JavaMethodExtractor("./my-languages.so")
    with open("./Example/AnonymousClass.java", "r") as f:
        code = f.read()
    tree = extractor.parser.parse(bytes(code, "utf8"))
    # methods = extractor.get_java_methods(code)
    pass

def test_enmu_anoymous_class():
    os.chdir(Path(__file__).resolve().parents[0])
    extractor = JavaMethodExtractor("./my-languages.so")
    with open("./Example/Enmu+AnonymousClass.java", "r") as f:
        code = f.read()
    # tree = extractor.parser.parse(bytes(code, "utf8"))
    methods = extractor.get_java_methods(code)
    pass

if __name__ == "__main__":
    # test_anoymous_class()
    # test_enmu_anoymous_class()
    test_nested_anoymous_class()