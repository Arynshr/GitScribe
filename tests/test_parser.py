import ast

from gitscribe.core.indexer.parser import SymbolVisitor


def parse_source(src: str):
    tree = ast.parse(src)
    visitor = SymbolVisitor("fake_file.py")
    visitor.visit(tree)
    return visitor.symbols


def test_top_level_function():
    symbols = parse_source("def foo():\n    pass\n")
    assert len(symbols) == 1
    assert symbols[0].name == "foo"
    assert symbols[0].kind == "function"
    assert symbols[0].parent is None


def test_class_with_method():
    src = """
class Foo:
    def bar(self):
        pass
"""
    symbols = parse_source(src)
    names = {(s.name, s.kind, s.parent) for s in symbols}
    assert ("Foo", "class", None) in names
    assert ("bar", "method", "Foo") in names


def test_nested_function_not_top_level():
    """Nested functions are tracked in calls but excluded from the
    top-level symbol list per the Stage 2 'track but don't embed' rule.
    """
    src = """
def outer():
    def inner():
        pass
    return inner
"""
    symbols = parse_source(src)
    names = [s.name for s in symbols]
    assert names == ["outer"]
    assert "inner" not in names


def test_method_calls_captured():
    src = """
class Foo:
    def bar(self):
        helper()
        self.other()
"""
    symbols = parse_source(src)
    method = next(s for s in symbols if s.name == "bar")
    assert "helper" in method.calls
    assert "other" in method.calls


def test_class_bases_captured():
    src = """
class Base:
    pass

class Child(Base):
    pass
"""
    symbols = parse_source(src)
    child = next(s for s in symbols if s.name == "Child")
    assert child.bases == ["Base"]


def test_import_and_import_from():
    src = "import os\nfrom pathlib import Path\n"
    symbols = parse_source(src)
    kinds = {s.name: s.kind for s in symbols}
    assert kinds["os"] == "import"
    assert kinds["pathlib.Path"] == "import"


def test_import_with_alias():
    symbols = parse_source("import numpy as np\n")
    assert symbols[0].name == "np"
    assert symbols[0].kind == "import"


def test_async_function_captured():
    symbols = parse_source("async def fetch():\n    pass\n")
    assert len(symbols) == 1
    assert symbols[0].name == "fetch"
    assert symbols[0].kind == "function"
