"""
AST-based symbol extraction
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from gitscribe.core.diff_parser import filter_ignored_files, load_ignore_spec

SymbolKind = Literal["function", "class", "method", "import"]


class Symbol(BaseModel):
    name: str
    kind: SymbolKind
    file: str
    lineno: int
    end_lineno: int | None = None
    parent: str | None = None 
    calls: list[str] = Field(default_factory=list)


class SymbolVisitor(ast.NodeVisitor):
    """Walks a single module's AST and collects symbols.
    Scope rule: methods are recorded with `parent` set to the enclosing
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.symbols: list[Symbol] = []
        self._class_stack: list[str] = []
        self._func_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(
            Symbol(
                name=node.name,
                kind="class",
                file=self.file_path,
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", None),
                parent=self._class_stack[-1] if self._class_stack else None,
            )
        )
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_nested = bool(self._func_stack)
        in_class = bool(self._class_stack) and not is_nested
        kind = "method" if in_class else "function"

        sym = Symbol(
            name=node.name,
            kind=kind,
            file=self.file_path,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            parent=self._class_stack[-1] if in_class else None,
        )
        sym.calls = self._extract_calls(node)

        # Nested functions are tracked but flagged, not treated as
        # first-class symbols for embedding (per Stage 2 scope decision).
        if not is_nested:
            self.symbols.append(sym)

        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.symbols.append(
                Symbol(
                    name=alias.asname or alias.name,
                    kind="import",
                    file=self.file_path,
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for alias in node.names:
            self.symbols.append(
                Symbol(
                    name=f"{module}.{alias.name}" if module else alias.name,
                    kind="import",
                    file=self.file_path,
                    lineno=node.lineno,
                    end_lineno=getattr(node, "end_lineno", None),
                )
            )

    @staticmethod
    def _extract_calls(node: ast.AST) -> list[str]:
        """Raw call names within a function body. Resolution to symbol
        IDs (intra-repo vs external/unresolved) happens in graph_builder,
        not here — parser stays single-responsibility: extraction only.
        """
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    calls.append(func.id)
                elif isinstance(func, ast.Attribute):
                    calls.append(func.attr)
        return calls


def parse_file(path: str) -> list[Symbol]:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=path)
    visitor = SymbolVisitor(path)
    visitor.visit(tree)
    return visitor.symbols


def discover_python_files(repo_root: str = ".") -> list[str]:
    """Reuses diff_parser's ignore-spec logic instead of reimplementing
    ignore handling — single source of truth for what gets excluded.
    """
    spec = load_ignore_spec(repo_root=repo_root)
    all_py = [str(p) for p in Path(repo_root).rglob("*.py")]
    return filter_ignored_files(all_py, spec)


def parse_repo(repo_root: str = ".") -> list[Symbol]:
    symbols: list[Symbol] = []
    for file_path in discover_python_files(repo_root):
        try:
            symbols.extend(parse_file(file_path))
        except SyntaxError as e:
            print(f"[skip] {file_path}: {e}")
    return symbols


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    for sym in parse_repo(root):
        scope = f"{sym.parent}." if sym.parent else ""
        print(f"{sym.kind:9} {sym.file}:{sym.lineno:<5} {scope}{sym.name}")
