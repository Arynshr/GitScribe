from gitscribe.core.indexer.graph_builder import SymbolResolver, build_edges
from gitscribe.core.indexer.parser import Symbol


def make_symbol(name, kind, file, parent=None, calls=None, bases=None):
    return Symbol(
        name=name,
        kind=kind,
        file=file,
        lineno=1,
        end_lineno=1,
        parent=parent,
        calls=calls or [],
        bases=bases or [],
    )


def test_resolve_unique_repo_wide_match():
    symbols = [(1, make_symbol("helper", "function", "a.py"))]
    resolver = SymbolResolver(symbols)
    assert resolver.resolve("helper", "b.py") == 1


def test_resolve_prefers_same_file():
    symbols = [
        (1, make_symbol("helper", "function", "a.py")),
        (2, make_symbol("helper", "function", "b.py")),
    ]
    resolver = SymbolResolver(symbols)
    assert resolver.resolve("helper", "a.py") == 1
    assert resolver.resolve("helper", "b.py") == 2


def test_resolve_ambiguous_repo_wide_returns_none():
    """Two same-named symbols in different files, called from a third
    file with no local match -> ambiguous, left unresolved rather than
    guessing.
    """
    symbols = [
        (1, make_symbol("helper", "function", "a.py")),
        (2, make_symbol("helper", "function", "b.py")),
    ]
    resolver = SymbolResolver(symbols)
    assert resolver.resolve("helper", "c.py") is None


def test_resolve_missing_name_returns_none():
    resolver = SymbolResolver([(1, make_symbol("foo", "function", "a.py"))])
    assert resolver.resolve("does_not_exist", "a.py") is None


def test_build_edges_import_always_unresolved():
    symbols = [(1, make_symbol("os", "import", "a.py"))]
    edges = build_edges(symbols)
    assert len(edges) == 1
    assert edges[0].edge_type == "imports"
    assert edges[0].resolved is False
    assert edges[0].target_id is None


def test_build_edges_resolved_call():
    symbols = [
        (1, make_symbol("helper", "function", "a.py")),
        (2, make_symbol("caller", "function", "a.py", calls=["helper"])),
    ]
    edges = build_edges(symbols)
    call_edges = [e for e in edges if e.edge_type == "calls"]
    assert len(call_edges) == 1
    assert call_edges[0].resolved is True
    assert call_edges[0].target_id == 1
    assert call_edges[0].source_id == 2


def test_build_edges_unresolved_call_still_recorded():
    """External/unknown call names still produce an edge row (per Stage 2
    scope decision) — not silently dropped.
    """
    symbols = [(1, make_symbol("caller", "function", "a.py", calls=["print"]))]
    edges = build_edges(symbols)
    assert len(edges) == 1
    assert edges[0].resolved is False
    assert edges[0].target_name == "print"


def test_build_edges_inherits():
    symbols = [
        (1, make_symbol("Base", "class", "a.py")),
        (2, make_symbol("Child", "class", "a.py", bases=["Base"])),
    ]
    edges = build_edges(symbols)
    inherit_edges = [e for e in edges if e.edge_type == "inherits"]
    assert len(inherit_edges) == 1
    assert inherit_edges[0].target_id == 1
    assert inherit_edges[0].resolved is True
