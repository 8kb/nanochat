"""
Mechanical guard for datacore's standalone-ness: AST-scans every .py file under datacore/
(package and its own tests) and asserts none of them import anything from the host application
(nanochat, scripts, tasks, dev) or modelcore (a sibling standalone component, not a dependency).
This is what makes `cp -r datacore /somewhere/else` a real, testable claim rather than an
aspiration -- mirrors modelcore/tests/test_standalone.py.

A docstring or comment mentioning "nanochat" is fine (and common); only actual import statements
are checked.

python -m pytest datacore/tests/test_standalone.py -v
"""
import ast
import os

FORBIDDEN_TOP_LEVEL_PACKAGES = {"nanochat", "scripts", "tasks", "dev", "modelcore"}

DATACORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _iter_python_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _imported_top_level_packages(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # level > 0 is a relative import, always internal
                found.add(node.module.split(".")[0])
    return found


def test_no_python_file_under_datacore_imports_the_host_application():
    violations = {}
    for file_path in _iter_python_files(DATACORE_ROOT):
        found = _imported_top_level_packages(file_path) & FORBIDDEN_TOP_LEVEL_PACKAGES
        if found:
            violations[os.path.relpath(file_path, DATACORE_ROOT)] = sorted(found)
    assert not violations, (
        f"datacore/ must have zero imports from its host application or modelcore, but found: {violations}"
    )
