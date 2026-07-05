from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_setup_py_declares_quant_extra_for_optimum_quanto():
    tree = ast.parse((PROJECT_ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    extras_keyword = next(kw for kw in setup_call.keywords if kw.arg == "extras_require")
    extras = ast.literal_eval(extras_keyword.value)

    assert "quant" in extras
    assert "optimum-quanto>=0.2.7,<0.3" in extras["quant"]
