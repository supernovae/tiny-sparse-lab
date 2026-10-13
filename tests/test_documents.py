"""The shared decoder has no schema, path-binding or execution authority."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from sparselab.documents import read_document
from sparselab.experiments.plan import read_document as plan_read_document


@pytest.mark.parametrize("suffix", ["yaml", "json", "JSON"])
def test_data_parity_and_no_path_interpretation(tmp_path, suffix):
    raw = {"plugin": {"path": "opaque:relative", "future_field": True}, "seed": "7"}
    path = tmp_path / f"input.{suffix}"
    path.write_text(yaml.safe_dump(raw) if suffix == "yaml" else json.dumps(raw))
    assert read_document(path) == raw
    assert plan_read_document is read_document
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "suffix,text,match",
    [
        ("yaml", "nested: {key: 1, key: 2}", "duplicate YAML key"),
        ("json", '{"nested":{"key":1,"key":2}}', "duplicate JSON key"),
        ("yaml", "nested: {1: value}", "keys must be strings"),
        ("yaml", "nested: [.nan]", "non-finite"),
        ("yaml", "nested: [.inf]", "non-finite"),
        ("json", '{"nested":[NaN]}', "non-finite"),
        ("json", '{"nested":[Infinity]}', "non-finite"),
        ("json", '{"nested":[1e999]}', "non-finite"),
        ("yaml", "nested: &cycle [*cycle]", "cyclic"),
        (
            "yaml",
            "nested: !!python/object/apply:os.system [echo]",
            "invalid experiment",
        ),
        ("yaml", "nested: [", "invalid experiment"),
        ("yaml", "<<: {seed: 7}\nseed: 8", "invalid experiment"),
        ("yaml", "[]", "must be a mapping"),
        ("yaml", "", "must be a mapping"),
    ],
)
def test_rejected_data_has_source(tmp_path, suffix, text, match):
    path = tmp_path / f"bad.{suffix}"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError), match=match) as error:
        read_document(path)
    assert str(path) in str(error.value)


def test_duplicate_diagnostic_has_yaml_line_and_column(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("nested:\n  key: 1\n  key: 2\n")
    with pytest.raises(ValueError, match="line 3, column 3"):
        read_document(path)


@pytest.mark.parametrize(
    "suffix,text",
    [
        ("yaml", "fraction: 0.12345678901234567890123456789"),
        ("json", '{"fraction":0.12345678901234567890123456789}'),
    ],
)
def test_exact_numbers_preserve_authored_precision(tmp_path, suffix, text):
    path = tmp_path / f"exact.{suffix}"
    path.write_text(text)
    assert read_document(path, exact_decimals=True)["fraction"] == Decimal(
        "0.12345678901234567890123456789"
    )
    assert isinstance(read_document(path)["fraction"], float)


def test_repeated_alias_is_not_a_cycle(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("a: &value [1, 2]\nb: *value\n")
    assert read_document(path) == {"a": [1, 2], "b": [1, 2]}


def test_document_import_and_read_do_not_reach_schema_or_runtime(tmp_path):
    path = tmp_path / "input.yaml"
    path.write_text("plugin: {path: relative}\n")
    code = """
import importlib.abc
import sys
from pathlib import Path
class Fence(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'datasets', 'tokenizers', 'pydantic'} or (
            fullname.startswith('sparselab.') and fullname != 'sparselab.documents'
        ):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Fence())
from sparselab.documents import read_document
assert read_document(Path(sys.argv[1])) == {'plugin': {'path': 'relative'}}
"""
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, cwd=tmp_path)
    assert list(tmp_path.iterdir()) == [path]


def test_missing_and_invalid_utf8_have_source(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match=str(path)):
        read_document(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match=str(path)):
        read_document(path)


def test_plan_schema_is_separate_and_does_not_open_artifacts(tmp_path):
    from sparselab.experiments.plan import load_plan

    path = tmp_path / "plan.yaml"
    path.write_text("plan_version: 1\nid: parsed\nbase_run: absent.yaml\n")
    plan = load_plan(path)
    assert plan.base_run == "absent.yaml"
    assert list(tmp_path.iterdir()) == [path]
    path.write_text(path.read_text() + "unknown: true\n")
    assert read_document(path)["unknown"] is True
    with pytest.raises(ValueError, match="extra_forbidden"):
        load_plan(path)
    assert list(tmp_path.iterdir()) == [path]


def test_model_components_do_not_import_workflow_consumers():
    """Fence the computational package, not adapters with known coupling debt."""
    import ast

    import sparselab

    root = Path(sparselab.__file__).parent / "model"
    forbidden = (
        "sparselab.experiments",
        "sparselab.campaign",
        "sparselab.lab_mode",
        "sparselab.research",
        "sparselab.cli",
    )
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
                if node.module == "sparselab":
                    imports += ["sparselab." + name.name for name in node.names]
            elif isinstance(node, ast.Import):
                imports = [name.name for name in node.names]
            else:
                continue
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in imports
                for prefix in forbidden
            ), (path, imports)
