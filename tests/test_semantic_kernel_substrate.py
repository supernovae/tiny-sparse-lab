"""Software gates only; these do not measure a neural controller."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/research/semantic-kernel-mvp-v1/substrate.py"
SPEC = importlib.util.spec_from_file_location("kernel_substrate", SCRIPT)
kernel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel)


@pytest.fixture
def repo(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "sample.py").write_text(
        'raise RuntimeError("must never execute")\n'
        "def orbit(mass: float, /, *, dt=1) -> float:\n"
        '    """A real declaration, not a verified physics claim."""\n'
        "    return helper(mass)\n"
        "class Other:\n"
        "    def orbit(self):\n"
        "        return x.helper()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    return tmp_path


def test_compile_records_syntax_without_execution(repo):
    bank = kernel.compile_bank(repo)
    assert not bank["dirty"]
    result = kernel.execute(bank, {"op": "find", "args": {"name": "orbit"}})
    assert result["total"] == 2  # Ambiguity remains visible.
    record = kernel.execute(
        bank, {"op": "get", "args": {"id": result["records"][0]["id"]}}
    )["records"][0]
    assert record["arguments"] == "mass: float, /, *, dt=1"
    assert record["returns"] == "float"
    assert record["epistemic_status"] == "observed_syntax"
    calls = kernel.execute(bank, {"op": "call_sites", "args": {"name": "helper"}})
    assert calls["total"] == 2
    assert calls["target_resolution"] == "not_performed"


def test_integrity_and_no_overwrite(repo, tmp_path):
    bank = kernel.compile_bank(repo)
    path = tmp_path / "bank.json"
    kernel.write_new(path, bank)
    assert kernel.load_bank(path) == bank
    with pytest.raises(FileExistsError):
        kernel.write_new(path, bank)
    bank["records"][0]["name"] = "tampered"
    path.unlink()
    kernel.write_new(path, bank)
    with pytest.raises(ValueError, match="integrity"):
        kernel.load_bank(path)


@pytest.mark.parametrize(
    "action",
    [
        {"op": "execute", "args": {"name": "orbit"}},
        {"op": "get", "args": {"name": "orbit"}},
        {"op": "find", "args": {"name": "orbit", "path": "/etc/passwd"}},
        {"op": "find", "args": {"name": ""}},
        {"op": [], "args": {}},
        {"op": "find", "args": {"name": "orbit"}, "extra": True},
    ],
)
def test_invalid_actions_are_retained(repo, action):
    trace = kernel.replay(kernel.compile_bank(repo), [action])
    assert trace["invalid_actions"] == 1
    assert trace["events"][0]["action"] == action


def test_missing_and_step_limit(repo):
    bank = kernel.compile_bank(repo)
    action = {"op": "find", "args": {"name": "absent"}}
    assert kernel.execute(bank, action)["status"] == "not_found"
    with pytest.raises(ValueError, match="steps"):
        kernel.replay(bank, [action] * 17)


def test_bank_changes_when_source_changes(repo):
    first = kernel.compile_bank(repo)
    path = repo / "src/sample.py"
    path.write_text(path.read_text() + "\ndef newly_added(): pass\n")
    second = kernel.compile_bank(repo)
    assert second["dirty"]
    assert first["sha256"] != second["sha256"]
    assert first["sources"] != second["sources"]
    assert (
        kernel.execute(second, {"op": "find", "args": {"name": "newly_added"}})["total"]
        == 1
    )


def test_source_symlink_rejected(repo):
    (repo / "src/link.py").symlink_to(repo / "src/sample.py")
    with pytest.raises(ValueError, match="symlink"):
        kernel.compile_bank(repo)


def test_real_repository_has_grounded_prompt_definition():
    bank = kernel.compile_bank(ROOT)
    found = kernel.execute(bank, {"op": "find", "args": {"name": "format_chat_prompt"}})
    assert found["total"] >= 1
    for record in found["records"]:
        line = (ROOT / record["path"]).read_text().splitlines()[record["line"] - 1]
        assert "def format_chat_prompt" in line
