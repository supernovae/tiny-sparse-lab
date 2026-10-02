"""Read-only AST interface pilot; no source execution, model, or training."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import time
from pathlib import Path

ABI = "sparselab-source-abi-v1"
MAX_STEPS = 16
MAX_RESULTS = 64
MAX_FILE_BYTES = 2_000_000
MAX_BANK_BYTES = 64_000_000
MAX_ACTION_BYTES = 65_536


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        )
        stream.write("\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, timeout=10
    ).strip()


class Inventory(ast.NodeVisitor):
    def __init__(self, path: str, source_sha256: str) -> None:
        self.path = path
        self.source_sha256 = source_sha256
        self.scope: list[str] = []
        self.records: list[dict] = []
        self.calls: list[dict] = []

    def definition(self, node: ast.AST, kind: str) -> None:
        name = node.name
        qualified = ".".join([*self.scope, name])
        record = {
            "id": f"{self.path}::{qualified}@{node.lineno}",
            "name": name,
            "qualified_name": qualified,
            "kind": kind,
            "path": self.path,
            "line": node.lineno,
            "end_line": node.end_lineno,
            "source_sha256": self.source_sha256,
            "epistemic_status": "observed_syntax",
            "docstring": (ast.get_docstring(node) or "")[:2000],
            "arguments": ast.unparse(node.args) if kind != "class" else None,
            "returns": ast.unparse(node.returns)
            if kind != "class" and node.returns
            else None,
        }
        self.records.append(record)
        self.scope.append(name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.definition(node, "class")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.definition(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.definition(node, "async_function")

    def visit_Call(self, node: ast.Call) -> None:
        # A spelling is evidence of syntax, NOT a resolved target identity.
        spelling = ast.unparse(node.func)
        if isinstance(node.func, (ast.Name, ast.Attribute)):
            self.calls.append(
                {
                    "spelling": spelling[:500],
                    "path": self.path,
                    "line": node.lineno,
                    "scope": ".".join(self.scope),
                    "source_sha256": self.source_sha256,
                    "epistemic_status": "observed_syntax_unresolved_target",
                }
            )
        self.generic_visit(node)


def compile_bank(repo: Path) -> dict:
    repo = repo.resolve()
    if not (repo / "src").is_dir():
        raise ValueError("repository must contain src/")
    if git(repo, "rev-parse", "--show-toplevel") != str(repo):
        raise ValueError("--repo must be the Git repository root")
    records, calls, sources = [], [], []
    for path in sorted((repo / "src").rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to(repo / "src"):
            raise ValueError("source symlinks are not allowed")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"source exceeds byte limit: {path.name}")
        raw = path.read_bytes()
        relative = path.relative_to(repo).as_posix()
        sha = hashlib.sha256(raw).hexdigest()
        # ast.parse accepts Python encoding declarations in bytes.
        visitor = Inventory(relative, sha)
        visitor.visit(ast.parse(raw, filename=relative))
        records.extend(visitor.records)
        calls.extend(visitor.calls)
        sources.append({"path": relative, "sha256": sha, "bytes": len(raw)})
    body = {
        "format": "sparselab-source-bank",
        "version": 1,
        "abi": ABI,
        "source_commit": git(repo, "rev-parse", "HEAD"),
        "dirty": bool(git(repo, "status", "--porcelain", "--untracked-files=normal")),
        "compiler_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_license": "See repository LICENSE; no independent rights determination",
        "sources": sources,
        "records": records,
        "call_sites": calls,
    }
    body["sha256"] = digest(body)
    if len(canonical(body)) > MAX_BANK_BYTES:
        raise ValueError("compiled bank exceeds byte limit")
    return body


def read_json(path: Path, max_bytes: int) -> object:
    if path.stat().st_size > max_bytes:
        raise ValueError("input exceeds byte limit")
    return json.loads(path.read_text(encoding="utf-8"))


def load_bank(path: Path) -> dict:
    bank = read_json(path, MAX_BANK_BYTES)
    if not isinstance(bank, dict):
        raise TypeError("bank must be an object")
    body = {key: value for key, value in bank.items() if key != "sha256"}
    if bank.get("sha256") != digest(body):
        raise ValueError("bank integrity mismatch")
    if (bank.get("format"), bank.get("version"), bank.get("abi")) != (
        "sparselab-source-bank",
        1,
        ABI,
    ):
        raise ValueError("unsupported bank/ABI")
    for field in ("sources", "records", "call_sites"):
        if not isinstance(bank.get(field), list):
            raise TypeError(f"invalid bank field: {field}")
    ids = [record["id"] for record in bank["records"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate definition IDs")
    return bank


def execute(bank: dict, action: object) -> dict:
    if not isinstance(action, dict) or set(action) != {"op", "args"}:
        raise ValueError("action requires exactly op and args")
    op, args = action["op"], action["args"]
    if not isinstance(op, str) or op not in {"find", "get", "call_sites"}:
        raise ValueError("unknown operation")
    expected = "id" if op == "get" else "name"
    if not isinstance(args, dict) or set(args) != {expected}:
        raise ValueError(f"operation requires exactly {expected}")
    value = args[expected]
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("argument must be a nonempty bounded string")
    if op == "get":
        found = [record for record in bank["records"] if record["id"] == value]
    elif op == "find":
        found = [
            {
                key: record[key]
                for key in ("id", "name", "kind", "path", "line", "source_sha256")
            }
            for record in bank["records"]
            if record["name"] == value
        ]
    else:
        found = [
            record
            for record in bank["call_sites"]
            if record["spelling"] == value or record["spelling"].endswith("." + value)
        ]
    return {
        "status": "found" if found else "not_found",
        "bank_sha256": bank["sha256"],
        "total": len(found),
        "truncated": len(found) > MAX_RESULTS,
        "records": found[:MAX_RESULTS],
        "scope": "compiled_src_python_only",
        "target_resolution": "not_performed" if op == "call_sites" else None,
    }


def replay(bank: dict, actions: object) -> dict:
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_STEPS:
        raise ValueError(f"actions must contain 1..{MAX_STEPS} steps")
    events = []
    for index, action in enumerate(actions):
        start = time.perf_counter()
        try:
            result = execute(bank, action)
        except ValueError as exc:
            result = {"status": "invalid_action", "error": str(exc)}
        events.append(
            {
                "step": index,
                "action": action,
                "observation": result,
                "operation_wall_seconds": time.perf_counter() - start,
            }
        )
    body = {
        "format": "sparselab-source-replay",
        "version": 1,
        "abi": ABI,
        "bank_sha256": bank["sha256"],
        "actions_sha256": digest(actions),
        "controller": "externally_supplied_actions_no_model",
        "events": events,
        "steps": len(events),
        "invalid_actions": sum(
            event["observation"]["status"] == "invalid_action" for event in events
        ),
    }
    return {**body, "sha256": digest(body)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compile_parser = sub.add_parser("compile")
    compile_parser.add_argument("--repo", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    query_parser = sub.add_parser("query")
    query_parser.add_argument("--bank", type=Path, required=True)
    query_parser.add_argument("--action", required=True)
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--bank", type=Path, required=True)
    replay_parser.add_argument("--actions", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compile":
        start = time.perf_counter()
        bank = compile_bank(args.repo)
        write_new(args.output, bank)
        print(
            json.dumps(
                {
                    "bank_sha256": bank["sha256"],
                    "definitions": len(bank["records"]),
                    "call_sites": len(bank["call_sites"]),
                    "compiler_wall_seconds": time.perf_counter() - start,
                    "output_bytes": args.output.stat().st_size,
                }
            )
        )
    elif args.command == "query":
        if len(args.action.encode("utf-8")) > MAX_ACTION_BYTES:
            raise ValueError("action exceeds byte limit")
        print(
            json.dumps(execute(load_bank(args.bank), json.loads(args.action)), indent=2)
        )
    else:
        trace = replay(load_bank(args.bank), read_json(args.actions, MAX_ACTION_BYTES))
        write_new(args.output, trace)
        print(
            json.dumps(
                {
                    "trace_sha256": trace["sha256"],
                    "steps": trace["steps"],
                    "invalid_actions": trace["invalid_actions"],
                }
            )
        )


if __name__ == "__main__":
    main()
