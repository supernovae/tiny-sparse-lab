"""Measure unique retained v3 corpus text by upstream source, code language and shape.

No tokenizer or training mixture is selected here. Classification uses pinned source
IDs and source paths, not NLP guesses about scholarly or generic-web content.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LANGUAGES = {
    ".py": "Python",
    ".go": "Go",
    ".rs": "Rust",
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".hh": "C++",
    ".hxx": "C++",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "SQL",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".rb": "Ruby",
    ".lua": "Lua",
    ".tcl": "Tcl",
    ".s": "assembly",
}
CONFIG_EXTENSIONS = {
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".conf",
    ".cfg",
    ".proto",
    ".xml",
}
SOURCE_FAMILIES = {
    "v2_git_technical",
    "v2_go_faq",
    "v2_go_standard",
    "v2_k8s_concepts",
    "v2_linux_fs",
    "v2_oci_runtime_specification",
    "v2_omp_harness",
    "v2_otel_collector",
    "v2_otel_specification",
    "v2_postgres_backend",
    "v2_prometheus_promql",
    "v2_python_library",
    "v2_rust_core",
}


def shape(path: str, suffix: str) -> str:
    lower = path.lower()
    name = Path(lower).name
    if re.search(r"(^|/)(tests?|testing|__tests__)(/|$)|(^|/)(test_|.*_test\.)", lower):
        return "tests"
    if re.search(r"(^|/)(examples?|samples?|tutorials?)(/|$)", lower):
        return "examples"
    if re.search(r"(^|/)(rfcs?|adrs?|design|architecture|proposals?)(/|$)", lower):
        return "design"
    if re.search(
        r"(^|/)(admin|ops|operation|runbooks?|deploy|installation|upgrade|troubleshoot)(/|$)",
        lower,
    ):
        return "operations"
    if re.search(r"(^|/)(schemas?|api-spec|specs?)(/|$)", lower) or suffix == ".proto":
        return "schemas"
    if re.search(r"(error|status|diagnostic|troubleshoot)", name):
        return "errors_status"
    if name in {
        "makefile",
        "cmakelists.txt",
        "meson.build",
        "build",
        "build.bazel",
        "dockerfile",
    } or suffix in {".mk", ".bzl", ".gradle"}:
        return "build"
    if suffix in CONFIG_EXTENSIONS:
        return "config"
    if suffix in LANGUAGES or suffix in {".h", ".cs", ".pl", ".m"}:
        return "implementation"
    return "manual_reference"


def measure(release: Path) -> dict:
    sources = {
        item["id"]: item for item in json.loads((release / "sources.json").read_text())
    }
    counters = defaultdict(Counter)
    source_summary = defaultdict(Counter)
    retained_content = 0
    with (release / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            source_id = item["source_id"]
            source_summary[source_id]["normalized_documents"] += 1
            if item["drop_reason"] is not None:
                source_summary[source_id]["removed_documents"] += 1
                continue
            text_bytes = len(item["text"].encode("utf-8"))
            source_summary[source_id]["retained_documents"] += 1
            source_summary[source_id]["retained_utf8_bytes"] += text_bytes
            split = item["split"]
            source_summary[source_id][split + "_utf8_bytes"] += text_bytes
            source_summary[source_id][split + "_documents"] += 1
            if split == "train":
                retained_content += text_bytes
            # Only first-party engineer-authored sources count as developer data.
            if not (source_id.startswith("v3_") or source_id in SOURCE_FAMILIES):
                if split == "train":
                    counters["other_train"]["documents"] += 1
                    counters["other_train"]["utf8_bytes"] += text_bytes
                continue
            pool = "developer_train" if split == "train" else "developer_heldout"
            counters[pool]["documents"] += 1
            counters[pool]["utf8_bytes"] += text_bytes
            path = item["source_location"].split("#", 1)[0]
            suffix = Path(path).suffix.lower()
            family = shape(path, suffix)
            counters["shapes_by_split"][split + ":" + family + ":documents"] += 1
            counters["shapes_by_split"][split + ":" + family + ":utf8_bytes"] += (
                text_bytes
            )
            if split == "train":
                counters["shapes"][family + ":documents"] += 1
                counters["shapes"][family + ":utf8_bytes"] += text_bytes
            language = LANGUAGES.get(suffix)
            if suffix == ".h":
                language = (
                    "C++"
                    if "cpp" in path or "cxx" in path
                    else "C/C++ header (ambiguous)"
                )
            if language:
                counters["languages_by_split"][
                    split + ":" + language + ":documents"
                ] += 1
                counters["languages_by_split"][
                    split + ":" + language + ":utf8_bytes"
                ] += text_bytes
                if split == "train":
                    counters["languages"][language + ":documents"] += 1
                    counters["languages"][language + ":utf8_bytes"] += text_bytes
    developer = counters["developer_train"]["utf8_bytes"]
    other = counters["other_train"]["utf8_bytes"]
    exposure = {
        f"{round(percent * 100)}_percent": {
            "developer_passes_needed_relative_to_one_other_pass": percent
            * other
            / ((1 - percent) * developer)
            if developer
            else None,
            "note": "Byte-proxy planning only; no sampling schedule or tokenizer tokens selected",
        }
        for percent in (0.2, 0.3, 0.4)
    }
    return {
        "release_id": json.loads((release / "manifest.json").read_text())["release_id"],
        "train_retained_utf8_bytes": retained_content,
        "source_counts": {
            k: dict(sorted(v.items())) for k, v in sorted(source_summary.items())
        },
        "source_kinds": {k: v["kind"] for k, v in sorted(sources.items())},
        "developer_selection": "Retained train counts v3 train sources and nine inherited v2 train engineering sources; heldout shape/language counts separately include the four inherited v2 engineering heldouts. No peS2o reclassification.",
        "pool": {k: dict(sorted(v.items())) for k, v in sorted(counters.items())},
        "exposure_byte_proxy": exposure,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(measure(args.release), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
