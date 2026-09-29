"""Measure frozen v4 retained text, languages and engineering shapes without loading the corpus.

Shapes are path/extension evidence, not claims about every document's instructional
quality. Primary shapes partition developer text; material labels intentionally
overlap and report their own denominators. No tokenizer or mixture is selected.
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
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".mts": "TypeScript",
    ".cts": "TypeScript",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "SQL",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".rb": "Ruby",
    ".lua": "Lua",
    ".tcl": "Tcl",
    ".s": "assembly",
}
DEV_V2 = frozenset(
    {
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
)
CONFIG_SUFFIXES = frozenset(
    {
        ".yaml",
        ".yml",
        ".json",
        ".jsonnet",
        ".toml",
        ".ini",
        ".conf",
        ".cfg",
        ".properties",
        ".hcl",
        ".tf",
        ".tfvars",
        ".service",
        ".socket",
        ".timer",
        ".target",
        ".mount",
        ".path",
    }
)
BUILD_SUFFIXES = frozenset({".cmake", ".mk", ".bzl", ".gradle"})
BUILD_NAMES = frozenset(
    {
        "makefile",
        "gnumakefile",
        "cmakelists.txt",
        "meson.build",
        "build",
        "build.bazel",
        "dockerfile",
        "cargo.toml",
        "go.mod",
        "package.json",
        "pom.xml",
        "build.gradle",
        "settings.gradle",
    }
)
CODE_SUFFIXES = frozenset(LANGUAGES) | {".h", ".cs", ".pl", ".m"}
TEST_PART = re.compile(r"(^|/)(tests?|testing|__tests__)(/|$)|(^|/)(test_|.*_test\.)")
EXAMPLE_PART = re.compile(r"(^|/)(examples?|samples?|tutorials?)(/|$)")
DESIGN_PART = re.compile(r"(^|/)(rfcs?|adrs?|design|architecture|proposals?)(/|$)")
OPERATIONS_PART = re.compile(
    r"(^|/)(admin|ops|operation|runbooks?|deploy|installation|install|"
    r"upgrade|backup|restore|recovery|troubleshoot|migration|migrate)(/|$)"
    r"|(^|/)(install|upgrade|backup|restore|recovery|troubleshoot|"
    r"deployment|runbook)[^/]*\."
)
ERROR_PART = re.compile(
    r"(error|status|diagnostic|troubleshoot|warning|failure|incident)"
)
SCHEMA_PART = re.compile(
    r"(^|/)(schemas?|api-spec|specs?|crds?)(/|$)|schema|openapi|swagger"
)


def describe_path(path: str, text: str) -> tuple[str | None, str, set[str]]:
    """Classify one source location; material tags intentionally overlap."""
    lower = path.lower()
    name = Path(lower).name
    suffix = Path(lower).suffix
    language = LANGUAGES.get(suffix)
    if suffix == ".h":
        language = (
            "C++" if "cpp" in lower or "cxx" in lower else "C/C++ header (ambiguous)"
        )
    if language is None and text.startswith("#!"):
        first = text.split("\n", 1)[0]
        if re.search(r"(?:/|\s)(?:ba|z|da)?sh(?:\s|$)", first):
            language = "shell"
    is_test = bool(TEST_PART.search(lower))
    is_example = bool(EXAMPLE_PART.search(lower))
    is_design = bool(DESIGN_PART.search(lower))
    is_operation = bool(OPERATIONS_PART.search(lower))
    is_error = bool(ERROR_PART.search(name))
    is_schema = suffix == ".proto" or bool(SCHEMA_PART.search(lower))
    is_build = name in BUILD_NAMES or suffix in BUILD_SUFFIXES
    is_config = suffix in CONFIG_SUFFIXES or is_build or is_schema
    tags = set()
    if is_test:
        tags.add("tests")
    if is_example:
        tags.add("examples")
    if is_design:
        tags.add("design")
    if is_operation:
        tags.add("operations_runbook")
    if is_error:
        tags.add("errors_status")
    if is_schema:
        tags.add("schemas_api")
    if is_build:
        tags.add("build")
    if is_config:
        tags.add("configuration")
    if (
        suffix in {".tf", ".tfvars", ".hcl"}
        or "playbook" in lower
        or "/charts/" in lower
    ):
        tags.add("infrastructure_as_code")
    if suffix == ".sql":
        tags.add("sql_file")
    if suffix == ".test" and (
        "/sql/" in lower or "/sqllogictest/" in lower or "mysql-test/" in lower
    ):
        tags.add("sql_mixed_test_harness")
    if suffix in CODE_SUFFIXES or language == "shell":
        tags.add("code")
    if suffix in {".md", ".rst", ".adoc", ".txt", ".man", ".1", ".5", ".7", ".8"}:
        tags.add("manuals_reference")
    if is_test:
        primary = "tests"
    elif is_example:
        primary = "examples"
    elif is_design:
        primary = "design"
    elif is_operation:
        primary = "operations_runbook"
    elif is_schema:
        primary = "schemas_api"
    elif is_error:
        primary = "errors_status"
    elif is_build:
        primary = "build"
    elif is_config:
        primary = "configuration"
    elif language or suffix in CODE_SUFFIXES:
        primary = "implementation"
    else:
        primary = "manuals_reference"
    return language, primary, tags


def measure(release: Path) -> dict:
    sources = {
        item["id"]: item for item in json.loads((release / "sources.json").read_text())
    }
    counters: dict[str, Counter] = defaultdict(Counter)
    source_summary: dict[str, Counter] = defaultdict(Counter)
    retained_content = 0
    with (release / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            source_id = item["source_id"]
            counts = source_summary[source_id]
            counts["normalized_documents"] += 1
            if item["drop_reason"] is not None:
                counts["removed_documents"] += 1
                continue
            text_bytes = len(item["text"].encode("utf-8"))
            split = item["split"]
            counts["retained_documents"] += 1
            counts["retained_utf8_bytes"] += text_bytes
            counts[split + "_utf8_bytes"] += text_bytes
            counts[split + "_documents"] += 1
            if split == "train":
                retained_content += text_bytes
            if not (source_id.startswith(("v3_", "v4_")) or source_id in DEV_V2):
                if split == "train":
                    counters["other_train"]["documents"] += 1
                    counters["other_train"]["utf8_bytes"] += text_bytes
                continue
            pool = "developer_train" if split == "train" else "developer_heldout"
            counters[pool]["documents"] += 1
            counters[pool]["utf8_bytes"] += text_bytes
            path = item["source_location"].split("#", 1)[0]
            language, primary, tags = describe_path(path, item["text"])
            counters["shapes_by_split"][split + ":" + primary + ":documents"] += 1
            counters["shapes_by_split"][split + ":" + primary + ":utf8_bytes"] += (
                text_bytes
            )
            if split == "train":
                counters["shapes"][primary + ":documents"] += 1
                counters["shapes"][primary + ":utf8_bytes"] += text_bytes
            for tag in tags:
                counters["material_by_split"][split + ":" + tag + ":documents"] += 1
                counters["material_by_split"][split + ":" + tag + ":utf8_bytes"] += (
                    text_bytes
                )
                if split == "train":
                    counters["material"][tag + ":documents"] += 1
                    counters["material"][tag + ":utf8_bytes"] += text_bytes
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
        f"{percent}%": {
            "developer_passes_needed_relative_to_one_other_pass": (
                percent * other / ((100 - percent) * developer) if developer else None
            ),
            "note": "Byte-proxy only; no tokenizer, training mixture or sampler selected",
        }
        for percent in (20, 25, 30, 35, 40)
    }
    return {
        "release_id": json.loads((release / "manifest.json").read_text())["release_id"],
        "train_retained_utf8_bytes": retained_content,
        "source_counts": {
            k: dict(sorted(v.items())) for k, v in sorted(source_summary.items())
        },
        "source_kinds": {k: v["kind"] for k, v in sorted(sources.items())},
        "developer_selection": "Only pinned v2 engineer sources and all v3/v4 sources; no peS2o relabeling. Heldout separately measured. Primary shapes partition developer text; material tags overlap.",
        "pool": {k: dict(sorted(v.items())) for k, v in sorted(counters.items())},
        "exposure_byte_proxy": exposure,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(measure(args.release), indent=2, sort_keys=True) + "\n"
    )
