"""Conservative title-only engineering incidence in a frozen corpus release.

This is a topical screening heuristic, not a paper subject classifier or a source
license decision. It reads one document at a time and never changes its split.
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TOPICS = {
    "software_engineering": (
        r"software (engineering|development|testing|maintenance|verification|architecture|defect|bug|repository|repositories)",
        r"(source|program) code (analysis|generation|search|completion|repair|review)",
        r"(code|program) (generation|completion|repair|review|analysis|synthesis|search)",
        r"(static|dynamic) (program|code) analysis",
        r"(automated|unit|integration|regression) (software )?testing",
        r"(program|software) (verification|synthesis|debugging)",
        r"(github|gitlab|stackoverflow|stack overflow) (repositories|projects|commits|issues|developers|posts)",
    ),
    "programming_languages": (
        r"programming languages?",
        r"(compiler|interpreter|type system|type checking|program semantics|runtime system)s?",
        r"(python|javascript|typescript|golang|rust|java|c\+\+|haskell) (programs?|libraries|packages|code|compiler|runtime|developer)",
    ),
    "operating_systems": (
        r"\boperating systems?\b",
        r"(linux|unix|posix) (kernel|filesystem|system call|scheduler|driver)",
        r"(filesystem|file system|system call|process schedul)(s|ing|er|ers)?",
        r"(container|virtual machine|hypervisor) (runtime|orchestration|isolation|security|performance)",
    ),
    "distributed_networking": (
        r"distributed (systems?|databases?|computing|storage|consensus|transaction|tracing)",
        r"(network|tcp|http|dns|routing|packet) (protocol|stack|performance|security|traffic)",
        r"(cloud native|kubernetes|microservice|service mesh|serverless) (applications?|architectures?|systems?|deployments?|orchestration|platforms?)",
        r"(raft|paxos) (consensus|algorithm|protocol|replication)",
    ),
    "data_security_tools": (
        r"(database|sql|query optimizer|transaction processing|storage engine|file system) (systems?|performance|design|implementation|benchmark|optimization)",
        r"(build system|continuous integration|deployment pipeline|package manager|version control) (systems?|tools?|implementation|performance)",
        r"(software|application|cloud|container|web application|network) (security|vulnerabilit|supply chain)",
        r"(vulnerabilit|malware) (detection|analysis|patching) (in|of|for) (software|code|programs?|packages?|libraries)",
    ),
}
PATTERNS = {
    topic: tuple(re.compile(p, re.IGNORECASE) for p in rules)
    for topic, rules in TOPICS.items()
}


def measure(path: Path) -> dict:
    counts = Counter()
    bytes_by = Counter()
    samples = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            source = document["source_id"]
            if source not in {"pes2o_v2_train", "pes2o_v2_fulltext"}:
                continue
            if document["split"] != "train" or document["drop_reason"] is not None:
                continue
            # The normalized `title` is the input shard filename for HF rows.
            # peS2o's authored paper title is the first line of `text`.
            title = document["text"].split("\n", 1)[0]
            topics = [
                name
                for name, rules in PATTERNS.items()
                if any(p.search(title) for p in rules)
            ]
            text_bytes = len(document["text"].encode("utf-8"))
            categories = ["all", *(topics or ["unmatched"])]
            if topics:
                categories.append("any_engineering_title_match")
            for category in categories:
                key = f"{source}:{category}"
                counts[key] += 1
                bytes_by[key] += text_bytes
                # Stable samples across input ordering; inspect false positives.
                score = hashlib.sha256(document["document_id"].encode()).hexdigest()
                sample = {
                    "score": score,
                    "id": document["document_id"],
                    "title": title[:300],
                }
                current = samples[key]
                current.append(sample)
                current.sort(key=lambda row: row["score"])
                del current[5:]
    return {
        "method": "v1_conservative_title_regex",
        "definitions": TOPICS,
        "counts": dict(sorted(counts.items())),
        "utf8_bytes": dict(sorted(bytes_by.items())),
        "samples": {key: value for key, value in sorted(samples.items())},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", type=Path, help="frozen release documents.jsonl")
    parser.add_argument("output", type=Path, help="small JSON incidence report")
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(measure(args.documents), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
