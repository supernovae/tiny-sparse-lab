"""Offline Hub metadata for integration tests above the acquisition boundary."""


def offline_hub_identity(source, **_options):
    return {
        "repo_id": source.repo_id,
        "revision": source.revision,
        "loader": "json",
        "loader_options": {},
        "files": {
            split: [{"path": f"{split}.jsonl", "size": 100, "blob_id": "a" * 40}]
            for split in source.splits
        },
    }
