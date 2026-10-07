"""Keep the approved Card 03 source and transport envelope pinned."""

from pathlib import Path

from sparselab.corpus.project import load_project

PROJECT = (
    Path(__file__).resolve().parents[1]
    / "experiments/research/kernel-memory-lab/corpus-scale/corpus-acquire.yaml"
)


def test_card03_scale_sources_and_budget_match_approved_attempt() -> None:
    project = load_project(PROJECT)
    assert project.config.id == "kernel-memory-lab-card03-scale-v1"
    assert project.config.transport_budget.attempt_id == "kml-card03-scale-v1"
    assert not project.release.lm.selected
    assert not project.release.chat.selected

    sources = {source.id: source for source in project.sources}
    assert set(sources) == {
        "kml_scale_project_gutenberg",
        "kml_scale_wikimedia",
        "kml_scale_pagerduty",
        "kml_scale_scoutflo",
    }
    shards = {
        source_id: source.acquisition.bounded_shards[0]
        for source_id, source in sources.items()
        if source.kind == "huggingface_dataset"
    }
    assert {
        source_id: (shard.path, shard.expected_sha256, shard.max_shard_bytes)
        for source_id, shard in shards.items()
    } == {
        "kml_scale_project_gutenberg": (
            "project_gutenberg-dolma-0014.json.gz",
            "bdcb5e7e0e48d42489949fe2492b765c71d4eae1e1cd6de34cddfb1b4360b3ee",
            362_929_669,
        ),
        "kml_scale_wikimedia": (
            "wikimedia-0027.json.gz",
            "0abe6c9be821ac100c604ac4b799faa00c90e39d139c7dccf9491e693b177ed0",
            480_242_750,
        ),
    }
    assert {
        source_id: (
            len(source.acquisition.bounded_blobs),
            sum(blob.max_bytes for blob in source.acquisition.bounded_blobs),
        )
        for source_id, source in sources.items()
        if source.kind == "git"
    } == {
        "kml_scale_pagerduty": (38, 317_526),
        "kml_scale_scoutflo": (433, 2_087_503),
    }
    assert all(shard.hash_modulus == 16 for shard in shards.values())
    assert all(shard.hash_remainders == (0,) for shard in shards.values())
    assert all(
        source.rights.training_eligibility == "review_required"
        for source in project.sources
    )

    budget = project.config.transport_budget
    first_pass = sum(shard.max_shard_bytes for shard in shards.values()) + sum(
        blob.max_bytes
        for source in sources.values()
        if source.kind == "git"
        for blob in source.acquisition.bounded_blobs
    )
    assert first_pass == 845_577_448
    assert budget.max_source_body_bytes == 2 * first_pass == 1_691_154_896
    assert budget.max_metadata_body_bytes == 4_194_304
    assert budget.max_transfers == 2 * (2 + 38 + 433) == 946
    assert budget.max_retries_per_shard == 1
    assert budget.max_wall_seconds == 7_200
    assert budget.max_disk_bytes == 8 * 1024**3
