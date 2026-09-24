"""Compose explicit research and lesson folders without executing experiments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from pydantic import ValidationError

from sparselab.config.loading import load_config
from sparselab.config.models import RunConfig, TokenizerTrainConfig
from sparselab.engram.packs import _rename_noreplace
from sparselab.experiments.matrix import _patchable_config, apply_patch, expand
from sparselab.experiments.study import plan_study
from sparselab.model.portable_engram import load_portable_engram
from sparselab.research.catalog import (
    MechanismLesson,
    ResearchEntry,
    ResearchRecipe,
    load_datasets,
    load_lesson,
    load_profiles,
    load_recipe,
    load_research,
    resource_sha256,
)
from sparselab.training.manifest import canonical_json, config_sha256, sha256_file

_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
_DATASETS = frozenset({"offline", "tinystories"})
_BACKENDS = frozenset({"cpu", "mps", "cuda", "rocm", "xpu"})
_SEEDS = (17, 41, 73)
_CARDS = (
    "chat-alias-retention-v1",
    "chat-alias-recall-v1",
    "chat-context-override-v1",
)
_PATH_KEYS = frozenset(
    {
        "path",
        "cache_dir",
        "root_dir",
        "output_dir",
        "memory_package_path",
        "train_path",
        "validation_path",
        "allocation_manifest_path",
    }
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()


def _write(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = data.encode("utf-8") if isinstance(data, str) else data
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _yaml_bytes(value: object) -> bytes:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True).encode("utf-8")


def _source_paths(
    reference: str | Path, entry: ResearchEntry
) -> tuple[Path | None, Path]:
    candidate = Path(reference).expanduser()
    if candidate.suffix.lower() == ".json" and candidate.is_file():
        entry_path = candidate.resolve(strict=True)
        recipe_path = (entry_path.parent / entry.recipe).resolve(strict=True)
        return entry_path, recipe_path
    return (
        _RESOURCE_ROOT / "catalog" / f"{entry.id}.json",
        _RESOURCE_ROOT / entry.recipe,
    )


def _dataset_mapping(name: str) -> dict[str, object]:
    try:
        profile = load_datasets().datasets[name]
    except KeyError as error:
        raise ValueError(
            f"unknown dataset profile {name!r}; choose offline or tinystories"
        ) from error
    return {
        "source": profile.source,
        "revision": profile.revision,
        "dataset_config": None,
        "cache_dir": "artifacts/data",
        "train_max_documents": profile.train_max_documents,
        "validation_max_documents": profile.validation_max_documents,
        "train_max_tokens": profile.train_max_tokens,
        "validation_max_tokens": profile.validation_max_tokens,
        "synthetic_seed": profile.dataset_seed,
        "train_path": None,
        "validation_path": None,
        "license": None,
        "allocation_manifest_path": None,
    }


def _tokenizer_values(name: str) -> dict[str, object]:
    profile = load_datasets().datasets[name]
    dataset = {
        "source": profile.source,
        "revision": profile.revision,
        "dataset_config": None,
        "cache_dir": "artifacts/data",
        "train_max_documents": profile.tokenizer_max_documents,
        "validation_max_documents": profile.tokenizer_validation_max_documents,
        "train_max_tokens": profile.tokenizer_train_max_tokens,
        "validation_max_tokens": profile.tokenizer_validation_max_tokens,
        "synthetic_seed": profile.dataset_seed,
        "train_path": None,
        "validation_path": None,
        "license": None,
    }
    return {
        "schema_version": 1,
        "vocab_size": profile.vocab_size,
        "min_frequency": profile.min_frequency,
        "max_documents": profile.tokenizer_max_documents,
        "output_dir": "artifacts/tokenizer",
        "dataset": dataset,
    }


def _scale_patch(scale: str, data: str, backend: str) -> dict[str, object]:
    try:
        profile = load_profiles().scales[scale]
    except KeyError as error:
        choices = ", ".join(load_profiles().scales)
        raise ValueError(
            f"unsupported runnable scale {scale!r}; choose {choices}. Larger classes are recommendation-only."
        ) from error
    dataset = load_datasets().datasets[data]
    return {
        "model.vocab_size": dataset.vocab_size,
        "model.hidden_dim": profile.hidden_dim,
        "model.num_layers": profile.num_layers,
        "model.num_heads": profile.num_heads,
        "model.ffn_dim": profile.ffn_dim,
        "model.max_seq_len": profile.max_seq_len,
        "tokenizer.path": "artifacts/tokenizer/tokenizer.json",
        "dataset": _dataset_mapping(data),
        "training.seq_len": profile.seq_len,
        "training.max_steps": profile.max_steps,
        "training.max_tokens": profile.max_tokens,
        "training.micro_batch_size": profile.micro_batch_size,
        "training.gradient_accumulation": profile.gradient_accumulation,
        "optimizer.peak": 0.003 if scale == "smoke" else 0.0003,
        "optimizer.floor": 0.0003 if scale == "smoke" else 0.00003,
        "optimizer.warmup_steps": 4 if scale == "smoke" else 10,
        "evaluation.every_steps": profile.validation_steps,
        "checkpoint.every_steps": profile.validation_steps,
        "runtime.backend": backend,
        "logging.root_dir": "runs",
    }


def _apply_overlay(
    config: dict[str, object], patch: dict[str, object]
) -> dict[str, object]:
    nested = {key: value for key, value in patch.items() if "." in key}
    direct = {key: value for key, value in patch.items() if "." not in key}
    if direct:
        for key, value in direct.items():
            if key not in config:
                raise ValueError(f"unknown top-level config patch: {key}")
            config[key] = value
    return apply_patch(config, nested) if nested else config


def _research_config_sha256(config: RunConfig) -> str:
    payload = config.model_dump(mode="json")
    allocation_path = config.dataset.allocation_manifest_path
    if allocation_path is not None:
        if allocation_path.is_symlink() or not allocation_path.is_file():
            raise ValueError("allocation research config requires a regular manifest")
        dataset = payload["dataset"]
        if not isinstance(dataset, dict):
            raise TypeError("allocation research config lacks a dataset mapping")
        dataset["allocation_manifest_sha256"] = sha256_file(allocation_path)
    return config_sha256(payload)


def _write_resolved_config(config: RunConfig, root: Path, directory: Path) -> bytes:
    value = config.model_dump(mode="json")

    def rebase(item: object, key: str | None = None) -> object:
        if isinstance(item, dict):
            return {name: rebase(child, name) for name, child in item.items()}
        if isinstance(item, list):
            return [rebase(child) for child in item]
        if key in _PATH_KEYS and isinstance(item, str):
            absolute = Path(item)
            if not absolute.is_absolute():
                absolute = root / absolute
            absolute = absolute.resolve()
            scaffold_root = root.resolve()
            workspace_root = Path(__file__).resolve().parents[3]
            if not absolute.is_relative_to(
                scaffold_root
            ) and not absolute.is_relative_to(workspace_root):
                raise ValueError(
                    f"generated config path escapes scaffold and workspace roots: {item}"
                )
            return Path(os.path.relpath(absolute, directory.resolve())).as_posix()
        return item

    return _yaml_bytes(rebase(value))


def _load_base() -> dict[str, object]:
    value = yaml.safe_load((_RESOURCE_ROOT / "base.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("packaged base config must contain a mapping")
    patchable = _patchable_config(value)
    if not isinstance(patchable, dict):
        raise TypeError("packaged base config must project to a mapping")
    return patchable


def _validate_run(value: dict[str, object]) -> RunConfig:
    try:
        return RunConfig.model_validate(value)
    except ValidationError as error:
        raise ValueError(f"generated run configuration is invalid: {error}") from error


def _destination(output: Path) -> tuple[Path, Path]:
    destination = output.expanduser().absolute()
    if os.path.lexists(destination):
        raise FileExistsError(f"output destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    return destination, temporary


def _publish(temporary: Path, destination: Path) -> Path:
    _rename_noreplace(temporary, destination)
    return destination


def _cleanup(temporary: Path) -> None:
    if temporary.exists():
        shutil.rmtree(temporary)


def _read_recipe(
    entry: ResearchEntry, reference: str | Path
) -> tuple[ResearchRecipe, Path | None, Path]:
    entry_path, recipe_path = _source_paths(reference, entry)
    local_root = (
        entry_path.parent
        if entry_path and entry_path.parent != (_RESOURCE_ROOT / "catalog")
        else None
    )
    recipe = load_recipe(entry, local_root=local_root)
    return recipe, entry_path, recipe_path


def _study_readme(
    entry: ResearchEntry,
    scale: str,
    data: str,
    backend: str,
    design: str,
    coordinates: list[dict[str, object]],
    run_count: int,
    pair_count: int,
    factorial_count: int,
) -> str:
    profile = load_datasets().datasets[data]
    choices = "\n".join(
        f"- `{size}` — {scale.hidden_dim}D, {scale.num_layers} layers, {scale.num_heads} heads, FFN {scale.ffn_dim}, {scale.max_steps} steps / {scale.max_tokens} target tokens"
        for size, scale in sorted(load_profiles().scales.items())
    )
    coordinate_rows = "\n".join(
        f"| `{json.dumps(item['coordinate'], sort_keys=True)}` | `configs/{item['config_sha256']}.yaml` |"
        for item in coordinates
    )
    prep = "\n".join(
        f"sparselab data prepare configs/{item['config_sha256']}.yaml"
        for item in coordinates
    )
    first = coordinates[0]
    return f"""# {entry.title}

**Question:** {entry.question}

{entry.hypothesis}

## First: inspect one mechanism without a campaign

```sh
sparselab inspect configs/{first["config_sha256"]}.yaml --json
sparselab learn probe configs/{first["config_sha256"]}.yaml --prompt \"A learner asks a short question.\"
sparselab train configs/{first["config_sha256"]}.yaml --run-id one-arm --stop-after-step 2
```

## Fixed data, scale, and runtime

- Dataset `{data}`: {profile.notes[0]} {profile.notes[-1]}
- Source and rights: {profile.license} [{profile.source_url}]({profile.source_url})
- Dataset seed `{profile.dataset_seed}` is independent of model seeds `{", ".join(map(str, _SEEDS))}`.
- Scale `{scale}`; backend `{backend}`; design `{design}`. The recipe has {run_count} planned coordinates, {pair_count} declared pairs, and {factorial_count} versioned 2×2 factorial designs.
- Other supported profiles (scale is independent of data and budget):
{choices}
- Tokenizer fitting uses only the declared training prefix. Held-out validation and capability-card examples are not tokenizer inputs. Preparation can access network/cache only when the explicit `data prepare` command is run for TinyStories.
- Capability cards: {", ".join(f"`{card}` ({profile.card_applicability.get(card, 'not applicable')})" for card in _CARDS)}. Cards are unchanged, unsealed development/stress evaluations; on TinyStories they are out-of-domain stress, not story-model quality. Held-out LM loss and each card stay separate. Generated stories are unscored examples.
- Configured endpoint stops at the first existing step or target-token limit. Periodic validation is not a preregistered milestone; primary thresholds are absent. Actual steps and targets must be reported.

## Prepare explicitly, then plan or train

```sh
sparselab tokenizer train tokenizer.yaml
{prep}
sparselab study plan study.yaml
```

Exported standalone configurations:

| Coordinate | Config |
|---|---|
{coordinate_rows}

The config files use full scientific SHA-256 names; labels never become paths. To run an arm directly, train any listed config with `--run-id YOUR_ID`. To dispatch the complete study, use an explicit store and worker:

```sh
sparselab worker register research-{backend} --backend {backend} --store runs
sparselab study submit study.yaml --receipt receipt.json --worker research-{backend} --store runs
sparselab controller run --store runs
sparselab study collect study.yaml receipt.json --runs-dir runs
sparselab study report study.yaml receipt.json --evidence PATH_FROM_COLLECT --research research.json --runs-dir runs --output research-reports
```

The scaffold itself performs no tokenization, data preparation/download, training, worker registration, submission, controller execution, inference, or evaluation. Use the independent train command above or the explicit plan→worker→submit→controller→collect→report route.

## Controls and interpretation

**Failure interpretation:** {entry.failure_interpretation}

**Fixed controls:** {"; ".join(entry.controls)}

**Varied fields:** {", ".join(entry.independent_variables)}

**Confounders:** {"; ".join(entry.confounders)}

**Can establish:** {"; ".join(entry.can_establish)}

**Cannot establish:** {"; ".join(entry.cannot_establish)}

**Prerequisites / unavailable arms:** {"; ".join(entry.prerequisites) if entry.prerequisites else "None for the declared recipe."}

The first displayed configuration is a runnable `{scale}`/`{data}`/`{backend}` example. The smoke/offline/cpu entry route remains available by scaffolding with `--scale smoke --data offline --backend cpu`; the public-data option remains independent, for example `--scale micro --data tinystories`. Larger reference-small, medium-research, and large-local profiles are recommendations only here; no GQA/reference aliases are generated.

"""


def _entry_payload(entry: ResearchEntry) -> dict[str, object]:
    return entry.model_dump(mode="json")


def scaffold_research(
    reference: str | Path,
    output: Path,
    *,
    scale: str = "micro",
    data: str = "offline",
    backend: str = "cpu",
    design: str = "default",
) -> Path:
    """Write a complete editable study scaffold; never prepare or execute it."""
    entry = load_research(reference)
    if data not in _DATASETS:
        raise ValueError(f"unsupported dataset {data!r}; choose offline or tinystories")
    if backend not in _BACKENDS:
        raise ValueError(
            f"unsupported backend {backend!r}; choose {', '.join(sorted(_BACKENDS))}"
        )
    profiles = load_profiles()
    if scale not in profiles.scales:
        raise ValueError(
            f"unsupported runnable scale {scale!r}; supported: {', '.join(profiles.scales)}; larger named classes are recommendation-only"
        )
    recipe, entry_path, recipe_path = _read_recipe(entry, reference)
    if design not in recipe.designs:
        raise ValueError(
            f"unsupported design {design!r} for {entry.id}; choose {', '.join(sorted(recipe.designs))}"
        )
    try:
        scale_recipe = recipe.designs[design][scale]
    except KeyError as error:
        raise ValueError(
            f"recipe {entry.id} has no {scale!r} scale for design {design!r}"
        ) from error
    if "seed" in scale_recipe.axes:
        raise ValueError("recipe axes must exclude seed; scaffold owns matched seeds")

    destination, temporary = _destination(output)
    try:
        base = _load_base()
        base = _apply_overlay(base, _scale_patch(scale, data, backend))
        base = _apply_overlay(base, scale_recipe.base_set)
        base["name"] = f"{entry.id}-{scale}-{data}"
        _validate_run(base)
        _write(temporary / "base.yaml", _yaml_bytes(base))
        tokenizer_values = _tokenizer_values(data)
        TokenizerTrainConfig.model_validate(tokenizer_values)
        _write(temporary / "tokenizer.yaml", _yaml_bytes(tokenizer_values))

        axes = {
            "seed": [{"label": f"s{seed}", "set": {"seed": seed}} for seed in _SEEDS],
            **{
                name: [option.model_dump(mode="json") for option in options]
                for name, options in scale_recipe.axes.items()
            },
        }
        matrix = {
            "matrix_version": 1,
            "base_config": "base.yaml",
            "scheduling": {"requirements": {"backend": [backend]}},
            "axes": axes,
        }
        study = {
            "study_version": 1,
            "name": entry.id,
            "matrix": "matrix.yaml",
            "cards": [card.reference for card in entry.cards],
            "comparisons": [item for item in scale_recipe.comparisons],
        }
        _write(temporary / "matrix.yaml", _yaml_bytes(matrix))
        _write(temporary / "study.yaml", _yaml_bytes(study))
        # Existing expansion and study planning are the only matrix/pair engines.
        expanded = expand(temporary / "matrix.yaml")
        planned = plan_study(temporary / "study.yaml")
        if len(expanded) != len(planned.expanded):
            raise ValueError("study planner and matrix expansion disagree")

        coordinates: list[dict[str, object]] = []
        seen_hashes: set[str] = set()
        config_payloads: dict[str, bytes] = {}
        for item in expanded:
            digest = _research_config_sha256(item.config)
            if digest in seen_hashes:
                raise ValueError(
                    f"duplicate scientific configuration hash at {item.coordinate}"
                )
            seen_hashes.add(digest)
            filename = f"configs/{digest}.yaml"
            config_bytes = _write_resolved_config(
                item.config, temporary, temporary / "configs"
            )
            # The file is published once after all matrix identities have been checked.
            config_payloads[filename] = config_bytes
            coordinates.append(
                {
                    "coordinate": item.coordinate,
                    "config_sha256": digest,
                    "path": filename,
                }
            )
        for filename, content in config_payloads.items():
            _write(temporary / filename, content)
        for item in coordinates:
            exported = load_config(temporary / str(item["path"]))
            if _research_config_sha256(exported) != item["config_sha256"]:
                raise ValueError(
                    f"exported config identity mismatch at {item['coordinate']}"
                )
        # The planner was run against the exact published scientific inputs.
        planned = plan_study(temporary / "study.yaml")
        if [_research_config_sha256(item.config) for item in planned.expanded] != [
            str(item["config_sha256"]) for item in coordinates
        ]:
            raise ValueError(
                "exported config identities differ from matrix coordinates"
            )

        readme = _study_readme(
            entry,
            scale,
            data,
            backend,
            design,
            coordinates,
            len(planned.expanded),
            len(planned.pairs),
            len(scale_recipe.factorial_designs),
        )
        _write(temporary / "README.md", readme)
        source_inputs = {
            "research_sources/catalog-entry.json": (
                entry_path
                if entry_path is not None
                else _RESOURCE_ROOT / "catalog" / f"{entry.id}.json"
            ),
            "research_sources/recipe.json": recipe_path,
            "research_sources/profiles.json": _RESOURCE_ROOT / "profiles.json",
            "research_sources/datasets.json": _RESOURCE_ROOT / "datasets.json",
        }
        for relative, source in source_inputs.items():
            _write(temporary / relative, source.read_bytes())
        input_names = [
            "base.yaml",
            "tokenizer.yaml",
            "matrix.yaml",
            "study.yaml",
            *source_inputs,
            *sorted(str(item["path"]) for item in coordinates),
        ]
        inputs = [
            {"path": relative, "sha256": sha256_file(temporary / relative)}
            for relative in input_names
        ]
        if entry_path is not None:
            entry_digest = sha256_file(entry_path)
            recipe_digest = sha256_file(recipe_path)
        else:
            entry_digest = resource_sha256(f"catalog/{entry.id}.json")
            recipe_digest = resource_sha256(entry.recipe)
        metadata: dict[str, object] = {
            "format": "sparselab-research-scaffold",
            "version": 1,
            "entry": _entry_payload(entry),
            "entry_sha256": entry_digest,
            "recipe_sha256": recipe_digest,
            "factorial_designs": [
                item.model_dump(mode="json") for item in scale_recipe.factorial_designs
            ],
            "selection": {
                "scale": scale,
                "data": data,
                "backend": backend,
                "design": design,
            },
            "inputs": inputs,
            "coordinates": coordinates,
        }
        metadata["research_sha256"] = hashlib.sha256(
            canonical_json(metadata)
        ).hexdigest()
        _write(temporary / "research.json", _json_bytes(metadata))
        return _publish(temporary, destination)
    except BaseException:
        _cleanup(temporary)
        raise


def _semantic_lesson_workspace(directory: Path) -> dict[str, object]:
    """Materialize a verified semantic pack and its executable lesson probe."""
    import numpy as np
    from safetensors.numpy import save_file

    from sparselab.engram.packs import compile_pack

    key_indices = {
        ("atlas", "maps_to"): 0,
        ("beacon", "maps_to"): 1,
        ("cipher", "maps_to"): 2,
        ("harbor", "label"): 3,
        ("vault", "label"): 4,
        ("fork", "maps_to"): 5,
    }
    value_indices = {
        "beacon": 0,
        "amber": 1,
        "delta": 2,
        "open": 3,
        "closed": 4,
        "left": 5,
        "right": 6,
    }
    facts = (
        ("lesson-atlas-maps-to-beacon", "atlas", "maps_to", "beacon", None, None),
        ("lesson-beacon-maps-to-amber", "beacon", "maps_to", "amber", None, None),
        ("lesson-cipher-maps-to-delta", "cipher", "maps_to", "delta", None, None),
        (
            "lesson-harbor-label-open",
            "harbor",
            "label",
            "open",
            "2025-01-01",
            "2025-12-31",
        ),
        ("lesson-vault-label-closed", "vault", "label", "closed", None, "2020-12-31"),
        ("lesson-fork-left", "fork", "maps_to", "left", None, None),
        ("lesson-fork-right", "fork", "maps_to", "right", None, None),
    )
    record_rows = [
        {
            "record_version": 1,
            "id": record_id,
            "namespace": "semantic-retrieval-lesson",
            "subject": subject,
            "relation": relation,
            "value": value,
            "license": "CC0-1.0",
            "source": "SparseLab original tutorial fixture",
            "source_revision": "frozen-v1",
            "created_at": "2026-09-23T00:00:00Z",
            "valid_from": valid_from,
            "valid_until": valid_until,
        }
        for record_id, subject, relation, value, valid_from, valid_until in facts
    ]
    records_bytes = b"".join(canonical_json(row) + b"\n" for row in record_rows)
    records_path = directory / "records.jsonl"
    _write(records_path, records_bytes)

    key_specification = {
        "algorithm": "fixed-one-hot-structured-key-v1",
        "dimension": len(key_indices),
        "indices": [
            {"subject": subject, "relation": relation, "index": index}
            for (subject, relation), index in sorted(key_indices.items())
        ],
    }
    value_specification = {
        "algorithm": "fixed-one-hot-value-v1",
        "dimension": len(value_indices),
        "indices": [
            {"value": value, "index": index}
            for value, index in sorted(value_indices.items())
        ],
    }
    key_encoder = {
        "name": "tutorial-fixed-structured-key",
        "revision": "frozen-v1",
        "sha256": hashlib.sha256(canonical_json(key_specification)).hexdigest(),
    }
    value_encoder = {
        "name": "tutorial-fixed-value",
        "revision": "frozen-v1",
        "sha256": hashlib.sha256(canonical_json(value_specification)).hexdigest(),
    }
    keys = np.zeros((len(facts), len(key_indices)), dtype=np.float32)
    values = np.zeros((len(facts), len(value_indices)), dtype=np.float32)
    for row_index, (_, subject, relation, value, _, _) in enumerate(facts):
        keys[row_index, key_indices[(subject, relation)]] = 1.0
        values[row_index, value_indices[value]] = 1.0
    keys_path = directory / "semantic_keys.safetensors"
    values_path = directory / "semantic_values.safetensors"
    save_file({"keys": keys}, keys_path)
    save_file({"values": values}, values_path)
    metadata_path = directory / "semantic.json"
    _write(
        metadata_path,
        canonical_json(
            {
                "format": "sparselab-semantic-assets",
                "format_version": 1,
                "record_ids": [row["id"] for row in record_rows],
                "key_encoder": key_encoder,
                "value_encoder": value_encoder,
                "key_normalization": "none",
            }
        )
        + b"\n",
    )
    pack_path = directory / "semantic-pack"
    manifest = compile_pack(
        records_path,
        pack_path,
        name="semantic-retrieval-lesson",
        namespace="semantic-retrieval-lesson",
        default_license="CC0-1.0",
        source_name="SparseLab original tutorial fixture",
        source_revision="frozen-v1",
        created_at="2026-09-23T00:00:00Z",
        semantic_keys=keys_path,
        semantic_values=values_path,
        semantic_metadata=metadata_path,
    )
    semantic = manifest.semantic
    if semantic is None:
        raise RuntimeError("compiled semantic lesson pack lacks its semantic component")
    demo = """from __future__ import annotations

import json
from pathlib import Path

import torch

from sparselab.config.models import AttentionConfig, ModelConfig
from sparselab.engram.semantic import SemanticQueryBatch, SemanticRetriever
from sparselab.model.transformer import DenseLM

ROOT = Path(__file__).resolve().parent
PACK_ID = "__PACK_ID__"
KEY_DIM = 6
KEY_INDEX = __KEY_INDEX__
VALUE_INDEX = __VALUE_INDEX__
VALUE_BY_INDEX = {index: value for value, index in VALUE_INDEX.items()}


def encode_key(subject: str, relation: str) -> torch.Tensor:
    vector = torch.zeros(KEY_DIM, dtype=torch.float32)
    index = KEY_INDEX.get((subject, relation))
    if index is not None:
        vector[index] = 1.0
    return vector


retriever = SemanticRetriever.from_pack(
    ROOT / "semantic-pack", expected_pack_id=PACK_ID
)
query_examples = {
    "hit": ("atlas", "maps_to", None),
    "unknown": ("missing", "maps_to", None),
    "conflict": ("fork", "maps_to", None),
    "temporal_hit": ("harbor", "label", "2025-06-01"),
    "temporal_miss": ("vault", "label", "2025-06-01"),
}
retrieval = {}
for name, (subject, relation, as_of) in query_examples.items():
    outcome = retriever.retrieve(
        encode_key(subject, relation),
        key_encoder=retriever.key_encoder,
        top_k=2,
        min_score=0.9,
        as_of=as_of,
    )
    retrieval[name] = {
        "status": outcome.status,
        "hits": [
            {
                "record_id": hit.record_id,
                "score": hit.score,
                "value_shape": list(hit.value_vector.shape),
            }
            for hit in outcome.hits
        ],
        "trace": {
            "pack_id": outcome.trace.pack_id,
            "metric": outcome.trace.metric,
            "comparison_count": outcome.trace.comparison_count,
            "candidate_count": outcome.trace.candidate_count,
            "temporal_excluded_count": outcome.trace.temporal_excluded_count,
            "best_record_id": outcome.trace.best_record_id,
            "best_score": outcome.trace.best_score,
            "tie_count": outcome.trace.tie_count,
            "tied_record_ids": outcome.trace.tied_record_ids,
        },
    }

first_hop = retriever.retrieve(
    encode_key("atlas", "maps_to"),
    key_encoder=retriever.key_encoder,
    min_score=0.9,
)
intermediate_index = int(first_hop.hits[0].value_vector.argmax())
intermediate = VALUE_BY_INDEX[intermediate_index]
second_hop = retriever.retrieve(
    encode_key(intermediate, "maps_to"),
    key_encoder=retriever.key_encoder,
    min_score=0.9,
)

model = DenseLM(
    ModelConfig(
        vocab_size=260,
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_dim=32,
        max_seq_len=8,
    ),
    AttentionConfig(),
)
adapter = model.add_semantic_memory(
    "lesson",
    retriever,
    site="after_block",
    block_index=0,
    min_score=0.9,
)
trainable_adapter_parameters = sum(
    parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
)
model.eval()
model.requires_grad_(False)
adapter.freeze()
tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
queries = SemanticQueryBatch(
    retriever.key_encoder, encode_key("atlas", "maps_to").reshape(1, KEY_DIM)
)
with torch.inference_mode():
    logits = model(tokens, semantic_queries=queries)
    cached_logits, _ = model.forward_cached(
        tokens, cache_capacity=8, semantic_queries=queries
    )
torch.testing.assert_close(cached_logits, logits, atol=1e-5, rtol=1e-5)
metrics = {
    name: float(value.detach().cpu())
    for name, value in model.architecture_metric_tensors().items()
}
report = {
    "pack_id": retriever.pack_id,
    "dimensions": {
        "keys": [retriever.entry_count, retriever.key_dim],
        "values": [retriever.entry_count, retriever.memory_dim],
        "queries": [1, retriever.key_dim],
        "hidden": [1, tokens.shape[1], model.config.hidden_dim],
        "logits": list(logits.shape),
    },
    "retrieval": retrieval,
    "two_hop": {
        "status": "retrieved" if second_hop.status == "hit" else second_hop.status,
        "intermediate": intermediate,
        "answer": VALUE_BY_INDEX[int(second_hop.hits[0].value_vector.argmax())],
        "record_ids": [first_hop.hits[0].record_id, second_hop.hits[0].record_id],
    },
    "adapter": {
        "site": adapter.site,
        "block_index": adapter.block_index,
        "trainable_parameter_count_before_freeze": trainable_adapter_parameters,
        "frozen_after_attachment": all(
            not parameter.requires_grad for parameter in model.parameters()
        ),
        "traces": [
            {
                "status": trace.status,
                "best_record_id": trace.best_record_id,
                "best_score": trace.best_score,
                "comparison_count": trace.comparison_count,
            }
            for trace in adapter.last_traces
        ],
        "metrics": metrics,
    },
    "cached_forward_parity": True,
    "parameter_gradients_absent": all(
        parameter.grad is None for parameter in model.parameters()
    ),
}
print(json.dumps(report, indent=2, sort_keys=True))
"""
    demo = (
        demo.replace("__PACK_ID__", manifest.pack_id)
        .replace("__KEY_INDEX__", repr(key_indices))
        .replace("__VALUE_INDEX__", repr(value_indices))
    )
    demo_path = directory / "demo.py"
    _write(demo_path, demo)

    inputs = [
        {"path": name, "sha256": sha256_file(directory / name)}
        for name in (
            "records.jsonl",
            "semantic_keys.safetensors",
            "semantic_values.safetensors",
            "semantic.json",
            "demo.py",
        )
    ]
    pack_manifest_path = pack_path / "manifest.json"
    inputs.append(
        {
            "path": "semantic-pack/manifest.json",
            "sha256": sha256_file(pack_manifest_path),
        }
    )
    inputs.extend(
        {
            "path": f"semantic-pack/{item.relative_path}",
            "sha256": item.sha256,
        }
        for item in manifest.files
    )
    return {
        "records_sha256": sha256_file(records_path),
        "inputs": inputs,
        "semantic_pack": {
            "path": "semantic-pack",
            "pack_id": manifest.pack_id,
            "manifest_sha256": sha256_file(pack_manifest_path),
            "entry_count": semantic.entry_count,
            "key_dim": semantic.key_dim,
            "value_dim": semantic.memory_dim,
            "key_encoder": key_encoder,
            "value_encoder": value_encoder,
        },
    }


def _lesson_readme(
    lesson: MechanismLesson, scale: str, data: str, backend: str, *, artifact: bool
) -> str:
    if artifact and lesson.id == "semantic-retrieval":
        walkthrough = "\n".join(
            f"- **{step.source_path}:{step.source_symbol}** — {step.explanation} "
            f"Shapes: {'; '.join(step.shapes)} Observe: {', '.join(step.observe)}."
            for step in lesson.steps
        )
        source_links = "\n".join(f"- {doc}" for doc in lesson.docs)
        return f"""# {lesson.title}

{lesson.summary}

This standalone workspace contains a verified semantic EngramPack and a runnable Python lesson. Query vectors are explicit structured one-hot fixtures, not natural-language embeddings.

## Run and inspect

```sh
python demo.py
sparselab engram pack inspect semantic-pack
sparselab engram pack verify semantic-pack
```

The demo reports verified pack identity, retrieval status, ordered record IDs and scores, candidate/comparison counts, temporal exclusions, deterministic tie IDs, adapter site, trainable adapter parameter count before freezing, frozen attachment state, observed model metrics, and full-prefix/cached parity. It also runs a two-edge structured lookup. No tokenizer, model training run, external dataset, text encoder, or network access is required.

## Tensor walkthrough

This fixture uses keys `[7,6]`, values `[7,7]`, queries `[1,6]`, and backbone hidden states `[1,3,16]`. The adapter maps value width `7` to hidden width `16`; the key encoder width `6` is independent. The demo prints actual output and diagnostic shapes.

{walkthrough}

**Try changes:** {"; ".join(lesson.try_changes)}

**Limits:** {"; ".join(lesson.limits)}

## Source and follow-up

{source_links}
"""
    if artifact:
        return f"""# {lesson.title}

{lesson.summary}

These records are original tutorial-only CC0 triples. They are not a sealed evaluation card, a dynamic-world dataset, or executable model retrieval.

```sh
sparselab engram pack compile records.jsonl --output artifacts/tutorial-pack --name tutorial-map --namespace tutorial --license CC0-1.0 --source-name original-tutorial-records --created-at 2026-09-23T00:00:00Z
sparselab engram pack inspect artifacts/tutorial-pack
sparselab engram pack verify artifacts/tutorial-pack
```

`compile`, `inspect`, and `verify` exercise the artifact format only. There is no `model.yaml`, `tokenizer.yaml`, or probe path in this artifact lesson.
"""
    return f"""# {lesson.title}

{lesson.summary}

**Selection:** `{scale}` / `{data}` / `{backend}`. This standalone path can be used without a campaign or worker.

## Predict, inspect, probe, change one knob

Before running the probe, predict the output shape and diagnostic values from the shape walkthrough in `lesson.json`. Then inspect the concrete recipe and compare the observed forward. Change exactly one config knob and explain any mismatch.

```sh
sparselab tokenizer train tokenizer.yaml
sparselab data prepare model.yaml
sparselab inspect model.yaml --json
sparselab learn probe model.yaml --prompt \"A short input asks about an object.\" --json
sparselab train model.yaml --run-id lesson-{lesson.id} --stop-after-step 2
```

Tokenizer fitting and dataset preparation are explicit; TinyStories may use the network/cache only at those commands. The probe is a freshly initialized CPU FP32 reference forward, not a hardware benchmark, learned result, or score prediction. Training/evaluation checkpoints and dashboard diagnostics provide learned observations later.

## Walkthrough

{chr(10).join(f"- **{step.source_path}:{step.source_symbol}** — {step.explanation} Shapes: {'; '.join(step.shapes)} Observe: {', '.join(step.observe)}." for step in lesson.steps)}

**Try changes:** {"; ".join(lesson.try_changes)}

**Limits:** {"; ".join(lesson.limits)}

## Source and follow-up

{chr(10).join(f"- {doc}" for doc in lesson.docs)}
"""


def scaffold_lesson(
    identifier: str,
    output: Path,
    *,
    scale: str = "smoke",
    data: str = "offline",
    backend: str = "cpu",
    memory_package: Path | None = None,
) -> Path:
    """Write a single-mechanism lesson configuration or an artifact walkthrough."""
    lesson = load_lesson(identifier)
    if data not in _DATASETS:
        raise ValueError(f"unsupported dataset {data!r}; choose offline or tinystories")
    if backend not in _BACKENDS:
        raise ValueError(
            f"unsupported backend {backend!r}; choose {', '.join(sorted(_BACKENDS))}"
        )
    if lesson.mechanism_kind == "model":
        if scale not in load_profiles().scales:
            raise ValueError(
                f"unsupported runnable scale {scale!r}; larger named classes are recommendation-only"
            )
        if scale not in lesson.patches_by_scale:
            raise ValueError(f"lesson {identifier} has no patch for scale {scale}")
    if identifier == "portable-engram":
        if memory_package is None:
            raise ValueError(
                "portable-engram requires --memory-package PATH. First run: "
                "sparselab learn scaffold byte-engram --scale smoke --data offline --backend cpu --output experiments/byte-engram; "
                "sparselab tokenizer train experiments/byte-engram/tokenizer.yaml; "
                "sparselab data prepare experiments/byte-engram/model.yaml; "
                "sparselab train experiments/byte-engram/model.yaml --run-id byte-engram --runs-dir experiments/byte-engram/runs; "
                "sparselab engram export byte-engram --runs-dir experiments/byte-engram/runs --output experiments/byte-engram/memory.engram; "
                "sparselab learn scaffold portable-engram --memory-package experiments/byte-engram/memory.engram --output experiments/portable-engram"
            )
        if memory_package.is_symlink() or not memory_package.is_file():
            raise ValueError(
                "--memory-package must be an existing regular nonsymlink exported table"
            )
        package = load_portable_engram(memory_package)
    else:
        package = None

    destination, temporary = _destination(output)
    try:
        lesson_payload: dict[str, object] = {
            "format": "sparselab-lesson-scaffold",
            "version": 1,
            "lesson": lesson.model_dump(mode="json"),
            "lesson_sha256": resource_sha256(f"lessons/{identifier}.json"),
            "selection": {"scale": scale, "data": data, "backend": backend},
        }
        if lesson.mechanism_kind == "artifact":
            if identifier == "semantic-retrieval":
                lesson_payload.update(_semantic_lesson_workspace(temporary))
            else:
                record_rows = [
                    {
                        "record_version": 1,
                        "id": "tutorial-atlas-maps-to-beacon",
                        "namespace": "tutorial-map",
                        "subject": "atlas",
                        "relation": "maps_to",
                        "value": "beacon",
                        "license": "CC0-1.0",
                    },
                    {
                        "record_version": 1,
                        "id": "tutorial-beacon-maps-to-amber",
                        "namespace": "tutorial-map",
                        "subject": "beacon",
                        "relation": "maps_to",
                        "value": "amber",
                        "license": "CC0-1.0",
                    },
                    {
                        "record_version": 1,
                        "id": "tutorial-cipher-maps-to-delta",
                        "namespace": "tutorial-map",
                        "subject": "cipher",
                        "relation": "maps_to",
                        "value": "delta",
                        "license": "CC0-1.0",
                    },
                ]
                records = b"".join(canonical_json(row) + b"\n" for row in record_rows)
                _write(temporary / "records.jsonl", records)
                records_sha256 = hashlib.sha256(records).hexdigest()
                lesson_payload["records_sha256"] = records_sha256
                lesson_payload["inputs"] = [
                    {"path": "records.jsonl", "sha256": records_sha256}
                ]
            _write(
                temporary / "README.md",
                _lesson_readme(lesson, scale, data, backend, artifact=True),
            )
        else:
            base = _load_base()
            base = _apply_overlay(base, _scale_patch(scale, data, backend))
            patch = lesson.patches_by_scale.get(scale, {})
            base = _apply_overlay(base, patch)
            base["name"] = f"lesson-{identifier}-{scale}-{data}"
            if package is not None:
                asset_dir = temporary / "assets"
                asset_dir.mkdir()
                if memory_package is None:
                    raise ValueError("portable package disappeared before copy")
                shutil.copyfile(memory_package, asset_dir / "memory.engram")
                source = package.manifest
                base = _apply_overlay(
                    base,
                    {
                        "model.memory": "portable",
                        "model.memory_injection": "final",
                        "model.memory_table_size": source.table_size,
                        "model.memory_ngram_size": source.ngram_size,
                        "model.memory_dim": source.embedding_dim,
                        "model.memory_package_path": "assets/memory.engram",
                        "model.memory_ngram_orders": [],
                        "model.memory_hash_heads": 1,
                    },
                )
                lesson_payload["memory_package"] = {
                    "table_sha256": source.table_sha256,
                    "table_size": source.table_size,
                    "embedding_dim": source.embedding_dim,
                    "ngram_size": source.ngram_size,
                }
                lesson_payload["inputs"] = [
                    {
                        "path": "assets/memory.engram",
                        "sha256": sha256_file(asset_dir / "memory.engram"),
                    }
                ]
            config = _validate_run(base)
            tokenizer_values = _tokenizer_values(data)
            TokenizerTrainConfig.model_validate(tokenizer_values)
            _write(temporary / "model.yaml", _yaml_bytes(base))
            _write(temporary / "tokenizer.yaml", _yaml_bytes(tokenizer_values))
            lesson_payload["config_sha256"] = config_sha256(
                config.model_dump(mode="json")
            )
            lesson_payload["tokenizer_sha256"] = hashlib.sha256(
                _yaml_bytes(tokenizer_values)
            ).hexdigest()
            _write(
                temporary / "README.md",
                _lesson_readme(lesson, scale, data, backend, artifact=False),
            )
        lesson_payload["lesson_scaffold_sha256"] = hashlib.sha256(
            canonical_json(lesson_payload)
        ).hexdigest()
        _write(temporary / "lesson.json", _json_bytes(lesson_payload))
        return _publish(temporary, destination)
    except BaseException:
        _cleanup(temporary)
        raise
