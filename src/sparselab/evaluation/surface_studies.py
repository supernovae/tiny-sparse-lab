"""Read-only, content-verified adapters for the two frozen generation studies."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

_STUDY = Path("experiments/research/dense-lm-decoding-v1")
_CAMPAIGN = Path("experiments/research/tinystories-dense-30m-data-rich-v1")
_WORK = Path("sparselab-work/experiments/tinystories-dense-30m-data-rich-v1")
_MODELS = ("30m", "50m", "new")
_SEEDS = (42, 17, 73)
_OLD_DIMENSIONS = (
    "prompt_adherence", "entities", "stated_attributes",
    "temporal_causal_consistency", "repetition", "local_readability",
)
_DIMENSION_MAP = {
    "prompt_adherence": "prompt_adherence",
    "entities": "entity_continuity",
    "stated_attributes": "attribute_consistency",
    "temporal_causal_consistency": "causal_temporal_coherence",
    "repetition": "repetition",
    "local_readability": "readability_coherence",
}
_COMMON = ("prompt_adherence", "repetition", "readability_coherence", "overall_preference")
_ENTITY = {"entity_continuity", "named_character_continuity"}
_ATTRIBUTE = {"object_continuity", "color_attribute_continuity"}
_CAUSAL = {"cause_effect", "temporal_ordering", "location_permanence"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _file(path: Path, root: Path, *, sha: str | None = None, size: int | None = None):
    """Reject symlinks, escapes and changed indexed bytes before parsing."""
    root = Path(root).absolute()
    path = Path(path).absolute()
    if not path.is_relative_to(root):
        raise ValueError(f"source outside declared workspace: {path}")
    for component in (root, *path.relative_to(root).parts):
        if isinstance(component, Path):
            current = component
        else:
            current = current / component
        if current.is_symlink():
            raise ValueError(f"symlinked source input: {current}")
    if not path.is_file():
        raise ValueError(f"required study input missing: {path}")
    data = path.read_bytes()
    actual = _sha(data)
    if (sha is not None and actual != sha) or (size is not None and len(data) != size):
        raise ValueError(f"changed study input hash/size: {path}")
    return data, {"relative_path": str(path.relative_to(root)), "sha256": actual, "size_bytes": len(data)}


def _json(path: Path, root: Path, *, sha: str | None = None, size: int | None = None):
    data, artifact = _file(path, root, sha=sha, size=size)
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique), artifact
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed study JSON: {path}") from exc


def _jsonl(path: Path, root: Path, *, sha: str, size: int | None = None):
    data, artifact = _file(path, root, sha=sha, size=size)
    try:
        lines = data.decode("utf-8").splitlines()
        if not lines or any(not line for line in lines):
            raise ValueError("empty row")
        return [json.loads(line, object_pairs_hook=_unique) for line in lines], artifact
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed study JSONL: {path}") from exc


def _expect(condition: bool, detail: str):
    if not condition:
        raise ValueError(detail)


def _prompt_index(root: Path, expected_sha: str):
    path = root / _STUDY / "test.json"
    payload, artifact = _json(path, root, sha=expected_sha)
    _expect(payload.get("format") == "dense_lm_decoding_prompts_v1" and payload.get("split") == "test", "wrong frozen test prompt header")
    prompts = payload.get("prompts")
    _expect(isinstance(prompts, list) and len(prompts) == 55, "incomplete frozen test prompts")
    result = {}
    for item in prompts:
        _expect(isinstance(item, dict) and isinstance(item.get("id"), str)
                and isinstance(item.get("text"), str) and isinstance(item.get("category"), str), "invalid frozen prompt")
        _expect(item["id"] not in result, "duplicate frozen prompt ID")
        result[item["id"]] = item
    return result, artifact


def _dimensions(category: str):
    dims = list(_COMMON)
    if category in _ENTITY:
        dims.append("entity_continuity")
    if category in _ATTRIBUTE:
        dims.append("attribute_consistency")
    if category in _CAUSAL:
        dims.append("causal_temporal_coherence")
    return dims


def _source(checkpoint: dict, model: str, *, decoder: dict | None = None, rng_seed: int | None = None):
    result = {"kind": "sparselab_checkpoint", "model": model,
              "run_id": checkpoint["run_id"], "checkpoint_digest": checkpoint["digest"],
              "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
              "run_manifest_sha256": checkpoint["run_manifest_sha256"],
              "tokenizer_sha256": checkpoint["tokenizer_sha256"]}
    if decoder is not None:
        result["decoder"] = decoder
        result["rng_seed"] = rng_seed
    return result


def import_decoding_v1(review_dir, repository_root, output_dir, profile="quick", selection_seed=0, presentation_seed=0):
    """Revalidate every original cell and every published old A/B pair without generating."""
    from sparselab.evaluation.surface_review import create_surface_bundle

    root = Path(repository_root).absolute()
    review = Path(review_dir).absolute()
    study = root / _STUDY
    pre, pre_art = _json(study / "preregistration.json", root)
    pre_sha = pre_art["sha256"]
    _expect(pre.get("format") == "dense_lm_decoding_preregistration_v1", "wrong decoding preregistration")
    artifacts = [pre_art]
    for name, expected in pre["frozen_sha256"].items():
        _, item = _file(root / name, root, sha=expected)
        artifacts.append(item)
    prompts, prompt_art = _prompt_index(root, pre["frozen_sha256"][str(_STUDY / "test.json")])
    # The same path is also in the frozen-input list; keep only one artifact entry.
    selection, sel_art = _json(study / "evidence/selection.json", root)
    summary, summary_art = _json(study / "evidence/summary.json", root)
    artifacts += [sel_art, summary_art]
    _expect(selection.get("format") == "dense_lm_decoder_selection_v1"
            and selection.get("preregistration_sha256") == pre_sha
            and summary.get("format") == "dense_lm_decoding_evidence_v1"
            and summary.get("preregistration_sha256") == pre_sha
            and summary.get("selection_sha256") == sel_art["sha256"]
            and summary.get("selected_non_greedy") == selection.get("selected"), "decoding study index mismatch")
    selected = selection["selected"]
    policies = {p["name"]: p for p in pre["policies"]}
    _expect(selected in policies and selected != "regression_decoder"
            and pre["sample_seeds"] == [11, 29] and pre["test_cells_per_checkpoint"] == 165,
            "wrong frozen decoding policy matrix")
    checkpoints = {(c["model"], c["seed"]): c for c in pre["checkpoints"]}
    _expect(len(checkpoints) == 6 and set(checkpoints) == set(itertools.product(("30m", "50m"), _SEEDS)), "incomplete frozen checkpoint matrix")
    expected_names = {f"{model}-seed{seed}.jsonl" for model, seed in checkpoints}
    _expect(set(summary["source_test_files"]) == expected_names, "incomplete indexed test source files")
    old_sha, old_sha_art = _json(review / "test_sha256.json", root)
    blind, blind_art = _json(review / "blind.json", root, sha=summary["blind_review"]["blind_sha256"])
    key, key_art = _json(review / "key.json", root, sha=summary["blind_review"]["key_sha256"])
    artifacts += [old_sha_art, blind_art, key_art]
    _expect(old_sha == summary["source_test_files"] and summary["blind_review"]["pair_count"] == 198,
            "old review source digest index mismatch")
    rows = {}
    for (model, seed), cp in checkpoints.items():
        name = f"{model}-seed{seed}.jsonl"
        raw, artifact = _jsonl(study / "evidence/test" / name, root, sha=old_sha[name])
        artifacts.append(artifact)
        _expect(len(raw) == 166, f"incomplete indexed test file: {name}")
        header = raw[0]
        _expect(header.get("format") == "dense_lm_decoding_cells_v1"
                and header.get("type") == "header" and header.get("split") == "test"
                and header.get("preregistration_sha256") == pre_sha
                and header.get("prompt_sha256") == prompt_art["sha256"]
                and header.get("checkpoint") == cp, f"test header/checkpoint mismatch: {name}")
        by_coordinate = {}
        for row in raw[1:]:
            _expect(isinstance(row, dict) and row.get("type") == "cell" and row.get("status") == "ok",
                    f"failed/invalid test cell: {name}")
            prompt = prompts.get(row.get("prompt_id"))
            _expect(prompt is not None and row.get("prompt") == prompt["text"]
                    and row.get("category") == prompt["category"]
                    and row.get("model") == model and row.get("training_seed") == seed
                    and row.get("run_id") == cp["run_id"]
                    and row.get("checkpoint_digest") == cp["digest"]
                    and isinstance(row.get("text"), str)
                    and row["text"] == row["prompt"] + row.get("completion", ""),
                    f"test cell source/prompt mismatch: {name}")
            decoder = row.get("decoder")
            _expect(isinstance(decoder, dict) and decoder == policies.get(decoder.get("name")), f"test decoder mismatch: {name}")
            rng = row.get("rng_seed")
            _expect((rng is None if decoder["temperature"] == 0 else rng in (11, 29))
                    and decoder["name"] in ("regression_decoder", selected), f"test RNG/policy mismatch: {name}")
            coord = (row["prompt_id"], decoder["name"], rng)
            _expect(coord not in by_coordinate, f"duplicate indexed test cell: {name}: {coord}")
            by_coordinate[coord] = row
        expected = {(id_, policy, rng) for id_ in prompts
                    for policy, seeds in (("regression_decoder", (None,)), (selected, (11, 29)))
                    for rng in seeds}
        _expect(set(by_coordinate) == expected, f"missing/extra indexed test cells: {name}")
        rows[model, seed] = by_coordinate
    _expect(isinstance(blind, list) and isinstance(key, list) and len(blind) == len(key) == 198,
            "incomplete old review pairs")
    chosen = [item for category in sorted({p["category"] for p in prompts.values()})
              for item in sorted((p for p in prompts.values() if p["category"] == category), key=lambda p: p["id"])[:2]]
    _expect(len(chosen) == 22, "incomplete selected old review prompts")
    cells = []
    pair_ids = set()
    for (item, seed), positions in zip(itertools.product(chosen, _SEEDS),
                                       range(0, len(blind), 3), strict=True):
        comparisons = [("model", rows["30m", seed][item["id"], selected, 11],
                        rows["50m", seed][item["id"], selected, 11])]
        for model in ("30m", "50m"):
            comparisons.append(("decoder", rows[model, seed][item["id"], "regression_decoder", None],
                                rows[model, seed][item["id"], selected, 11]))
        for offset, (kind, left, right) in enumerate(comparisons):
            code = _sha(f"{pre_sha}:{item['id']}:{seed}:{kind}:{left['model']}:{right['model']}".encode())
            if int(code[-1], 16) % 2:
                left, right = right, left
            pair_id = code[:20]
            old = blind[positions + offset]
            original_key = key[positions + offset]
            _expect(pair_id not in pair_ids and isinstance(old, dict) and isinstance(original_key, dict),
                    "duplicate/invalid old review pair")
            pair_ids.add(pair_id)
            side = lambda r: {name: r[name] for name in ("model", "training_seed", "decoder", "rng_seed", "checkpoint_digest")}
            _expect(original_key == {"pair_id": pair_id, "comparison": kind, "a": side(left), "b": side(right)}
                    and old == {"pair_id": pair_id, "prompt": item["text"], "category": item["category"],
                                "a": left["text"], "b": right["text"],
                                "dimensions": list(_OLD_DIMENSIONS),
                                "allowed_votes": ["A", "B", "tie", "uncertain"], "judgments": []},
                    f"old review pair/source response mismatch: {pair_id}")
            def endpoint(row):
                cp = checkpoints[row["model"], row["training_seed"]]
                return {"source": _source({**cp, "tokenizer_sha256": pre["tokenizer_sha256"]}, row["model"],
                                          decoder=row["decoder"], rng_seed=row["rng_seed"]),
                        "response": row["text"]}
            decoder = ({"comparison": "model", "policy": selected, "temperature": policies[selected]["temperature"],
                        "top_k": policies[selected]["top_k"], "max_new_tokens": pre["max_new_tokens"]}
                       if kind == "model" else
                       {"comparison": "decoder", "a": left["decoder"], "b": right["decoder"],
                        "max_new_tokens": pre["max_new_tokens"]})
            cells.append({"candidate_id": f"decoding-v1:{pair_id}", "prompt_id": item["id"],
                          "prompt": item["text"], "category": item["category"], "decoder": decoder,
                          "rng_seed": 11 if kind == "model" else None,
                          "a": endpoint(left), "b": endpoint(right),
                          "dimensions": [_DIMENSION_MAP[d] for d in _OLD_DIMENSIONS],
                          "legacy": {"pair_id": pair_id, "comparison": kind,
                                     "original_dimensions": list(_OLD_DIMENSIONS),
                                     "dimension_mapping": _DIMENSION_MAP,
                                     "allowed_votes": ["A", "B", "tie", "uncertain"],
                                     "original_a": side(left), "original_b": side(right)}})
    _expect(len(cells) == 198, "incomplete old review matrix")
    return create_surface_bundle(cells, profile=profile, selection_seed=selection_seed,
                                 presentation_seed=presentation_seed, output_dir=output_dir,
                                 source_artifacts=artifacts)


def import_data_rich_v1(campaign_root, output_dir, profile="quick", selection_seed=0, presentation_seed=0):
    """Import all three indexed test outputs, requiring their complete matched 165-cell panel."""
    from sparselab.evaluation.surface_review import create_surface_bundle

    root = Path(campaign_root).absolute()
    study = root / _CAMPAIGN
    index, index_art = _json(study / "evidence.json", root)
    # The campaign workspace is an input, not an authority that can redefine
    # the tracked evidence index by changing both its digest and its JSONLs.
    tracked_index = Path(__file__).resolve().parents[3] / _CAMPAIGN / "evidence.json"
    _, tracked_art = _file(tracked_index, tracked_index.parent)
    _expect(index_art["sha256"] == tracked_art["sha256"],
            "campaign evidence index differs from the checked-in study index")
    prereg, pre_art = _json(study / "preregistration.json", root)
    _expect(index.get("format") == "tinystories_dense_30m_data_rich_evidence_v1"
            and prereg.get("format") == "tinystories_dense_30m_data_rich_preregistration_v1",
            "wrong campaign evidence/preregistration")
    artifacts = [index_art, pre_art]
    for name, sha in prereg["frozen_sha256"].items():
        _, artifact = _file(root / name, root, sha=sha)
        artifacts.append(artifact)
    old_pre, old_pre_art = _json(root / prereg["reference_checkpoint_receipt"], root)
    artifacts.append(old_pre_art)
    prompts, prompt_art = _prompt_index(root, prereg["frozen_sha256"][str(_STUDY / "test.json")])
    artifacts.append(prompt_art)
    _expect(len(prompts) == 55, "incomplete campaign test prompt set")
    old_checkpoints = {(cp["model"], cp["seed"]): cp for cp in old_pre["checkpoints"]}
    _expect(("30m", 42) in old_checkpoints and ("50m", 42) in old_checkpoints,
            "campaign reference checkpoints missing")
    sources = {}
    panel = {}
    for model in _MODELS:
        name = f"generation/test-{model}"
        entry = index["outputs"].get(name)
        _expect(isinstance(entry, dict) and entry.get("cells") == 165
                and entry.get("policy_and_short_counts") == [
                    {"count": 55, "policy": "greedy", "short_subset": True},
                    {"count": 110, "policy": "temperature_0_8", "short_subset": True}],
                f"incomplete indexed campaign test matrix: {name}")
        indexed = entry["artifact"]
        relative = _WORK / "evaluation/generation" / f"test-{model}.jsonl"
        _expect(indexed["relative_path"] == str(relative), f"changed campaign artifact path: {name}")
        raw, artifact = _jsonl(root / relative, root, sha=indexed["sha256"], size=indexed["size_bytes"])
        artifacts.append(artifact)
        _expect(len(raw) == 166, f"incomplete campaign test file: {name}")
        header = raw[0]
        _expect(header.get("type") == "header" and header.get("format") == "tinystories_data_rich_generation_v1"
                and header.get("kind") == "test" and header.get("model") == model
                and header.get("max_new_tokens") == 48
                and header.get("preregistration_sha256") == pre_art["sha256"]
                and header.get("prompt_sha256") == prereg["frozen_sha256"][str(_STUDY / "test.json")]
                and header.get("selected_policy_sha256") == prereg["frozen_sha256"][str(_STUDY / "evidence/selection.json")],
                f"campaign header mismatch: {name}")
        cp = header.get("checkpoint")
        _expect(isinstance(cp, dict) and isinstance(cp.get("run_id"), str)
                and isinstance(cp.get("tokenizer_sha256"), str)
                and isinstance(cp.get("checkpoint_sha256"), str), f"invalid campaign checkpoint: {name}")
        if model == "new":
            _expect(cp["run_id"] == index["endpoint"]["run_id"]
                    and cp["checkpoint_sha256"] == index["endpoint"]["checkpoint_digest"]
                    and cp["step"] == index["endpoint"]["step"]
                    and cp["tokens_seen"] == index["endpoint"]["tokens_seen"]
                    and cp["checkpoint_relative_path"].endswith(index["endpoint"]["generation"]),
                    "campaign endpoint checkpoint mismatch")
        else:
            old = old_checkpoints[model, 42]
            _expect(cp["run_id"] == old["run_id"]
                    and cp["checkpoint_sha256"] == old["digest"]
                    and cp["checkpoint_relative_path"].endswith(old["generation"].split("/")[-1])
                    and cp["source_identity_sha256"] == old["training_source_identity_sha256"]
                    and cp["tokenizer_sha256"] == old_pre["tokenizer_sha256"],
                    f"campaign reference checkpoint mismatch: {model}")
        sources[model] = {"kind": "sparselab_checkpoint", "model": model,
                          "run_id": cp["run_id"], "checkpoint_digest": cp["checkpoint_sha256"],
                          "tokenizer_sha256": cp["tokenizer_sha256"],
                          "source_identity_sha256": cp["source_identity_sha256"],
                          "checkpoint_manifest_sha256": (
                              index["endpoint"]["checkpoint_manifest_sha256"] if model == "new"
                              else old_checkpoints[model, 42]["manifest_sha256"]),
                          "checkpoint_relative_path": cp["checkpoint_relative_path"]}
        found = {}
        policy_counts = Counter()
        early_eos = 0
        contradictions = 0
        for row in raw[1:]:
            _expect(isinstance(row, dict) and row.get("type") == "cell" and row.get("status") == "ok",
                    f"failed/invalid campaign test cell: {name}")
            prompt = prompts.get(row.get("prompt_id"))
            _expect(prompt is not None and row.get("prompt") == prompt["text"]
                    and row.get("category") == prompt["category"]
                    and row.get("short_subset") is True
                    and row.get("checkpoint_digest") == cp["checkpoint_sha256"]
                    and isinstance(row.get("text"), str)
                    and row["text"] == row["prompt"] + row.get("completion", ""),
                    f"campaign test cell identity/prompt mismatch: {name}")
            policy, rng = row.get("policy"), row.get("rng_seed")
            _expect((policy == "greedy" and rng is None and row.get("temperature") == 0 and row.get("top_k") == 0)
                    or (policy == "temperature_0_8" and rng in (11, 29)
                        and row.get("temperature") == 0.8 and row.get("top_k") == 0),
                    f"campaign test decoder/RNG mismatch: {name}")
            coord = (row["prompt_id"], row["prompt"], row["temperature"], row["top_k"],
                     header["max_new_tokens"], rng)
            _expect(coord not in found, f"duplicate campaign test coordinate: {name}: {coord}")
            found[coord] = row
            policy_counts[policy, row["short_subset"]] += 1
            _expect(type(row.get("early_eos")) is bool
                    and isinstance(row.get("diagnostics"), dict)
                    and type(row["diagnostics"].get("explicit_anchors", {}).get("explicit_contradiction")) is bool,
                    f"invalid campaign test diagnostics: {name}")
            early_eos += row["early_eos"]
            contradictions += row["diagnostics"]["explicit_anchors"]["explicit_contradiction"]
        _expect(policy_counts == {("greedy", True): 55, ("temperature_0_8", True): 110},
                f"campaign indexed policy counts mismatch: {name}")
        _expect(early_eos == entry["early_eos"] and contradictions == entry["explicit_contradictions"],
                f"campaign indexed diagnostic counts mismatch: {name}")
        expected = {(p["id"], p["text"], temp, 0, 48, rng)
                    for p in prompts.values() for temp, rng in ((0, None), (0.8, 11), (0.8, 29))}
        _expect(set(found) == expected, f"missing/extra campaign indexed test cells: {name}")
        panel[model] = found
    _expect(set(panel["30m"]) == set(panel["50m"]) == set(panel["new"]),
            "campaign test models have unmatched decoder/prompt coordinates")
    cells = []
    for coord in sorted(panel["30m"], key=lambda k: (k[0], k[2], -1 if k[5] is None else k[5])):
        id_, prompt, temperature, top_k, max_new_tokens, rng = coord
        exemplar = panel["30m"][coord]
        _expect(all(panel[m][coord]["category"] == exemplar["category"]
                    and panel[m][coord]["policy"] == exemplar["policy"] for m in _MODELS),
                f"campaign models disagree on matched test coordinate: {coord}")
        for left, right in itertools.combinations(_MODELS, 2):
            decoder = {"policy": exemplar["policy"], "temperature": temperature,
                       "top_k": top_k, "max_new_tokens": max_new_tokens}
            first = {"source": {**sources[left], "decoder": decoder, "rng_seed": rng},
                     "response": panel[left][coord]["text"]}
            second = {"source": {**sources[right], "decoder": decoder, "rng_seed": rng},
                      "response": panel[right][coord]["text"]}
            cells.append({"candidate_id": f"data-rich-v1:{id_}:{exemplar['policy']}:{rng}:{left}:{right}",
                          "prompt_id": id_, "prompt": prompt, "category": exemplar["category"],
                          "decoder": {"policy": exemplar["policy"], "temperature": temperature,
                                      "top_k": top_k, "max_new_tokens": max_new_tokens},
                          "rng_seed": rng, "a": first, "b": second,
                          "dimensions": _dimensions(exemplar["category"])})
    _expect(len(cells) == 495, "incomplete matched campaign model-pair matrix")
    return create_surface_bundle(cells, profile=profile, selection_seed=selection_seed,
                                 presentation_seed=presentation_seed, output_dir=output_dir,
                                 source_artifacts=artifacts)
