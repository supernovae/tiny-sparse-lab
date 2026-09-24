"""Strict deterministic Cartesian expansion for concrete run configurations."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel

from sparselab.config.loading import _PATH_KEYS, load_config
from sparselab.config.models import RunConfig
from sparselab.training.manifest import canonical_json, config_sha256
from sparselab.workers.models import SchedulingRequirements


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader which makes duplicate mapping keys an input error."""


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


@dataclass(frozen=True)
class ExpandedExperiment:
    """A concrete, validated configuration selected by one matrix coordinate."""

    config: RunConfig
    coordinate: dict[str, str]
    matrix_sha256: str
    preferred_worker: str | None
    requirements: SchedulingRequirements


def _load_matrix(path: Path) -> dict[str, object]:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as error:
        raise ValueError(
            f"cannot read matrix {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in matrix {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError("matrix must contain a YAML mapping")
    if type(raw.get("matrix_version")) is not int or raw["matrix_version"] != 1:
        raise ValueError("matrix_version must be exactly 1")
    return raw


def _requirements(value: object) -> SchedulingRequirements:
    if value is None:
        return SchedulingRequirements()
    if not isinstance(value, dict):
        raise TypeError("matrix scheduling.requirements must be a mapping")
    return SchedulingRequirements.model_validate(value)


def _scheduling(
    raw: Mapping[str, object],
) -> tuple[str | None, SchedulingRequirements]:
    scheduling = raw.get("scheduling", {})
    if not isinstance(scheduling, dict):
        raise TypeError("matrix scheduling must be a mapping")
    allowed = {"preferred_worker", "requirements"}
    unexpected = set(scheduling) - allowed
    if unexpected:
        raise ValueError(f"unknown matrix scheduling fields: {sorted(unexpected)!r}")
    preferred = scheduling.get("preferred_worker")
    if preferred is not None and (
        not isinstance(preferred, str) or not preferred.strip()
    ):
        raise ValueError("matrix scheduling.preferred_worker must be a nonempty string")
    return preferred, _requirements(scheduling.get("requirements"))


def _patchable_config(value: object) -> object:
    if isinstance(value, BaseModel):
        return {
            name: _patchable_config(getattr(value, name))
            for name in type(value).model_fields
        }
    if isinstance(value, Mapping):
        return {key: _patchable_config(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_patchable_config(item) for item in value)
    if isinstance(value, list):
        return [_patchable_config(item) for item in value]
    return value


def _validate_path(config: Mapping[str, object], dotted: str) -> None:
    if not dotted or dotted.startswith(".") or dotted.endswith("."):
        raise ValueError(f"unknown matrix path: {dotted!r}")
    cursor: object = config
    for component in dotted.split("."):
        if not isinstance(cursor, Mapping) or component not in cursor:
            raise ValueError(f"unknown matrix path: {dotted}")
        cursor = cursor[component]


def _conflicts(left: str, right: str) -> bool:
    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


def _axis_options(
    raw: Mapping[str, object], base: Mapping[str, object]
) -> tuple[list[str], list[list[dict[str, object]]]]:
    axes = raw.get("axes")
    if not isinstance(axes, dict) or not axes:
        raise ValueError("matrix axes must be a nonempty mapping")
    names: list[str] = []
    options: list[list[dict[str, object]]] = []
    paths_by_axis: dict[str, set[str]] = {}
    for axis, entries in axes.items():
        if not isinstance(axis, str) or not axis.strip():
            raise ValueError("matrix axis names must be nonempty strings")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"axis {axis} must be nonempty")
        labels: set[str] = set()
        typed: list[dict[str, object]] = []
        axis_paths: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"label", "set"}:
                raise ValueError(f"axis {axis} entries require exactly label and set")
            label = entry["label"]
            settings = entry["set"]
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"axis {axis} labels must be nonempty strings")
            if label in labels:
                raise ValueError(f"axis {axis} has duplicate label: {label}")
            labels.add(label)
            if not isinstance(settings, dict):
                raise TypeError(f"axis {axis} set must be a mapping")
            for dotted in settings:
                if not isinstance(dotted, str):
                    raise TypeError(f"axis {axis} patch paths must be strings")
                _validate_path(base, dotted)
                if any(
                    other != dotted and _conflicts(dotted, other)
                    for other in settings
                    if isinstance(other, str)
                ):
                    raise ValueError(f"overlapping patches within axis {axis}")
                axis_paths.add(dotted)
            typed.append(entry)
        for other_axis, other_paths in paths_by_axis.items():
            for dotted in axis_paths:
                for other in other_paths:
                    if _conflicts(dotted, other):
                        raise ValueError(
                            f"overlapping matrix patches: {axis}.{dotted} conflicts with "
                            f"{other_axis}.{other}"
                        )
        paths_by_axis[axis] = axis_paths
        names.append(axis)
        options.append(typed)
    return names, options


def expand(path: Path, max_runs: int = 1000) -> list[ExpandedExperiment]:
    """Expand a strict v1 matrix into ordered, concrete, validated configurations."""
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs <= 0:
        raise ValueError("max_runs must be a positive integer")
    raw = _load_matrix(path)
    if set(raw) - {"matrix_version", "base_config", "axes", "scheduling"}:
        raise ValueError(
            f"unknown matrix fields: {sorted(set(raw) - {'matrix_version', 'base_config', 'axes', 'scheduling'})!r}"
        )
    base_config = raw.get("base_config")
    if not isinstance(base_config, str) or not base_config:
        raise ValueError("matrix base_config must be a nonempty path string")
    base_path = Path(base_config)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    resolved_base = load_config(base_path)
    base = resolved_base.model_dump(mode="python")
    patchable_base = _patchable_config(resolved_base)
    if not isinstance(patchable_base, dict):
        raise TypeError("resolved configuration must project to a mapping")
    names, options = _axis_options(raw, patchable_base)
    count = 1
    for axis in options:
        count *= len(axis)
        if count > max_runs:
            raise ValueError(f"matrix expansion exceeds max_runs ({max_runs})")
    preferred_worker, requirements = _scheduling(raw)
    matrix_sha256 = hashlib.sha256(
        canonical_json(
            {
                "matrix_version": 1,
                "base_config_sha256": config_sha256(base),
                "axes": [
                    {"name": name, "options": entries}
                    for name, entries in zip(names, options, strict=True)
                ],
                "scheduling": {
                    "preferred_worker": preferred_worker,
                    "requirements": requirements.model_dump(mode="json"),
                },
            }
        )
    ).hexdigest()
    matrix_dir = path.parent.resolve()
    expanded: list[ExpandedExperiment] = []
    for selected in product(*options):
        patch: dict[str, object] = {}
        coordinate: dict[str, str] = {}
        for axis_name, entry in zip(names, selected, strict=True):
            label = cast(str, entry["label"])  # validated by _axis_options
            settings = cast(dict[str, object], entry["set"])
            coordinate[axis_name] = label
            for dotted, value in settings.items():
                if (
                    isinstance(value, str)
                    and dotted.rsplit(".", maxsplit=1)[-1] in _PATH_KEYS
                ):
                    axis_path = Path(value)
                    value = (
                        axis_path
                        if axis_path.is_absolute()
                        else (matrix_dir / axis_path).resolve()
                    )
                patch[dotted] = value
        concrete = RunConfig.model_validate(apply_patch(patchable_base, patch))
        expanded.append(
            ExpandedExperiment(
                config=concrete,
                coordinate=coordinate,
                matrix_sha256=matrix_sha256,
                preferred_worker=preferred_worker,
                requirements=requirements,
            )
        )
    return expanded


def apply_patch(
    config: Mapping[str, object], patch: Mapping[str, object]
) -> dict[str, object]:
    """Return a copied concrete config after validating dotted existing paths."""
    result = copy.deepcopy(dict(config))
    for dotted, value in patch.items():
        if not isinstance(dotted, str):
            raise TypeError("matrix patch paths must be strings")
        _validate_path(result, dotted)
        cursor: dict[str, object] = result
        components = dotted.split(".")
        for component in components[:-1]:
            nested = cursor[component]
            if not isinstance(nested, dict):  # guarded by _validate_path
                raise TypeError(
                    f"matrix path does not address a config mapping: {dotted}"
                )
            cursor = nested
        cursor[components[-1]] = value
    return result
