"""Strict deterministic Cartesian expansion for concrete run configs."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml


def expand(path: Path, max_runs: int = 1000) -> list[tuple[tuple[str, ...], dict[str, object]]]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or raw.get("matrix_version") != 1:
        raise ValueError("invalid matrix v1")
    axes = raw.get("axes")
    if not isinstance(axes, dict):
        raise TypeError("matrix v1 requires axes")
    names = list(axes)
    options: list[list[dict[str, object]]] = []
    for name in names:
        values = axes[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"axis {name} must be nonempty")
        if not all(isinstance(value, dict) for value in values):
            raise TypeError(f"axis {name} entries must be mappings")
        typed = list(values)
        labels = [value.get("label") for value in typed]
        if any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f"axis {name} labels must be unique")
        options.append(typed)
    result: list[tuple[tuple[str, ...], dict[str, object]]] = []
    for combination in product(*options):
        patch: dict[str, object] = {}
        coordinate: list[str] = []
        for item in combination:
            coordinate.append(str(item["label"]))
            settings = item.get("set", {})
            if not isinstance(settings, dict):
                raise TypeError("matrix set must be a mapping")
            for key, value in settings.items():
                if key in patch:
                    raise ValueError(f"overlapping patch: {key}")
                patch[key] = value
        result.append((tuple(coordinate), patch))
        if len(result) > max_runs:
            raise ValueError("matrix expansion exceeds max_runs")
    return result

def apply_patch(config: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    """Return a copied concrete config after validating dotted existing paths."""
    import copy

    result = copy.deepcopy(config)
    for path, value in patch.items():
        cursor: dict[str, object] = result
        parts = path.split(".")
        for part in parts[:-1]:
            nested = cursor.get(part)
            if not isinstance(nested, dict):
                raise TypeError(f"unknown matrix path: {path}")
            cursor = nested
        if parts[-1] not in cursor:
            raise ValueError(f"unknown matrix path: {path}")
        cursor[parts[-1]] = value
    return result
