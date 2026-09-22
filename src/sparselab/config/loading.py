"""YAML configuration loading with paths anchored to the YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from sparselab.config.models import RunConfig, TokenizerTrainConfig

_PATH_KEYS = {"path", "cache_dir", "root_dir", "output_dir", "memory_package_path"}


def _resolve_paths(value: Any, base: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            item_key: _resolve_paths(item_value, base, item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_paths(item, base) for item in value]
    if key in _PATH_KEYS and isinstance(value, str):
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()
    return value


def _load[T: BaseModel](path: Path, model_type: type[T]) -> T:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read config {path}: {error.strerror or error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise TypeError(f"config {path} must contain a YAML mapping")
    try:
        return model_type.model_validate(_resolve_paths(raw, path.parent.resolve()))
    except ValidationError as error:
        raise ValueError(f"invalid config {path}: {error}") from error


def load_config(path: Path) -> RunConfig:
    """Load a fully resolved run configuration without touching runtime artifacts."""
    return _load(path, RunConfig)


def load_tokenizer_config(path: Path) -> TokenizerTrainConfig:
    """Load a standalone tokenizer training configuration."""
    return _load(path, TokenizerTrainConfig)
