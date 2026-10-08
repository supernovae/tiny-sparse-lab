"""Opt-in, shared native-budget hooks for the bounded KML CPU CI dispatch.

Python loads this before pytest collection and in inherited Python children.
No hook is installed in ordinary test or research processes.
"""

from __future__ import annotations

import builtins
import contextvars
import functools
import json
import os
import sys
import uuid
from pathlib import Path

_ledger_path = os.environ.get("KML_CI_BUDGET_LEDGER")
if _ledger_path:
    _ledger = Path(_ledger_path)
    _identity = os.environ["KML_CI_CONTENT_SHA256"]
    _kind = os.environ["KML_CI_JOB_KIND"]
    _events = Path(os.environ["KML_CI_EVENTS_DIR"])
    _update_context = contextvars.ContextVar("kml_ci_update", default=None)
    _optimizer_context = contextvars.ContextVar("kml_ci_optimizer", default=None)
    _generation_context = contextvars.ContextVar("kml_ci_generation", default=None)
    _original_import = builtins.__import__
    _patching = False

    def _budget():
        from sparselab.training.attempt_budget import AttemptBudget

        return AttemptBudget(_ledger)

    def _charge(prefix: str, **vector: int) -> str:
        if os.environ.get("KML_CI_PREFLIGHT_ONLY") == "1" and any(vector.values()):
            raise RuntimeError("model work is forbidden during CI collection preflight")
        label = f"{prefix}:{os.getpid()}:{uuid.uuid4().hex}"
        _budget().reserve_vector(label, content_identity_sha256=_identity, **vector)
        return label

    def _event(label: str, **values: object) -> None:
        destination = _events / (label.replace(":", "-") + ".json")
        with destination.open("x", encoding="utf-8") as output:
            json.dump(
                {"label": label, "pid": os.getpid(), **values}, output, sort_keys=True
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())

    def _train_shape(config, run_id: str) -> tuple[str, int]:
        training = config.training
        shape = (
            training.max_steps,
            training.seq_len,
            training.micro_batch_size,
            training.gradient_accumulation,
            training.max_tokens,
            config.runtime.backend,
            config.runtime.precision,
        )
        expected = {
            "serving": {
                "original": ((2, 16, 2, 1, 384, "cpu", "fp32"), 32, 1),
                "snapshot-model": ((2, 16, 1, 2, 640, "cpu", "fp32"), 32, 9),
            },
            "archive": {
                "archive-model": ((2, 16, 1, 1, 32, "cpu", "fp32"), 16, 1),
            },
            "zero": {},
        }[_kind]
        if run_id not in expected or shape != expected[run_id][0]:
            raise RuntimeError(f"unreviewed CI training fixture: {run_id} {shape}")
        status = _budget().status()
        prior = sum(
            row["label"].startswith(f"train:{run_id}:")
            for row in status["reservations"]
        )
        if prior >= expected[run_id][2]:
            raise RuntimeError(f"CI training fixture invocation limit: {run_id}")
        return run_id, expected[run_id][1]

    def _patch_trainer(module) -> None:
        if getattr(module.train, "_kml_ci_guarded", False):
            return
        original = module.train

        @functools.wraps(original)
        def guarded(config, *args, **kwargs):
            if os.environ.get("KML_CI_PREFLIGHT_ONLY") == "1":
                raise RuntimeError("CI collection preflight cannot initialize training")
            run_id = kwargs.get("run_id")
            if not isinstance(run_id, str):
                raise TypeError("CI train requires an explicit reviewed run ID")
            label_kind, targets = _train_shape(config, run_id)
            label = _charge(f"train:{label_kind}")
            token = _update_context.set((label_kind, targets))
            try:
                return original(config, *args, **kwargs)
            finally:
                _update_context.reset(token)
                _event(label, invocation=run_id)

        guarded._kml_ci_guarded = True
        module.train = guarded

    def _patch_engine(module) -> None:
        cls = module.PyTorchEngine
        if getattr(cls.train_update, "_kml_ci_guarded", False):
            return
        original = cls.train_update

        @functools.wraps(original)
        def guarded(self, microbatches, update_index, valid_targets):
            context = _update_context.get()
            if context is None or valid_targets != context[1]:
                raise RuntimeError("unmetered or drifted CI optimizer update")
            label = _charge("update", updates=1, target_positions=valid_targets)
            optimizer_calls = [0]
            optimizer_token = _optimizer_context.set(optimizer_calls)
            try:
                result = original(self, microbatches, update_index, valid_targets)
            except BaseException:
                _event(label, result="unknown_after_exception")
                raise
            finally:
                _optimizer_context.reset(optimizer_token)
            if result.status == "APPLIED" and optimizer_calls[0] != 1:
                _event(
                    label,
                    result="optimizer_call_mismatch",
                    optimizer_calls=optimizer_calls[0],
                )
                raise RuntimeError(
                    "CI applied update did not perform exactly one optimizer.step"
                )
            _event(
                label,
                result=result.status,
                optimizer_calls=optimizer_calls[0],
                actual_updates=int(result.status == "APPLIED"),
                actual_targets=(valid_targets if result.status == "APPLIED" else 0),
            )
            return result

        guarded._kml_ci_guarded = True
        cls.train_update = guarded

    def _patch_generation(module, name: str) -> None:
        original = getattr(module, name)
        if getattr(original, "_kml_ci_guarded", False):
            return

        @functools.wraps(original)
        def guarded(*args, **kwargs):
            if len(args) <= 4 and "max_new_tokens" not in kwargs:
                raise RuntimeError("CI generation lacks requested-token bound")
            requested = kwargs.get("max_new_tokens", args[4] if len(args) > 4 else None)
            if type(requested) is not int or not 0 <= requested <= 5:
                raise RuntimeError("unreviewed CI generation token request")
            parent = _generation_context.get()
            if parent is not None:
                if requested > parent[1]:
                    raise RuntimeError("nested generation exceeds parent allowance")
                return original(*args, **kwargs)
            label = _charge(
                "generation", generation_calls=1, generated_tokens=requested
            )
            draws = [0]
            token = _generation_context.set((draws, requested))
            completed = False
            try:
                result = original(*args, **kwargs)
                completed = True
                return result
            finally:
                _generation_context.reset(token)
                _event(
                    label,
                    requested_tokens=requested,
                    actual_sampled_tokens=draws[0],
                    completed=completed,
                )

        guarded._kml_ci_guarded = True
        setattr(module, name, guarded)

    def _patch_sampler(module) -> None:
        original = module._next_token
        if getattr(original, "_kml_ci_guarded", False):
            return

        @functools.wraps(original)
        def guarded(*args, **kwargs):
            context = _generation_context.get()
            if context is None:
                raise RuntimeError("unmetered CI token sampling")
            if context[0][0] >= context[1]:
                raise RuntimeError("CI generated-token allowance exceeded")
            result = original(*args, **kwargs)
            context[0][0] += 1
            return result

        guarded._kml_ci_guarded = True
        module._next_token = guarded

    def _patch_optimizer(module) -> None:
        if getattr(module, "_kml_ci_guarded", False):
            return
        base = module.Optimizer
        for cls in (
            base,
            *(value for value in vars(module).values() if isinstance(value, type)),
        ):
            if not issubclass(cls, base) or getattr(cls.step, "_kml_ci_guarded", False):
                continue
            original = cls.step

            @functools.wraps(original)
            def guarded(self, *args, _original=original, **kwargs):
                calls = _optimizer_context.get()
                if calls is None or calls[0] >= 1:
                    raise RuntimeError("unmetered CI optimizer.step")
                calls[0] += 1
                return _original(self, *args, **kwargs)

            guarded._kml_ci_guarded = True
            cls.step = guarded
        module._kml_ci_guarded = True

    def _patch_loaded() -> None:
        global _patching
        if _patching:
            return
        _patching = True
        try:
            modules = sys.modules
            if "sparselab.training.trainer" in modules and hasattr(
                modules["sparselab.training.trainer"], "train"
            ):
                _patch_trainer(modules["sparselab.training.trainer"])
            if "sparselab.engines.pytorch" in modules and hasattr(
                modules["sparselab.engines.pytorch"], "PyTorchEngine"
            ):
                _patch_engine(modules["sparselab.engines.pytorch"])
            if "sparselab.evaluation.generation" in modules and hasattr(
                modules["sparselab.evaluation.generation"], "generate_with_token_ids"
            ):
                generation = modules["sparselab.evaluation.generation"]
                _patch_sampler(generation)
                _patch_generation(generation, "generate_with_token_ids")
            if "sparselab.evaluation.generation_request" in modules and hasattr(
                modules["sparselab.evaluation.generation_request"], "generate_result"
            ):
                request = modules["sparselab.evaluation.generation_request"]
                generation = modules.get("sparselab.evaluation.generation")
                if generation is not None:
                    request._next_token = generation._next_token
                _patch_generation(request, "generate_result")
            if "torch.optim" in modules and all(
                hasattr(modules["torch.optim"], name) for name in ("AdamW", "Optimizer")
            ):
                _patch_optimizer(modules["torch.optim"])
        finally:
            _patching = False

    @functools.wraps(_original_import)
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        result = _original_import(name, globals, locals, fromlist, level)
        if name == "torch" or name.startswith(
            (
                "sparselab.training",
                "sparselab.engines",
                "sparselab.evaluation",
                "torch.optim",
            )
        ):
            _patch_loaded()
        return result

    builtins.__import__ = _import
    _patch_loaded()
