"""Optional Apple MLX dense execution engine."""

from __future__ import annotations

from importlib.util import find_spec

from sparselab.config.models import RunConfig
from sparselab.model.mlx_dense import MLXDenseLM
from sparselab.runtime import RuntimeInfo, discover_runtimes


def validate(config: RunConfig) -> RuntimeInfo:
    if config.runtime.engine != "mlx" or config.runtime.backend != "metal":
        raise ValueError("MLX engine requires runtime.engine=mlx and backend=metal")
    if (
        config.model.ffn != "dense"
        or config.model.memory != "none"
        or config.attention.kind != "dense"
    ):
        raise ValueError("MLX currently supports dense attention and dense FFN only")
    if config.runtime.precision not in {"fp32", "auto"}:
        raise ValueError("MLX precision is currently fp32 only")
    if find_spec("mlx.core") is None:
        raise ValueError("MLX runtime is unavailable")
    return next(info for info in discover_runtimes() if info.engine == "mlx")


class MLXEngine:
    def __init__(self, config: RunConfig) -> None:
        import mlx.optimizers as optim
        from mlx import nn

        validate(config)
        if config.optimizer.name != "adamw":
            raise ValueError("MLX currently supports AdamW only")
        self.config = config
        self.model = MLXDenseLM(config.model)
        self.optimizer = optim.AdamW(
            learning_rate=config.optimizer.peak,
            betas=list(config.optimizer.betas),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )
        self._value_and_grad = nn.value_and_grad(self.model, self._loss)

    def _loss(self, input_ids, targets):
        from mlx import nn

        return nn.losses.cross_entropy(self.model(input_ids), targets, reduction="mean")

    def logits(self, input_ids):
        return self.model(input_ids)

    def train_update(self, input_ids, targets) -> float:
        import mlx.core as mx

        loss, gradients = self._value_and_grad(input_ids, targets)
        self.optimizer.update(self.model, gradients)
        mx.eval(loss, self.model.parameters(), self.optimizer.state)
        return float(loss)

    def save_state(self, directory):
        """Persist model and optimizer trees for same-engine continuation."""
        import json
        from pathlib import Path

        import mlx.core as mx
        from mlx import nn

        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(
            str(destination / "weights.safetensors"),
            dict(nn.utils.tree_flatten(self.model.parameters())),
        )
        mx.save_safetensors(
            str(destination / "optimizer.safetensors"),
            dict(nn.utils.tree_flatten(self.optimizer.state)),
        )
        (destination / "state.json").write_text(
            json.dumps({"codec": "mlx_native", "version": 1}) + "\n"
        )

    def load_state(self, directory) -> None:
        from pathlib import Path

        import mlx.core as mx
        from mlx import nn

        source = Path(directory)
        self.model.update(
            nn.utils.tree_unflatten(mx.load(str(source / "weights.safetensors")))
        )
        self.optimizer.state = nn.utils.tree_unflatten(
            mx.load(str(source / "optimizer.safetensors"))
        )
