"""MLX preflight must work in an interpreter with no PyTorch installation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "src"


def test_mlx_preflight_and_inventory_without_torch() -> None:
    script = r"""
import importlib.abc
import importlib.machinery
import json
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, sys.argv[1])

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
        return None

sys.meta_path.insert(0, NoTorch())
from sparselab import runtime
from sparselab.engines import mlx_policy
from sparselab.engines.base import EngineCapabilityError

assert runtime.torch is None
assert "sparselab.engines.mlx" not in sys.modules
config = NS(
    runtime=NS(engine="mlx", backend="metal", device_index=0, precision="fp32",
               memory=NS(activation_offload=NS(enabled=False),
                         activation_checkpointing=NS(enabled=False))),
    model=NS(ffn="dense", memory="none"),
    attention=NS(kind="dense"),
    optimizer=NS(name="adamw", state_offload=False),
)

def rejected(action):
    try:
        action()
    except EngineCapabilityError as error:
        return str(error)
    raise AssertionError("invalid MLX configuration accepted")

errors = {}
config.model.memory = "ngram"
errors["memory"] = rejected(lambda: runtime._validate_mlx_runtime(config))
config.model.memory = "none"
config.runtime.precision = "bf16"
errors["precision"] = rejected(lambda: mlx_policy.validate_config(config))
config.runtime.precision = "fp32"
config.optimizer.name = "adafactor"
errors["optimizer"] = rejected(lambda: mlx_policy.validate_config(config))
config.optimizer.name = "adamw"
errors["unavailable"] = rejected(lambda: runtime._validate_mlx_runtime(config))
assert "sparselab.engines.mlx" not in sys.modules

# The package is observable without a Torch build. Do not claim device readiness.
original_find_spec = runtime.importlib.util.find_spec
runtime.importlib.util.find_spec = (
    lambda name: importlib.machinery.ModuleSpec("mlx", loader=None)
    if name == "mlx" else original_find_spec(name)
)
infos = runtime.discover_runtimes()
frameworks = [info for info in infos if info.engine == "pytorch"]
metal = next(info for info in infos if info.engine == "mlx")
assert len(frameworks) == 5
assert all(info.framework_version is None and info.runtime_version is None
           and info.torch_device is None and not info.precision_capabilities
           and "runtime unavailable" in info.limitations for info in frameworks)
assert metal.backend == "metal" and metal.measurement_source == "mlx package metadata"
assert not metal.tested_features and "MLX execution has not been validated" in metal.limitations

# Fake installed SDK only to reach the native child boundary; never run a native update.
mlx_policy._mlx_available = lambda: True
runtime._probe_runtime = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("native child boundary"))
try:
    runtime._validate_mlx_runtime(config)
except RuntimeError as error:
    assert str(error) == "native child boundary"
else:
    raise AssertionError("MLX validation did not reach the external probe")
assert "torch" not in sys.modules
assert "sparselab.engines.mlx" not in sys.modules
print(json.dumps({"errors": errors, "metal": metal.backend}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(_SOURCE)],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert "does not support memory modules" in observed["errors"]["memory"]
    assert "fp32 only" in observed["errors"]["precision"]
    assert "AdamW only" in observed["errors"]["optimizer"]
    assert "MLX runtime is unavailable" in observed["errors"]["unavailable"]
    assert observed["metal"] == "metal"
