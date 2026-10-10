from __future__ import annotations

from pathlib import Path

from sparselab.config.loading import load_config
from sparselab.model.inspection import inspection_report

CUDA = Path("configs/first_model_20m_cuda.yaml")
CPU = Path("configs/first_model_20m_cpu.yaml")
FINEWEB_EDU_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


def test_first_model_configs_share_one_model_and_eval_protocol() -> None:
    cuda, cpu = load_config(CUDA), load_config(CPU)

    assert cuda.model == cpu.model
    assert cuda.dataset == cpu.dataset
    assert cuda.tokenizer == cpu.tokenizer
    assert cuda.training.seq_len == cpu.training.seq_len == 512
    assert cuda.dataset.source == "fineweb_edu"
    assert cuda.dataset.revision == FINEWEB_EDU_REVISION
    assert inspection_report(cuda)["total"] == 17_308_032


def test_first_model_runtime_split_is_one_gpu_or_cpu() -> None:
    cuda, cpu = load_config(CUDA), load_config(CPU)

    assert (cuda.runtime.backend, cuda.runtime.precision) == ("cuda", "bf16")
    assert (cpu.runtime.backend, cpu.runtime.precision) == ("cpu", "fp32")
    assert cuda.training.max_tokens == 49_152_000
    assert cpu.training.max_tokens == 4_915_200
    for config in (cuda, cpu):
        tokens_per_step = (
            config.training.micro_batch_size
            * config.training.gradient_accumulation
            * config.training.seq_len
        )
        assert config.training.max_steps * tokens_per_step == config.training.max_tokens
