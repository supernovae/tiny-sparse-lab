# Prospective fresh run configuration

The YAML below is a review template, not a runnable config. Replace every
`${...}` slot only from a verified upstream receipt, save as a new `.yaml`,
validate with native `RunConfig` and `sparselab inspect`, and record its digest
before staging. The values with unresolved identity slots are not evidence of
a built corpus or completed preparation.

```yaml
schema_version: 2
name: kernel-memory-lab-card05-base-50m-v1
seed: 17
runtime:
  engine: pytorch
  backend: rocm
  device_index: 0
  precision: bf16
  memory:
    policy: balanced
    max_device_memory_fraction: 0.9
    budget_bytes: 21474836480
    activation_checkpointing:
      enabled: true
      strategy: transformer_block
    activation_offload:
      enabled: false
    allowed_sequence_lengths: []
    allowed_optimizers: []
model:
  vocab_size: 32768
  hidden_dim: 1024
  num_layers: 24
  num_heads: 16
  ffn_dim: 2816
  max_seq_len: 1024
  rms_norm_eps: 1.0e-06
  tie_embeddings: true
  ffn: dense
  num_experts: 1
  experts_per_token: 1
  shared_expert: false
  router_aux_loss_coefficient: 0.0
  memory: none
  memory_table_size: 0
  memory_ngram_size: 0
  memory_dim: 0
  memory_package_path: null
  memory_ngram_orders: []
  memory_hash_heads: 1
  semantic_memory_dim: null
tokenizer:
  path: /srv/sparselab/state/experiments/kernel-memory-lab/corpora/kernel-memory-lab-card03-scale-retry1/exports/e893cb2e3c65f7f6360f73e4daa159dcec5da1b4c2c6f911ff5d9d549b815818/lm/b57de1d5857b918120baa86d0caef8c7813cf50d422b621adee3d12a0561c944/tokenizer/tokenizer.json
dataset:
  source: local_token_mixture
  revision: ${NEW_RELEASE_ID}
  dataset_config: null
  cache_dir: ${NEW_CACHE_DIR}
  train_max_documents: ${VERIFIED_MIXTURE_TRAIN_DOCUMENTS}
  validation_max_documents: ${VERIFIED_RELEASE_VALIDATION_DOCUMENTS}
  train_max_tokens: 50000000
  validation_max_tokens: ${VERIFIED_RELEASE_VALIDATION_TOKENS}
  synthetic_seed: 17
  train_path: ${VERIFIED_MIXTURE_DIR}/train.tokens.jsonl
  validation_path: ${VERIFIED_NEW_RELEASE_DIR}/lm/validation.jsonl
  license: Apache-2.0 repository claim, per-file review required; Apache-2.0; private
    local US research with notice obligations; CC-BY-SA-4.0 claim, per-page review
    required; CC-BY-SA-4.0; private local research with attribution obligations; MIT
    repository claim, per-file review required; MIT; private local US research with
    notice obligations; Public Domain claim, per-work review required; US public-domain
    selected component; private local research only
  allocation_manifest_path: null
  mixture_declaration_path: ${SEALED_MIXTURE_DECLARATION}
  mixture_output_path: ${VERIFIED_MIXTURE_DIR}
training:
  micro_batch_size: 1
  gradient_accumulation: 1
  seq_len: 1024
  max_steps: 48829
  max_tokens: 50000000
  grad_clip_norm: 1.0
  neural_loss_weight: 1.0
  deterministic: true
optimizer:
  name: adamw
  peak: 0.0003
  floor: 3.0e-05
  warmup_steps: 500
  decay_steps: 48829
  weight_decay: 0.1
  betas:
  - 0.9
  - 0.95
  eps: 1.0e-08
  state_offload: false
attention:
  kind: dense
  rope_base: 10000.0
  window_size: null
  latent_dim: null
  block_size: null
  selected_blocks: null
evaluation:
  every_steps: 5000
  max_batches: 1
checkpoint:
  every_steps: 5000
  every_tokens: null
  every_minutes: null
  keep_periodic: true
staging:
  smoke_steps: 2
  warmup_steps: 5
logging:
  root_dir: ${NEW_ATTEMPT_ROOT}/runs
  every_steps: 1
  architecture_diagnostics: scalar
```
