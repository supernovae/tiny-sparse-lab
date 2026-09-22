# Raw UTF-8 byte addressing

Milestone 0.4 adds a prepared-data byte-address path for optional `model.memory: byte`. Data preparation records one causal address per input token position in `train_byte_addresses.npy` and `validation_byte_addresses.npy`. Each address hashes the raw UTF-8 suffix ending at the token's source-text offset; the manifest records `raw-utf8-suffix-v1`, table size, n-gram size, and array hashes.

No Unicode normalization, case folding, whitespace normalization, or escaping occurs. Equal raw byte spans yield equal pre-reduction hashes regardless of how text is segmented; visually similar strings with distinct UTF-8 bytes do not. EOS receives address zero because it is a structural model token, not source text.

The byte-memory table is local trainable state, checkpointed with the model. Train and standalone evaluation use prepared address blocks. Greedy generation is deliberately unavailable for byte-memory runs until prompt byte-address preparation is implemented. This milestone does not claim byte-hash retrieval quality, cross-tokenizer transfer, or withheld-fact learning.

```sh
uv run sparselab data prepare configs/smoke_byte_memory_cpu.yaml
uv run sparselab train configs/smoke_byte_memory_cpu.yaml --run-id byte-memory
uv run sparselab eval byte-memory
```
