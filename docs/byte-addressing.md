# Raw UTF-8 byte addressing

Milestone 0.4 adds a prepared-data byte-address path for optional `model.memory: byte`. Data preparation records one causal address per input token position in `train_byte_addresses.npy` and `validation_byte_addresses.npy`. Each address hashes the raw UTF-8 suffix ending at the token's source-text offset; the manifest records `raw-utf8-suffix-v1`, table size, n-gram size, and array hashes.

No Unicode normalization, case folding, whitespace normalization, or escaping occurs. Equal raw byte spans yield equal pre-reduction hashes regardless of text segmentation; visually similar strings with distinct UTF-8 bytes do not. EOS receives address zero because it is a structural model token, not source text.

Milestone 0.5 extends this contract to greedy generation. Prompt token offsets produce causal prompt addresses. Each generated non-special token contributes its ByteLevel-decoded UTF-8 bytes to the prompt byte stream; the next model call receives the aligned cropped ID/address suffix. This is deterministic greedy generation, not a proof of byte-retrieval quality or cross-tokenizer transfer.

The byte-memory table is local trainable state, checkpointed with the model. Train, standalone evaluation, and greedy generation use byte addresses. This milestone does not claim withheld-fact learning or tokenizer-agnostic training.

```sh
uv run sparselab data prepare configs/smoke_byte_memory_cpu.yaml
uv run sparselab train configs/smoke_byte_memory_cpu.yaml --run-id byte-memory
uv run sparselab eval byte-memory
uv run sparselab generate byte-memory --prompt "Once upon a time" --max-new-tokens 24
```
