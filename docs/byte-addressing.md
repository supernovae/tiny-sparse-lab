# Raw UTF-8 byte addressing

Optional `model.memory: byte` uses one causal address per input token in `train_byte_addresses.npy` and `validation_byte_addresses.npy`. Each address hashes the raw-byte suffix ending at that token's exact ByteLevel byte boundary; the manifest records `raw-utf8-suffix-v1`, table size, n-gram size, and array hashes. Packing version `contiguous-eos-v3` introduced the corrected boundary semantics. Current `contiguous-eos-v4` caches additionally bind source/tokenizer/settings identity, a manifest digest, and each array's dtype, shape, token-stream length, and file hash. Byte-address arrays must align with the corresponding token arrays. Older cache directories remain untouched.

No Unicode normalization, case folding, whitespace normalization, or escaping occurs. Equal raw byte spans yield equal pre-reduction hashes regardless of text segmentation; visually similar strings with distinct UTF-8 bytes do not. EOS receives address zero because it is a structural model token, not source text.

Packing and generation share the same reversible ByteLevel token-byte conversion. A multibyte character may span several tokens: each partial UTF-8 byte sequence contributes only bytes actually consumed so far, never replacement characters or future bytes from a character-end offset. The next model call receives aligned cropped ID/address suffixes. This applies to greedy and sampled generation, including chat.

The byte-memory table is local trainable state, checkpointed with the model. Byte addressing is not proof of retrieval quality, withheld-fact learning, or cross-tokenizer semantic transfer. Prepared-data caches bind tokenizer serialization, source-code identity, packing version, source contents where local, and array digests; changing a generator cannot silently reuse old packed data.

```sh
uv run sparselab data prepare configs/smoke_byte_memory_cpu.yaml
uv run sparselab train configs/smoke_byte_memory_cpu.yaml --run-id byte-memory
uv run sparselab eval byte-memory
uv run sparselab generate byte-memory --prompt "Once upon a time" --max-new-tokens 24
```
