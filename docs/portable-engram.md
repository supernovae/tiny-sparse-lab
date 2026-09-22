# Portable Engram

A portable Engram package stores only the tokenizer-independent byte-addressed latent table plus a versioned manifest. The manifest records normalization, polynomial hashing identity, N-gram size, table shape, latent width, and the table SHA-256. It intentionally excludes a backbone-specific projection or gate.

Export from a byte-memory checkpoint:

```sh
uv run sparselab engram export RUN_ID --runs-dir runs --output artifacts/memory.engram
uv run sparselab engram inspect artifacts/memory.engram
```

`PortableEngramAdapter` freezes the exported table and trains only an output projection and context gate for a target hidden width. Byte-equivalent addressing alone does not prove transfer; use held-out facts and controlled baseline comparisons.
