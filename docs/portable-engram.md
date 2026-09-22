# Portable Engram

A portable Engram package stores only the tokenizer-independent byte-addressed latent table plus a versioned manifest. The manifest records normalization, polynomial hashing identity, N-gram size, table shape, latent width, and the table SHA-256. It intentionally excludes a backbone-specific projection or gate.

Export from a byte-memory checkpoint:

```sh
uv run sparselab engram export RUN_ID --runs-dir runs --output artifacts/memory.engram
uv run sparselab engram inspect artifacts/memory.engram
```

`PortableEngramAdapter` freezes the exported table and trains only an output projection and context gate for a target hidden width. Byte-equivalent addressing alone does not prove transfer; use held-out facts and controlled baseline comparisons.

Held-out evaluation records exact greedy completion plus length-normalized expected-value log probability and reciprocal rank among the manifest's held-out candidate values. Candidate values are evaluation labels only; they are never used in source or target training. New reports use the immutable `withheld-v2-<manifest-sha>.json` filename.

The current two-case control remains negative: baseline, random frozen-table, and trained portable-adapter runs each have zero exact matches and reciprocal rank `0.75`. Mean expected-value log probability is `-5.0596`, `-4.9939`, and `-5.1395`, respectively. The trained adapter did not outperform the random-table control under this protocol.
