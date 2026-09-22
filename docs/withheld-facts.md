# Withheld-fact diagnostic

Milestones 0.6–0.7 provide an offline deterministic fixture and an immutable split manifest for testing fact-identity data separation before transfer experiments. `split_facts(seed)` creates six training facts and two held-out facts. The split is by complete `(subject, relation)` identity; held-out values are absent from training documents. Evaluation cases expose a prompt and expected value, with the expected value excluded from the prompt.

This is not a training source, benchmark score, or evidence of byte-memory transfer. It deliberately does not add held-out facts to backbone training, memory-adapter supervision, or router objectives. Its purpose is to make accidental leakage detectable before a future trained diagnostic is designed.

```python
from sparselab.data.withheld_facts import evaluation_cases, training_documents

train = training_documents(seed=0)
held_out = evaluation_cases(seed=0)
```

Before running a diagnostic, write and retain a manifest:

```sh
uv run sparselab facts manifest --seed 0 --output artifacts/withheld-facts-seed-0.json
```

The manifest records the ordered training statements, held-out prompt/expected-value cases, seed, format version, and a SHA-256 digest of its canonical payload. Re-running with the same seed accepts byte-identical evidence; a different seed cannot overwrite the existing manifest. It contains no model outputs, training examples, or score.

Record the manifest alongside any future experiment result. It proves the fixture split only; data-provenance evidence is still required to prove a training pipeline obeyed that boundary.
