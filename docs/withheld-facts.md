# Withheld-fact diagnostic

Milestone 0.6 provides an offline deterministic fixture for testing fact-identity data separation before transfer experiments. `split_facts(seed)` creates six training facts and two held-out facts. The split is by complete `(subject, relation)` identity; held-out values are absent from training documents. Evaluation cases expose a prompt and expected value, with the expected value excluded from the prompt.

This is not a training source, benchmark score, or evidence of byte-memory transfer. It deliberately does not add held-out facts to backbone training, memory-adapter supervision, or router objectives. Its purpose is to make accidental leakage detectable before a future trained diagnostic is designed.

```python
from sparselab.data.withheld_facts import evaluation_cases, training_documents

train = training_documents(seed=0)
held_out = evaluation_cases(seed=0)
```

Record the seed and exact fact split with any future experiment result.
