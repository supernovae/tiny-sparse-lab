# Compiled Engram initialization v1 — staged bridge definition

**Status: documentation only.** This bridge is not part of Experiment A and is not an executable catalog entry. It does not authorize a compiler, corpus download, or training run.

## Question

Does compiled external memory improve learning efficiency when used to initialize a system, and does neural refinement preserve or improve its value compared with learning from random memory?

## Prespecified comparison

- **Random → train:** initialize random external values and train under the declared ordinary learning budget.
- **Compiled → freeze:** initialize from the compiled artifact and keep its values frozen while the neural recipient learns to use them.
- **Compiled → refine:** initialize from the same compiled artifact and allow explicitly declared refinement.

Use identical recipient architecture, data exposure, optimizer policy, and stopping rule wherever ownership permits. Record any unavoidable compute or information asymmetry rather than treating the arms as equivalent.

## Outcomes

Measure held-out factual and lexical capability, novel application, multi-hop composition, and ordinary LM loss. The eventual primary efficiency outcomes are **training tokens and neural FLOPs to a fixed capability threshold**. Record compilation cost, neural training cost, and refinement cost as separate quantities. Include memory size, exact artifact identity, source coverage, per-case predictions, and unseen-pack replacement behavior. Do not use expected evaluation answers to build or align compiled values.

A positive result would justify a separately scoped study; it does not follow from artifact integrity or exact lookup alone. Experiment A tests a distinct question: whether a table learned jointly by a source model can transfer to independent widths using recipient-local calibration. Natural-language query integration and teacher representations remain separate deferred questions and are not bridge prerequisites.
