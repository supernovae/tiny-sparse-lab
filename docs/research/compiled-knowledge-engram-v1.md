# Compiled knowledge Engrams v1 — staged research definition

**Status: documentation only.** This is Experiment B, not the active learned-Engram portability implementation. No corpus is downloaded, no compiler or dataset builder is implemented, and this draft is not executable from the research catalog.

## Question

Can static lexical, factual, and relational knowledge be compiled into sparse external memory instead of acquired through language-model SGD, while preserving the neural model's ability to use and compose that knowledge?

The comparison must distinguish stored content from the recipient's ability to interpret it. Direct retrieval success is not evidence of neural composition or useful generated behavior.

## Conditions

- **Learned baseline:** acquire the same knowledge through ordinary LM training; record all training tokens and neural FLOPs.
- **Compiled/frozen:** compile licensed source material into external memory, freeze it, and train a recipient to use it without refinement of compiled values.
- **Compiled + refinement:** initialize the same compiled artifact, then permit predeclared neural refinement; report refinement separately from compilation.
- Include random-memory and conventional SGD controls. Keep corpus, recipient, trainable ownership, evaluation, and fixed capability threshold matched.

The primary eventual outcome is **training tokens and neural FLOPs to a fixed capability threshold**. Report compilation cost and refinement cost separately; neither is hidden inside neural training cost. Include held-out factual and lexical capability, novel application, multi-hop composition, and ordinary LM loss. Measure unseen-pack replacement without using test answers to construct memory.

## Staged source families

1. **Lexical:** WordNet, corpus-mined N-grams, aliases, and morphology. Preserve source, license, extraction rule, and lexical relation type.
2. **Factual and symbolic:** Wikidata under its applicable CC0 terms, generated worlds, and explicit formulas, constants, and units. Keep facts, qualifiers, units, and provenance structured.
3. **Relations:** ConceptNet only with item-level provenance and license review; ATOMIC-style relations with their specific source and license caveats. Do not merge incompatible provenance into an unqualified common corpus.
4. **Technical knowledge:** Python standard-library specifications and Kubernetes/OpenShift or other explicitly licensed technical specifications, with version and license pinned.

Do not call flattened lexical N-grams semantic memory. Relational examples remain typed records with explicit entities, predicates, arguments, qualifiers, and provenance.

## Evaluation and limits

Predeclare training/validation/test ownership before compilation. Hold out facts and relation compositions from both compilation and refinement where required by the question. Report exact per-case behavior, answer NLL, source coverage, collisions, memory bytes, compilation throughput/cost, and neural compute. Separate retrieval diagnostics from unrestricted recipient predictions. This draft supplies no license determination for any future source and authorizes no downloads or implementation.

The existing bounded MiniLM result is retained only as **preliminary bounded evidence that externally constructed memory can causally influence a jointly trained small recipient**. It used widths 32/64, seeds 17/41/73, six rows, correct-pack accuracy 1.0, baseline/random/disabled 0.125, incomplete 0.5625, conflicting 0.0, and 200 updates on 16 facts. Held-out wording reused training facts; identical synthetic token IDs were used per example while precomputed memory values supplied example-specific information. It is not learned-table portability, frozen-backbone adaptation, or unseen-fact evidence.

The earlier compiled portability smoke remains historical: its 120-arm protocol, 0.95 threshold, identities, and recorded scores are unchanged. The newly noticed answer-prefix alignment issue must be addressed prospectively; do not rewrite or rescore its immutable artifacts.
