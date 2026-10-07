# Day-2 Research Track: Calibrated Sensemaking and Recovery

**Status: planning only.** This document proposes future experiments. This docs-only PR does not implement a generator, change training, run experiments, or authorize production actions.

## Foundation boundary

Preserve the foundation plan: a proposed, from-scratch approximately 342M language kernel, subject to its existing data, correctness and measured-fit gates. No such checkpoint is established by this document. Expert/sensemaking behavior is a separate, gated day-2 research track. Judge it by decisions and verified outcomes, not expert-sounding prose. None of the cited work establishes expert reliability for this model size or architecture.

Before day 2: establish and verify the fresh foundation baseline through the existing cards, preserve its checkpoint/configuration/tokenizer/data manifest, establish held-out language/comprehension and simple-task tests (plus code tests for any approved CI/CD scope), and freeze evaluation splits and budgets. No prior MODEL-0 checkpoint or reproduction is a dependency. No day-2 checkpoint replaces the foundation without evidence of benefits and acceptable retention. Keep an easy rollback to the baseline.

The [source catalog](SOURCE_CATALOG.md) can inform the existing Card 03 corpus review now; it does not authorize ingestion. Behavior training and generator implementation remain later work, with current state recorded only in [STATUS.md](STATUS.md).

## Training stages and evidence

1. **Base corpus:** broad language/code competence plus rights-cleared technical material. Resilience literature can supply concepts and vocabulary; its inclusion does not establish operational judgment.
2. **Optional domain-adaptive pretraining (DAPT):** compare a selected personal/OSS corpus against the baseline, with general-capability retention tests. DAPT improved downstream classification in the original study; transfer to CI/CD action selection or a 342M decoder is an experimental hypothesis. [Gururangan et al.](https://aclanthology.org/2020.acl-main.740/)
3. **Supervised fine-tuning (SFT):** demonstrate direct answers for simple work; evidence inspection when useful; decision-relevant questions; bounded actions; outcome verification; and recovery or escalation. Instruction tuning is a stronger precedent for behavior shaping than merely adding domain documents. [Ouyang et al.](https://arxiv.org/abs/2203.02155)
4. **Optional preference tuning:** compare matched responses differing in correctness, unnecessary ceremony, evidence use, reversibility, and recovery. DPO is a candidate, not a guaranteed improvement. [Rafailov et al.](https://arxiv.org/abs/2305.18290)
5. **Tool trajectories:** teach tool choice, arguments, actual observations, and evidence-driven next actions. Include failed calls and unavailable tools. Toolformer and ReAct provide precedents, not proof of competence at this scale. [Toolformer](https://arxiv.org/abs/2302.04761), [ReAct](https://arxiv.org/abs/2210.03629)

Recommended ablations: baseline; baseline + SFT; baseline + DAPT + SFT; best preceding variant + preference tuning. Hold evaluation prompts, task families, inference budgets, and tool access constant; report training compute separately. Keep each change separately identifiable rather than combining all interventions in the first experiment.

## Parameters and system costs

Continued pretraining, full SFT, and full-parameter preference tuning update existing weights without adding parameters, assuming architecture and vocabulary stay fixed.

LoRA adds trainable low-rank matrices while freezing base weights. Unmerged deployment carries adapters; compatible updates can be merged into original weight matrices, preserving original deployed tensor shapes and parameter count. Adapter files, optimizer state, checkpoints, and training compute still cost resources. [LoRA](https://arxiv.org/abs/2106.09685)

Retrieval/external memory need not enlarge the approximately 342M generator. They do add index/storage costs, retrieval latency, context tokens, and potentially separate embedding/reranking models. Report total system resources, not only generator parameters. Stored facts do not replace learned language competence, evidence use or judgment. [RAG](https://arxiv.org/abs/2005.11401)

## Behavior contract

Treat Cynefin as a fallible interpretive lens, not a ground-truth classification oracle. The framework motivates contextual responses, not automatic operational correctness. [Snowden and Boone](https://hbr.org/2007/11/a-leaders-framework-for-decision-making)

Train a revisable repertoire:

- **Direct response:** sufficient evidence, familiar low-consequence task
- **Inspect then act:** bounded diagnosis can establish an answer
- **Probe and observe:** uncertain interactions call for a reversible experiment
- **Stabilize then reassess:** active disruption calls for authorized containment before deeper diagnosis
- **Clarify or escalate:** material missing information, irreversibility, or authority blocks safe progress

Modes are not mandatory output labels or fixed personality settings. New observations can change the appropriate mode. A chaotic situation does not imply permission to perform chaos engineering.

For deliberate work, record only concise observable decisions: goal/success check; material assumption or uncertainty; evidence consulted; selected action and limit; observed result; and next step, recovery, or stop condition. Simple tasks should usually bypass this expanded record.

Do not require hidden chain-of-thought, exhaustive private deliberation, or theatrical expert panels. A decision record is an auditable claim about evidence and actions, not proof of internal reasoning. Generated explanations can rationalize incorrect answers. [Turpin et al.](https://arxiv.org/abs/2305.04388)

Resilience-inspired evaluation should test adaptation when assumptions break, not only replay of known recovery recipes. Woods distinguishes rebound, robustness, graceful extensibility, and sustained adaptability. [Woods](https://doi.org/10.1016/j.ress.2015.03.018)

## Proposed generator contract

This is a future specification, not an implemented generator.

### Inputs

- Original personally authored scenarios or explicitly permitted OSS fixtures
- Source provenance and use restrictions
- Scenario-family identifier and random seed
- Latent actual state/fault, dependencies, permitted actions, available observations, action costs, reversibility, timing, and transitions
- Visibility boundaries distinguishing model-visible evidence from evaluator-only facts
- Counterfactual dimensions and intended evaluation split

### Outputs

- Model-visible task, evidence, permissions, and tool interface
- Evaluator-only world state and acceptable outcome/action constraints
- Verified tool/action/outcome trajectory, including failure and recovery where applicable
- Alternative acceptable choices when judgment admits more than one good answer
- Paired counterfactuals with one material fact changed and most wording retained
- Provenance, generator/template version, seed, validation result, and split membership

### Invariants and rejection criteria

- No employer material, credentials, private incident data, or unreviewed copied text
- No evaluator-only facts leaked into the prompt or answer key explanations shown to the model
- No invented tool observations or impossible gold actions
- No label determined solely by mentioning a framework, risk vocabulary, or urgency
- Reject inconsistent worlds, unsupported conclusions, duplicates, and split contamination
- Hold out whole scenario families, projects, fault mechanisms, and templates
- Validate executable outcomes independently; a generator model must not be the sole evaluator
- Preserve ambiguous cases and multiple reasonable choices rather than forcing a false oracle label

Generation plus filtering has a research precedent, but synthetic labels still require independent checks. [Self-Instruct](https://arxiv.org/abs/2212.10560)

Provenance fields should include source URL, revision/hash, license, attribution requirements, allowed use, acquisition date, personal/OSS origin, generator/template version, model/version and applicable generation terms, seed, validation result, and split membership. Public availability alone is not a training license.

## Evaluation matrix

| Family | Personal/OSS CI/CD example | Expected observable behavior | Negative control |
| --- | --- | --- | --- |
| Simple | Documented formatter failure with exact command | Give/run the authorized fix, verify narrowly, stop | No framework recital, broad investigation, or unnecessary question |
| Complicated | Deterministic dependency incompatibility | Inspect versions, reproduce, make evidence-supported bounded fix | No speculative rewrite or blind upgrade |
| Complex | Flaky concurrency/retry/shared-cache interaction | Choose discriminating bounded probes; revise assumptions with observations | No confident single-cause claim from incomplete logs |
| Chaotic | Simulated runaway job exhausts disposable environment | Apply authorized stop action, verify stabilization, then diagnose | No unapproved destructive action or prolonged analysis before available containment |
| Ambiguous | Insufficient evidence distinguishes two causes | State material uncertainty and select useful next observation | No forced Cynefin label or arbitrary certainty |
| Recovery | Initial fix fails or rollback is unavailable | Report observed failure; revise or escalate within authority | No repeat-until-success loop or fabricated recovery |
| Routine success | Healthy pipeline or correct user diagnosis | Confirm evidence and finish | No manufactured criticism, forced failure narrative, or gratuitous doubt |
| Authority boundary | Useful action exceeds permissions | Pause that action and request specific authorization | No urgency-based permission expansion |

Matched adversarial variants should include misleading urgency, correct and incorrect confident user hypotheses, stale runbooks, incomplete logs, irrelevant uncertainty, failed rollback, and cases where waiting or asking a question is worse than an authorized reversible action. Vary wording independently from the underlying decision to expose superficial shortcuts.

### Measures

- Task success and correctness
- Unauthorized/destructive action rate
- Evidence-supported claims and fabricated observations
- Appropriate action choice and recovery success
- Unnecessary clarification, refusal, escalation, and tool calls
- Response length, latency, tokens, and total tool cost
- Answer coverage versus error among answered cases
- Calibration, including Brier score and reliability plots where binary outcomes support probabilistic scoring
- General language/code regressions and simple-task overhead

Measure on familiar and shifted distributions. Verbal confidence can be trained, but calibration may degrade on new tasks; high-confidence wording is not a safety mechanism. [Lin et al.](https://arxiv.org/abs/2205.14334), [Kadavath et al.](https://arxiv.org/abs/2207.05221)

Evaluate abstention jointly with coverage. Refusing everything is not success. Domain shift matters because raw model probabilities can be overconfident outside training distributions. [Kamath et al.](https://aclanthology.org/2020.acl-main.503/)

### Advancement gates

Before training, pre-register baseline-relative acceptance margins, evaluation budgets, and per-family safety thresholds. Use mechanical checks for executable outcomes and blinded human review for judgment. Report per-family results and uncertainty intervals instead of a single expertise score.

A candidate advances only if it demonstrates the intended decision/recovery benefit while satisfying safety, provenance, foundation-retention, and simple-task overhead gates. Increased verbosity, blanket abstention, or better framework naming alone does not count as improvement. Investigate regressions; keep the baseline when benefits are inconclusive.

## Failure modes and limits

- **Performative expertise:** rewards for framework vocabulary, lengthy checklists, or confidence
- **Overskepticism:** challenging correct assumptions, delaying straightforward work, or escalating everything
- **Sycophancy:** agreeing with a confident user/evaluator despite contrary evidence
- **Unproductive review:** changing answers without new evidence or useful checks
- **Proxy optimization:** reward hacking, synthetic-label artifacts, and evaluator leakage
- **Transfer overclaim:** treating simulator performance as production competence or permission

Preference judgments can favor persuasive agreement over truth; proxy overoptimization can degrade the underlying objective. [Sycophancy study](https://arxiv.org/abs/2310.13548), [Reward overoptimization](https://arxiv.org/abs/2210.10760)

Review should seek external checks where available. Intrinsic self-correction sometimes failed or worsened answers in tested settings; this cautions against unconditional review loops rather than proving universal impossibility. [Huang et al.](https://arxiv.org/abs/2310.01798)

Future fault injection is restricted to disposable personal/OSS environments with explicit scope, stop conditions, and verified cleanup. No autonomous production chaos. Chaos engineering supplies an experimental inspiration, not operational permission. [Original chaos-engineering paper](https://arxiv.org/abs/1702.05843)
