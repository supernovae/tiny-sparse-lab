# DevMind v0 source project

This is a proposed real research corpus, **not** `devmind-sample-v0` and not a training-readiness claim. The source-of-truth is `corpus.yaml` and its versioned source, split, transform, and release declarations. Immutable acquired bytes, builds, frozen releases, tokenizer outputs and run artifacts remain under `sparselab-work/experiments/devmind-v0/`. No source snapshot or model checkpoint belongs in Git.

## Pinned sources and rights

| Source family | Revision | Terms and required notice | Split |
|---|---|---|---|
| Kubernetes `content/en/docs/concepts/`, `tasks/`, `tutorials/` | `69f4fc12ef397cacb60d3b6830ef7c6d6b87245e` | [CC BY 4.0](https://github.com/kubernetes/website/blob/69f4fc12ef397cacb60d3b6830ef7c6d6b87245e/LICENSE); credit Kubernetes Authors, link terms, indicate extraction/normalization | train / validation / test |
| CPython `Doc/library/`, `Lib/`, `Doc/tutorial/` | `333071231d3a46cccc32d7f44b99328c3299d0b1` | [PSF 2.0](https://github.com/python/cpython/blob/333071231d3a46cccc32d7f44b99328c3299d0b1/LICENSE), including dual PSF/0BSD documentation-example notice | train / train / validation |
| Go `doc/`, `src/net/http/`, `src/encoding/json/` | `2a2f42823b1877f5f9189609ba7b4d2ebdcfaf48` | [BSD 3-Clause](https://github.com/golang/go/blob/2a2f42823b1877f5f9189609ba7b4d2ebdcfaf48/LICENSE); preserve notice and disclaimer | train |
| Rust Book `src/ch*.md` | `1500248d8f230566e4ec9f27fcbb8fe9e2898ab1` | [MIT](https://github.com/rust-lang/book/blob/1500248d8f230566e4ec9f27fcbb8fe9e2898ab1/LICENSE-MIT); retain author/license notice | train |
| OpenStax Physics `modules/*/index.cnxml` | `dfdfd7a5356ecdd42e504de3df50d9153e33ea49` | [CC BY 4.0](https://github.com/openstax/osbooks-physics/blob/dfdfd7a5356ecdd42e504de3df50d9153e33ea49/LICENSE); credit OpenStax/Rice University, link terms, disclose CNXML text extraction | train, except module `m54081` validation and `m54082` test |
| SparseLab deterministic inert scenarios | generator version `1` | [MIT](../../LICENSE); never executed as tools | train / validation / test by world and template family |

Git selectors are limited to text suffixes and bounded bytes. This inventory is **not** a blanket legal clearance: review each selected file for embedded third-party terms and attribution before marking an acquired release training-eligible. Exclude any conflicting document and issue a new declaration; do not silently alter a frozen release. XML images and linked media are excluded. The [college physics bundle](https://github.com/openstax/osbooks-college-physics-bundle/blob/fd1b25dfd5d8c6580c6e2b2b34a19e29cc69ada9/LICENSE) is explicitly **rejected**: CC BY-NC-SA 4.0, not this source. Unknown-license crawls, logos, repositories' media and unreviewed third-party material are not a shortcut to the token target. Full notices and any third-party per-file exclusions belong in the checked-in audit report before training.

Each source family has exactly one declared split; derived siblings inherit source/world/template lineage. Synthetic worlds are inert descriptions of possible observations and judgments, not independent real developer knowledge or permission to run commands. `release.yaml` weights are descriptive, not sampling quotas. Token mix (including general-education share), unique eligible train content and actual train-view capacity require selected-tokenizer measurement. No C lock if distinct train content <150 million tokens, general train tokens >10%, any required text kind missing, rights unresolved, or source/evaluation family leakage occurs.

Run the acquire/build/freeze/audit commands in the [research protocol](../../experiments/research/devmind-pretrain-v0/protocol.md) using the locked environment. All run and release IDs are observation-dependent; do not replace them with guessed paths.
