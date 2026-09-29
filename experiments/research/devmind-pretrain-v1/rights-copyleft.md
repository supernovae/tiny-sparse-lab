# DevMind v1 copyleft and compiler-source rights survey

**Disposition: survey only; no source approved for acquisition, training, release, or model-weight redistribution.** This is a bounded path/notice inspection of local Git objects, not legal advice or a blanket license clearance. GPL/LGPL/copyleft files remain *licensed with obligations*, not automatically rejected; an incompatible or unidentified component requires per-file review. The immutable v0 artifacts are unaffected.

## Reproduction and scope

Local bare mirrors: `sparselab-work/experiments/devmind-v0/source-survey/{git-full,linux-full,rust-full}`. `git --git-dir=<mirror> rev-parse HEAD` returned respectively:

| Repository | Observed HEAD | Upstream exact-pin URL |
|---|---|---|
| Git | `a018953688f1b10bddf91bff8747068f5f4746a4` | https://github.com/git/git/tree/a018953688f1b10bddf91bff8747068f5f4746a4 |
| Linux | `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` | https://github.com/torvalds/linux/tree/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e |
| Rust | `c1070d69382b8d2f2eb65119c738a77d9e324c9e` | https://github.com/rust-lang/rust/tree/c1070d69382b8d2f2eb65119c738a77d9e324c9e |

Method: enumerate exact-HEAD blobs with `git ls-tree -rz -l HEAD`, count raw Git blob sizes under the *listed prospective patterns*, retrieve each selected blob with `git cat-file --batch`, inspect the first 45 lines for SPDX, inspect representative complete notices and root rights documents with `git show HEAD:<path>`, and screen selected full text for a bounded set of explicit AI/ML/training prohibition phrases. Rust's `license-metadata.json` was traversed as a nested directory/file/group tree and cross-checked against `REUSE.toml`. This is **not** a comprehensive authorship, fixture-content, copyright, license-compatibility, training-use, or data-protection review. Counts below are raw files/bytes, not audited eligible files or distinct tokens; shell scripts were inspected as inert bytes, never executed. Reproduce an individual file with `git --git-dir=sparselab-work/experiments/devmind-v0/source-survey/git-full show a018953688f1b10bddf91bff8747068f5f4746a4:t/t0000-basic.sh` (substitute exact mirror/SHA/path for the other rows).

## Git — GPL-2.0-only root, first-party candidates

Pinned [`COPYING`](https://github.com/git/git/blob/a018953688f1b10bddf91bff8747068f5f4746a4/COPYING) explicitly says *this* GPL version 2, not a later version absent an express statement. `Documentation/git-commit.adoc` (21,772 bytes), `Documentation/technical/hash-function-transition.adoc` (36,295), `Documentation/howto/recover-corrupted-blob-object.adoc` (5,510) and `t/t0000-basic.sh` (36,975; `Copyright (c) 2005 Junio C Hamano`) exist at this commit. First-45-line SPDX scan found **0 SPDX declarations** in the 621 files below; root COPYING is a prospective default, **not** evidence that embedded examples/patches or authorship are fully cleared. Preserve GPL-2.0 text, copyright/author notices and source/revision/path with any distributed source or extracted text; obligations and downstream trained-weight treatment need a dedicated decision before release.

| Prospective include pattern (file extension and depth are significant) | Raw files / bytes | Split candidate |
|---|---:|---|
| `Documentation/*.adoc` (immediate children only) | 252 / 2,700,387 | reference-manual train |
| `Documentation/howto/**/*.adoc` | 16 / 126,313 | howto validation; review email/patch provenance |
| `Documentation/technical/**/*.adoc` | 35 / 462,046 | architecture/test candidate, independent section review |
| `t/t0*.sh` (direct children only) | 81 / 624,146 | core/plumbing shell train |
| `t/t1*.sh` (direct children only) | 99 / 916,052 | index/tree shell validation |
| `t/t9*.sh` (direct children only) | 138 / 869,510 | peripheral/integration shell test candidate |

Total candidate envelope: **621 files / 5,698,454 bytes**, of which **0 have a first-45-line SPDX tag**; this is not 621 rights-cleared files. Exclude by default `Documentation/RelNotes/**`, `Documentation/**` outside the three explicit patterns, `t/` helpers/fixtures/other numbered suites, `contrib/**`, generated data, binary/media files, and shell-derived downloaded fixtures. `t/t9116-git-svn-log.sh` (4,884 bytes) has Eric Wong copyright; keep its notice if ever selected. Since `t/t0*`, `t/t1*`, and `t/t9*` are disjoint *paths* but potentially share helpers, tests and facts, do not claim independent examples until cross-family deduplication and shared-fixture review. GPL is a rights/redistribution-obligation review state, **not an exclusion triggered by the word GPL**.

## Linux — GPL-2.0-only kernel with per-file exceptions

Pinned [`COPYING`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/COPYING) identifies GPL-2.0 with Linux-syscall-note and expressly points to [`Documentation/process/license-rules.rst`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/Documentation/process/license-rules.rst) for additional file licenses. Consult the exact-pin [`LICENSES/preferred/GPL-2.0`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/preferred/GPL-2.0) and [`LICENSES/exceptions/Linux-syscall-note`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/exceptions/Linux-syscall-note), rather than claiming all kernel documentation/code has one uniform SPDX expression. `fs/ext4/super.c` says `GPL-2.0`; `kernel/sched/core.c` and `security/selinux/hooks.c` say `GPL-2.0-only`. `fs/ext4/inline.c` says `LGPL-2.1` and credits Taobao; `net/ipv4/tcp_bbr.c` says `GPL-2.0 OR BSD-3-Clause`. These are licensed exceptions, not an unlicensed bucket.

| Prospective include pattern | Raw files / bytes | First-45-line SPDX observations beyond plain GPL-2.0, GPL-2.0-only, GPL-2.0-or-later or GPL-2.0+ |
|---|---:|---|
| `Documentation/process/**/*.rst` | 57 / 783,661 | 36 without tag; 1 `(GPL-2.0+ OR CC-BY-4.0)` |
| `Documentation/filesystems/**/*.rst` | 148 / 2,128,560 | 26 without tag |
| `Documentation/networking/**/*.rst`, **excluding** `Documentation/networking/device_drivers/**` | 183 / 2,052,101 | 18 without tag; 7 dual-licensed BSD/MIT expressions |
| `fs/ext4/**/*.{c,h}` | 51 / 1,994,621 | 3 `LGPL-2.1`; 1 without tag |
| `fs/xfs/**/*.{c,h}` | 355 / 6,278,916 | 1 `LGPL-2.1` |
| `net/ipv4/**/*.{c,h}` | 130 / 3,017,632 | 2 `((GPL-2.0 WITH Linux-syscall-note) OR BSD-3-Clause)`, 1 `GPL-2.0 OR BSD-3-Clause` |
| `net/core/**/*.{c,h}` | 75 / 2,363,054 | 2 syscall-note/BSD-3-Clause dual expressions |
| `kernel/sched/**/*.{c,h}` | 53 / 2,163,866 | no nonstandard/headerless count in this screen |
| `mm/**/*.{c,h}` | 195 / 6,319,595 | 1 `LGPL-2.1` |
| `security/selinux/**/*.{c,h}` | 56 / 736,105 | no nonstandard/headerless count in this screen |

Total envelope: **1,303 files / 27,838,111 bytes**. Header screen: **1,204 plain-GPL-expression**, **81 no first-45-line SPDX**, **18 other SPDX expressions** (including **5 LGPL-2.1** and **13 alternative-license-expression files**). Headerless means *needs file/parent-rule review*, not necessarily unlicensed; none of these counts certify file origin. `Documentation/process/handling-regressions.rst` (48,055 bytes) includes an explicit bottom-of-file CC-BY-4.0 attribution/source instruction and warns that processed versions might incorporate more restrictive material; use only reviewed original RST if selecting that alternative. `Documentation/filesystems/ext4/about.rst` explicitly says the book is GPL v2. Do not collapse either to a generic root-license claim. Exclude `drivers/**` entirely, nested `Documentation/networking/device_drivers/**`, `Documentation/devicetree/bindings/**`, media, generated source, and any excerpt with unmatched third-party content or unidentified author/rights until reviewed. GPL/LGPL paths may be prospectively included subject to notice/source-distribution obligations; no decision about derivative weights is inferred.
Exact-pin additional Linux license texts for observed exceptions: [`LGPL-2.1`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/preferred/LGPL-2.1), [`BSD-3-Clause`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/preferred/BSD-3-Clause), [`BSD-2-Clause`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/preferred/BSD-2-Clause), [`MIT`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/preferred/MIT), and [`CC-BY-4.0`](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/LICENSES/dual/CC-BY-4.0). Each dual-expression choice requires review, not automatic relicensing of adjacent material.

For family splits, use `Documentation/process/` (development-process prose) vs `Documentation/filesystems/` (storage prose) vs non-driver `Documentation/networking/` (network prose), and separately `fs/ext4/` vs `fs/xfs/` vs `net/ipv4/` vs `net/core/` vs `kernel/sched/` vs `mm/` vs `security/selinux/` by whole subtree. Example *candidate* assignment: process/filesystems and ext4/ipv4/sched for train, networking and xfs/core for validation, mm/selinux for test. Topic overlap (especially filesystem docs vs ext4/xfs implementation) defeats mere path disjointness: near-duplicate, upstream quotation and answer-fact leakage checks must precede any locked split.

## Rust compiler/library — dual default, nested exceptions

Pinned [`COPYRIGHT`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/COPYRIGHT) says Apache-2.0 **OR** MIT except as otherwise noted and points to [`REUSE.toml`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/REUSE.toml) and the committed nested [`license-metadata.json`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/license-metadata.json). Exact license texts: [`LICENSE-APACHE`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/LICENSE-APACHE), [`LICENSE-MIT`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/LICENSE-MIT). Preserve selected choice's license text, notices and credits, plus exceptional component terms where applicable.

| Prospective `.rs` subtree | Raw files / bytes | Potential family |
|---|---:|---|
| `library/core/src/**/*.rs` | 291 / 6,531,117 | core primitives, prospective train |
| `library/alloc/src/**/*.rs` | 95 / 2,409,555 | allocation/collections, separate validation |
| `library/std/src/**/*.rs` | 646 / 5,165,560 | standard-library platform APIs, separate test |
| `compiler/rustc_ast/src/**/*.rs` | 25 / 413,888 | AST train |
| `compiler/rustc_hir/src/**/*.rs` | 10 / 300,045 | HIR validation |
| `compiler/rustc_middle/src/**/*.rs` | 139 / 2,548,800 | MIR/middle train, exception below |
| `compiler/rustc_borrowck/src/**/*.rs` | 62 / 1,530,726 | borrow-checking test |
| `compiler/rustc_driver_impl/src/**/*.rs` | 8 / 102,782 | compiler-driver validation |

Total envelope: **1,276 files / 19,002,473 bytes**. `REUSE.toml` gives the `compiler/**` and `library/**` default `MIT OR Apache-2.0`; cached metadata spells its equivalent `Apache-2.0 OR MIT`. Most selected Rust files do **not** contain first-45-line SPDX (1,273/1,276); this is expected for this tree's REUSE annotations, **not** a reason to invent a per-file SPDX header. Nested exceptions in the envelope: `library/core/src/unicode/unicode_data.rs` (95,875 bytes) has **Unicode-3.0** in metadata and says it is generated; its sibling `library/core/src/unicode/mod.rs` (1,890) has an explicit nested override back to Apache-2.0 OR MIT. `library/std/src/sys/sync/mutex/fuchsia.rs` (6,178) has **BSD-2-Clause AND (Apache-2.0 OR MIT)** and credits Fuchsia in the metadata; `compiler/rustc_middle/src/ptrauth/llvm_siphash/tests.rs` (2,695) has **Apache-2.0 WITH LLVM-exception AND (Apache-2.0 OR MIT)** in metadata/REUSE, identifies derivation from LLVM, and has an explicit LLVM-exception SPDX line. These **three** files are exception-review/exclude before any default-license allowlist; none is automatically incompatible merely for having another license.
Pinned Rust exception license texts: [`Unicode-3.0`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/LICENSES/Unicode-3.0.txt), [`BSD-2-Clause`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/LICENSES/BSD-2-Clause.txt), and [`LLVM-exception`](https://github.com/rust-lang/rust/blob/c1070d69382b8d2f2eb65119c738a77d9e324c9e/LICENSES/LLVM-exception.txt).

Additional embedded-origin flags not captured by a root default: `library/alloc/src/boxed/thin.rs` credits a specified external RFC example; `library/std/src/os/unix/net/ucred.rs` says it is heavily based on a tokio-uds PR and credits Martin Habovštiak and contributors. Put both in **review/exclude** pending source/original-license tracing even though the metadata default matches. Exclude `src/gcc/**` (nested GPL-3.0-or-later and GPL-2.0-only/ISC subtrees in metadata), `src/llvm-project/**` (LLVM/NCSA metadata), `src/librustdoc/html/static/fonts/**` (OFL), vendored crates, generated Unicode data, and compiler/library paths outside the bounded allowlist. These are *scope/metadata exclusions*, not assertions of universal incompatibility. Distinct Rust library/compiler subtree names do not prevent overlap with Rust Book or embedded examples; compare source parents and near duplicates before assigning any train/validation/test identity.

## Decision and remaining blockers

The keyword screen of the **selected envelopes** returned **zero matches** for explicit `no-AI`/`no-ML`, `not for training`, or nearby training-prohibition constructions. This is limited negative evidence, **not a finding of no training restrictions**: notices can occur outside selected paths, in linked original works or external contributor agreements. Explicitly observed third-party/atypical terms are assigned review/exclude above. Unknown embedded material must not become train-eligible by inheritance from a repository root.

For any eventual redistributable extraction, retain exact upstream URI/revision/path, actual applicable license expression/text and copyright/attribution/exception notices, mark transformed excerpts, and resolve GPL/LGPL source/notice obligations and downstream weight-release policy separately. Needed before declaration or fit: full selected-file notice/origin audit (including Rust external derivations and Git shell fixtures), review of Linux's 81 headerless and 18 exceptional-SPDX files, exact licensing of held-out candidates, tests for shared source parents and lexical/semantic leakage, sensitive fixture screening, and human decision on distribution/training/weights. **Prospective state: all three repositories `reference_only` pending those gates; zero declared training-ready files, no v1 release ID, and no tokenizer or model run follows from this survey.**
