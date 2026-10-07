# Card 03 scaled preparation proposal — metadata sizing v1

Status: **APPROVED FOR ONE PREPARATION ATTEMPT** under
[KML-D15](DECISIONS.md#kml-d15--one-scaled-card-03-preparation-attempt-approved-2026-10-07),
subject to the owner's clarifications recorded there. This is a bounded
preparation allocation, not model training authority. It preserves the accepted
source-admission policy v1, 65/25/10 realized target-token mix, 32,768-entry
train-only byte-level BPE, 80/10/10 independent-family split, 200 language and
400 evidence evaluation items, and 341,885,952-parameter architecture. The
first candidate is **5,000,000 training target positions**, at no more than two
exposures per unique train position. This is half of the earlier aspirational
10-million-position candidate because the pinned incident sources do not yet
support a credible 500,000-unique-position claim. It is not a Card 05 run
approval.

## Exact pinned input selection

| Source | Immutable selection | First-pass response-body bytes |
| --- | --- | ---: |
| Project Gutenberg filtered | `3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7` / `project_gutenberg-dolma-0014.json.gz`; SHA-256 `bdcb5e7e0e48d42489949fe2492b765c71d4eae1e1cd6de34cddfb1b4360b3ee` | 362,929,669 |
| Wikimedia filtered | `0641bb84bd9b7162bcddf8be7836822161a9a342` / `wikimedia-0027.json.gz`; SHA-256 `0abe6c9be821ac100c604ac4b799faa00c90e39d139c7dccf9491e693b177ed0` | 480,242,750 |
| PagerDuty incident-response docs | commit `464fc9d3e47e19e9d8da17cec1a41dc09624e95a`, tree `0a6d6a7e05784260ddd649513af6d42f003c5927`; 36 `docs/**/*.md` plus LICENSE and README, 38 pinned blobs | 317,526 |
| Scoutflo SRE Playbooks | commit `acc55da0224fda5eda580939a15bc60990e8e69f`, tree `1e77048f12715248fd7a1cab6c4d0bdc4eb2ec86`; 431 AWS/Kubernetes/Sentry playbook Markdown files plus LICENSE and root README, 433 pinned blobs | 2,087,503 |
| **Total** | **2 complete HF shards and 471 Git blobs** | **845,577,448** |

The two complete HF files were deleted after the earlier pilot; only their
small selected-row snapshots remain. Charge the **entire 843,172,419-byte
redownload** to this request. Preserve the pilot snapshots, declarations,
receipts, historical screen and failed/negative evidence. Create new project,
transport-ledger, snapshot, admission and release identities for this tranche.

The HF file sizes and digests come from the pinned [Gutenberg tree](https://huggingface.co/api/datasets/common-pile/project_gutenberg_filtered/tree/3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7?recursive=true)
and [Wikimedia tree](https://huggingface.co/api/datasets/common-pile/wikimedia_filtered/tree/0641bb84bd9b7162bcddf8be7836822161a9a342?recursive=true).
The Git selections come from the pinned [PagerDuty repository](https://github.com/PagerDuty/incident-response-docs/tree/464fc9d3e47e19e9d8da17cec1a41dc09624e95a)
and [Scoutflo repository](https://github.com/Scoutflo/Scoutflo-SRE-Playbooks/tree/acc55da0224fda5eda580939a15bc60990e8e69f).
Scoutflo's [MIT license](https://github.com/Scoutflo/Scoutflo-SRE-Playbooks/blob/acc55da0224fda5eda580939a15bc60990e8e69f/LICENSE)
is source-level evidence to review under policy v1, not blanket admission of
third-party inserts. PagerDuty's pinned Apache-2.0 policy is already proposed;
its file contents are still unreviewed. Git blob IDs bind selected bytes, and
the native acquisition receipt must also record SHA-256 of each downloaded
file. No PDF, image, generated file or unselected repository path enters the
tranche.

## Selection, expected capacity and measured stop

Use the existing bounded-HF path with one pinned shard per Common Pile source.
Scan each selected gzip to EOF, subject to `max_scanned_rows` 3,000 for
Gutenberg and 1,000,000 for Wikimedia, and deterministically retain indices
whose existing revision/path/index hash has remainder zero modulo **16**.
Set Gutenberg `max_rows` 500 and Wikimedia `max_rows` 100,000; a ceiling hit is
a stopped attempt, not permission to truncate invisibly. Use source-level
policy inheritance and automated per-row exceptions under accepted policy v1.
Keep the previously flagged Wikimedia privacy row, any conflicting-license
row and genuinely unknown coverage quarantined **as rows**, while allowing
other qualifying rows from that source. Log optional metadata issues without
automatic quarantine. Review all flags and a preregistered stratified spot
sample before release. Bind a new source policy for Scoutflo's commit, paths,
use, attribution and exception rules before admitting its files; check
third-party text, secrets, personal data and template/near-duplicate content.

Metadata and pilot extrapolation suggest roughly **120–160 Gutenberg work
candidates**, **15,000–25,000 Wikimedia page candidates**, and **467 incident
Markdown documents** before exclusions. The planning estimate after admission,
family split and deduplication is **10–20 million unique Gutenberg train target
positions across about 100–140 independent work families**, **4–10 million
Wikimedia positions across about 8,000–18,000 page families**, and
**0.25–0.38 million incident positions across about 150–300 independent
scenario/topic families**. These are deliberately broad estimates from source
bytes, the pilot's selected rows and an assumed 4–5 UTF-8 bytes per eventual
BPE target; they are **not measured model tokens or verified family counts**.
The incident estimate is most fragile because Scoutflo playbooks share a
template and some describe proactive monitoring. Exclude files outside the
incident-preparation/response scope; do not count boilerplate or mirrored
topics as independent families.

Before publishing any prepared mix, verify at least **1,625,000 unique
general**, **625,000 unique explanatory**, and **250,000 unique incident**
*train* target positions, with at least **100, 1,000, and 75 eligible
independent families**, respectively, and nonempty validation and test
families in each stratum. Freeze 80/10/10 assignments by eligible family count
before chunking, group work/page mirrors and incident topic siblings, and
check cross-split near duplicates. If any floor fails after admission,
deduplication and tokenizer measurement, stop and return a revised tranche
request; do not fill the deficit from held-out rows, another stratum or extra
exposures. The selected Card 05 input remains unavailable until all three
floors and evaluation criteria are met.

## One bounded preparation allocation

| Resource or output | Hard ceiling for this one preparation attempt |
| --- | ---: |
| Source response bodies, including at most one interrupted-transfer retry for each of 473 files | **1,691,154,896 bytes** |
| Metadata, redirect and error response bodies, shared across all sources/retries/resumes | **4,194,304 bytes** |
| Combined response bodies | **1,695,349,200 bytes** |
| Transfer count; per-file interrupted retry count | **946; 1** |
| Transport wall time from durable ledger initialization | **7,200 seconds** |
| HF decompressed JSONL read, separately per pinned shard | **4 GiB each; 8 GiB aggregate** |
| Retained selected-row snapshot, Gutenberg / Wikimedia | **256 MiB / 256 MiB** |
| Retained pinned Git files, including notices and README files | **4 MiB** |
| Combined retained acquisition content | **516 MiB (541,065,216 bytes)** |
| Added physical disk / inodes across acquisition and preparation | **8 GiB / 100,000** |
| Preparation after transport: admission, split/dedup, tokenizer, mixture and evaluation processing | **28,800 CPU wall seconds; 24 GiB peak RSS; 516 MiB tokenizer input** |
| Main tokenizer / prepared token bundle / evaluation artifacts within the disk ceiling | **32,768 entries / 5,000,000 target positions / 200 + 400 items** |
| Optimizer updates, GPU, cloud use and spend | **zero** |

The first source pass is 845,577,448 bytes; the doubled source ceiling reserves
one whole-size interrupted retry for *every* file, including both deleted HF
shards. Terminal checksum, metadata, rights, monitoring or numerical failures
stop the attempt; they are not retried as if successful. The durable native
ledger must be initialized before any acquisition metadata request and enforce
the shared response-body and transport wall ceilings across resumes.
Preflight actual free bytes/inodes and monitor the separate disk/RSS/CPU-prep
ceilings; stop before crossing any cap. Decompression is streamed and does not
authorize 8 GiB of retained text. The 8 GiB physical-disk cap allows transient
compressed input, bounded snapshots, verified releases, tokenizer and prepared
outputs; no training checkpoint is included.

Use native acquisition, snapshot verification, v2 record admission and release
for the two single-shard Common Pile sources. The 433-blob Scoutflo declaration
needs an offline/mock scale test of the existing bounded-Git adapter before
network use. No multi-shard admission or parallel ingestion system is needed
for this tranche. Use explicit `unit: document` split assignments from a
frozen work/page/topic-family inventory and test sibling consistency. The one
missing native operation is tokenizer-bound deterministic materialization of
the **realized 65/25/10 target-token mix** with per-document ordering,
target counts, repeat counts, shortfall and fail-closed admission/leakage
checks; `requested_mixture` currently records intent only. Implement and test
that narrow operation before prepared-input publication. Author and review the
frozen 200/400 evaluation suites from held-out or original fictional cases;
budget **up to 20 aggregate agent-assisted reviewer-hours** in addition to
machine time. This is not a commitment of the owner's time or permission to
hire anyone. Escalate decisions that specifically require the owner. No real-data
Card 05 forecast or training is authorized here; after this preparation,
measure exact supply and request the separate run allocation.
