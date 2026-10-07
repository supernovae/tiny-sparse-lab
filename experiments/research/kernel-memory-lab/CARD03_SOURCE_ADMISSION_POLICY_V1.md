# Card 03 source-level admission policy v1

Status: **ACCEPTED for current local research use** by the owner on 2026-10-07
in [KML-D14](DECISIONS.md#kml-d14--source-level-admission-policy-v1-accepted-2026-10-07).
This version updates the prospective admission rule in the accepted
[Card 03 contract](CARD03_DATA_CONTRACT.md). The conservative
[source preflight](CARD03_SOURCE_RIGHTS_PREFLIGHT.md) and
[pilot row screen](CARD03_PILOT_ROW_SCREEN.md) remain historical evidence for
the earlier quarantined pilot. Accepting this policy does not itself admit a
row, select a training release, approve acquisition, or authorize tokenizer
fitting or training.

## Decision rule

Approve a rights policy for a *pinned source component and intended use*, then
let a record inherit that policy when its immutable provenance places it inside
the reviewed scope and no exception applies. Independent permission evidence is
not required for every covered record. A dataset wrapper's general license label
alone is not a policy: the review must document the underlying source rules,
selection/filtering rules and known failure modes. Training eligibility and
redistribution remain separate decisions.

Each reviewed policy must bind a source ID, upstream revision, exact included
files or paths, use/jurisdiction, license or public-domain basis and its evidence
URLs, required attribution/notices, excluded content, automated exception rules,
reviewer/date, and policy digest. A changed upstream revision, use, or material
license term requires a new policy decision. Preserve the pilot's original
declarations, manifests and receipts; issue a new immutable admission decision
and release identity for any later selected content.

| Candidate policy scope | Inherited rule and required exception review |
| --- | --- |
| [Common Pile Project Gutenberg filtered](https://huggingface.co/datasets/common-pile/project_gutenberg_filtered), pinned `3cdf6879c807f4e4e063f2ceb23bc268d8c29ab7`, English works selected as public domain, internal US research | The component's selection and [Project Gutenberg's terms](https://www.gutenberg.org/policy/license.html) are the source-level evidence, not a new permission grant to every book. Record the book URL/ID, component and shard provenance. Flag permission-only or restricted-work notices, contradictory copyright/license metadata, known third-party inserts, unresolved jurisdiction or use, and material uncertainty about whether the work falls in the selected public-domain scope. Keep the applicable Gutenberg name/header/redistribution treatment in the policy; do not call public domain an SPDX license or infer redistribution rights from private training eligibility. The filtered component's license-error warning makes exception screening and a spot audit necessary. |
| [Common Pile Wikimedia filtered](https://huggingface.co/datasets/common-pile/wikimedia_filtered/blob/0641bb84bd9b7162bcddf8be7836822161a9a342/README.md), pinned `0641bb84bd9b7162bcddf8be7836822161a9a342`, English content namespace | Covered text inherits the source's CC BY-SA rule and [Wikimedia attribution route](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use) through a canonical page/history URL. Exclude non-content namespaces. Flag contrary license, imported/third-party/fair-use notices, absent or unusable page/history route, and material privacy or source-coverage problems. A one-entry `authors` field or absent revision query key is an issue to record, not a reason by itself to quarantine a record with a valid page/history route, pinned dump/shard identity and row digest. Retain attribution, license and change obligations for later uses. |
| [PagerDuty incident-response docs](https://github.com/PagerDuty/incident-response-docs/blob/464fc9d3e47e19e9d8da17cec1a41dc09624e95a/LICENSE), pinned `464fc9d3e47e19e9d8da17cec1a41dc09624e95a`, proposed `docs/**/*.md` | The repository's Apache-2.0 LICENSE can cover ordinary authored documentation with license/NOTICE and change obligations. Flag file-level SPDX or notice conflicts, third-party/vendor material, embedded media or copied text outside the repository grant, and sensitive content. The 36 pinned Markdown blobs are only a metadata inventory; no file content has been acquired or admitted. |

## Inherited provenance and exception disposition

For each covered record, retain `source_id`, policy ID/digest, upstream revision,
snapshot/manifest and shard digests, row index and row/content SHA-256, canonical
source locator, document/family identity, applicable license/rights decision and
notice/attribution route. A repository file instead binds commit, path and blob
digest. The pilot already retains source shard, row index and row digest; verify
the remaining mandatory binding fields in a new admission record rather than
altering those snapshots. Record extraction version and reviewer of the policy or
exception decision. The record need not have a separate rights reviewer when it
passes a reviewed policy and automated screen.

Run deterministic checks on **all** candidates for source/policy and digest
binding, scope, excluded namespace or path, contradictory license/copyright
metadata, exception notices, missing mandatory locator/attribution route, and
sensitive content. A material rights conflict or genuinely unknown policy
coverage is `QUARANTINE` with a reason; an out-of-scope record is `EXCLUDE`.
Optional descriptive gaps (for example a full author list, a revision query key,
or an edition detail when scope is otherwise established) are `ISSUE` entries
with field, owner, remedy and decision impact. They do not turn an otherwise
covered record into quarantine. A missing field becomes mandatory if it is the
only way to establish identity, coverage or an obligation for that record.

Before release, perform a reproducible, stratified spot audit: for this pilot,
at least three of the eight Gutenberg candidates and three of the twelve
Wikimedia content candidates, selected by a pre-recorded seed across source and
metadata-risk strata, plus **every** automatically flagged exception. Verify
the upstream source/notice and attribution route and compare with retained
metadata. This is a policy-application audit, not a demand for bespoke
permission. If an audit finds a material error, quarantine the affected
record/stratum, expand the audit to determine whether the policy still covers
the population, and do not release that stratum until resolved. Record sampled
IDs, findings and unresolved issues; do not replace negative samples.

## What this would change and what it would not

The current [row screen](CARD03_PILOT_ROW_SCREEN.md) reports 29 acquired rows:
nine `EXCLUDE_NONCONTENT`, twenty `REVIEW_REQUIRED`, zero admitted. A reviewed
source policy could remove the blanket per-work/per-page independent-clearance
requirement for the 8 Gutenberg and 12 Wikimedia content candidates. It could
also turn the Wikimedia one-entry-author and missing revision-query observations
into tracked issues when a page/history route and pinned record identity suffice.
It does **not** pre-clear those twenty rows; their exception screen, mandatory
provenance and spot audit have not been performed under this proposed rule. The
nine non-content rows remain excluded. PagerDuty remains unacquired and
unadmitted. Material conflicts and unknown coverage remain quarantined.

The accepted contract's per-item rights paragraph now permits policy
inheritance and nullable *optional* descriptive metadata with issue tracking.
This v1 policy supersedes the preflight's per-work/per-page stop wording **only
for future admission decisions**; do not rewrite that preflight or the historical
0-admitted pilot screen. Preserve the split, mixture, train-only tokenizer and
evaluation requirements. Keep the original source declarations
`review_required`; admission requires a separately verified manifest and native
record-level enforcement.

## Release gate under policy v1

1. Review and version a source policy for each included source/pin/use; verify
   the immutable pilot snapshots and new admission inputs against their receipts.
2. Bind every candidate to mandatory provenance and run the complete exception
   screen; exclude or quarantine all out-of-scope, conflicting or unknown records.
   Log optional gaps and their resolution priority separately.
3. Pass the predeclared spot audit, investigate flags and discrepancies, and
   freeze an admission manifest with covered/excluded/quarantined counts, rights
   obligations, reviewer decisions and digests. Zero unresolved material
   conflicts may enter a selected release.
4. Create a new rights-admitted release using only covered rows, with its own
   declaration and verification. Existing quarantined receipts stay unchanged.
   Distribution of source text needs a separate redistribution decision and
   obligation check.
5. Then satisfy the rest of Card 03: domain-source admission, family splits and
   leakage review, realized mixture, train-only main tokenizer, and frozen
   language/evidence suites. A real-data timing/forecast and Gate 0 review are
   separate prerequisites for Card 05. This policy alone grants no Card 05 run.

The current native rights resolver inherits source rights at **file** scope and
can stop conflicting files, while the pilot's selected rows live inside large
shards. Before selecting a release, verify that the supported native path can
apply record-level exceptions and bind the reviewed policy/manifest; otherwise
make the smallest typed native extension and test fail-closed release behavior.
Do not mark the source declarations eligible merely to bypass that check.
