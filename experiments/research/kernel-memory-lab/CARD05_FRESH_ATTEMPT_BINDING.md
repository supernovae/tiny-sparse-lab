# Card 05 fresh-attempt execution binding

**Status:** owner-authorized, conditional on the tested adaptation, clean pushed checkout and original preflight. This is a new attempt under the unchanged [full-tranche contract](CARD05_FULL_TRANCHE_PROPOSAL.md), not a revision of that contract or a resume of C05-F1.

- New unused run ID: `kml-card05-full-tranche-v2` under `/srv/sparselab/state/experiments/kernel-memory-lab/card04-synthetic/runs/`.
- New unused attempt root: `/srv/sparselab/state/experiments/kernel-memory-lab/card05-full-tranche-v2/`.
- Scientific input: unchanged `card05-full-tranche-v1.yaml`, semantic SHA-256 `4f22d18d80ba308b7358fc3d7375da5ecf4470f0d3bbddbb8cbc7e2e3d9e6a53`, seed 17, accepted local Card 03 release, tokenizer, prepared 65/25/10 mixture and frozen 200-item language suite. All original checkpoint, selection, decoding, scoring, eligibility, no-retry and stop rules apply.
- One new common-root live-sampler storage baseline and independent ledger: 10,800 seconds, 4,883 reserved optimizer updates, exactly 5,000,000 nonmasked targets for success. Stage/train/selection at most 3,000 seconds; evaluation at most 7,200 seconds. Whole-device VRAM 20 GiB; process-tree RSS 24 GiB; added common-root disk 64 GiB and 2,000 inodes; evaluation output 64 MiB and 25,600 new tokens maximum. Zero acquisition, cloud spend, resume, extra attempt or Card 06 progression.
- The failed `kml-card05-full-tranche-v1` run and `card05-full-tranche-v1` attempt root, including the fully charged ledger, baseline and partial files, remain immutable historical evidence. Their names are not rebound.

The launcher, validator, checkpoint selector and evaluator must all use the new run ID before runtime. The original config and frozen scientific declarations remain unchanged. Any adaptation, offline test or preflight failure stops before ledger creation; any later declared failure stops this one attempt with evidence preserved.
