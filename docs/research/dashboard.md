# Read-only Learn and Research pages

Start the dashboard with an optional report root:

```sh
sparselab dashboard --runs-dir runs --reports-dir artifacts/research-reports
```

Learn and Research are browseable before a run database exists. Learn shows the packaged lesson cards, scale/data choices, shape/source walkthroughs, and copyable standalone commands; its controls do not write a scaffold or submit training. Research shows the declarative question, hypothesis, failure interpretation, controls, varied fields, cards, resource class, recommended scale, and HTTPS paper links.

The Research page discovers at most 200 **direct** children of `--reports-dir`. A child must be a non-symlink content-addressed directory with a safe `manifest.json`, and the shared bundle loader validates it before rendering. Invalid manifests, hash mismatches, unsafe children, and malformed bundles appear as rejected rows and their charts are not rendered. The page does not open the writer-side experiment store for report browsing, execute bundled Markdown/HTML, or dereference report-controlled paths.

A valid bundle presents raw outcomes, endpoint status, observed pair deltas, calculated architecture/cache quantities, available local telemetry, costs/unavailable values, limitations, and missing/rejected evidence. An unlinked historical bundle is labeled as evidence without recorded hypothesis metadata. “Observed delta” is not a winner label: partial/inconclusive data and unavailable values remain explicit.
