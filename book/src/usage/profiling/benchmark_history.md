# Benchmark history keeps each run and its raw measurements

The dashboard publishes CPU measurements at `dev/prover-history/` on the existing
`gh-pages` branch. Standard main history and nightly history have separate views.
The legacy `dev/bench/` files are neither converted nor replaced. The entire
publication job is skipped unless `JOLT_BENCH_PAGES_ENABLED` is `true`. An
administrator configures Pages before enabling that variable; merging this code
does not change repository settings or activate publication.

## Compare one input and configuration at a time

The producer owns the version-one `benchmark-results.json` contract. The publisher
archives that JSON without rewriting it. The browser derives a median and a
minimum–maximum range for each complete case using only its measured repetitions.
Warmups remain in the download. A failed case, failed build, missing metric, or
incomplete repetition set has no plotted value and breaks the line.

The three measurement tiers have different purposes.

| Tier | Purpose | History view |
| --- | --- | --- |
| PR | Fast comparison subset | Kept by the producer; not published as main history |
| Main | Standard suite on each main push, including its new first-parent commits | Standard main |
| Nightly | Wider suite with larger inputs and more repetitions | Nightly |

Selectors separate workload, polynomial commitment scheme, CPU backend, fixed
input, input-size target, and exactly 1, 8, or 24 physical cores. Host settings,
CPU affinity, compiler version, target, flags, profile, build host, and measurement
harness settings separate configurations. Source revisions, lockfile hashes,
Akita schedule hashes, binary hashes, and guest hashes remain per-point provenance:
source and dependency changes do not by themselves start a new configuration.
The runtime reading `settings.swap_free_bytes` is excluded from both measurement-
and build-host configuration keys. Swap capacity and all other host settings stay
in those keys. Both host observations remain in each run's raw JSON; the original
build host also remains in **Build and source**.

**Build and source** shows the original construction manifests. Their `build_s`
and timestamps are provenance, not current preparation time when a bundle is
reused. Current preparation, when recorded, is in the raw result's
`wall_time.build_s`; it includes cache lookup, fetch, validation and copying, or a
new build. A prebuilt-only run can leave that field null.

The primary `e2e_s` timer includes prebuilt guest loading, both sizing and proof
traces, preprocessing, commitment setup, proving, and serialization. It excludes
compilation and verification. `prove_s` covers only the prove call. `process_s`
includes process startup, both verifications, logging, and teardown; peak resident
memory spans that same process. The metric selector states these boundaries.
Verification time, serialized proof bytes, and actual trace rows are available
without pooling different inputs or configurations.

Each rerun attempt gets its own record. The page initially loads the newest 30
records in the selected view; **Load older runs** reads earlier records without
deleting any history. A manual preview remains in **Manual previews**, with its
original PR/main/nightly tier and comparison/null/settings/history purpose.
Pilot data does not become an authenticated main or nightly trend.

## Publication runs only trusted code

`.github/workflows/prover-history.yml` runs on completion of **End-to-end prover
benchmarks** (`.github/workflows/e2e-bench.yml`) and **CI Benchmarks**
(`.github/workflows/ci-bench.yml`). It also supports an explicit manual dispatch.
Both source workflows must retain those names and paths. GitHub requires the
receiving `workflow_run` file on the default branch and emits completion events
regardless of the source conclusion. [GitHub workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#workflow_run)

`JOLT_BENCH_PAGES_ENABLED` gates the entire publication job, before the
`github-pages` environment is evaluated. Unset or `false` skips producer, legacy,
and manual publication. `JOLT_BENCH_HISTORY_ENABLED` controls only new history
processing. After Pages activation, legacy **CI Benchmarks** completions deploy
the existing site with
that variable either `true` or `false`. They invoke neither the history collector
nor the append parser, so a rejected artifact or invalid history index cannot
block legacy publication. PR-originated completions remain excluded.

The write-token job checks out executable code and dashboard assets from `main`,
not the producer revision or a pull request. The history path accepts only this
repository's completed main-branch push, schedule, or manual producer runs. It
checks the run ID, attempt, workflow, repository, branch, artifact identity, and recorded source
revisions against GitHub metadata. The exact artifact name is
`jolt-e2e-RUN_ID-RUN_ATTEMPT`. Only JSON members are read; downloaded scripts,
HTML, binaries, and manifest file paths are never executed or dereferenced.
Artifact reads require `actions: read`. [GitHub artifact API](https://docs.github.com/en/rest/actions/artifacts#list-workflow-run-artifacts)

For a multi-commit main push, the producer's root `index.json` declares the
oldest-first documents. The tip uses `benchmark-results.json`; other entries must
use `commits/FULL_SHA/benchmark-results.json`. GitHub's commit graph verifies
first-parent ancestry and order, including gaps for absent documents. The
producer's `push_before` is retained as a recorded lower bound, **not** an
independently verified original push-event field. With a null lower bound, the
oldest declared revision sets the checked range; earlier coverage is unknown.
The original push index is downloadable beside each result.

The publisher writes only `dev/prover-history/` in a fresh Pages checkout. Its
immutable key includes measured revision, source run ID, and rerun attempt.
Replaying identical data adds nothing. Conflicting bytes are refused; an expired
upstream artifact cannot erase archived data. Missing, expired, disabled, and
rejected results retain explicit records instead of zero-valued timings.

The publication job uses `concurrency.queue: max` and does not cancel an active
publisher. GitHub allows at most 100 pending jobs in that group; further jobs can
be canceled. Ordering is by when jobs begin waiting, not original dispatch time.
The publisher also retries a rejected non-force Git push against the newest
`gh-pages` tree, preserving updates made by the independent legacy writer.
[GitHub concurrency limits](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)

## An administrator enables the existing Pages site

This setup preserves the old site while changing how it is deployed. The workflow
uploads the whole `gh-pages` site with `actions/upload-pages-artifact`
and deploys it with `actions/deploy-pages`. Completion of the legacy benchmark
workflow also triggers this full-site deployment. There is no reliance on a
`gh-pages` push event, and no second workflow copy on the data branch.
[GitHub publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site),
[custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)

1. Merge the reviewed producer and publisher changes into `main`. Enable Actions
   for **End-to-end prover benchmarks**, **CI Benchmarks**, and **Publish prover
   benchmark history**. The producer's runner setup and execution gate are
   separate prerequisites; the publisher itself uses a hosted Ubuntu runner.
2. Preserve the existing `gh-pages` branch and its complete site tree. In the
   repository's **Settings → Pages → Build and deployment → Source**, select
   **GitHub Actions**, following GitHub's publishing-source instructions above.
   Do not replace the branch or delete `dev/bench/`.
3. Ensure the `github-pages` deployment environment permits `main`. Retain the
   repository's existing deployment protections. The allowed-action policy must
   permit the pinned `actions/checkout`, `astral-sh/setup-uv`,
   `actions/upload-pages-artifact`, and `actions/deploy-pages` actions.
4. After configuring Pages and its environment, set the repository Actions
   variable `JOLT_BENCH_PAGES_ENABLED` to `true` to enable publication. The job
   still requires the Pages API to report `build_type: workflow`; that check
   fails if activation is incomplete. The workflow does not enable or
   reconfigure Pages itself. Legacy deployment and no-run-ID manual redeployment
   now work without history processing.
5. Separately set `JOLT_BENCH_HISTORY_ENABLED` to `true` to enable history
   collection and updates. Dispatch the publisher on `main` without a source run
   ID for the initial full-site refresh. Confirm both `dev/bench/` and
   `dev/prover-history/` after deployment. Normal producer completions then
   archive and deploy new results.

This implementation makes a site-wide tradeoff: one workflow owns
deployment of both the legacy site and the new history. Turning off history
processing leaves legacy deployment active. To roll back, first disable Pages
activation, then restore branch-based Pages publication for `gh-pages` at `/`.
Live deployment remains unverified until an administrator activates and checks
this route.

The publisher needs only the following job permissions. It does not need issue,
pull-request, administration, or `deployments: write` permissions. The PR test job
has only `contents: read` and never runs with the publisher's credentials.

| Permission | Use |
| --- | --- |
| `contents: write` | Read source and append to the existing `gh-pages` branch |
| `actions: read` | Read source-run metadata and download its artifact |
| `pages: write` | Deploy the static Pages artifact |
| `id-token: write` | Authenticate that Pages deployment through GitHub's identity token |

GitHub documents the last two permissions and the deployment environment as
requirements of `actions/deploy-pages`. Its uploader excludes repository metadata
such as `.git` and `.github`; the branch's legacy files remain untouched.
[Pages deployment requirements](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages),
[uploader implementation](https://github.com/actions/upload-pages-artifact/blob/7b1f4a764d45c48632c6b24a0339c27f5614fb0b/action.yml)

## Replay publication before source artifacts expire

These are administrator operations, not commands run by a preview. Use a real
producer run ID and its specific attempt when backfilling a missed publication.

```sh
# Refresh the existing full site, including legacy assets.
gh workflow run prover-history.yml --repo a16z/jolt --ref main

# Archive one existing producer attempt; set these variables to its numeric IDs.
gh workflow run prover-history.yml --repo a16z/jolt --ref main \
  -f run_id="$RUN_ID" -f run_attempt="$RUN_ATTEMPT"
```

All dispatches require `JOLT_BENCH_PAGES_ENABLED=true`. A dispatch with a run ID
also requires `JOLT_BENCH_HISTORY_ENABLED=true`. Without a run ID, it redeploys
the existing whole site even when history is disabled; when history is enabled,
it also initializes or refreshes the history assets.

Queue overflow, a failed publication job, a disabled publisher, or an expired
source artifact can leave gaps. The page does not claim that every merge ran.
GitHub's default artifact retention is 90 days, but repository and producer
settings can shorten it. Once a receipt has archived valid JSON, source-artifact
expiry does not remove that copy. A recorded invalid or missing attempt is not
overwritten by a later replay; repair the producer and create a new run attempt.
[Artifact access and retention](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts)

There is no rolling deletion. JSON documents are limited to 32 MiB and downloaded
producer archives to 128 MiB; exceeding either produces a rejected receipt.
GitHub currently limits a published Pages site to 1 GB and a deployment to 10
minutes. Custom Actions publication is exempt from the soft ten-builds-per-hour
limit, not from all Pages limits. If the archive reaches a provider limit, arrange
an explicit retention or storage decision instead of deleting old points.
[GitHub Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)

Setting `JOLT_BENCH_HISTORY_ENABLED` to `false` stops new history collection and
updates. While Pages activation remains enabled, legacy completions and
no-run-ID manual refreshes continue to deploy the existing site without parsing
history JSON. To restore the previous publication mode, first set
`JOLT_BENCH_PAGES_ENABLED` to `false` or remove it. Then an administrator selects
**Deploy from a branch**, with `gh-pages` and `/`, in the publishing-source
settings and verifies the legacy site. The legacy writer can still update
`gh-pages`; the branch and its existing files are retained.

## Preview and test without publishing

Use a task-owned directory. This command reads one producer JSON document and
copies static assets; it never contacts GitHub, runs benchmarks, or pushes a
branch. Keep the label faithful to the measurement purpose.

```sh
uv run --no-project python scripts/ci/benchmark_history.py preview \
  --input "$RESULTS_JSON" --site "$TMPDIR/jolt-history-preview" \
  --label "Manual pilot; partial suite"
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 \
  --directory "$TMPDIR/jolt-history-preview"
```

Open `http://127.0.0.1:8000/dev/prover-history/`. This is unauthenticated manual
preview evidence, not proof that GitHub publication is enabled. The HTML page
loads data over HTTP; opening it with a `file:` URL is not supported.

Focused checks use only synthetic fixtures, never checked-in example trends.

```sh
uv run --no-project python -m unittest scripts.tests.test_benchmark_history -v
uv run --no-project --with playwright==1.61.0 playwright install webkit
uv run --no-project --with playwright==1.61.0 python -m unittest scripts.tests.test_benchmark_history_browser -v
```

Browser checks use the bundled headless WebKit. Playwright is pinned because
1.64.1's WebKit 27.2 reports false CSP errors for native selects; the page keeps
its strict content policy and does not filter those errors. Tests check actual
controls, raw downloads, failure gaps, hostile strings, and mobile/desktop layout.
[Upstream reproduction](https://github.com/microsoft/playwright/issues/43617)

Collect screenshots separately from functional console assertions. Playwright's
WebKit screenshot preparation injects an inline `body {}` stylesheet, which this
page's CSP rejects even on a script-free control page. Preserve those capture
events as tool telemetry; do not disable the page's CSP or ignore application
errors to obtain screenshots.
