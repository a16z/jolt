# Prover benchmark history

The dashboard lives at `dev/prover-history/` on the existing `gh-pages` branch.
It appends each result and its raw JSON without changing `dev/bench/` or other
site files. Main, nightly and opted-in PR comparisons have separate views.
Benchmark execution requires the restricted runner and producer setup in
`scripts/bench/README.md`; publication needs neither a new repository variable
nor a Pages-source change.

## Read one benchmark point

| View | Source and build route |
| --- | --- |
| Standard main | Every main push's new first-parent commits; hosted build, target run-only |
| Nightly | Wider suite on main at 07:00 UTC; hosted build, target run-only |
| Pull requests | Same-repository PRs with a maintainer/admin-applied `benchmark` label; local build and proof |
| Manual previews | Explicitly labelled imported measurements, never authenticated main history |

The PR label persists across pushes. The trusted producer checks the actual label
applier, current label and merge parents; a write-only collaborator or fork is
not admitted. The target rechecks the label and PR head after queue/lock waits.
The dashboard shows the PR link, submitted head, base and measured merge, source
run/attempt, admission event, and raw result. Select a PR, workload/input/core
count, base or head leg, host/build configuration, and metric. PR legs do not enter
the main trend. Automatic calibration remains unavailable: these are observed
measurements, not statistical significance or a future false-positive guarantee.

A point is the median of complete measured repetitions. The table shows its
minimum–maximum range; raw JSON retains warmups. Failed cases/builds, incomplete
repetitions and missing metrics have no plotted value and break the line. The
initial view loads the newest 30 records; **Load older runs** retains and reads
earlier records. Each rerun attempt remains distinct.

Inputs, PCS, core count (1/8/24), scale, tier and leg separate points. Host,
CPU affinity, compiler, target, flags, profile and harness separate configurations.
Source revisions, binary/guest/lockfile/schedule hashes remain per-point metadata.
Only the changing `swap_free_bytes` observation is omitted from configuration
identity; capacity and other settings remain. **Build and source** retains the
original full build manifest. Its construction time is not the current cost of
reusing a bundle; current preparation is `wall_time.build_s`.

`e2e_s` covers prebuilt guest loading, tracing, preprocessing, setup, proving and
serialization, excluding compilation and verification. `prove_s` covers only the
prove call. `process_s` includes launch, both verifications, logging and reaped
exit; peak RSS spans that process. The metric selector states each boundary.
Local PR build time and total time are separate from proof metrics. Historical
artifact-only timings do not predict changed-source PR compilation time.

## Trusted collection and append-only publication

`prover-history.yml` reacts to completed **End-to-end prover benchmarks** runs,
including failed and cancelled ones, or a manual dispatch on `main`. It ignores
skipped runs, which are PR pushes and label events that were never admitted, and
the collector refuses them. Its executable collector and UI assets are checked
out from `main`, never from the source PR or an artifact.
The workflow can only publish the configured legacy site: `gh-pages` at `/`.
It does not edit that setting or a deployment environment.

The collector verifies repository, producer workflow/name/path, event, attempt,
artifact ID/name and source revisions. Main/nightly sources must be main-branch
push, schedule or dispatch runs. PR sources must be trusted `pull_request_target`
runs and carry a matching `jolt-e2e-plan-RUN_ID-ATTEMPT` artifact whose PR number
is in the run's `pull_requests`. GitHub lists only open PRs there, so a run whose
PR closed before collection is rejected. The plan binds label admission,
base/head/merge/harness, native build inputs and merge parents; GitHub's run
head is the submitted PR head, not the trusted harness revision.
Current PR state can change after a valid run; that does not rewrite its history.
Only JSON is read. Downloaded scripts, HTML and binaries are never executed.

For multi-commit pushes, `index.json` declares the oldest-first documents. The
tip uses `benchmark-results.json`; earlier commits use
`commits/FULL_SHA/benchmark-results.json`. GitHub commit ancestry checks the chain
and order, including absent document gaps. The producer-recorded `push_before`
is not independent proof of the original event boundary. The original index is
archived with the results. The page shows received records, not proof that no
provider job was missed.

Each immutable key includes measured revision, source run ID and attempt.
Identical replay is a no-op; conflicting bytes are refused. Expired upstream
artifacts cannot erase archived results. Missing, rejected, unsupported and
failed runs remain visible rather than becoming zero timings. JSON is limited
to 32 MiB and result archives to 128 MiB. No automatic retention deletes history.

The publisher serializes with `concurrency.queue: max` (100 pending jobs), then
retries non-force pushes against the latest `gh-pages` tree. Only
`dev/prover-history/` is staged. This preserves concurrent updates by the legacy
benchmark writer. Queue overflow or provider limits still require an explicit
replay; no custom scheduler is introduced.
[GitHub concurrency limits](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)

## Pages deployment without changing its source

A `GITHUB_TOKEN` push alone is not the deployment contract: GitHub documents that
such pushes do not trigger Pages builds. The existing legacy benchmark job has
nevertheless produced built Pages receipts for its pushed commits; those
observations do not establish a supported trigger guarantee.
[Branch publishing](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)

After appending to `gh-pages`, this publisher explicitly requests a Pages build
with `POST /repos/a16z/jolt/pages/builds`. That endpoint accepts GitHub App
installation tokens with `pages: write`; `GITHUB_TOKEN` is such an installation
token. The workflow also needs `contents: write` for the branch and `actions: read`
for producer artifacts. It uses no personal token, OIDC deployment, new enable
variable or Pages-source switch. The request's acceptance is not proof the later
Pages build completed: check that build and both URLs after the first merged run.
[Pages build API](https://docs.github.com/en/rest/pages/pages#request-a-github-pages-build),
[GITHUB_TOKEN type](https://docs.github.com/en/actions/concepts/security/github_token)

Merge the reviewed producer and publisher, preserving the existing branch-based
site and allowed actions. Then a producer completion appends and requests a
build automatically. A user with workflow-dispatch permission can refresh assets
or replay an existing producer attempt before its artifact expires:

```sh
# Refresh history assets and request a build of the existing site.
gh workflow run prover-history.yml --repo a16z/jolt --ref main

# Append a specific existing producer attempt, preserving its original identity.
gh workflow run prover-history.yml --repo a16z/jolt --ref main \
  -f run_id="$RUN_ID" -f run_attempt="$RUN_ATTEMPT"
```

A valid recorded attempt is immutable. Repair a rejected producer result with a
new attempt, not relabelled bytes. If the Pages build API is denied by repository
policy, keep the failed request visible; do not substitute a token or change
settings. Pages size, build-frequency and build-duration limits still apply.
[Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)

## Local preview and focused checks

Preview preserves manual provenance; it never contacts GitHub or publishes.
Use a task-owned directory and a label describing the actual measurement:

```sh
uv run --no-project python scripts/ci/benchmark_history.py preview \
  --input "$RESULTS_JSON" --site "$TMPDIR/jolt-history-preview" \
  --label "Manual hosted-artifact comparison; not an automated PR run"
uv run --no-project python -m http.server 8000 --bind 127.0.0.1 \
  --directory "$TMPDIR/jolt-history-preview"
```

Open `http://127.0.0.1:8000/dev/prover-history/`; `file:` URLs are unsupported.
The same renderer handles real campaign previews and explicit synthetic route
fixtures. Never relabel a manual sample as authenticated PR/main/nightly data.

```sh
uv run --no-project python -m unittest scripts.tests.test_benchmark_history -v
uv run --no-project --with playwright==1.61.0 playwright install chromium
uv run --no-project --with playwright==1.61.0 python -m unittest scripts.tests.test_benchmark_history_browser -v
```

Browser checks use bundled headless Chromium and the page's unchanged content
policy. They exercise PR/main separation, filters, details, raw downloads,
failure gaps, hostile strings and mobile/desktop layout. A live dashboard still
needs merged workflows, an enabled restricted runner and actual received runs;
source and preview checks do not establish live execution or deployment.
