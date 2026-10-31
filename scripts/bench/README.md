# Reproducible CPU proof benchmarks

These scripts reuse `jolt-prover profile --backend optimized`. They build real
guest ELFs before measurement, run fresh prover processes, verify every successful
proof, and retain raw samples. There is no reference-backend timing lane.

**The sizes and repetition counts are proposals, not a calibrated performance
gate.** The PR budget is less than 600 seconds, with a 300–420 second aim, for the
actual workflow path on a real changed-source PR update. It includes every build,
preparation, transfer, and result-processing interval, measured from the successful
request's completion through the result artifact and, if enabled, the trusted
comment job. A target-host `suite` run measures the local
build/preparation-through-results component; it does not
include workflow transfers or hosted publication. Runtime-only or prebuilt-only
runs cannot pass this gate, and a preparation interval already above 600 seconds
is a miss before downstream work is added.

## Workloads and tiers

`suites.json` is the source of the case matrix. Standard-main and nightly run both
Dory and Akita at exactly 1, 8, or 24 physical cores; PRs run only Akita.
The ordinary instruction profile and clear proofs are
used; no protocol parameter is adjusted for the benchmark.

| Workload | Fixed operations at 1 / 8 / 24 cores in `main` | Input |
|---|---:|---|
| Fibonacci | 16000 / 64000 / 250000 | Postcard-encoded iteration count |
| SHA2 chain | 64 / 256 / 1024 | `[5; 32]`, then the count |
| SHA3/Keccak chain | 48 / 192 / 768 | `[5; 32]`, then the count |
| BTreeMap | 128 / 512 / 2048 | Insertion count; guest-owned deterministic keys |
| secp256k1 ECDSA batch | 1 / 4 / 16 | One valid signature/public-key vector, verified repeatedly |

`examples/secp256k1-batch/guest` calls the real `ecdsa_verify` precompile. Its
signature vector comes from `examples/secp256k1-ecdsa-verify/src/main.rs`; it does
not replace ECDSA with a synthetic loop. Input bytes and guest ELF hashes are
recorded for each build. `seed: null` means the input generator uses no PRNG.

| Tier | Proposed cases | Warmups / measured repetitions per leg |
|---|---|---:|
| `pr` | Fibonacci at 1 core, SHA2 at 8, BTreeMap at 24; Akita only: three points | 1 / 3 |
| `main` | All five workloads, all three core counts, both PCS: 30 points | 1 / 3 |
| `nightly` | Two larger inputs for each main point: 60 points | 1 / 5 |

The fast PR subset has no Dory performance signal. It launches 24 fresh proof
processes, including warmups and both legs; this count does not establish a wall
time or build saving. The full changed-source workflow, including preparation,
transfers and result processing, must still be measured below 600 seconds.

Main trace ceilings are `2^18`, `2^20`, and `2^22` at 1, 8, and 24 cores.
Nightly uses twice and four times each main operation count and raises the trace
ceiling by one and two bits respectively. These are ceilings, not invented row
measurements. An input that exceeds its ceiling fails visibly.

Main and nightly each retain six unsupported `ethereum-block` rows. The tracked
`examples/sig-recovery/guest/src/lib.rs::verify_txs` verifies signed transactions;
it does not execute an Ethereum block. Its host input generator uses
`rand::rng()`. The tracked guest inventory contains no full Ethereum execution
guest, so this suite makes no block-execution claim. A future transaction-shaped
case must pin serialized signed transactions instead of regenerating them.
`akita,zk` is rejected by the compile-time feature gate in
`crates/jolt-prover/src/lib.rs`; this suite does not claim that combination.

## Measurement boundaries

```text
suite wall: [ pinned build + guest preparation ][ run preparation + all processes + results ]
process:                                          [ launch ... verification ... exit ]
e2e:                                                [ load, trace, preprocess, setup, prove, serialize ]
verify:                                                                                                [ parallel ][ single ]
```

`e2e_s` starts before loading the prebuilt ELF and input bytes. It includes the
profile harness's first sizing trace, program decoding/expansion and
preprocessing, its second modular trace, witness construction, PCS setup,
proving, and proof serialization. It stops before verification. Both existing
verification calls must succeed before the JSON metric file is written.
`prove_s` and `setup_s` surround the existing prove and setup calls, respectively;
neither substitutes for `e2e_s`.

`verify_s` is the existing parallel verifier; `verify_single_s` is its separate
single-threaded run. Both verify the in-memory proof; there is no deserialize
step. `process_s` is parent-observed launch through reaped exit,
including both verifications, output, and teardown. `peak_rss_bytes` is Linux
`wait4` maximum RSS for that process, converted from KiB to bytes; it is not a
heap allocation estimate or a host-wide memory reading. `proof_bytes` is the
serialized proof length. `trace_rows` is the modular trace's `proof_len()` before
padding, not a raw RISC-V instruction count. `padded_trace_rows` is the actual
`ProverConfig.trace_length`, including the required final no-op and minimum size.

All times are full-resolution numeric seconds. Build time is separate from the
proof metrics. `suite` measures one continuous build/preparation-through-results
interval on its executing host. `run` records `prebuilt-execution-only`, with
`build_s: null`; it never adds historical build timings to claim a fresh total.
Individual build manifests retain their original construction intervals and host
identity, including when a completed bundle is reused. Each new `preparation.json`
records the current validation/copy or build interval. The suite's `build_s` is
the freshly observed preparation interval, never a sum of old manifest durations.
The workflow's transfer and trusted result-processing intervals must be measured
separately before claiming the PR total. Cargo target existence or a configured
compiler cache does not establish cold or cached state; hit state stays unknown
without evidence for the actual invocation.

Each warmup and repetition launches a new process. The paired order is base/head
for repetition zero, head/base for repetition one, and then alternates. The runner
selects one allowed hardware thread per `(socket, core)` from `lscpu` topology and
applies both `taskset` and `RAYON_NUM_THREADS`. It never calls 24 SMT threads 24
physical cores. A process/file lock serializes participating suite commands.

The Dory URS cache begins empty in each result directory and is shared by the two
legs. Setup may use system entropy; retained URS files are hash-identified.
Warmup means an excluded first invocation, not a proven Cargo or URS cache hit.
Akita's external schedule files are copied into the bundle and hash-checked.
No host governor, SMT, boost, swap, RAID, or other setting is changed.
`host.microcode` comes from `/proc/cpuinfo`. `host.control_plane` records the
readable `/opt/pika/bin/pika-cli` and `pika-shim` SHA256s and the shim's `--version`
output. Unavailable observations are null, never an assumed match. These are
runtime observations; they are not part of native bundle cache identity.

## Manual commands

Run on Linux with Python 3.11+ through `uv`, Git, `taskset`, `lscpu`, the pinned
Rust toolchain, its `rust-src` component and configured RISC-V targets. The five
prepared guests use `riscv64imac-unknown-none-elf`. The build produces its own
`jolt` CLI; a stale globally installed CLI is not used.

On Ubuntu, have the administrator provide `gcc`, `g++`, `make`, `dpkg-dev`,
`libcrypt-dev`, `cmake`, `pkg-config`, and `libssl-dev` through the normal
package-management process. Keep `libc6-dev` for libc headers/startup objects;
`libssl-dev` supplies OpenSSL headers. The native tools and headers follow
[setup_machine.sh](../setup_machine.sh#L11-L13) and the
[devcontainer Dockerfile](../../.devcontainer/Dockerfile#L7).

Use the direct packages on Ubuntu 26.04: `build-essential` pulls
`coreutils-from-uutils`, which would replace this runner's `coreutils-from-gnu`.
The administrator transaction must use `--no-remove` and
`--no-install-recommends` to preserve the installed coreutils provider and avoid
bringing the metapackage back through recommendations. Existing `ld`/`ar` alone
do not satisfy the native compiler requirement. Do not run the broad setup
script for this repair: it also upgrades the OS and changes firewall/services.

After provisioning, check availability as `runner` without compiling anything:

```bash
set -eu
for jolt_tool in cc c++ make cmake pkg-config ld ar; do
  command -v "$jolt_tool"
  "$jolt_tool" --version
done
pkg-config --modversion openssl
test -r /usr/include/stdio.h
test -r /usr/include/openssl/ssl.h
jolt_crt=$(cc -print-file-name=Scrt1.o)
test -r "$jolt_crt"
dpkg-query -W -f='${binary:Package}\t${Version}\t${db:Status-Abbrev}\n' \
  gcc g++ make dpkg-dev libcrypt-dev cmake pkg-config libssl-dev libc6-dev
```

Build state lives under
`${XDG_CACHE_HOME:-$HOME/.cache}/jolt-bench/`, outside the action workspace.
Its `builds/<cpu>/<profile>/source` is an independent Git clone with its own
objects, fetched from the current trusted checkout; Cargo targets stay inside
that clone. The driver forwards no `CARGO_TARGET_DIR` override. Normal Cargo job settings are honored;
the manifest records `CARGO_BUILD_JOBS` if supplied, without inferring concurrency
when it is absent.

Preparation builds the real `jolt` guest-builder CLI from the selected source
under the existing `ci` Cargo profile, without fat LTO or debug information.
The benchmark executables still use the requested prover profile; the workflow
uses `release` with fat LTO. Guest builds still receive `--release`. The manifest
records `guest_builder_profile: "ci"` separately from the measured prover's
`profile` and retains the CLI's hash. PR preparation
builds `jolt-prover-akita`; main and nightly build both variants. Guest preparation
uses an executable present in that bundle. The CLI still depends on the Dory host
SDK (`Cargo.toml` and `jolt-sdk/src/host_utils.rs`); omitting the Dory benchmark
executable does not eliminate Dory dependency compilation. No CLI dependency,
prover optimization or guest optimization is changed by the builder-profile split.

The harness revision must be committed, and `run.py` must be that revision's
file. Both source revisions receive the same benchmark-only profile harness and
direct guest dependencies. Existing locked package identities may not change.
Build manifests distinguish source revision, harness revision, original lockfile,
effective lockfile, and all overlaid files. A same-source null run uses identical
bundles, not differently instrumented main-vs-main binaries.

Completed bundles persist in the same locked store. Reuse requires an exact
source and harness revision, Rust/Cargo versions, target CPU/triple/features,
flags, prover and guest-builder profiles, lockfiles, harness files, runner and
suite-definition match.
Every required PCS and guest/input must exist. All recorded binary, guest, input,
runner, configuration and schedule hashes are checked before copying. A main-tier bundle
does not qualify merely because its source SHA matches: it may lack a PR input.
Different PCS/input sets can coexist; a valid main-tier superset may serve a PR
base. An Akita-only bundle cannot serve a request that also requires Dory.
An invalid or incomplete entry is recorded as a cache miss and rebuilt.

Reuse preserves the construction manifest byte-for-byte. The fresh preparation
receipt identifies the copied manifest, required inputs and current elapsed
interval; `construction.log` retains the original build log. Git synchronization,
lock resolution, validation, copying and pruning still count toward preparation.
The completed-bundle store retains at most eight entries. Successful reuse updates
an entry's recency; each completed store prunes the oldest entries and removes
empty identity directories. Pruning leaves the source clone, Cargo target, result
outputs and lock file intact. Other cache maintenance is administrator-owned and
must hold the same `host.lock`; do not remove the lock file during maintenance.

`build.log` records each command's argument vector, elapsed seconds, exit code
and `complete` or `failed` outcome in a flushed JSON line when the attempt finishes.
The initial argument-vector record is flushed before execution. A start without
an outcome record is interrupted or unknown, never a completed duration.
This covers the CLI build, prover builds and guest preparation, including failures;
a launch failure has no exit code. These are command durations, not separate
compiler/linker timings. Reuse keeps the original durations in `construction.log`;
the current commands remain in `build.log`. Validation, copying and all other
preparation still count in the fresh preparation and full wall-time receipts.

Build a transportable bundle (no proving occurs):

```bash
uv run --no-project python scripts/bench/run.py build \
  --revision "$REVISION" --harness-revision "$HARNESS" \
  --target-cpu "$TARGET_CPU" --out "$BENCH_OUT/bundle"
```

`TARGET_CPU` must be explicit, never `native`. Record the actual target's CPU
flags before accepting portability. The runner compares the compiler's recorded
target triple/features with Linux's observed ISA before launching a prover; that
does not replace the target's first executable compatibility check. Both the
`release` and existing `ci` Cargo profiles are available through `--profile`;
the default and workflow profile are `release`. Paired legs must use the same
profile and flags. `ci` disables release's fat LTO and debug information; it is
a separate performance identity, not a measured substitute for release.

For an early bundle, repeat `--workload fibonacci --workload sha2-chain
--workload sha3-chain --workload btreemap`. This changes which guests are prepared,
not the proving harness. Missing guests remain missing results, never zeroes.

On the measurement host, use the bundled script so its hash matches the manifest:

```bash
uv run --no-project python "$BENCH_OUT/bundle/scripts/bench/run.py" run \
  --tier main --kind null \
  --base "$BENCH_OUT/bundle/manifest.json" \
  --head "$BENCH_OUT/bundle/manifest.json" --out "$BENCH_OUT/null"
```

`--case ID` is repeatable, and `--warmups N --repetitions N` records explicit pilot
overrides. Select IDs directly from `suites.json`; partial/pilot results are not a
full PR comparison. Every case uses the exact same profile invocation as CI.

For the local component of a PR dry run, use an exact current-main base, exact
candidate merge revision, and one committed harness:

```bash
uv run --no-project python scripts/bench/run.py suite \
  --tier pr --base-revision "$BASE" --head-revision "$MERGE" \
  --harness-revision "$HARNESS" --target-cpu "$TARGET_CPU" \
  --out "$BENCH_OUT/pr-dry-run"
```

The runner never builds or measures on another host implicitly. Manual JSON uses
null GitHub provenance. Never compare a historical run on another host with a PR.
This manual command does not replace the actual workflow acceptance run, which
must also include checkout/fetch, transfers and result processing.
To render a comment without posting:

```bash
uv run --no-project python scripts/bench/run.py render-comment \
  --results "$BENCH_OUT/pr-dry-run/benchmark-results.json" \
  --out "$BENCH_OUT/pr-comment.md"
```

`run`, `suite`, and `render-comment` accept `--calibration "$CALIBRATION_JSON"`.
The file supplies the observational reference at execution or rendering time;
`suite.calibration_sha256` records the SHA256 of its exact bytes. A render-only
override changes the comment, not the original raw result. The option does not
enter build identity or change any native bundle's construction manifest.
Keep the suite configuration and bundles unchanged when adding observations.

Collect the standard-main population first on the released target runtime, with
`$BASE` set to its exact main revision:

```bash
uv run --no-project python scripts/bench/run.py suite \
  --tier main --head-revision "$BASE" --harness-revision "$HARNESS" \
  --target-cpu "$TARGET_CPU" --out "$MAIN_OUT"
```

Then use its returned bundle and bundled collector for the ten-pair reference:

```bash
uv run --no-project python "$MAIN_OUT/builds/head/scripts/bench/run.py" run \
  --tier pr --kind null --warmups 1 --repetitions 10 \
  --base "$MAIN_OUT/builds/head/manifest.json" \
  --head "$MAIN_OUT/builds/head/manifest.json" --out "$NULL_OUT"
```

After deriving the reference, pass it to the actual three-repetition comparison
without rebuilding that baseline just to add observations:

```bash
uv run --no-project python scripts/bench/run.py suite \
  --tier pr --base-revision "$BASE" --head-revision "$MERGE" \
  --harness-revision "$HARNESS" --target-cpu "$TARGET_CPU" \
  --calibration "$CALIBRATION_JSON" --out "$BENCH_OUT/pr-calibrated"
```

The source revisions, native harness revision, collector hash and calibration
source remain distinct. A later renderer does not change the collector that
recorded an earlier result. A changed collector hash makes its old reference
unavailable; there is no source-equivalence exception registry.

## Result and failure contract

The schema-1 `benchmark-results.json` contains the expected `cases`, each with all
raw warmup and measured `samples`. A successful sample is `complete`, exit code
zero, and has all numeric metrics. Failed, incomplete, unsupported, and missing
points are explicit, with reasons and null metrics. Non-executed workflows emit
a `disabled` result instead of scheduling an absent runner. The CI artifact keeps
raw logs, manifests, fixed inputs, compiler flags, toolchain and lockfile hashes,
binary/guest hashes, CPU topology, and host settings. It excludes executable
payloads and the runtime URS cache. A transportable `build` bundle keeps the
actual binaries, guest ELFs, and Akita schedules for off-host execution.

The comment shows the base and head medians for end-to-end time and peak RSS,
with signed percentage changes computed as `100 * (head_median / base_median - 1)`.
Only the configured measured repetitions contribute; warmups are excluded.
Missing repetitions, failed processes and zero denominators remain unavailable
comparisons. Proof sizes remain in the raw samples.

Without an optional file, `suite.calibration` comes from the supplied configuration.
The current default is explicitly unavailable. The comment identifies the suite
configuration and optional input by their SHA256s; a manual rendering does not
authenticate either source. Trusted publication requires both the calibration
object and `calibration_sha256` to equal its pinned expected configuration.
An unapproved manual override is rejected, even if the artifact claims the trusted
suite hash. The hosted workflow continues to use only that pinned configuration.

The measured method is `null-median3-symmetric-envelope-v2`: ten same-binary null
pairs supply eight overlapping windows of three pairs. Each window compares the
two leg medians in both directions and retains the larger percentage increase.
The informational threshold is 1.5 times the largest window increase, with floors
of 3% for time and 2% for peak RSS, rounded up to 0.5 percentage points. The
windows are not independent observations; no statistical significance or pass/fail
claim follows. The comment displays each cell's threshold and observed maximum.

References apply only to three measured repetitions. Match each cell against the
base leg's revision, binary, guest, input, trace rows, compiler settings and
effective lockfile, plus its workload, PCS, core count, ordered CPU mask and scale.
The collector hash, metric contract, host identity, kernel, microcode, control-plane
identity, boost, SMT, governor map, huge-page setting and total swap must also match.
Missing evidence or a mismatch leaves the real medians and signed deltas visible,
with an unavailable reference and its reason. Candidate binary, guest and trace
changes are shown, not rejected. A harness Git revision difference alone is not
a mismatch when the required binary, collector and metric identities match.

The GitHub artifact is `jolt-e2e-RUN_ID-RUN_ATTEMPT`. Its root always contains
`benchmark-results.json`. For a multi-commit main push, that document is the
pushed tip; earlier new first-parent commits have
`commits/FULL_SHA/benchmark-results.json`. `index.json` lists the actual event's
`push_before`, `push_after`, and every expected document, oldest first. A failed
or interrupted batch keeps missing documents visible in that index. An ancestry
check alone cannot independently attest that `push_before` matches GitHub's event
boundary; consumers must keep that provenance limit explicit.

## Administrator enablement

The draft adds no runner, setting, secret, or live comment. Keep
`JOLT_E2E_ENABLED` unset until steps 1–3 are satisfied. Step 4 enables controlled
acceptance runs; it is not evidence that the time budget has passed.

1. Use a dedicated Linux runner account and the observed target machine. That
   account must have no write-capable GitHub, SSH, cloud, or Pika credentials in
   its environment, home, helpers, or accessible files. Do not use a general
   development account. Confirm CPU topology/ISA and host storage readiness.
2. Create or select organization runner group `jolt-bench`. Set `visibility` to
   `selected` and its repository access list to **only `a16z/jolt`**. Set
   `allows_public_repositories=true` because this repository is public. Set
   `restricted_to_workflows=true` and `selected_workflows` to **only**
   `a16z/jolt/.github/workflows/e2e-bench.yml@refs/heads/main`. Register the runner
   into that organization group with label `jolt-bench`, as shown below. If this
   restriction is unavailable for the account, leave execution disabled: a
   PR-editable YAML `if` is not a security boundary. GitHub documents these
   fields in its [runner-group API](https://docs.github.com/en/rest/actions/self-hosted-runner-groups#update-a-self-hosted-runner-group-for-an-organization).
3. Install the repository's build prerequisites and `uv` through the normal
   administrator process. Set `JOLT_E2E_TARGET_CPU` only after observing the
   target ISA. Keep the runner cache outside the action workspace, with ownership
   restricted to that account. Complete a real null pilot and calibrate
   sizes/repetitions without changing protocol parameters.
4. Enable `JOLT_E2E_ENABLED=true` for an actual changed-source PR workflow run.
   Inspect every expected point, physical affinity, hashes and verified exits.
   Measure the full request-completion-through-result interval, including build
   or reuse preparation, checkout/fetch, transfers and result processing. Include
   trusted comment publication when enabled. Keep a miss as a miss; a prebuilt
   run or manual local receipt cannot establish the less-than-600-second gate.
5. After reviewing trusted publication separately, optionally set
   `JOLT_E2E_COMMENTS_ENABLED=true`. Until then, comments are rendered artifacts
   only. This task does not enable or post them.

### Registration uses the organization group

These are proposed administrator commands, not actions performed by this task.
Wait for the benchmark lead to release the host for administration. Configure
the group in step 2 before registering. Use an authorized administrator
workstation for GitHub API commands; do not put its credentials on the runner.
For an existing group, repository selection uses the separate
[repository-access API](https://docs.github.com/en/rest/actions/self-hosted-runner-groups#set-repository-access-for-a-self-hosted-runner-group-in-an-organization),
not the group-update payload. Read back the group and its selected repositories:

```bash
jolt_bench_group_id=$(gh api --paginate orgs/a16z/actions/runner-groups \
  --jq '.runner_groups[] | select(.name == "jolt-bench") | .id')
gh api "orgs/a16z/actions/runner-groups/$jolt_bench_group_id" \
  --jq '{id, name, visibility, allows_public_repositories, restricted_to_workflows, selected_workflows}'
gh api --paginate "orgs/a16z/actions/runner-groups/$jolt_bench_group_id/repositories" \
  --jq '.repositories[].full_name'
```

Require the exact settings in step 2 and only `a16z/jolt` in the repository
readback. If the group or workflow restriction is unavailable, leave execution
disabled. On the administrator workstation, obtain an organization registration
token using the [documented API](https://docs.github.com/en/rest/actions/self-hosted-runners#create-a-registration-token-for-an-organization).
It expires after one hour:

```bash
gh api --method POST orgs/a16z/actions/runners/registration-token --jq .token
```

Then, in an interactive root terminal on the prepared host, register as the
dedicated `runner` account (home `/home/runner`):

```bash
runuser -u runner -- bash -lc '
  cd /home/runner/actions-runner &&
  ./config.sh --url https://github.com/a16z \
    --runnergroup jolt-bench --name jolt-bench \
    --labels jolt-bench,jolt-bench-cherry-9254p --work _work
'
```

Supply an **organization** registration token at the runner's masked local
prompt. Omitting `--unattended` and `--token` keeps the token out of the command
line; do not store it as a repository secret or supply an administrator PAT on
this account. Runner 2.338.2 supports
[`--runnergroup`](https://github.com/actions/runner/blob/d1eb12d4b115c301201de27d667284e1a80f2ab5/src/Runner.Listener/Runner.cs#L1141-L1148)
and its [secret prompt](https://github.com/actions/runner/blob/d1eb12d4b115c301201de27d667284e1a80f2ab5/src/Runner.Listener/CommandSettings.cs#L389-L425).
The organization URL is intentional: repository-scoped registration at
`https://github.com/a16z/jolt` does not select this organization group. The
hardware label is additional; `jolt-bench` is the workflow's required label.

After registration succeeds, use the same root terminal to install and start the
native runner service as `runner`, then inspect its status. Runner 2.338.2's
[service script](https://github.com/actions/runner/blob/d1eb12d4b115c301201de27d667284e1a80f2ab5/src/Misc/layoutbin/systemd.svc.sh.template#L48-L108)
requires root and accepts the service account as the `install` argument:

```bash
cd /home/runner/actions-runner &&
./svc.sh install runner &&
./svc.sh start &&
./svc.sh status
```

Back on the administrator workstation, read the group's actual runner membership
and labels. Before enabling execution, repeat the group/repository readbacks
above and require runner `jolt-bench` in this group, online, with label `jolt-bench`:

```bash
gh api --paginate "orgs/a16z/actions/runner-groups/$jolt_bench_group_id/runners" \
  --jq '.runners[] | select(.name == "jolt-bench") | {id, name, status, labels: [.labels[].name]}'
```

The hosted PR request workflow runs on code PRs and skips docs-only changes. A
default-branch `workflow_run` producer validates the request against GitHub,
rejects forks, and resolves the PR merge ref's exact current-main parent. No
`pull_request_target` workflow executes PR code. The persistent measurement job
has only read permissions; subprocess environments exclude tokens. The runner's
non-secret `RUNNER_TRACKING_ID` is retained for cancellation cleanup. Trusted
action checkouts are cleaned normally. The locked source clone, its repo-local
Cargo target and completed bundles survive outside that workspace. Removing the
store causes fresh preparation; a compiler-cache hit is never inferred from
directory existence or from a completed-bundle reuse receipt.

The hosted plan is transferred by its captured immutable artifact ID, not just
its name. The hosted read-only validator checks actual workflow/artifact/PR
records, revisions, matrix, sampling, affinity, and metrics before rendering text.
A separate hosted job receives only that validated text and a PR-write token,
rechecks current head/base, and updates its own sticky comment. It never executes
a downloaded script or measured binary. These checks establish recorded
provenance, not independent cryptographic attestation of a persistent host.

PR requests and comparisons cancel superseded runs per PR branch. Main and
nightly use unique run groups; GitHub's single-pending replacement behavior
cannot silently discard main commits. The dedicated runner and owned file lock
serialize execution. Nightly is scheduled at 07:00 UTC (03:00 EDT / 02:00 EST).
The self-hosted job has a 10-minute timeout for PRs and 180 minutes for main and
nightly. A timeout is a failure, not evidence of a completed sub-600-second job.
Interruption can prevent finalization and artifact upload; missing artifacts or
results remain unavailable, not passing. A complete manual trial must finish in
less than 600 seconds, including `ci.py execute`, `ci.py finalize`, preparation,
transport and result processing. Hosted-only overhead remains unverified.

The measured pilot ran each of the 60 supported nightly cases once. It does not establish the full
nightly suite's duration, variation between runs, or completion within 180
minutes. Hosted-job and queue overhead remain unmeasured; no finish guarantee
is established.

Rerun all jobs together: a plan from an older run attempt is rejected.

Focused checks, without proof timings:

```bash
uv run --no-project python -m unittest \
  scripts.tests.test_e2e_bench scripts.tests.test_e2e_ci \
  scripts.tests.test_e2e_publication -v
```

Process fixtures own and reap their children. They await completed readiness
records rather than scheduling sleeps or host-wide PID assumptions. Rust test
changes must follow the same ownership rule; Tokio timer tests use paused time.
