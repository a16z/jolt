# Reproducible CPU proof benchmarks

These scripts reuse `jolt-prover profile --backend optimized`. They build real
guest ELFs before measurement, run fresh prover processes, verify every successful
proof, and retain raw samples. There is no reference-backend timing lane.

PR benchmarks are opt-in. A maintainer or administrator adds `benchmark` to a
same-repository PR; subsequent pushes retain that admission while the label stays
applied. The target builds both legs using its local Cargo state, then proves.
Every main push's new first-parent commits and the nightly run instead build on
GitHub-hosted runners; the target downloads, verifies and runs those artifacts
without any Cargo/toolchain fallback. Both routes keep the repository's `release`
profile (fat LTO and debug information) and explicit `-C target-cpu=znver4`.

The PR matrix has 17 cells and launches 136 proof processes, including warmups.
`pr-baseline.json` preserves the initial three-cell, 24-proof matrix. Measured
timing and noise references apply to their exact artifact inputs, not to an
unmeasured later kernel or a future run. Automatic references use the exact-base
lifecycle below; the measured bands below belong to the pinned manual experiment. No host installation, target setting change, or live check publication is part of this draft.

## Workloads and tiers

`suites.json` is the source of the case matrix. Standard-main and nightly run both
Dory and Akita at exactly 1, 8, or 24 physical cores. PRs use all 15 Akita
main-size cells plus Dory SHA2 and BTreeMap at 24 cores.
All three tiers use `release` without profile overrides.
The ordinary instruction profile and clear proofs are
used; no protocol parameter is adjusted for the benchmark.

| Workload | Fixed operations at 1 / 8 / 24 cores in `main` | Input |
|---|---:|---|
| Fibonacci | 16000 / 64000 / 250000 | Postcard-encoded iteration count |
| SHA2 chain | 64 / 256 / 1024 | `[5; 32]`, then the count |
| SHA3/Keccak chain | 48 / 192 / 768 | `[5; 32]`, then the count |
| BTreeMap | 128 / 512 / 2048 | Insertion count; guest-owned deterministic keys |
| secp256k1 ECDSA batch | 1 / 4 / 16 | One valid signature/public-key vector, verified repeatedly |

The `verify_batch` function in `examples/secp256k1-ecdsa-verify/guest` calls the
real `ecdsa_verify` precompile. The profiling harness enables `batch-bench` and
selects `verify_batch` through `JOLT_FUNC_NAME`; builds without that feature keep
the existing single-signature program. The signature vector comes from
`examples/secp256k1-ecdsa-verify/src/main.rs`; each operation verifies it once.
Input bytes and guest ELF hashes are recorded for each build. `seed: null` means
the input generator uses no PRNG.

| Tier | Cases | Warmups / measured repetitions per leg |
|---|---|---:|
| `pr` | All five workloads at 1 / 8 / 24 cores with Akita; Dory SHA2 and BTreeMap at 24 cores: 17 points | 1 / 3 |
| `main` | All five workloads, all three core counts, both PCS: 30 points | 1 / 3 |
| `nightly` | Two larger inputs for each main point: 60 points | 1 / 5 |

The historical sub-600-second interval below excludes compilation. The new PR
route includes local builds and has no measured sub-600-second promise. A warm
Cargo directory is not proof that a changed PR will compile quickly. Process
count alone establishes neither a duration nor a future timing guarantee.

To select the preserved 24-proof baseline, use
`ci.py execute --config scripts/bench/pr-baseline.json` with artifacts built by
the matching harness. The original measurement artifacts retain their original
configuration bytes and source revisions. Larger inputs remain bundled for
separately scoped measurements.

### Measured PR interval

[Hosted run 41362804476](https://github.com/a16z/jolt/actions/runs/41362804476)
built base `e37e6f24a4276fb76e85d0db150cdb196cbc26e5` and candidate/harness
`535840a97b0678a95e47df5c3adf71c5faa04d85`. Both hosted legs contain identical
native prover bytes. The following measurements used those artifacts on
`jolt-bench`, not the later kernel merged into the PR.

| Matrix | Fresh processes, including warmups | Manual on-host checkout through packaging | Download request through finalize | Suite execution |
|---|---:|---:|---:|---:|
| Preserved three-cell baseline | 24 | 78.999932 s | 67.783 s | 46.969 s |
| Adopted 17-cell matrix | 136 | 458.583460 s | 448.842 s | 429.834 s |

The manual on-host interval uses one target clock: checkout start through result
packaging end. It includes downloads, ZIP checks, extraction, payload verification,
the proof processes, finalization and remote-orchestration gaps. It excludes
result upload and runner queue/setup; no self-hosted runner was registered. The
narrower download-to-finalize interval is shown separately. Suite execution
includes fresh-process setup and gaps; it is not pure proving time. Hosted
preparation took 2438.71 s for the whole command, outside these target intervals.

The hosted and matched local binaries are not byte-identical as packaged. Their
GNU build-ID and GCC-version comment sections differ; removing those two metadata
sections from comparison copies yields byte-identical files for both provers.
The original artifacts were left unchanged and used for the proof measurements.

The matched hosted/local control had equal public-output bytes, proof sizes and
trace dimensions, with end-to-end and process-time observations inside the frozen
per-cell margins. The three secp256k1 cells return no public bytes, so output
equality alone says nothing about their result. The secp256k1 one-core RSS
deviation was 4.9399%, outside its 2% margin; its cause is unresolved. RSS was
not an execution/timing falsifier, and no RSS equivalence is claimed.

### Hosted build replay

[Run 41384450297](https://github.com/a16z/jolt/actions/runs/41384450297) replayed
the same `e37e6f24`/`535840a9` native inputs on one standard `ubuntu-24.04` runner.
Whole-command preparation took 2943.57 s with empty build state and 26.29 s on
the second pass, with source and Cargo state retained but completed bundles
removed. Both passes called Cargo; the second compiled no native prover units.
These intervals exclude checkout and artifact upload. Upload took 36 s and 34 s
respectively, using the job API's one-second timestamps.

The cold prover and guest hashes match the original artifact. The second pass
checks retained-output identity, not an independent rebuild. This is an
unchanged-input preparation floor, not a code-changing PR or cross-job cache
measurement. No cross-job cache is configured. The earlier 2438.71 s preparation
used an AMD EPYC runner; this replay used an Intel Xeon runner with the same
toolchain, image and build inputs. One cold run per host does not isolate a
hardware speed difference. The temporary replay job was removed after capture.

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
hosted: [ pinned release build + guest preparation + artifact upload ]
target:                                                              [ download ][ verify artifacts + processes + results ]
process:                                                                          [ launch ... verification ... exit ]
e2e:                                                                                [ load, trace, preprocess, setup, prove, serialize ]
verify:                                                                                                                              [ parallel ][ single ]
```

`e2e_s` starts before loading the prebuilt ELF and input bytes. It includes the
profile harness's shared trace, program decoding/expansion and
preprocessing, witness construction, PCS setup,
proving, and proof serialization. It stops before verification. Both existing
verification calls must succeed before the JSON metric file is written.
`prove_s` and `setup_s` surround the existing prove and setup calls, respectively;
neither substitutes for `e2e_s`.

`verify_s` is the existing parallel verifier; `verify_single_s` is its separate
single-threaded run. Both verify the in-memory proof; there is no deserialize
step. `public_output` is an array of the exact public output bytes passed to
those verifiers, emitted after they both succeed. Same-source hosted/local
controls compare these bytes, proof sizes and row counts; they do not claim
byte-identical serialized proofs. `process_s` is parent-observed launch through reaped exit,
including both verifications, output, teardown, and the GNU time launcher.
`peak_rss_bytes` is Linux `ru_maxrss`, collected by GNU `/usr/bin/time` (`time`
package) and converted from KiB to bytes. GNU time forks the measured process
after exec, excluding the Python harness's address space from the child's RSS
high-water mark. This retains metric contract 1: resident anonymous and
file-backed memory, not an allocation estimate or cgroup total. Unmapped page
cache is excluded; reaped descendants contribute their largest peak, not a sum.
It uses exit-time kernel accounting, without polling or a cgroup. The collector's
extra fork/exec is included in `process_s`. Per-sample `process.rss` retains the
raw KiB value; stdout/stderr remain in `process.log`. Nonzero command exits stay
nonzero; signal termination uses GNU time's `128 + signal` exit status.
`proof_bytes` is the serialized proof length. `trace_rows` is the modular trace's `proof_len()` before
padding, not a raw RISC-V instruction count. `padded_trace_rows` is the actual
`ProverConfig.trace_length`, including the required final no-op and minimum size.

All times are full-resolution numeric seconds. Build time is separate from the
proof metrics. `ci.py prepare` records its hosted interval in `builds.json`.
`ci.py execute` records `artifact-verification-through-results-on-this-host`,
with `build_s: 0` and `artifact_verification_s`. Download, checkout, upload and any
trusted publication are outside that Python interval. `run` records
`prebuilt-execution-only`, with `build_s: null`. PR CI uses `ci.py local`, which
reuses preparation and execution under one host lock and records
`sequential-build-preparation-through-results-on-this-host`. Its current build
time and total interval include both legs; no hosted build is part of that route.
None of these commands adds historical build durations to a fresh target total.
Individual build manifests retain their original construction intervals and host
identity, including when a completed bundle is reused. Each new `preparation.json`
records the current validation/copy or build interval. The suite's `build_s` is
the freshly observed preparation interval, never a sum of old manifest durations.
The target's transfer and result-processing intervals must be measured separately
before claiming its PR budget. Cargo target existence or a configured
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
No benchmark-target governor, SMT, boost, swap, RAID, or other setting is changed.
`host.microcode` comes from `/proc/cpuinfo`. `host.control_plane` records the
readable `/opt/pika/bin/pika-cli` and `pika-node` SHA256s and the node's `--version`
output. Unavailable observations are null, never an assumed match. These are
runtime observations; they are not part of native bundle cache identity.

The disposable GitHub-hosted builder checks for 40 GB of free disk space, then
adds a 24 GB swap file, reserving 16 GB of disk space for the build. It uses one
Cargo job and retains the repository's release profile, including fat LTO and
debug information. Its existing 120-minute job timeout is separate from the
target's 120-minute PR build-plus-run timeout. Main/nightly/calibration target
jobs allow 180 minutes. These are failure limits, not duration estimates.

## Manual commands

Run-only execution needs Linux, Python 3.11+ through `uv`, Git for the pinned harness
checkout, GNU `/usr/bin/time` (Debian/Ubuntu package `time`), `taskset` and `lscpu`. The target also builds opted-in PRs, so it needs the pinned Rust toolchain,
its `rust-src` component and configured RISC-V targets. The five
prepared guests use `riscv64imac-unknown-none-elf`. The build produces its own
`jolt` CLI; a stale globally installed CLI is not used.

On an Ubuntu build host, have the administrator provide `gcc`, `g++`, `make`, `dpkg-dev`,
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

Preparation builds the real `jolt` guest-builder CLI and a host-runnable
`jolt-prover prepare` under the existing `ci` Cargo profile, without target-CPU
flags. These helpers never prove, and the hosted CPU need not support the target
ISA. Guest builds receive `--release`. The measured prover binaries use `release`
with explicit `znver4` code generation, not the host-generic helper. The manifest
records both helper hashes, their profile/flags, Rust/Cargo versions and native
compiler/linker versions. Every bundle prepares both prover variants and the
union of all configured tiers' guest inputs. Its host-generic tools live in
`target/host-tools`, separate from the measured release outputs. The CLI still depends on the Dory host
SDK (`Cargo.toml` and `jolt-sdk/src/host_utils.rs`); omitting the Dory benchmark
executable does not eliminate Dory dependency compilation. No CLI dependency,
prover optimization or guest optimization is changed by the builder-profile split.

The collector revision must be committed, and `run.py` must be that revision's
file. Admitted PR builds prefer main's native profiling harness. When its declared
file set differs from a leg's own harness, or its build attempt fails, that leg
uses its own native harness instead. Failed build output is published in
`executor.log` and the last 8 MiB of each `builds/*/build.log`. Other build payloads are excluded. The collector
and suite configuration still come from main; the PR's collector code is never
executed.

Build manifests record the collector revision as `harness_revision`, the actual
native code as `native_harness_revision`, its file hashes, and any fallback reason
and diff against the collector's native harness. Results record the native
revision per leg. When the legs use differing harness code, the comment says so
and includes escaped diff excerpts, each capped at 12000 characters after
escaping. Existing locked package identities may not change. A same-source null
run uses identical binaries, guests and native harness code.

Git checkout and cleanup can rewrite identical harness files and patched manifests.
After overlay and lock resolution, the builder preserves their prior modification
times only when the effective bytes are identical. Changed or removed inputs keep
normal invalidation behavior; tracked edits and untracked debris are still reset.
This avoids timestamp-only invalidation, but does not establish Cargo cache hits
or predict the time for a changed-source build.

Completed bundles persist in the same locked store. Reuse requires an exact
source and harness revision, Rust/Cargo versions, target CPU/triple/features,
flags, prover profile and `profile_config`, guest-builder profile, lockfiles, harness files, runner and
suite-definition match.
Every required PCS and guest/input must exist. All recorded binary, guest, input,
runner, configuration and schedule hashes are checked before copying. A main-tier bundle
does not qualify merely because its source SHA matches: it may lack a PR input.
Different PCS/input sets can coexist; a superset may serve a smaller request
only under the same profile and settings. An Akita-only bundle cannot serve a
request that also requires Dory.
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
These command receipts cover the CLI build, prover builds and guest preparation,
including failures; a launch failure has no exit code. They are command durations,
not separate compiler/linker timings. Reuse keeps the original durations in
`construction.log`; current commands remain in `build.log`. Validation, copying
and all other preparation count in the fresh preparation and wall-time receipts.
Commands, child output and completion records also stream to the CI step log,
so a lost runner does not erase all build progress. During hosted preparation,
memory, swap, free disk bytes and the kernel's `oom_kill` counter are sampled
every ten seconds. A final sample and readable kernel warnings/errors are logged
on exit. A denied kernel-log read is reported, not evidence that no OOM occurred.
These observations describe the build host, never the proof timing process.
GNU time records the preparation command's duration and largest reaped process's
resident high-water mark. That RSS excludes swapped-out pages and is not the
concurrent process-tree total. Maximum observed swap is a sampled aggregate,
not a true high-water mark.
Native Cargo builds emit stable JSON into the same log. Each `compiler-artifact`
record's `fresh` field distinguishes reused units from executed rustc units;
`package_id`, `target`, `features` and `profile` identify the unit. Preserve these
records and candidate binary hashes for a warm-path receipt. A missing completion
record is not a completed compilation, and unit freshness does not separate
LLVM, LTO or native linking. Log emission remains inside the measured interval.

Build a transportable bundle (no proving occurs):

```bash
uv run --no-project python scripts/bench/run.py build \
  --revision "$REVISION" --harness-revision "$HARNESS" \
  --target-cpu "$TARGET_CPU" --out "$BENCH_OUT/bundle"
```

`TARGET_CPU` must be explicit, never `native`. Record the actual target's CPU
flags before accepting portability. The runner compares the compiler's recorded
target triple/features with Linux's observed ISA before launching a prover; that
does not replace the target's first executable compatibility check. `--profile`
accepts `release` and `ci`; `build` and `suite` default to `release` for every tier.
CI always selects `release`. `profile_config` records the selected source's
profile table, without changing it through Cargo overrides. Paired legs must use
the same profile, settings and flags. Manual `ci` experiments remain a separate
performance identity and cannot replace release results or their references.

For an early bundle, repeat `--workload fibonacci --workload sha2-chain
--workload sha3-chain --workload btreemap`. This changes which guests are prepared,
not the proving harness. Missing guests remain missing results, never zeroes.

For a manual pilot, use the collector at the recorded harness revision after
validating the artifact hashes. The operational entrypoint instead reads a
separately pinned plan and the hosted job's build-index digest:

```bash
uv run --no-project python scripts/bench/ci.py execute \
  --plan "$PLAN/plan.json" --bundles "$BUNDLES" \
  --artifact-id "$BUILD_ARTIFACT_ID" --index-sha256 "$BUILD_INDEX_SHA256" \
  --out "$BENCH_OUT/pr-run"
```

`ci.py execute` checks the planned source/harness revisions, release settings,
lockfile and harness hashes, suite configuration, required binaries, guests,
inputs and schedules before starting a proof. Missing or changed artifacts fail;
there is no build fallback. It restores executable permission only on verified
binaries because zipped Actions artifacts do not preserve that permission.
The downloaded Python files are checked as data, not imported or executed.
`ci.py verify` takes the same plan, bundle, artifact-ID and index-hash arguments
without `--out`; it checks every payload and restores binary modes without
starting a proof. Use it before the first cheap executable/proof readiness check.

The lower-level collector supports separately labelled pilots and references:

```bash
uv run --no-project python "$BENCH_OUT/bundle/scripts/bench/run.py" run \
  --tier main --kind null \
  --base "$BENCH_OUT/bundle/manifest.json" \
  --head "$BENCH_OUT/bundle/manifest.json" --out "$BENCH_OUT/null"
```

`--case ID` is repeatable, and `--warmups N --repetitions N` records explicit pilot
overrides. Select IDs directly from `suites.json`; partial/pilot results are not a
full PR comparison. Every case uses the exact same profile invocation as CI.

For an explicitly authorized local build-plus-run diagnostic only, use an exact
base, candidate merge revision, and one committed harness:

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

This option is manual-only. It can render a saved comparison against its exact
measured reference without changing the original results or build manifests.
It does not refresh calibration for the trusted workflow. Committing a reference
is not the external refresh procedure described below. A separately authorized local build remains only the off-host equivalence
control, not a replacement for the hosted base.

The source revisions, native harness revision, collector hash and calibration
source remain distinct. A later renderer does not change the collector that
recorded an earlier result. A changed collector hash makes its old reference
unavailable; there is no source-equivalence exception registry.

## Result and failure contract

The schema-1 `benchmark-results.json` contains the expected `cases`, each with all
raw warmup and measured `samples`. A successful sample is `complete`, exit code
zero, and has all numeric metrics. Failed, incomplete, unsupported, and missing
points are explicit, with reasons and null metrics. Non-executed workflows emit
a `disabled` result and no dispatch request. The CI artifact keeps
raw sample logs, embedded manifests, compiler flags, toolchain and lockfile
hashes, binary/guest hashes, CPU topology, and host settings. It excludes the
runtime URS cache and local build payloads. For main/nightly, the separate
`jolt-e2e-builds-RUN_ID-RUN_ATTEMPT` artifact keeps
the actual binaries, guest ELFs, fixed inputs, Akita schedules and construction
logs. `builds.json` binds their manifest hashes to the exact plan bytes.

If hosted preparation fails, no dispatch artifact is published and no target job
or App check is created. The hosted workflow failure remains visible. Producing
a separate build-failure receipt remains debt to resolve before enabling target
execution. Target execution failures with a result go through validation; failures
before any result complete the App check directly.

The comment shows the base and head medians for end-to-end time and peak RSS,
with signed percentage changes computed as `100 * (head_median / base_median - 1)`.
Only the configured measured repetitions contribute; warmups are excluded.
Missing repetitions, failed processes and zero denominators remain unavailable
comparisons. Proof sizes remain in the raw samples.

### External calibration lifecycle

Every main push calibrates its tip once, before measuring all new first-parent
commits. Nightly and manual main/nightly runs refresh their pinned tip. The null
uses the verified hosted tip bundle for both legs and the 17 PR cells, with one
warmup and ten measured pairs per cell: 374 proof processes. There is no extra
build or timeout increase for ordinary refreshes. Failed calibration preserves
independent history results; only a complete null produces a reference.

The target binds `calibration/raw/benchmark-results.json` by path and SHA256 in
its root result. Trusted ingest validates the plan, hosted manifest digests,
identical legs, all 374 zero-exit samples, alternating order, affinity and metrics.
It recomputes eight overlapping three-pair windows. For each metric and window,
`d = 100 * (max(median(A)/median(B), median(B)/median(A)) - 1)`; nonpositive
denominators are unavailable. The envelope is `0.5 * ceil(max(floor, 1.5 * max(d))
/ 0.5)`, with time/RSS floors of 3%/2%. Rational arithmetic keeps rounding exact
at half-percentage-point boundaries. Method: `null-median3-symmetric-envelope-v2`.
These are observed informational bands, not confidence intervals or verdicts.

Each schema-1 cell key is SHA256 of compact, sorted JSON with no nonfinite
numbers. Matching requires the exact base revision, metric contract, collector,
calibration helper and committed suite hashes; Rust/Cargo versions, target ISA,
flags, profile/config, effective lockfile, native harness hashes and schedules;
cell/input/guest identities, ordered physical CPU IDs, and every measured base
sample's trace and padded-row counts. Host identity includes machine ID, kernel,
microcode, Pika management version and CLI/node hashes, boost, SMT, the complete
governor map, THP and total swap. These must agree before and after both
measurements. The host's management node is not a benchmark agent; a Pika update
invalidates existing references. Missing or unreadable identity evidence makes
references unavailable. Construction hostname, compiler/linker labels,
transient free swap, random URS hashes and harness Git SHA alone do not bind;
the initially empty suite-local Dory cache policy does.

An RSS collector change alters the committed runner hash, so references from the
direct Python `wait4` collector do not match and must be regenerated, even when
their hosted-artifact samples were not visibly limited by the parent's RSS.

For lookup only, `elf-no-build-id-comment-v1` uses GNU objcopy to remove exactly
`.note.gnu.build-id` and `.comment` on a copy, refusing either section if writable
or executable. Hash that copy; execute and byte-verify the untouched originals.
Normalization failure makes the reference unavailable. This admits metadata
differences, not RSS equivalence; the measured one-core secp256k1 discrepancy
above remains unresolved.

Ingest stores recomputed `calibration/reference.json` with full keys, envelopes,
raw hash, measurement times, producer/ingest identities and plan/build digests.
Canonical ingest artifacts are retained 30 days. The history publisher archives
reference and raw evidence under `dev/prover-history/calibration/<reference-sha>/`
and appends `calibration/index.json`. The catalog is discovery-only. PR ingest
captures its gh-pages commit, fetches candidates by immutable artifact ID, checks
successful main App ingest metadata and attempt, follows `ingest.json` to the
trusted non-PR plan/dispatch, and validates the raw evidence and recomputed
reference again. Catalog hashes alone grant no authority; expired artifacts have
no fallback. No measured reference is committed into main or build identity.

Choose the newest authenticated matching reference finished before comparison
start and valid through comparison finish; ties use producer then ingest
run/attempt. Expiry is 48 hours after calibration finished, not publication.
Keep older references for queued older bases; never select the widest band.
Apply references only to an in-memory rendering copy. Source result JSON stays
unchanged; `comparison.json` archives the validated summary, reference SHA/source,
catalog revision and availability reasons. Missing, expired, mismatched or
unauthenticated references preserve medians and signed deltas, with an explicit
unavailable reason. Target-supplied calibration overrides remain rejected.

### Bootstrap acceptance before ordinary enablement

With the trusted workflows and history publisher on main, a maintainer dispatches
`e2e-bench.yml` on main with `tier=calibration`. This tier alone is permitted while
`JOLT_E2E_ENABLED` is disabled. It runs the hosted null, then a three-pair
same-source probe built with the local PR recipe: 374 + 136 = 510 proof processes
plus a local build, with no PR source or comments. The root binds the probe
sidecar too. Bootstrap probes do not enter ordinary history trends.

After its ingest artifact has been published, run the trusted acceptance command
with an Actions/Contents-read token in `GH_TOKEN`:

```bash
uv run --no-project python scripts/bench/publication.py verify-calibration \
  --ingest-run-id "$INGEST_RUN_ID" --ingest-run-attempt "$INGEST_RUN_ATTEMPT"
```

It authenticates the bootstrap source and resolves/renders all 17 probe cells
through the same published catalog/artifact path used by PR ingest. Acceptance
requires all 17 reference cells, 374 verified exits, estimator contract vectors,
matching reference/raw hashes, and all 17 probe bands. Record actual build,
target and queue intervals; process counts are not timing acceptance evidence.
After a real main advance, repeat bootstrap: the new reference must publish and
resolve without a reference commit, and the old exact-base reference must refuse
the new probe. Only then enable ordinary tiers. These live acceptance steps have
not been performed by this source change.

The verified manual reference remains evidence for the 17-cell hosted null at
base `e37e6f24`, harness `535840a9` and collector `c6afb17a`. Its original JSON has
SHA256 `09b4fa3f076d1042724959b8af7479f134fb131b3c4d667b401e6d38cd9dae6b`.
Its time envelopes are 3–5%, and RSS envelopes are 2–12.5%. The saved reference
and raw results retain their original identities; they are not an automatic
reference for a later base. Manual output identifies the supplied reference by
SHA256, but that rendering does not authenticate the source.

The measured method is `null-median3-symmetric-envelope-v2`: ten same-binary null
pairs supply eight overlapping windows of three pairs. Each window compares the
two leg medians in both directions and retains the larger percentage increase.
The informational threshold is 1.5 times the largest window increase, with floors
of 3% for time and 2% for peak RSS, rounded up to 0.5 percentage points. The
windows are not independent observations; no statistical significance or pass/fail
claim follows. A manual comment using the matching reference displays each cell's
threshold and observed maximum.

References apply only to three measured repetitions. Match each cell against the
base leg's revision, binary, guest, input, trace rows, compiler settings and
effective lockfile, plus its workload, PCS, core count, ordered CPU mask and scale.
The collector hash, metric contract, host identity, kernel, microcode, control-plane
identity, boost, SMT, governor map, huge-page setting and total swap must also match.
Missing evidence or a mismatch leaves the real medians and signed deltas visible,
with an unavailable reference and its reason. Candidate binary, guest and trace
changes are shown, not rejected. A harness Git revision difference alone is not
a mismatch when the required binary, collector and metric identities match.

When produced, the GitHub artifact `jolt-e2e-RUN_ID-RUN_ATTEMPT` contains
`benchmark-results.json`. For a multi-commit main push, that document is the
pushed tip; earlier new first-parent commits have
`commits/FULL_SHA/benchmark-results.json`. `index.json` lists the actual event's
`push_before`, `push_after`, and every expected document, oldest first. A failed
or interrupted batch keeps missing documents visible in that index. An ancestry
check alone cannot independently attest that `push_before` matches GitHub's event
boundary; consumers must keep that provenance limit explicit.

## Execution through the Pika App

GitHub-hosted jobs plan each run and build main/nightly/calibration artifacts.
Their final job uploads `jolt-e2e-dispatch-RUN_ID-RUN_ATTEMPT/dispatch.json`. The Pika GitHub
App lists that run's artifacts and downloads only the exact plan, dispatch and
optional builds names for the same attempt. It moves bytes; it never checks out,
builds or executes Jolt code on the daemon host. A run with no dispatch artifact
starts no target work. No self-hosted GitHub runner or organization registration
token is needed.

The hosted planner reads the current PR and all issue-event pages. It requires
an open same-repository PR into main, its current `benchmark` label application,
and an applier whose current API `role_name` is `maintain` or `admin`, not merely
`write`. It resolves and pins the current main base, submitted head and merge
parents. Replays older than the label application are refused. Later pushes
remain admitted while the same label application stays current.

The App repeats admission when enqueueing and immediately before submitting.
Every request must be a successful `.github/workflows/e2e-bench.yml` run whose
repository and head repository are `a16z/jolt`. Main push, schedule and manual
requests must have head branch `main`; the API comparison of the measured SHA
against main must be `identical` or `ahead`. PR requests must have exactly one
payload PR with base `main` and the dispatch's PR number. The App rechecks the
open PR, same-repository head, main base, exact head SHA and current label. All
issue-event pages must identify the dispatch's latest label event and actor,
with no later removal. The actor must still hold `maintain` or `admin`. API
errors refuse admission. For `pull_request_target`, the run's head SHA and branch
are the submitted PR's; the payload PR's base ref binds it to main.

The App connects to `jolt-bench` as `runner` with a dedicated forced-command SSH
key. Its root-owned entry accepts only `submit`, `status`, `fetch` and `remove`
with a bounded job ID. Submission carries an uncompressed POSIX tar containing
only `dispatch.json`, `plan.zip` and, for main/nightly/calibration, `builds.zip`.
The entry limits those files to 1 MiB, 8 MiB and 32 GiB, respectively, rejects links,
duplicates and traversal, and caps extracted builds at 64 GiB. It fetches the
planned harness SHA anonymously from `https://github.com/a16z/jolt`. The worker runs in a transient `runner` systemd user unit named
`jolt-bench-JOB.service`, so SSH disconnects and daemon restarts do not restart it.
`RuntimeMaxSec` bounds each tier, `ExitType=cgroup` includes separate-session
descendants, and `KillMode=control-group` terminates the whole job. Completed and
failed unit states are retained for status reads; `remove` stops and resets the
unit before deleting its directory. A repeated job ID returns `exists` without
re-executing it. A reboot that loses the transient unit reports `lost`.

The worker uses a fixed environment with the `runner` Rust and uv directories
on PATH and no inherited GitHub, SSH, cloud or Pika credentials. It calls
`ci.py target`, which uses `local` for PRs and `execute` for
main/nightly/calibration, then always finalizes the result. The PR path anonymously rechecks the same label,
branch and head after acquiring the kernel lock and again before proving.
Removing the label does not terminate a build or proof already running. The App
submits one job at a time in FIFO order; the kernel lock also serializes manual
target commands. Submission is size-capped and staged before acquiring the jobs
lock, so an incomplete upload cannot block status/fetch/remove. PR timeouts are
120 minutes; main/nightly/calibration timeouts are 180. Results include an
`executor.log` tail of at most 8 MiB from the unit journal and capped per-build logs. Normal completion
finalizes in the worker; systemd finalizes a killed/timed-out worker after its
process tree has stopped.

PR source, including `build.rs`, executes only on `jolt-bench` as `runner`.
While a label remains, anyone able to push that same-repository branch can change
the code that runs without a new label application. Environment filtering is
not a sandbox: PR code can change that account's files, warm Cargo state and
future measurements. Hash validation authenticates neither a persistent host nor
locally produced PR binaries. This remains accepted persistent-account trust
debt. The account must have no write-capable GitHub, SSH, cloud or Pika credentials
in its environment, home, helpers or accessible files. Pika agents never run there. SSH invokes the forced command through the account
login shell; a runner-owned Bash startup file can run first. That startup hook
has the same trust level as the accepted persistent-account state, and clearing
the entry environment does not remove it.

### One-time host installation

The `jolt-bench` operator installs the reviewed entry and the dispatcher's public
key. No a16z organization administrator is involved. Keep the private key only
on the daemon host. Supply the reviewed script as `/root/jolt-bench-entry.reviewed`
and its dedicated Ed25519 public key as `/root/jolt-bench-dispatch.pub`, then run
these commands in a root shell on `jolt-bench`:

```sh
set -eu
loginctl enable-linger runner
install -d -o root -g root -m 0755 /usr/local/libexec
install -o root -g root -m 0755 /root/jolt-bench-entry.reviewed /usr/local/libexec/jolt-bench-entry
install -d -o root -g root -m 0755 /etc/ssh/authorized_keys
read -r jolt_key_type jolt_key_blob jolt_key_comment < /root/jolt-bench-dispatch.pub
test "$jolt_key_type" = ssh-ed25519
printf 'restrict,command="/usr/local/libexec/jolt-bench-entry" %s %s\n' \
  "$jolt_key_type" "$jolt_key_blob" > /etc/ssh/authorized_keys/jolt-bench-runner
chown root:root /etc/ssh/authorized_keys/jolt-bench-runner
chmod 0644 /etc/ssh/authorized_keys/jolt-bench-runner
cat > /etc/ssh/sshd_config.d/60-jolt-bench-runner.conf <<'SSH_CONFIG'
Match User runner
    AuthorizedKeysFile /etc/ssh/authorized_keys/jolt-bench-runner
    PasswordAuthentication no
    KbdInteractiveAuthentication no
Match all
SSH_CONFIG
chown root:root /etc/ssh/sshd_config.d/60-jolt-bench-runner.conf
chmod 0600 /etc/ssh/sshd_config.d/60-jolt-bench-runner.conf
/usr/sbin/sshd -t
systemctl reload ssh
```

The key file is readable by `runner` because sshd reads it as the target user.
The root-owned authorized-key path prevents PR code from changing this key's
restrictions through the account's home directory. It does not prevent code
already running as `runner` from modifying other account state. The entry stores
private jobs under `/home/runner/.local/share/jolt-bench/jobs/`; no persistent
benchmark service unit is installed. Linger keeps the runner user manager
available without an interactive login. The entry sets
`XDG_RUNTIME_DIR=/run/user/<runner uid>`. From the dispatcher host, use the
dedicated key to run `status preflight` through the forced command: it must return
`absent` without a user-manager connection error. That status path calls
`systemctl --user`, so an interactive-shell-only success is insufficient.

Install the build and run prerequisites above and uv through the normal host
operator process. Confirm the fixed `znver4` ISA and physical CPU topology. Run
the bootstrap acceptance steps above before ordinary enablement. The saved manual
null experiment does not satisfy that requirement.

### Enablement and result publication

After those prerequisites, a repository maintainer sets `JOLT_E2E_ENABLED=true`.
The daemon operator configures the App dispatcher for `a16z/jolt`, host
`jolt-bench`, SSH user `runner`, the dedicated private key and pinned host key,
and a public base URL for immutable result downloads. App registration permissions
and subscribed events stay unchanged. This source change installs nothing and
sets no live variable or daemon setting.

The App creates **End-to-end benchmark** on the PR head or measured main commit,
queued at admission and in progress at submission. It fetches the finished
result ZIP, excluding `runtime-cache` and build payloads but retaining build logs, and serves it for 14 days at
`<public_base_url>/github-bench/<sha256>.zip`. It dispatches
`e2e-bench-ingest.yml` on main with the job, producer run/attempt, URL, digest and
executor exit code. The ingest workflow accepts only `0xandoroid-ai-agent[bot]`
on main. It caps the download at 1 GiB, verifies SHA-256 before opening the ZIP,
and treats all downloaded files as data.

The default branch's `publication.py` validates provenance, planned revisions,
build inputs, matrix, sampling, physical affinity and successful metrics. The
plan and dispatch come from that producer attempt's own artifacts. Main/nightly
manifest and index digests come from the hosted dispatch. PR manifests are checked
against the plan's pinned native inputs; their binary hashes are target observations,
not independent hosted-build attestations. The ingest run publishes canonical
`jolt-e2e-INGEST_RUN_ID-ATTEMPT` results, including `ingest.json` binding them to
the original plan, and `jolt-e2e-check-INGEST_RUN_ID-ATTEMPT/check.json`.

The App completes its check with validated Markdown. `success` means complete
validated results; `failure` means execution or validation failed; `neutral`
means partial results or unavailable comparisons. A complete PR without an
authenticated matching reference is neutral. Measured
deltas and calibration context never produce a regression verdict. An ingest
failure or missing/invalid `check.json` fails the check with the ingest link.
Bootstrap checks succeed with the validated reference SHA, or fail with the
reference-unavailable reason; executor failures remain failures. Published
bootstrap acceptance then requires all 17 probe bands.
PR-tier ingests create or update one sticky `github-actions[bot]` comment linking
the full table in the ingest attempt's job summary and the history dashboard.
Only cells with more than 3% change in either time or peak RSS are listed, using
the check summary's values and rounding, independently of the noise references.
Validated failed or partial PR results replace the comment with
a comparison-unavailable message. A failed ingest leaves the previous comment
unchanged. Comment-posting errors do not fail ingest or the benchmark check.
Executor failures before a result and failed ingests leave no history record;
their App check or hosted workflow failure is the visible outcome.

The history publisher collects the ingest workflow's canonical artifacts, keeps PR comparisons separate, and appends only
under `dev/prover-history/` on the existing `gh-pages` source. It never modifies
`dev/bench/`. Nightly remains at 07:00 UTC. An interrupted or missing result is
unavailable, never a pass.

The measured pilot ran each of the 60 supported nightly cases once. It does not establish the full
nightly suite's duration, variation between runs, or completion within 180
minutes. Hosted-job and queue overhead remain unmeasured; no finish guarantee
is established.

Rerun all jobs together: a plan from an older run attempt is rejected.

Focused checks, without proof timings:

```bash
uv run --no-project python -m unittest \
  scripts.tests.test_e2e_bench scripts.tests.test_e2e_ci \
  scripts.tests.test_e2e_publication scripts.tests.test_e2e_entry \
  scripts.tests.test_e2e_ingest -v
```
