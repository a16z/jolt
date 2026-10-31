# Reproducible CPU proof benchmarks

These scripts reuse `jolt-prover profile --backend optimized`. They build real
guest ELFs before measurement, run fresh prover processes, verify every successful
proof, and retain raw samples. There is no reference-backend timing lane.

**The sizes and repetition counts are proposals, not a calibrated performance
gate.** Only a full target-host `suite` run can establish the PR budget: less than
600 seconds including build and preparation, with a 300–420 second aim. A
prebuilt run or a build on another host cannot establish that budget.

## Workloads and tiers

`suites.json` is the source of the case matrix. Both Dory and Akita run at exactly
1, 8, or 24 physical cores. The ordinary instruction profile and clear proofs are
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
| `pr` | Fibonacci at 1 core, SHA2 at 8, BTreeMap at 24; both PCS: six points | 1 / 3 |
| `main` | All five workloads, all three core counts, both PCS: 30 points | 1 / 3 |
| `nightly` | Two larger inputs for each main point: 60 points | 1 / 5 |

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
Individual build manifests retain their own elapsed intervals and host identity.

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

## Manual commands

Run on Linux with Python 3.11+ through `uv`, Git, `taskset`, `lscpu`, the pinned
Rust toolchain and its `rust-src` component, and the RISC-V compiler required by
the repository's guest builder. The build produces its own `jolt` CLI; a stale
globally installed CLI is not used. Cargo targets stay inside private source
checkouts under this checkout's `target/e2e-builds/`.

The harness revision must be committed, and `run.py` must be that revision's
file. Both source revisions receive the same benchmark-only profile harness and
direct guest dependencies. Existing locked package identities may not change.
Build manifests distinguish source revision, harness revision, original lockfile,
effective lockfile, and all overlaid files. A same-source null run uses identical
bundles, not differently instrumented main-vs-main binaries.

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
the default is `release`. Paired legs must use the same profile and flags.

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

For a build-inclusive PR dry run, use an exact current-main base, exact candidate
merge revision, and one committed harness:

```bash
uv run --no-project python scripts/bench/run.py suite \
  --tier pr --base-revision "$BASE" --head-revision "$MERGE" \
  --harness-revision "$HARNESS" --target-cpu "$TARGET_CPU" \
  --out "$BENCH_OUT/pr-dry-run"
```

The runner never builds or measures on another host implicitly. Manual JSON uses
null GitHub provenance. Never compare a historical run on another host with a PR.
To render a comment without posting:

```bash
uv run --no-project python scripts/bench/run.py render-comment \
  --results "$BENCH_OUT/pr-dry-run/benchmark-results.json" \
  --out "$BENCH_OUT/pr-comment.md"
```

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
`JOLT_E2E_ENABLED` unset until all of the following are satisfied:

1. Use a dedicated Linux runner account and the observed target machine. That
   account must have no write-capable GitHub, SSH, cloud, or Pika credentials in
   its environment, home, helpers, or accessible files. Do not use a general
   development account. Confirm CPU topology/ISA and host storage readiness.
2. Put the runner in organization runner group `jolt-bench`, label `jolt-bench`.
   Restrict that group to **only**
   `a16z/jolt/.github/workflows/e2e-bench.yml@refs/heads/main`. Read back the
   group's `restricted_to_workflows` and `selected_workflows` settings. If this
   restriction is unavailable for the account, leave execution disabled: a
   PR-editable YAML `if` is not a security boundary. GitHub documents these
   fields in its [runner-group API](https://docs.github.com/en/rest/actions/self-hosted-runner-groups#update-a-self-hosted-runner-group-for-an-organization).
3. Install the repository's build prerequisites and `uv` through the normal
   administrator process. Set `JOLT_E2E_TARGET_CPU` only after observing the
   target ISA. Complete a real null pilot and a full fresh build-inclusive PR
   dry run, calibrating sizes/repetitions without changing protocol parameters.
4. Enable `JOLT_E2E_ENABLED=true`. Inspect the first run's complete artifact,
   every expected point, physical affinity, hashes, verified exits, and measured
   total. Target measurements, not this document, determine readiness.
5. After reviewing trusted publication separately, optionally set
   `JOLT_E2E_COMMENTS_ENABLED=true`. Until then, comments are rendered artifacts
   only. This task does not enable or post them.

The hosted PR request workflow runs on code PRs and skips docs-only changes. A
default-branch `workflow_run` producer validates the request against GitHub,
rejects forks, and resolves the PR merge ref's exact current-main parent. No
`pull_request_target` workflow executes PR code. The persistent measurement job
has only read permissions; subprocess environments exclude tokens. The runner's
non-secret `RUNNER_TRACKING_ID` is retained for cancellation cleanup. Trusted
checkouts are cleaned normally; correctness does not depend on a retained source
tree. Cargo cache reuse is recorded, never assumed or required for a time claim.

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
serialize execution. Nightly starts at 09:00 UTC (05:00 EDT / 04:00 EST); finishing
before ET work hours still requires measured suite and queue durations. Rerun
all jobs together: a plan from an older run attempt is rejected.

Focused checks, without proof timings:

```bash
uv run --no-project python -m unittest \
  scripts.tests.test_e2e_bench scripts.tests.test_e2e_ci \
  scripts.tests.test_e2e_publication -v
```

Process fixtures own and reap their children. They await completed readiness
records rather than scheduling sleeps or host-wide PID assumptions. Rust test
changes must follow the same ownership rule; Tokio timer tests use paused time.
