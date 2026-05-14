# Jolt-on-Bolt Equivalence Gates

The first full-field, non-zk Jolt-on-Bolt implementation is in equivalence,
hardening, and perf-gating mode. The active objective is in
`crates/jolt-equivalence/GOAL.md`: keep `jolt-equivalence` as a thin oracle and
gate suite while semantic construction lives in Bolt, generated crates,
`jolt-kernels`, or `jolt-witness`.

## Fast Local Gates

Set up and source the Bolt dev environment first:

```bash
scripts/setup-bolt-dev.sh
source .bolt-dev-env
```

Run:

```bash
cargo fmt --check
cargo check -p bolt -p jolt-verifier -p jolt-prover -p jolt-equivalence --quiet
cargo nextest run -p bolt --test verifier_cleanup --no-capture
cargo nextest run -p bolt --test commitment_ir --cargo-quiet
cargo nextest run -p jolt-equivalence --test generated_role_crates --cargo-quiet
```

`commitment_ir` can also materialize ignored MLIR/Rust scratch fixtures for
local inspection:

```bash
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt --test commitment_ir --cargo-quiet
```

These gates cover:

- MLIR dialect registration and schema validation.
- Concrete transcript threading.
- Prover/verifier party projection.
- `compute` and `cpu` schema validation.
- Prover-only kernel resolution.
- Kernel-free verifier CPU IR.
- Generated Rust compilation.
- Generated role-crate layout and import boundaries.
- Matching generated stage registries.
- Generated verifier LOC, duplication, relation-string, and boundary metrics.

## Real-Data Equivalence Gate

Run:

```bash
cargo nextest run -p jolt-equivalence --test bolt_commitment --no-capture
```

This is the main semantic oracle. It should continue to prove:

- Bolt verifier accepts Bolt proof artifacts on real trace data.
- Core accepts the corresponding Bolt proof path.
- Bolt/core transcript histories match.
- Bolt/core observable proof artifacts match.
- Generated standalone and top-level verifier paths agree.
- Representative tampering is rejected by the generated verifier.

The `Bolt equivalence` workflow runs the generated role parity and real-data
tamper gates on pull requests, including stacked modular PRs whose base is not
`main`. It also has an optional full
`jolt-equivalence` sweep that runs on the nightly schedule, or manually through
`workflow_dispatch` with `include_full_sweep=true`:

```bash
cargo nextest run -p jolt-equivalence --cargo-quiet
```

## Required Hardening Coverage

The verifier hardening suite should cover these negative cases:

```text
tampered commitment
missing commitment
tampered sumcheck coefficient
tampered sumcheck point
tampered named eval
missing stage proof
reordered stage proof
stage proof in wrong slot
wrong transcript state
missing opening claim
extra opening claim
opening claims in wrong order
opening equality mismatch
wrong evaluation proof
missing evaluation setup
missing evaluation proof
PCS proof mismatch
```

The MLIR/compiler hardening suite should cover:

```text
unknown dialect rejection
prover-only op rejection in verifier pipeline
verifier-only op rejection in prover pipeline
unthreaded transcript op rejection
hidden/reordered opening batch claim rejection
unsupported equality mode rejection
duplicate proof slot rejection
invalid point arity rejection
invalid round schedule rejection
invalid relation kind rejection
kernel attr rejection in verifier CPU IR
forbidden generated verifier imports
```

## LOC And Readability Gates

Track these metrics before and after each cleanup iteration:

```text
total generated jolt-verifier LOC
verifier.rs LOC
stage6 + stage7 generated LOC
number of stage-local generic plan structs
number of stage-local helper/interpreter functions
number of field-expression operand constants
number of relation string-dispatch sites
forbidden imports
```

Targets:

```text
generated verifier surface:        <= 4k-6k LOC
stretch generated surface:         <= 2k-3k LOC
verifier.rs orchestration:         <= 350-500 LOC
stage6 + stage7 generated surface: <= 2k-3k LOC
```

Do not accept a LOC reduction that hides semantics in opaque runtime code. The
generated surface should shrink because generic mechanics moved into named,
reviewable runtime modules and the remaining generated code became declarative
plan data.

## Regeneration Gate

Checked-in generated role crates must stay synchronized with the artifact rail:

```bash
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules --cargo-quiet
```

After regenerating, rerun the fast local gates and the real-data equivalence
gate.

## Perf Oracle Guard

New Jolt-on-Bolt changes should preserve a core-vs-Bolt perf oracle that uses
`jolt-profiling` as the shared instrumentation layer. The gate should run the
same program, inputs, trace length, PCS setup size, and transcript mode through:

```text
core reference path:
  setup, prove, verify, proof size, peak RSS

Bolt generated path:
  setup, prove, verify, proof size, peak RSS
```

Both paths must emit the same named tracing spans through `jolt-profiling`, at
minimum:

```text
core.setup
core.prove
core.verify
bolt.setup
bolt.prove
bolt.commitment
bolt.commitment.batch
bolt.commitment.dory_commit
bolt.stage1 ... bolt.stage8
bolt.evaluate
bolt.evaluate.claims
bolt.evaluate.materialize_joint_polynomial
bolt.evaluate.joint_opening_hint
bolt.evaluate.dory_open
bolt.verify
bolt.verify.evaluation_state
bolt.verify.dory_verify
```

The checked-in CI smoke programs are:

```text
PR gate: bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle
PR gate: bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle
```

Both tests live in `jolt-equivalence/tests/bolt_perf.rs` because they reuse the
real semantic oracle fixture and pass paired `PerfMetrics` into the sampled
core-vs-Bolt gate. The workflow sets `JOLT_BOLT_PERF_TRACE=1` so the same run
writes Perfetto JSON traces under `benchmark-runs/perfetto_traces/`. It also
sets `JOLT_BOLT_PERF_SAMPLES=3`; sampled runs gate `prove_ms` against the 1.3x
target by failing only when the 95% confidence interval is fully above the
threshold. The perf workflow runs on pull requests, including stacked modular
PRs, so core-vs-Bolt regressions gate before the stack lands in `main`.

To run them locally after `source .bolt-dev-env`:

```bash
JOLT_BOLT_PERF_TRACE=1 JOLT_BOLT_PERF_SAMPLES=3 cargo nextest run -p jolt-equivalence --test bolt_perf --release --cargo-quiet --run-ignored only --no-capture bolt_sha2_chain_2_16_core_vs_bolt_perf_oracle
JOLT_BOLT_PERF_TRACE=1 JOLT_BOLT_PERF_SAMPLES=3 cargo nextest run -p jolt-equivalence --test bolt_perf --release --cargo-quiet --run-ignored only --no-capture bolt_sha2_chain_2_20_core_vs_bolt_perf_oracle
```
