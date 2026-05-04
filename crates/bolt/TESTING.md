# Bolt Verifier Cleanup Gates

The first full-field, non-zk Jolt-on-Bolt implementation is in cleanup and
hardening mode. The active objective is in `GOAL.md`: make the generated
`jolt-verifier` compact, readable, auditable, and security-hardened while
preserving current semantics.

## Fast Local Gates

Run:

```bash
cargo fmt --check
cargo check -p bolt -p jolt-verifier -p jolt-prover -p jolt-equivalence --quiet
cargo test -p bolt --test verifier_cleanup -- --nocapture
cargo test -p bolt --test commitment_ir --quiet
cargo test -p jolt-equivalence --test generated_role_crates --quiet
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
cargo test -p jolt-equivalence --test bolt_commitment -- --nocapture
```

This is the main semantic oracle. It should continue to prove:

- Bolt verifier accepts Bolt proof artifacts on real trace data.
- Core accepts the corresponding Bolt proof path.
- Bolt/core transcript histories match.
- Bolt/core observable proof artifacts match.
- Generated standalone and top-level verifier paths agree.
- Representative tampering is rejected by the generated verifier.

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

## Perf Regression Guard

Verifier cleanup should not move prover performance, but keep the e2e
`sha2-chain` gate available as a regression check:

```bash
cargo run --release -p jolt-bench --bin bolt-stage -- \
  --program sha2-chain --stage stage8 --log-t 20 \
  --num-iters 1 --iters 1 --warmup 0 \
  --trace-output benchmark-runs/perfetto_traces/sha2_chain_e2e_2p20_bolt.json
```

Use the existing core-vs-Bolt Perfetto traces for inspection when cleanup
changes touch shared prover/verifier orchestration.
