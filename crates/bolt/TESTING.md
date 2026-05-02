# Bolt Testing Pattern

The Bolt compiler earns correctness one protocol stage at a time. The full
stage-addition algorithm lives in `JOLT_PROTOCOL_IMPLEMENTATION.md`; this file
defines the concrete gates that make a stage acceptable.

## Stage Done Means

A stage is complete only when all of these are true on real trace data:

- **Bolt acceptance**: generated/Bolt prover artifacts are accepted by the
  generated/Bolt verifier.
- **Bolt transcript parity**: prover and verifier transcript states match
  step-for-step through the stage boundary.
- **Core acceptance**: `jolt-core` accepts the proof prefix after Bolt-produced
  artifacts are spliced into the matching core proof fields.
- **Core transcript/artifact parity**: Bolt matches `jolt-core` transcript
  states and observable proof components through the stage boundary.
- **Tamper rejection**: generated verifier rejects representative mutations for
  every new soundness obligation introduced by the stage.
- **Perf parity**: Bolt prover time for the newly added stage is within 20% of
  `jolt-core` on the agreed `sha2-chain` workload, with perf gates capped at
  three iterations.

Synthetic fixtures are allowed for early unit tests, but they do not count as
stage acceptance.

## Local Compiler Gates

Run:

```bash
cargo nextest run -p bolt --cargo-quiet
```

This verifies:

- IRDL dialect registration and parsing.
- Jolt protocol schema validation.
- Concrete transcript threading.
- Prover/verifier role projection.
- `compute` and `cpu` schema validation.
- Kernel resolution only on prover IR.
- Golden MLIR fixtures for every implemented stage.
- Generated Rust compilation.
- Canonical generated artifact layout:
  `crates/jolt-prover/src/stages/<stage>.rs` and
  `crates/jolt-verifier/src/stages/<stage>.rs`.
- Whole generated role-crate assembly and `cargo check` for `jolt-prover` and
  `jolt-verifier`.
- Checked-in generated crate source stays synchronized with
  `assemble_jolt_workspace_generated_crates` and the artifact writer can
  materialize the same layout under a `crates/` root.
- Top-level generated `prover.rs`/`verifier.rs` APIs are emitted by the same
  artifact rail. `jolt-verifier` owns proof types and must not import
  `jolt-prover`; `jolt-prover` may import verifier-owned proof types but not
  verifier stage internals.
- Generated stage registries match between prover and verifier so
  `jolt-equivalence` and `jolt-bench` can discover the implemented prefix.
- Generated verifier import policy: no `jolt-kernels`, `jolt-core`,
  `jolt-equivalence`, `jolt-bench`, or tracer internals.

## Equivalence Gates

Run:

```bash
cargo check -p jolt-equivalence --tests --quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
```

`jolt-equivalence` is the real-data oracle. Stage tests should be named by stage
and should check both internal Bolt parity and Bolt-vs-core parity. Each newly
wired stage should add or extend tests that:

- Generate the same real trace data for Bolt and core.
- Run Bolt prover and Bolt verifier through the complete implemented prefix.
- Assert Bolt proof acceptance and transcript-state parity.
- Splice Bolt artifacts into the corresponding `jolt-core` proof prefix.
- Assert `jolt-core` accepts through the same prefix.
- Assert transcript states and proof components match core through the stage.
- Mutate representative proof components and assert the generated verifier
  rejects them.

## Kernel Gates

Run:

```bash
cargo nextest run -p jolt-kernels --cargo-quiet
```

Kernel tests can use synthetic data for arithmetic-local coverage, but stage
completion still requires real-data `jolt-equivalence` coverage.

## Perf Gate

Run the stage perf oracle after correctness is green:

```bash
cargo run --release -p jolt-bench --bin bolt-stage -- \
  --program sha2-chain --stage <stage> --log-t 16 \
  --num-iters 16 --iters 3 --warmup 1 \
  --json perf/bolt-<stage>-last.json
```

Run core first for a fair timeout baseline. If Bolt exceeds 10x the core time,
stop and profile rather than waiting for a full timing run. Adding the stage to
the `bolt-stage` bench selector is part of the stage implementation if the
selector does not support it yet.

## Current Stage Status

- **Commitment**: active compiler/equivalence rails are green.
- **Stage 1**: compiler/generated-artifact rails are green; the real-data
  equivalence gate currently fails on core acceptance with
  `UniSkipVerificationError` and must be repaired before Stage 1 is marked
  complete again.
- **Stage 2**: compiler/generated-artifact rails are green; the real-data
  equivalence gate currently fails on product-uniskip/sumcheck coefficient
  parity and must be repaired before Stage 2 is marked complete again.
- **Stage 3**: protocol/codegen/arithmetic work exists; current work is
  optimization plus real-data parity, tamper hardening, and perf closure.
