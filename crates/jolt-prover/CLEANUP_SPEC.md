# jolt-prover Readability Cleanup Spec

## Goal

`jolt-prover` should read like a direct proof recipe. It should coordinate
checked inputs, preprocessing, witness providers, backend kernels, transcript
ordering, verifier stage contracts, PCS openings, and BlindFold. It should not
look like an independent protocol implementation.

The new priority is readability first, then LOC reduction. A smaller file that
is still hard to follow is not done.

## Current Problem

The largest files, especially `prover.rs` and several stage `prove.rs` files,
are too verbose and mix too many jobs in one body:

- dependency validation,
- witness row extraction,
- backend request construction,
- sumcheck round loops,
- clear/ZK transcript recording,
- field-inline branching,
- output-opening evaluation,
- expected-output checks,
- verifier output construction,
- final proof packaging.

That makes local cleanup easy to verify but hard to read. Future work should
make the proof flow obvious from the top of each file.

## Design Rule

Each stage should be organized around the reader's question:

1. What does this stage need?
2. What does it ask the backend or witness for?
3. What proof component does it produce?
4. What verifier-owned output does it hand to later stages?

If a helper does not make one of those answers clearer, it should not exist in
`jolt-prover`.

## Standard Stage Layout

Each stage directory should use more than one file when the stage has enough
surface area to make a single `prove.rs` hard to audit. Large monolithic
stage files are explicitly not the target.

```text
stageN/
  mod.rs
  io.rs                 # stage input/output/prepared/batch data structures
  prove.rs              # entrypoints and high-level stage recipe, about 800 LOC
  prepare.rs            # dependency checks, witness access, backend requests
  batch.rs              # temporary local sumcheck runner, until moved out
  verifier_output.rs    # temporary verifier-output adapter, until verifier-owned
```

Rules:

- `prove.rs` should be readable top to bottom and should generally stay near
  800 LOC or less. Exceeding 800 LOC is a review smell unless the extra code is
  visibly linear orchestration.
- `io.rs` owns the stage-local data model: inputs, prepared state, proof
  components, commitment components, retained prover state, and batch outputs.
- `io.rs` should mostly be structs and small constructors. It should not contain
  protocol formulas, backend request construction, or sumcheck loops.
- `prepare.rs` should not contain sumcheck round logic.
- `batch.rs` should not construct final verifier outputs.
- `verifier_output.rs` is temporary. Prefer moving this construction to
  `jolt-verifier`.
- Do not recreate `assembly.rs`, `builder.rs`, `proof.rs`, `commitments.rs`,
  `transcript.rs`, `timing.rs`, or `bounds.rs`.
- Do not add a replacement module that only collects generic trait bounds; keep
  caller-facing API constraints next to the public entrypoints.
- Do not add files that only wrap one or two type aliases.

## Preferred Stage Flow

Every stage should converge toward this visible structure in `prove.rs`:

```rust
pub(crate) fn prove(...) -> Result<StageNProofComponent<...>, ProverError> {
    validate_mode_and_inputs(...)?;
    let prepared = prepare_stageN(...)?;
    let batch = prove_stageN_batch(prepared.batch_input(), recorder)?;
    let verifier_output = stageN_verifier_output(...)?;
    Ok(stageN_proof_component(...))
}
```

For ZK mode:

```rust
pub(crate) fn prove_commitment_component(...) -> Result<StageNCommitmentComponent<...>, ProverError> {
    validate_mode_and_inputs(...)?;
    let prepared = prepare_stageN(...)?;
    let batch = prove_stageN_batch(prepared.batch_input(), committed_recorder)?;
    let verifier_output = stageN_verifier_output(...)?;
    Ok(stageN_commitment_component(...))
}
```

The top-level flow should not hide protocol ordering behind a generic object.
It should be explicit and short.

Mode-specific entrypoints should be small wrappers over the same preparation and
batch machinery. A stage should not have one 250-line clear function and one
250-line committed function with the same loop.

## Naming

Use names that describe the role of the data.

Preferred:

- `StageNInput`
- `StageNPrepared`
- `StageNBatch`
- `StageNProofComponent`
- `StageNCommitmentComponent`
- `StageNClaims`
- `StageNVerifierOutputInput`
- `StageNBatchRequest`
- `StageNBatchOutput`
- `StageNProofRecorder`

Avoid:

- `ProofArtifact`
- `CommittedStageOutput`
- `Boundary`
- `Assembly`
- `Builder`
- `Sink`
- `prove_with_output`
- `shape`

If the name needs "temporary" or "helper" to explain it, it probably belongs in
another crate or should be folded into the call site.

## Public Surface

`jolt-prover` should export only the user-facing proving surface:

- prover configuration,
- preprocessing bundle,
- primary prove entrypoints,
- proof result,
- backend trait alias or adapter needed by callers,
- error type.

Stage modules should remain private implementation details unless another crate
has a concrete reason to depend on them.

## Main Driver Target

`prover.rs` should become a linear recipe:

```rust
let checked = validate_inputs(...)?;
let stage0 = stage0::prove(...)?;
let mut transcript = initialize_transcript(...)?;

let stage1 = stage1::prove(...)?;
let stage2 = stage2::prove(...)?;
let stage3 = stage3::prove(...)?;
let stage4 = stage4::prove(...)?;
let stage5 = stage5::prove(...)?;
let stage6 = stage6::prove(...)?;
let stage7 = stage7::prove(...)?;
let stage8 = stage8::prove(...)?;

assemble_result(...)
```

Target size: 500-900 LOC.

Allowed in `prover.rs`:

- config-derived parameter calculation,
- mode routing,
- stage sequencing,
- final proof/result packaging.

Not allowed in `prover.rs`:

- stage algebra,
- backend request construction,
- transcript preamble internals,
- verifier output reconstruction details,
- large ZK witness assignment.

## Per-Stage File Template

Use this template unless a stage has a specific reason to deviate.

```text
stageN/
  mod.rs
  io.rs
  prove.rs
  prepare.rs
  batch.rs
  verifier_output.rs
```

`mod.rs`:

- declare modules,
- expose only `prove`, `prove_commitment_component`, and the stage component
  types needed by `prover.rs`,
- no protocol logic.

`io.rs`:

- `StageNInput`,
- `StageNPrepared`,
- `StageNBatchInput`,
- `StageNBatchOutput`,
- `StageNProofComponent`,
- `StageNCommitmentComponent`,
- retained prover-only state.

`prove.rs`:

- validate mode and checked inputs,
- call `prepare`,
- call `batch`,
- call verifier-output assembly,
- return proof or commitment component.

`prepare.rs`:

- dependency validation calls into `jolt-verifier`,
- witness row access,
- backend request construction through `jolt-backends`,
- no sumcheck loop.

`batch.rs`:

- local sumcheck runner while the runner has not moved to an owner crate,
- proof recorder abstraction for clear/ZK,
- no final verifier-output construction.

`verifier_output.rs`:

- temporary adapter from batch output to verifier-owned output,
- should shrink as `jolt-verifier` gains direct constructors.

## Stage Size Targets

These are review signals, not hard quotas. The target for each `prove.rs` is
about 800 LOC or less. A stage can exceed its target only when the ownership
reason is explicit and the file remains easy to scan.

| Stage | `prove.rs` target | Main remaining pressure |
|---|---:|---|
| Stage 0 | 300-500 | Commitment request/result normalization |
| Stage 1 | 500-800 | Spartan row algebra and clear/ZK duplication |
| Stage 2 | 500-800 | Product/remainder plumbing and field-inline extension setup |
| Stage 3 | 500-800 | Batch runner and opening conversion |
| Stage 4 | 500-800 | RAM init/advice plumbing and remaining batch mechanics |
| Stage 5 | 500-800 | Specialized batch runner and field-register value plumbing |
| Stage 6 | 700-1000 | Bytecode, booleanity, RA, increment, advice cycle-phase flow |
| Stage 7 | 400-700 | Hamming/advice batch internals |
| Stage 8 | 300-500 | Final opening orchestration only |

For the whole stage directory, additional LOC in `io.rs`, `prepare.rs`,
`batch.rs`, or `verifier_output.rs` is acceptable when it makes ownership and
navigation clearer. The point is not to hide code; it is to make each file have
one obvious job.

## Ownership Rules

Move computation to the crate that owns the concept.

- `jolt-verifier` owns verifier transcript order, verifier outputs, expected
  output formulas, and stage dependency contracts.
- `jolt-claims` owns shared claim semantics, dimensions, address/chunk
  derivation, and formula-level helpers.
- `jolt-backends` owns backend relation ids, request construction, row
  projection, state materialization contracts, and backend result validation.
- `jolt-witness` owns witness row access and witness-specific row projections.
- `jolt-prover` owns orchestration, mode selection, transcript sequencing calls,
  PCS opening calls, and final result packaging.

Use owner crates aggressively:

- Prefer verifier-owned dependency and output objects over duplicate prover
  structs.
- Prefer `jolt-claims` dimensions and formula helpers over local arithmetic.
- Prefer backend-owned request constructors and row projections over local
  request assembly.
- Prefer witness-owned row accessors over reinterpreting witness data locally.

When code stays in `jolt-prover`, document whether it is:

- orchestration that belongs here,
- temporary adapter waiting for an owner crate,
- prover-only retained data such as hidden evaluations or blindings.

## Clear and ZK Paths

Clear and ZK paths should share stage preparation and sumcheck mechanics.
Separate only the proof recording behavior:

- clear mode records input claims and transparent round polynomials,
- ZK mode records committed rounds and hidden output-claim values,
- both modes should produce the same verifier-facing stage output.

Local recorder traits such as `StageNProofRecorder` are acceptable while they
remove duplicate stage loops. They should stay small and private.

This should mirror the `jolt-verifier` style: common stage semantics first,
mode-specific proof material second. If clear and committed paths differ only in
how rounds are recorded, the loop must be shared.

## Field-Inline Paths

Field-inline should be represented as a narrow extension bundle, not a second
copy of the stage. Follow the `jolt-verifier` approach: a common base request or
prepared value, plus field-inline extension data where needed.

Preferred pattern:

```rust
let prepared = prepare_stageN(base_input, extension)?;
let batch = prove_stageN_batch(prepared, recorder)?;
```

Avoid parallel function bodies where only one extra field-register instance is
added. If the field-inline path needs extra state, put that state in a small
typed extension value.

The target is one readable flow:

```rust
let extension = StageNExtension::from_field_inline(...)?;
let prepared = prepare_stageN(input, extension)?;
let batch = prove_stageN_batch(prepared.batch_input(), recorder)?;
```

Not four independent bodies for:

- clear,
- ZK,
- field-inline,
- ZK field-inline.

## Testing Gate

No smoke-test-only confidence. The canonical prover gate is real e2e data:

- `muldiv`
- `sha2-chain`

### Microbenchmarks

Keep a lightweight bench suite under `crates/jolt-prover/benches/` for quick
proof-system feedback during cleanup. These are not correctness replacements
for nextest; every bench iteration must still produce a real proof and run the
verifier on real guest data.

Initial suite:

- `sha2-chain` with modular prover target padded trace length `2^16`
- `sha2-chain` with modular prover target padded trace length `2^20`, gated
- default sample count 3, override with `JOLT_PROVER_MICROBENCH_SAMPLES=N`
- derive cycles per input from tracer calibration and scale `sha2-chain`
  iterations toward 90% of the requested padded trace length
- choose the largest calibrated input count that stays under the target
  unpadded cycle count and lands in the requested padded trace-length tier
- print the selected iteration count, target unpadded cycles, calibration
  cycles per input, selected unpadded cycles, and padded trace length
- setup, guest tracing, preprocessing, and witness construction outside the
  measured loop
- measured loop is proof plus verifier acceptance

The `2^20` case must not become the default feedback cost until `2^16` looks
healthy. Run it automatically only when the default transparent run has
`jolt-prover` within 15% of legacy `jolt-core` on the `2^16` case. For manual
large-case investigation, force it with `JOLT_PROVER_MICROBENCH_RUN_2_20=1`.

Comparison model:

- default features: `jolt-prover` transparent vs legacy `jolt-core` transparent
- `zk`: `jolt-prover` ZK proof plus verify
- `field-inline`: `jolt-prover` field-inline overhead vs default modular prover
- `zk,field-inline`: combined ZK plus field-inline overhead
- no field-inline comparison against `jolt-core`, because legacy core does not
  provide that path

Useful commands:

```bash
cargo bench -p jolt-prover --bench e2e_micro
cargo bench -p jolt-prover --bench e2e_micro --features zk
cargo bench -p jolt-prover --bench e2e_micro --features field-inline
cargo bench -p jolt-prover --bench e2e_micro --features zk,field-inline
JOLT_PROVER_MICROBENCH_RUN_2_20=1 cargo bench -p jolt-prover --bench e2e_micro
```

Baseline from June 9, 2026, local bench profile, `samples=3`:

Input calibration for `sha2-chain-2^16`:

- selected iterations: 18
- target unpadded cycles: 58982
- calibration base cycles: 1084.0
- calibration cycles per input: 3064.0
- unpadded cycles: 56236
- padded trace length: 65536

| Mode | Command | Min | Median | Mean |
| --- | --- | ---: | ---: | ---: |
| `jolt-prover` transparent | `cargo bench -p jolt-prover --bench e2e_micro` | 2.749s | 2.752s | 2.753s |
| legacy `jolt-core` transparent | `cargo bench -p jolt-prover --bench e2e_micro` | 2.085s | 2.127s | 2.129s |
| `jolt-prover` ZK | `cargo bench -p jolt-prover --bench e2e_micro --features zk` | 3.325s | 3.430s | 3.397s |
| `jolt-prover` field-inline | `cargo bench -p jolt-prover --bench e2e_micro --features field-inline` | 2.782s | 2.800s | 2.862s |
| `jolt-prover` ZK field-inline | `cargo bench -p jolt-prover --bench e2e_micro --features zk,field-inline` | 3.766s | 3.776s | 3.777s |

The transparent modular/core mean ratio was `1.293`, so `sha2-chain-2^20`
was skipped by the 15% gate. Treat forced `2^20` runs as investigation until
the `2^16` ratio is at or below `1.15`.

For any stage-flow rewrite, run:

```bash
cargo fmt -q
cargo clippy --no-deps -p jolt-prover --all-targets -q -- -D warnings
cargo clippy --no-deps -p jolt-prover --all-targets --features zk -q -- -D warnings
cargo clippy --no-deps -p jolt-prover --all-targets --features field-inline -q -- -D warnings
cargo clippy --no-deps -p jolt-prover --all-targets --features zk,field-inline -q -- -D warnings
cargo nextest run -p jolt-prover --cargo-quiet
cargo nextest run -p jolt-prover --cargo-quiet --features zk
cargo nextest run -p jolt-prover --cargo-quiet --features field-inline
cargo nextest run -p jolt-prover --cargo-quiet --features zk,field-inline
git diff --check
```

If a slice touches `jolt-verifier`, `jolt-backends`, `jolt-claims`, or
`jolt-witness`, include the touched crates in clippy and nextest.

## Current Progress

- Phase 0 is in place: the readability contract exists, true e2e tests exist,
  and the microbench harness records the `sha2-chain-2^16` feature matrix.
- Phase 1 driver cleanup is in target range. `prover.rs` now delegates BlindFold row-committer
  proof generation to `zk.rs`, and Stage 4 RAM value-check initial-evaluation
  preparation lives in `stages/stage4/prepare.rs`.
- Checked-input construction is now verifier-owned through
  `jolt_verifier::validate_inputs_from_parts`; `prover.rs` only passes the
  proof parameters and mode.
- Stage 6, Stage 7, and Stage 8 prover-config assembly now lives in each
  stage's `prepare.rs` module.
- Stage 0 prover-config assembly now lives in `stages/stage0/prepare.rs`.
- Public prove API wrappers and caller-facing proving requirements now live in
  `api.rs`, outside the driver recipe.
- Prover config validation now lives with `ProverConfig` in `config.rs`.
- Clear and ZK proof-material packaging now uses small stage-payload helpers,
  so the recipes are less interrupted by struct assembly.
- Current `prover.rs` size after these slices is 869 LOC. The duplicated
  clear/ZK stage call sequence remains explicit while stage-local flow is split.
- Phase 2 has started with Stage 6: config/input/component/batch data types now
  live in `stages/stage6/io.rs`, and clear/ZK proof recording now lives in
  `stages/stage6/batch.rs`. Stage 6 prefix derivation, bytecode request inputs,
  and backend state materialization now live in `stages/stage6/prepare.rs`.
  Backend-output claim packaging lives in `stages/stage6/verifier_output.rs`.
  The local batch context and verifier-aligned evaluation helpers also moved to
  `stages/stage6/batch.rs`, bringing `stage6/prove.rs` to 567 LOC. The next
  Stage 6 pressure point is reducing the temporary size of `batch.rs`.

## Implementation Phases

### Phase 0: Freeze Readability Contract

Add this spec and use it to evaluate future cleanup. Stop optimizing for small
internal refactors that leave the same giant file structure in place.

### Phase 1: Driver Recipe

Rewrite `prover.rs` until the stage sequence is obvious without scrolling
through stage-local details.

Deliverables:

- one clear proof recipe,
- one clear ZK routing strategy,
- final packaging helper under 100 LOC,
- no protocol algebra in the driver.

### Phase 2: Stage Table of Contents

For every stage, split or reorder code into:

- `io.rs` for data structures,
- `prove.rs` for entrypoints and high-level flow,
- `prepare.rs` for dependency/witness/backend setup,
- `batch.rs` for temporary local sumcheck runners,
- `verifier_output.rs` for temporary verifier-output adapters.

The first screen of each `prove.rs` should tell the reader what the stage does,
and `prove.rs` should be around 800 LOC or less.

### Phase 3: Split Only Large Responsibilities

After the table-of-contents pass, split only files that remain hard to read.

Allowed splits:

- `io.rs`
- `prepare.rs`
- `batch.rs`
- `verifier_output.rs`

Each split must remove a real navigation problem.

### Phase 4: Move Owners Out

For each temporary local adapter, move it to the owning crate:

- verifier output and expected-output helpers to `jolt-verifier`,
- claim semantics to `jolt-claims`,
- backend request construction to `jolt-backends`,
- witness row projection to `jolt-witness`.

### Phase 5: Final Condensed Path

Accept the cleanup only when:

- `prover.rs` is a readable recipe,
- every large stage file has a clear responsibility split,
- stage outputs use verifier-owned objects directly where possible,
- no forbidden modules or terms have been reintroduced,
- the full e2e feature matrix passes.

## Review Checklist

Before merging any cleanup slice:

- Does the top-level flow read more clearly?
- Did the change reduce a mixed-responsibility block?
- Are names role-based and protocol-specific?
- Did any computation stay in `jolt-prover` only for convenience?
- Are clear, ZK, and field-inline paths sharing the right mechanics?
- Are tests real e2e tests, not smoke tests?
- Is the spec updated with the new state and next pressure point?
