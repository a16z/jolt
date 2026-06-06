# Stage 0 Commitment Frontier Spec

## Scope

Stage 0 commits all verifier-visible witness polynomials and returns the verifier
commitment component plus prover-only opening hints. It does not prove a
sumcheck, derive random challenges, or run opening proofs. The stage boundary is:

1. Consume normalized `jolt-witness` providers and verifier-derived dimensions.
2. Build hardware-neutral commitment backend requests.
3. Run an optimized backend commitment kernel.
4. Assemble `jolt-verifier` commitment/advice proof fields.
5. Hand the commitments to the Fiat-Shamir owner in the exact verifier order.

This spec is intentionally scoped to one monitored stage. Do not start Stage 1
work from this document. Stage 0 is accepted only when the code, canonical
tests, correctness evidence, and performance evidence are simple enough to
inspect without relying on broad end-to-end proof runs.

Stage 0 should land as one complete stage slice. Do not accept separate partial
completion for "transparent only", "advice later", "ZK later", or
"field-inline later". Temporary substeps are fine during development, but the
reviewable implementation slice for Stage 0 must cover every required feature
surface and include its parity justification before the stage is considered
done.

Every stage must explicitly support the active feature surfaces:

- advice, including trusted and untrusted advice where the stage observes them;
- BlindFold/ZK mode, including the mode-specific verifier proof shape and
  transcript behavior;
- field-inline, behind the `field-inline` feature and absent from disabled
  builds.

For Stage 0 this means advice commitments are first-class outputs, ZK mode
selects the PCS hiding commitment path for every committed witness polynomial
while preserving the verifier's ZK proof/config shape, and field-inline
commitments are compiled and certified only under the field-inline feature.

## Monitored Workflow

Stage work proceeds in a single reviewable Stage 0 slice:

1. Confirm current inventory for this stage: `jolt-prover` stage code,
   `jolt-backends` kernels, harness frontiers, and ledger evidence.
2. Define the clean target API and proof fields before changing code.
3. Implement all missing prover-side orchestration for this stage, including
   advice, ZK/BlindFold shape, and field-inline coverage.
4. Run the stage's canonical focused tests.
5. Run the stage's verifier replay/correctness parity checks for every enabled
   surface.
6. Run or validate the stage's performance evidence for every required backend
   kernel.
7. Stop for review before moving to the next stage.

Every stage spec should make it obvious which facts come from:

- verifier-owned proof types in `jolt-verifier`;
- semantic identifiers, dimensions, and formulas in `jolt-claims`;
- witness views in `jolt-witness`;
- heavy compute requests/results in `jolt-backends`;
- temporary migration evidence in `jolt-prover-harness`.

The stage implementation should look like the prover analogue of the verifier:
linear, explicit, and easy to audit. Prefer a few direct steps over clever
abstractions while the migration is establishing patterns.

## Current Inventory

### jolt-prover

Current implementation lives in `crates/jolt-prover/src/stages/stage0/`.

- `input.rs` defines `CommitmentStageConfig` with the RA layout and advice
  presence flags.
- `request.rs` builds a `CommitmentRequest` from
  `CommittedWitnessProvider::committed_oracle_order`, preserving slot order and
  requesting streaming materialization.
- `prove.rs` owns the single public stage entry point, `prove`. Under
  `field-inline`, the same entry point runs the field-inline namespace
  commitment request after the Jolt VM request.
- `output.rs` assembles `jolt_verifier::proof::JoltCommitments`, optional advice
  commitments, and `CommitmentStageProverState` opening hints keyed by verifier
  committed-polynomial IDs. It uses deterministic `BTreeMap` state and a shared
  commitment collection path for Jolt VM and field-inline backend results.
- `transcript.rs` has `Stage0TranscriptContext` and
  `absorb_stage0_transcript` for verifier-order commitment absorption.

The current code is stage-shaped, verifier-type-oriented, and exposes the
Stage 0 transcript boundary later stages should consume without rebuilding a
whole proof just to absorb commitments.

### jolt-backends

Relevant commitment infrastructure is in `crates/jolt-backends/`.

- `CommitmentRequest` and `CommitmentRequestItem` carry slot-keyed
  `ViewRequirement`s plus an explicit `CommitmentMode`.
- `CpuBackend` implements `CommitmentBackend` for streaming commitments.
- `cpu/commitments/stream.rs` supports dense, compact integer, and one-hot
  witness streams. `CommitmentMode::Zk` routes dense/compact streams through
  `ZkStreamingCommitment::finish_zk_with_hint` and one-hot materialization
  through `ZkOpeningScheme::commit_zk`.
- `CpuBackendConfig::preserve_core_fast_path` infers the core row layout for
  through-stage8 dense/compact and one-hot commitments.

Certified backend kernels:

- `cpu_streaming_commitments`: covers `OPT-COM-001`, `OPT-COM-006`.
- `cpu_advice_commitment_contexts`: covers `OPT-COM-005`, `OPT-COM-006`.
- `cpu_field_inline_commitments`: covers `OPT-COM-001`, `OPT-COM-006`.
- `cpu_one_hot_commitments`: covers `OPT-COM-007`, `OPT-COM-008`; useful
  backend evidence for one-hot behavior, though not the named Stage 0 frontier
  kernel.

Required ongoing guardrail:

- ZK/BlindFold Stage 0 coverage must be kept in sync with the proof shell,
  verifier input validation, and the Dory hiding commitment path. Stage 0 does
  not run hidden sumchecks, but its commitments must be compatible with later
  ZK opening proofs.

### Harness State

Current frontier manifest entries:

- `stage0_commitments`: transparent `muldiv` fixture, backend port
  `cpu_streaming_commitments`.
- `stage0_advice_commitments`: transparent advice fixture, backend port
  `cpu_advice_commitment_contexts`.
- `stage0_field_inline_commitments`: field-inline fixture, backend port
  `cpu_field_inline_commitments`.

Current harness tests cover:

- `validate_frontier_replacement_ready` for transparent Stage 0 commitments.
- `validate_frontier_replacement_ready` for transparent advice commitments.
- `validate_frontier_replacement_ready` for field-inline Stage 0 commitments.
- verifier replay against core fixtures using native `jolt-verifier` checks.
- field-inline Stage 0 proof-shell replay through native `jolt-verifier`
  validation and transcript initialization, including verifier bytecode metadata
  absorption.
- local CPU backend tests for real one-hot RA streaming, advice commitments, and
  field-inline namespace output commitment contents.

## Clean Final Product

### Code Shape

Stage 0 should keep the local `input.rs`, `request.rs`, `prove.rs`, `output.rs`
module pattern and use the crate-level `builder.rs` for reusable verifier-proof
component assembly:

- `input.rs`: stage config and already-validated input bundle only.
- `request.rs`: translate stage input into backend request types.
- `prove.rs`: linear orchestration; no semantic reimplementation and no public
  field-inline mini-stage.
- `output.rs`: convert backend results into verifier proof components and
  prover-local hints.
- `builder.rs`: reusable proof-component assembly helpers for taking
  semantically keyed backend outputs into verifier-owned proof fields while
  retaining prover-only state.
- tests: one canonical stage test plus narrow regression tests for malformed
  backend output.

Imports should communicate ownership:

- use `jolt-verifier` for verifier proof components;
- use `jolt-claims` for committed-polynomial IDs, RA layout, dimensions, and
  protocol semantics;
- use `jolt-witness` for witness providers and view requirements;
- use `jolt-backends` for request/result traits and compute;
- keep `jolt-core` out of `jolt-prover`.

Avoid stage-local helper types that duplicate verifier proof structs or claim
semantics. Temporary harness-only adapters belong in `jolt-prover-harness`, not
in the production stage.

### Code Ordering

Order the code so a reviewer can read it top-to-bottom in prover execution
order.

Within the stage module:

1. `input.rs`: public input/config structs first, constructors second,
   validation helpers last.
2. `request.rs`: public request builder first, then small private helpers in the
   order they are called.
3. `prove.rs`: public stage entrypoint first. The body should read as:
   validate/normalize inputs, build backend request, call backend, assemble
   verifier component, retain prover-local state.
4. `output.rs`: verifier-facing output structs first, conversion from backend
   result second, and use `builder.rs` for deterministic take/finish mechanics.
5. tests: canonical happy-path test first, feature-surface tests next,
   malformed-output regression tests last.

Within functions, keep the same ordering as the prover pipeline. Avoid helpers
that force the reader to jump between semantic phases. If a helper is needed,
name it after the exact phase it performs, and keep it in the same file as the
phase owner unless it is reused by another stage.

The implementation should make ownership visually obvious:

- verifier fields are assembled in one contiguous block;
- backend calls are isolated to one contiguous block;
- Fiat-Shamir absorption is isolated and ordered exactly like verifier
  absorption;
- prover-only opening hints/state are never interleaved with verifier proof
  field construction.

### Inputs

The stage should accept a single explicit input bundle rather than scattered
arguments once the full prover orchestrator is wired:

- normalized `JoltVm` witness provider from `jolt-witness`;
- field-inline witness provider behind `field-inline`;
- `CommitmentStageConfig` derived from verifier dimensions and advice presence;
- proof mode, including transparent versus BlindFold/ZK verifier config;
- `PCS::ProverSetup`;
- mutable commitment backend;
- no `jolt-core` types.

The config must be derived from the same preprocessing and public I/O values that
the verifier validates. Stage 0 must not infer advice presence from backend
outputs; it must reject missing, duplicate, or unexpected outputs.

The desired input shape is:

- verifier-derived dimensions and one-hot/RA layout;
- advice presence flags from validated public/private input shape;
- normalized witness provider;
- backend and PCS setup handles.

The input type should not contain fixture data, core proof fragments, or
pre-assembled verifier proof placeholders.

### Backend Request

Stage 0 owns request construction; the backend owns execution.

- Iterate `committed_oracle_order` exactly once to assign stable slots.
- Request `MaterializationPolicy::Streaming`.
- Preserve each oracle's encoding and retention hint.
- Map `JoltProtocolConfig::zk` to `CommitmentMode::Transparent` or
  `CommitmentMode::Zk` and include that mode in every Jolt VM and field-inline
  commitment request item.
- Use backend results only through slot/oracle IDs, never by vector position
  alone.
- Return prover-only streamed witness data/opening hints for later opening
  stages without exposing them to verifier proof fields.

### Fiat-Shamir

No Stage 0 random challenge is needed before commitments. After proof-field
assembly, `jolt-prover` must absorb commitments in the exact `jolt-verifier`
order:

1. preamble values validated by `jolt-verifier`;
2. `rd_inc`, `ram_inc`;
3. instruction, RAM, and bytecode RA commitments in layout order;
4. field-inline commitments behind `field-inline`;
5. untrusted advice commitment if present;
6. trusted advice commitment if present.

The backend must not append to or sample from the transcript. The clean boundary
should expose a Stage 0 transcript method that accepts `CommitmentStageOutput`
and the already-checked verifier input context, instead of requiring a complete
proof object.

### Verifier Component Assembly

Stage 0 output should assemble only verifier-owned proof fields:

- `JoltCommitments<PCS::Output>`;
- proof-carried untrusted advice commitment;
- externally supplied trusted advice commitment, when present;
- field-inline commitment block under `field-inline`.

Opening hints remain prover-local. Later opening stages consume them through a
typed accumulator/request, not by reading verifier proof fields back.

Field-inline should not introduce a parallel proof-fragment DTO. The clean
pattern is:

1. `CommitmentStageInput` carries the field-inline witness provider under the
   feature.
2. `prove` runs the Jolt VM commitment request, then the field-inline commitment
   request under the feature.
3. `output.rs` drains both backend results through the same deterministic
   collect/take/check helpers.
4. `CommitmentStageOutput` exposes direct verifier fields plus
   `CommitmentStageProverState`; there is no local proof-mode enum and no
   field-inline-specific public output wrapper.

The output contract must compile cleanly in all relevant surfaces:

- transparent without advice;
- transparent with trusted and untrusted advice;
- ZK/BlindFold without advice;
- ZK/BlindFold with advice;
- field-inline builds, including no field-inline names in disabled builds.

### Incremental Proof Assembly

The full modular prover should grow a `JoltProof` incrementally. For Stage 0,
the assembled fragment is:

- commitments copied into the proof builder;
- untrusted advice commitment copied into the proof builder when present;
- trusted advice commitment retained as verifier input, not hidden in the proof;
- opening hints retained in prover state for Stage 8.

The canonical Stage 0 check should therefore inspect the Stage 0 fragment
directly and also verify that the partial proof shell can be passed through the
same verifier input validation and commitment transcript order used by the final
proof.

### Correctness And Parity

Stage 0 is replacement-ready when the transparent commitment, advice, and
field-inline frontiers all have certified kernel evidence and focused verifier
correctness coverage.

The simple canonical Stage 0 test should answer four questions:

1. Are the committed polynomials requested in the intended stable order?
2. Does the backend output assemble into exactly the verifier commitment fields?
3. Does verifier replay accept the Stage 0 proof shell with native
   `jolt-verifier` validation and transcript absorption?
4. Does the relevant certified kernel evidence satisfy the frontier perf gate?

Required checks for accepting the clean Stage 0 product:

- `cargo nextest run -p jolt-prover-harness optimization_inventory source_drift --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness backend_kernel_ledger_covers_every_cpu_backend_inventory_id --cargo-quiet`
- focused Stage 0 harness tests in `frontier_stage0_commitments`
- kernel evidence validation for `cpu_streaming_commitments`,
  `cpu_advice_commitment_contexts`, and `cpu_field_inline_commitments`
- verifier replay using native `jolt-verifier` types/flows
- both standard and ZK-aware compile modes when the stage surface is touched
- advice and field-inline focused checks when those surfaces are touched
- focused backend coverage proving `CommitmentMode::Zk` actually selects the
  ZK PCS commitment path for both streamed dense/compact values and one-hot
  commitments

The canonical test must stay focused. It should not run the full prover as the
inner loop and should not accept fixture replay without the certified backend
kernel evidence.

## Implementation Slice Justification Log

At the end of the Stage 0 implementation slice, append a dated entry here
explaining why the whole stage passes correctness parity and performance parity.
Do not mark the stage complete from intent, fixture replay alone, or broad E2E
success that does not isolate this stage.

The entry must include:

- slice name and code touched;
- correctness claim and the exact command or verifier replay proving it;
- performance claim and the exact kernel/frontier evidence proving it;
- backend kernel ledger status and evidence file paths, when applicable;
- remaining risk or explicit statement that no stage-local risk remains.

### 2026-05-28 - Stage 0 Contract And Transcript Boundary

Code touched:

- `crates/jolt-prover/src/stages/stage0/input.rs`: added explicit
  `CommitmentStageInput` carrying the verifier-owned `JoltProtocolConfig`
  rather than a stage-local proof-mode enum, with the field-inline witness
  provider as a feature-gated input field.
- `crates/jolt-prover/src/stages/stage0/prove.rs`: made `prove` the canonical
  stage entry point and made field-inline a feature-gated namespace commit
  inside the same public orchestration path.
- `crates/jolt-prover/src/stages/stage0/output.rs`: kept verifier-facing output
  as direct `JoltCommitments` plus advice commitments, grouped prover-only
  opening hints under `CommitmentStageProverState`, and keyed the prover state
  with deterministic `BTreeMap`s.
- `crates/jolt-verifier/src/verifier.rs`: added `verify_until_stage1`, the
  verifier-owned pre-Stage-1 prefix used by focused Stage 0 replay tests.
- `crates/jolt-prover/src/transcript.rs`: added `Stage0TranscriptContext` and
  `absorb_stage0_transcript`, then routed full-proof initialization through that
  Stage 0 boundary.
- `crates/jolt-prover/src/stages/stage0/tests.rs` and
  `crates/jolt-prover/src/transcript.rs`: added canonical Stage 0 contract and
  verifier-order transcript tests.
- `crates/jolt-prover/src/stages/stage5/output.rs`: added the existing
  field-inline placeholder claim block so `jolt-prover --features field-inline`
  compiles far enough to run Stage 0 tests. This is not Stage 5 acceptance.
- `crates/jolt-prover-harness/benches/frontier_perf.rs`: added
  `cpu_field_inline_commitments` evidence generation for
  `frontier_perf/stage0_field_inline_commitments`, comparing the CPU backend
  streamed `FieldRdInc` commitment against the same streaming Dory reference and
  checking direct-Dory equality once for correctness.
- `crates/jolt-prover-harness/src/optimization.rs`: marked
  `cpu_field_inline_commitments` parity-certified with its canonical evidence
  artifact path.
- `crates/jolt-prover-harness/tests/frontier_stage0_commitments.rs`: added the
  field-inline replacement-ready check and a native `jolt-verifier` replay that
  wraps the modular Stage 0 field-inline commitments in a verifier proof shell,
  validates field-inline bytecode metadata, and initializes the verifier-owned
  Stage 0 transcript.
- `crates/jolt-verifier/src/compat/claims.rs`: exposed a hidden empty clear
  opening-claims constructor for focused partial-proof replay fixtures.
- `crates/jolt-prover-harness/Cargo.toml`: made the field-inline harness feature
  enable the verifier-side transcript, sumcheck, and vector-commitment test
  dependencies needed by that replay.
- `crates/jolt-witness/src/protocols/jolt_vm/stage5.rs` and later-stage prover
  cfg cleanups: removed field-inline compile/lint blockers so Stage 0 feature
  checks can run under `field-inline` and `zk,field-inline`.

Correctness evidence:

- `cargo nextest run -p jolt-prover --cargo-quiet`
- `cargo nextest run -p jolt-prover --features zk --cargo-quiet`
- `cargo nextest run -p jolt-prover --features field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover --features zk,field-inline --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features zk -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features zk,field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features core-fixtures,zk zk_core_fixture_initializes_prover_owned_transcript --cargo-quiet`
- `stage0_field_inline_commitments_are_accepted_by_jolt_verifier_replay` passes
  inside the field-inline Stage 0 harness run. This focused replay uses the real
  modular field-inline Stage 0 commitments, real verifier field-inline bytecode
  metadata, and `jolt-verifier::verify_until_stage1` for proof config, input
  consistency, proof-shape consistency, preamble absorption, and commitment
  absorption before Stage 1.

Performance evidence:

- `stage0_commitment_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes for `cpu_streaming_commitments`, using
  `target/frontier-metrics/kernel-evidence/cpu_streaming_commitments/frontier_perf_stage0_commitments.json`.
- `stage0_advice_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes for `cpu_advice_commitment_contexts`, using
  `target/frontier-metrics/kernel-evidence/cpu_advice_commitment_contexts/frontier_perf_stage0_advice_commitments.json`.
- `stage0_field_inline_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes for `cpu_field_inline_commitments`, using
  `target/frontier-metrics/kernel-evidence/cpu_field_inline_commitments/frontier_perf_stage0_field_inline_commitments.json`.
- Field-inline evidence was generated by
  `JOLT_WRITE_KERNEL_EVIDENCE=cpu_field_inline_commitments cargo bench -p jolt-prover-harness --features core-fixtures,field-inline --bench frontier_perf --quiet`;
  the canonical gate returned `Warn` with time ratio `1.066179793780585` and
  memory ratio `1.0016274859816854`, both below the 15% failure threshold.
- `registered_parity_certified_kernel_evidence_files_are_valid` passes.

Static rails:

- `cargo nextest run -p jolt-prover-harness optimization_inventory source_drift --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness backend_kernel_ledger_covers_every_cpu_backend_inventory_id --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness ported_backend_kernel_microbenchmarks_are_declared_in_bench_sources prover_ready_frontiers_require_ported_or_certified_cpu_kernels --cargo-quiet`
- `cargo clippy -p jolt-prover -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`

Acceptance state:

- Transparent Stage 0 and advice Stage 0 pass correctness parity and certified
  performance parity.
- ZK/BlindFold mode compiles and uses the verifier-owned `JoltProtocolConfig`
  boundary, but Stage 0 has no hidden sumcheck kernel of its own.
- Field-inline Stage 0 compiles, is accepted by a focused native
  `jolt-verifier` Stage 0 replay with verifier bytecode metadata and transcript
  initialization, and passes `validate_frontier_replacement_ready` with
  certified timing and memory evidence for `cpu_field_inline_commitments`.
- No Stage 0-local risk remains. Stage 5 field-inline placeholder completion is
  explicitly outside this Stage 0 acceptance.

### 2026-05-28 - Stage 0 Public API Cleanup

Code touched:

- `crates/jolt-prover/src/stages/stage0/input.rs`: removed the separate
  field-inline Stage 0 input type. The canonical `CommitmentStageInput` now
  carries the field-inline witness provider directly when `field-inline` is
  enabled.
- `crates/jolt-prover/src/stages/stage0/prove.rs`: removed the public
  `prove_field_inline` path. The public `prove` function now commits Jolt VM
  polynomials first and field-inline polynomials second under the feature,
  matching the verifier-order assembly boundary.
- `crates/jolt-prover/src/stages/stage0/output.rs`: removed the
  field-inline-specific output wrapper and collapsed Jolt VM and field-inline
  backend result parsing onto `VerifierCommitmentBuilder`.
- `crates/jolt-prover/src/builder.rs`: added a reusable
  `VerifierComponentBuilder<Id, VerifierValue, ProverValue>` plus a
  `VerifierComponentSpec<Builder>`/`build(spec)` pattern and a commitment-result
  adapter. The generic builder is independent of concrete
  `JoltProof<PCS, VC, ZkProof>` shapes; stage specs map it into the relevant
  `ProverError` variant and verifier proof component type.
- `crates/jolt-backends/src/traits.rs` and
  `crates/jolt-backends/src/cpu/commitments/mod.rs`: allowed commitment backends
  to accept unsized witness providers so the field-inline witness can be passed
  as a feature-gated trait-object input without adding another public stage
  generic.

Correctness evidence:

- `cargo nextest run -p jolt-prover --features field-inline stage0 --cargo-quiet`
- `cargo nextest run -p jolt-prover --features zk,field-inline stage0 --cargo-quiet`
- `cargo nextest run -p jolt-prover stage0 --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features field-inline stage0_field_inline_commitments_are_accepted_by_jolt_verifier_replay --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features zk,field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo clippy -p jolt-prover -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover-harness --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-verifier --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-backends -q --all-targets -- -D warnings`

Performance evidence:

- No kernel implementation changed. The field-inline and Jolt VM commitment
  requests still call the same `jolt-backends` CPU commitment kernels.
- The field-inline frontier replacement-ready harness test still passes inside
  the full `frontier_stage0_commitments` field-inline run, using the existing
  certified `cpu_field_inline_commitments` evidence.
- The transparent and advice frontier replacement-ready harness tests still pass
  inside the same run for `cpu_streaming_commitments` and
  `cpu_advice_commitment_contexts`.

Acceptance state:

- Stage 0 now has one public prover entry point, direct verifier-owned output
  fields, deterministic `BTreeMap` prover state, a reusable generic verifier
  component builder in `jolt-prover/src/builder.rs`, and no
  field-inline-specific public proof-fragment DTO. No new Stage 0-local risk
  remains from this cleanup.

### 2026-05-28 - Stage 0 ZK Commitment Mode Fix

Code touched:

- `crates/jolt-openings/src/schemes.rs`: added `ZkStreamingCommitment`, the
  streaming analogue of `ZkOpeningScheme` for PCS implementations whose hiding
  commitment path differs from transparent streaming.
- `crates/jolt-dory/src/streaming.rs`: implemented `ZkStreamingCommitment` by
  routing to `DoryScheme::finish_zk`, matching core's `dory::ZK` commitment
  mode under the `zk` feature.
- `crates/jolt-backends/src/commitments/request.rs`: added
  `CommitmentMode::{Transparent, Zk}` to every commitment request item.
- `crates/jolt-backends/src/cpu/commitments/`: propagated request mode through
  the CPU commitment kernel. ZK dense/compact streams use
  `finish_zk_with_hint`; ZK one-hot commitments use `commit_zk` after
  one-hot materialization.
- `crates/jolt-prover/src/stages/stage0/request.rs` and `prove.rs`: mapped
  `JoltProtocolConfig::zk` to the backend commitment mode and passed it through
  both Jolt VM and field-inline namespace requests.
- `crates/jolt-openings/src/mock.rs` and focused tests: made the mock PCS expose
  whether the ZK path was selected so the backend test can catch accidental
  transparent fallback.

Correctness evidence:

- Sibling `jolt-core` was checked against
  `jolt-core/src/poly/commitment/dory/commitment_scheme.rs`; its
  `DoryCommitmentScheme::commit`, `prove`, and streaming `aggregate_chunks`
  select `dory::ZK` under the `zk` feature.
- Local modular `jolt-dory` was checked against
  `crates/jolt-dory/src/scheme.rs` and `crates/jolt-dory/src/streaming.rs`;
  transparent commitments are `commit`/`finish_with_hint`, while ZK commitments
  are `commit_zk`/`finish_zk`.
- `cargo nextest run -p jolt-backends cpu_commitment_backend --cargo-quiet`
- `cargo nextest run -p jolt-prover stage0 --cargo-quiet`
- `cargo nextest run -p jolt-prover --features field-inline stage0 --cargo-quiet`
- `cargo nextest run -p jolt-prover --features zk,field-inline stage0 --cargo-quiet`
- `cargo nextest run -p jolt-dory streaming_zk_commitment_is_blinded_and_verifies --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`
- `cargo nextest run -p jolt-prover-harness --features zk,field-inline -E 'binary(frontier_stage0_commitments)' --cargo-quiet`

Static rails:

- `cargo clippy -p jolt-backends -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-prover --features zk,field-inline -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-openings --features test-utils -q --all-targets -- -D warnings`
- `cargo clippy -p jolt-dory -q --all-targets -- -D warnings`

Performance evidence:

- No Stage 0 kernel algorithm changed; the fix selects the existing Dory ZK
  finalization path when the verifier/proof config is BlindFold. The row
  streaming work remains the same CPU backend path.
- Dedicated ZK commitment evidence was generated by
  `JOLT_WRITE_KERNEL_EVIDENCE=cpu_zk_streaming_commitments cargo bench -p jolt-prover-harness --features core-fixtures,zk --bench frontier_perf --quiet`.
  It compares core `DoryCommitmentScheme` ZK streaming aggregation against the
  modular CPU backend with `CommitmentMode::Zk`.
- The evidence file is
  `target/frontier-metrics/kernel-evidence/cpu_zk_streaming_commitments/frontier_perf_stage0_zk_commitments.json`.
  The canonical gate returned `Pass` with core time `71.96520866666667 ms`,
  modular time `73.23209733333333 ms`, time ratio `1.0176041824951099`, core
  peak `4424360` bytes, modular peak `4338284` bytes, and memory ratio
  `0.9805449827771701`.
- Existing certified Stage 0 frontier gates still pass in the focused
  field-inline and `zk,field-inline` harness runs above, including
  `stage0_commitment_frontier_is_replacement_ready_with_certified_kernel_evidence`,
  `stage0_advice_frontier_is_replacement_ready_with_certified_kernel_evidence`,
  and `stage0_field_inline_frontier_is_replacement_ready_with_certified_kernel_evidence`.
- `stage0_zk_commitment_frontier_is_replacement_ready_with_certified_kernel_evidence`
  passes under `--features zk`, and
  `registered_parity_certified_kernel_evidence_files_are_valid` accepts the new
  `cpu_zk_streaming_commitments` ledger entry.

Acceptance state:

- The previous ZK statement was incomplete: Stage 0 could not treat BlindFold as
  a proof-shell-only flag. ZK mode now reaches the same hiding Dory commitment
  path that core uses, while transparent mode remains the default request mode.
  ZK commitment performance is now separately benchmark-certified for the Stage
  0 commitment frontier.

## Remaining Work

No Stage 0 work remains under this spec. Move to the next stage only under a new
stage spec and monitored slice.

## Non-Goals

- Do not add a generic fallback commitment path as an acceptance route.
- Do not use core fixture replay alone as replacement evidence.
- Do not merge Stage 0 with later sumcheck or opening stages.
- Do not move to Stage 1 until Stage 0's API, tests, correctness evidence, and
  performance evidence have been reviewed.
