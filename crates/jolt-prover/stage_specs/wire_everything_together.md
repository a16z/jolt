# Wire Everything Together Critical Path

Status: 2026-05-31 latest local branch inspection. This refresh is based on
code/spec inspection plus focused non-benchmark acceptance/static checks; no
benchmarks were run.

This document replaces the broad frontier-expansion strategy with the shortest
path from the current committed-boundary migration state to `jolt-prover`
producing verifier-accepted `JoltProof`s.

The goal is not to prove that every historical optimization row is polished.
The goal is to get the real top-level prover orchestration working for the
feature combinations that matter:

- Clear proofs without advice.
- Clear proofs with trusted and untrusted advice.
- Clear field-inline proofs.
- ZK proofs with BlindFold.
- ZK field-inline proofs.

## Current State

The stage frontier work is no longer ahead of the top-level prover for the
feature paths covered by the current public acceptance tests. Clear, clear
field-inline, default ZK, and `zk + field-inline` now all have top-level
`prove_with_output` verifier-acceptance coverage.

- Stages 0 through 8 have enough local infrastructure to produce and verify
  clear proofs through the native verifier path.
- Stages 1 through 7 now have real modular committed-boundary prover entrypoints
  that emit verifier-owned committed proofs, hidden output-claim values, and
  retained `CommittedSumcheckWitness` material for BlindFold.
- Stage 8 now has a ZK final-opening prover entrypoint using `open_zk_poly`; it
  returns the verifier-visible PCS proof/joint commitment plus the hidden
  evaluation commitment and blinding needed by BlindFold.
- The older `zk + field-inline` SDK frontier replay verifies committed Stage 3
  through Stage 8 with real modular data and shell Stage 1/2 inputs. It remains
  frontier coverage, not full top-level `JoltProof` production; Stage 1/2 now
  have separate production committed-boundary prover coverage.
- Stage 1 now has a production `prove_committed_boundary` entrypoint for both
  standard ZK and `zk + field-inline`. It commits real uni-skip/remainder round
  polynomials, commits hidden output-claim rows in verifier order, retains the
  `CommittedSumcheckWitness` rows for BlindFold, and is accepted by native
  Stage 1 verifier tests.
- Stage 2 now has a production `prove_committed_boundary` entrypoint for both
  standard ZK and `zk + field-inline`. It commits the real product uni-skip and
  regular-batch round polynomials, commits product and batch output-claim rows
  in verifier order, retains the `CommittedSumcheckWitness` rows for BlindFold,
  and is accepted by focused native Stage 2 verifier tests.
- Stage 8 now has clear field-inline final-opening verifier acceptance and ZK
  final-opening committed-boundary verifier acceptance through Stage 8.
- Stage 4, Stage 5, Stage 6 field-inline CPU kernels are parity-certified.
- Stage 7 regular sumcheck and Stage 8 final-opening kernel evidence are also
  within parity.
- The default non-field-inline clear `prove_with_output` path now runs Stage 0
  through Stage 8, assembles `JoltStageProofs` and `ClearProofClaims`, calls
  `JoltProof::new(...)`, and returns `ProverOutput`.
- The ZK path no longer deliberately stops at `FrontierNotImplemented` after
  Stage 8. The default ZK and `zk + field-inline` paths now complete
  BlindFold proof generation, final ZK `JoltProof` assembly, and verifier
  acceptance in focused top-level tests.
- `prove_with_output_inner` now branches before clear orchestration when
  `config.features.zk` is true. The ZK branch runs committed Stage 1 through
  Stage 7, assembles committed `JoltStageProofs`, runs Stage 8 ZK final
  opening, records the hidden final-opening commitment/blinding, constructs the
  shared BlindFold protocol, assembles and R1CS-checks the prover-side
  BlindFold witness rows, calls `jolt_blindfold::prove_with_row_committer` at
  the verifier-aligned transcript position, routes BlindFold row commitments,
  folded row openings, relaxed-R1CS error row materialization, and
  post-challenge folding arithmetic through the configured backend, and
  finalizes a ZK `JoltProof` with `JoltProofClaims::Zk`.
- Full BlindFold proof generation is wired through the top-level public prover
  API. Both non-field-inline ZK and `zk + field-inline` acceptance tests now
  pass.
- Stage 8 is now registered as a first-class harness manifest frontier for
  transparent/advice, ZK final opening, and field-inline final opening.
- The post-clean benchmark evidence files for registered `ParityCertified`
  rows are restored and validated by `optimization_inventory`.
- Stage 6 manifest optimization IDs now include the bytecode read-RAF evidence
  owners `OPT-REL-004` and `OPT-REL-005`.
- Stage 7 regular-batch manifest fixtures now include `AdviceConsumer`, matching
  the exercised acceptance and evidence path.
- The top-level API now has `ProverOutput` and `prove_with_output`, so the
  prover returns the external trusted-advice commitment alongside the proof for
  the default clear path. The existing `prove` API remains as a proof-only
  compatibility wrapper.
- The top-level prover API now takes `&JoltDevice` explicitly, matching the
  verifier's public-I/O input boundary instead of hiding public I/O behind the
  generic witness trait.
- `crates/jolt-prover/src/assembly.rs` now owns the private top-level
  `ProofAssembly` state. It has slots for Stage 0 commitments/opening hints,
  Stage 1-7 proofs, clear claims, ZK BlindFold witness material, Stage 8 joint
  opening proof, advice commitments, trace/config metadata, and clear proof
  finalization.
- `ProofAssembly` now also has committed ZK staging hooks: Stage 1-7 committed
  recorders, `assemble_zk_stage_payloads`, `record_stage8_zk`, and
  `into_zk_proof`. These hooks are now used by the top-level
  `prove_zk_stages` path through Stage 8, and `into_zk_proof` now constructs
  the final ZK `JoltProof` from the generated BlindFold proof.
- `ProofAssembly::build_blindfold_protocol` now derives verifier-equivalent
  `Stage1ZkOutput` through `Stage8ZkOutput` from the assembled committed proof
  data, calls `jolt_verifier::stages::zk::blindfold::build`, and verifies that
  the top-level ZK branch can construct the exact shared BlindFold protocol.
- `ProofAssembly::assemble_blindfold_witness` now maps retained committed
  round/output-claim rows into the BlindFold witness-row layout, derives the
  sumcheck claim variables from the round polynomials, inserts the Stage 8
  hidden final-opening value/blinding, solves the auxiliary R1CS witness values,
  and checks the result against the generated BlindFold R1CS.
- `jolt-blindfold` now exposes a crate-level prover API:
  `BlindFoldWitness` plus `prove(...)`, which consumes preassembled witness
  rows, row blindings, final-opening evaluation values/blindings, the shared
  `BlindFoldProtocol`, VC setup, transcript, and RNG to produce a real
  `BlindFoldProof`. The integration proof pipeline now exercises this exported
  API instead of a private test-only proof constructor.
- `jolt-verifier` now exposes the Jolt-specific BlindFold protocol builder
  surface under `jolt_verifier::stages::zk::{blindfold, inputs, outputs}`.
  The builder takes a small `BlindFoldProofContext` instead of a full
  `JoltProof`, so `jolt-prover` can reuse the verifier's exact protocol
  construction without fabricating a placeholder ZK proof.
- `prove_with_output` now calls the real Stage 0 commitment prover, derives the
  Stage 0 RA/final-opening layout from `ProverProofShape`, and records the
  resulting commitments, opening hints, and advice commitments in
  `ProofAssembly`.
- `ProverProofShape` now carries the trace length, RAM size, read/write config,
  one-hot config, and trace polynomial order needed to populate proof metadata
  and derive Stage 0 formula/final-opening dimensions.
- The clear paths now initialize the verifier-compatible `Blake2bTranscript`,
  reconstruct `CheckedInputs` from explicit public I/O and `ProverProofShape`,
  absorb the Stage 0 transcript preamble, run the real Stage 1 through Stage 8
  provers in the same transcript stream, record verifier outputs, claims,
  sumcheck proofs, and the joint opening proof in `ProofAssembly`, then
  construct the clear `JoltProof`.
- Stage 4 top-level orchestration now computes `ram_val_check_init`
  self-contained from verifier preprocessing, normalized public I/O, Stage 2
  RAM opening coordinates, and trusted/untrusted advice bytes instead of
  importing fixture state.
- Stage 6 top-level orchestration now derives its config from proof shape and
  preprocessing, including formula dimensions, booleanity dimensions, advice
  cycle-phase layouts, and bytecode context.
- Stage 7 top-level orchestration now derives hamming-weight and advice
  address-phase layouts and records real verifier outputs, claims, and proof
  objects.
- Stage 8 top-level orchestration now orders retained Stage 0 commitments and
  opening hints in the verifier final-opening order and records the real joint
  opening proof.
- `top_level_clear_prover_outputs_verify` now proves and verifies the public
  `prove_with_output` path for transparent `MuldivSmall` and `AdviceConsumer`,
  including the external trusted-advice commitment returned through
  `ProverOutput`.
- Field-inline Stage 1 now returns a verifier-compatible `Stage1ClearOutput`;
  the focused Stage 1 test checks it against the native verifier output.
- The top-level field-inline clear `prove_with_output` path now runs Stage 0
  through Stage 8 through the shared `ProofAssembly` state and is accepted by
  `jolt-verifier` in `top_level_field_inline_clear_prover_outputs_verify`.
- The top-level default ZK `prove_with_output` path now produces a real
  `JoltProof` with `JoltProofClaims::Zk` and is accepted by `jolt-verifier` in
  `top_level_zk_prover_outputs_verify`.
- The top-level `zk + field-inline` acceptance test now passes in
  `top_level_field_inline_zk_prover_outputs_verify`. The runtime mismatch was in
  the BlindFold Stage 6 public-value builder: it used raw field-inline bytecode
  rows for the base Jolt bytecode read-RAF contribution, while the prover and
  normal Stage 6 verifier both first project those rows through
  `base_jolt_bytecode_row` and then add the field-inline extension.
- The field-inline ZK top-level and committed-frontier acceptance tests now run
  inside a dedicated 128 MiB test thread plus a local Rayon pool with 128 MiB
  worker stacks, matching the existing large ZK frontier pattern and avoiding
  default worker-stack aborts.
- Fresh non-benchmark validation covers `cargo clippy -D warnings` for
  `jolt-prover` in standard, `zk`, `field-inline`, and `zk,field-inline` modes;
  an explicit `cargo check` for the combined `zk,field-inline` mode; focused
  top-level nextest acceptance for clear, advice, field-inline clear, default
  ZK, and `zk + field-inline`; focused harness static checks; `cargo fmt -q`;
  `jolt-blindfold` nextest/clippy including row-committer hook coverage;
  focused `jolt-backends` BlindFold validation plus CPU
  formula/opening-reference nextest/clippy; and `git diff --check`.
- Focused BlindFold benchmark evidence now exists for both critical CPU rows:
  `cpu_blindfold_round_commitments` passed
  `frontier_perf/zk_blindfold_core_fixture` at 0.845x time / 1.004x measured
  allocation, and `cpu_blindfold_backend_kernels` passed
  `frontier_perf/blindfold_witness_rows` at 0.800x time / 1.000x measured
  allocation. Both evidence files are registered in the backend kernel ledger.
- The current frontier is therefore no longer top-level ZK runtime wiring. The
  remaining critical-path gates are focused top-level acceptance/static reruns
  before commit and any larger confirmation run the frontier owner requires.
- The BlindFold backend-certification handoff is now closed in the static rails:
  `zk_blindfold_core_fixture` owns `OPT-ZK-001`, `OPT-ZK-002`, `OPT-ZK-003`,
  and `OPT-ZK-006` through `cpu_blindfold_round_commitments` and
  `cpu_blindfold_backend_kernels`. The
  current CPU backend BlindFold module now owns the transcript-free row
  commitment, folded row-opening, relaxed-R1CS error-row, and folded row/scalar
  hooks used by top-level ZK proof generation, while `jolt-prover` and
  `jolt-blindfold` keep transcript ownership. The backend request contract
  validates labels, round slots, coefficient slots, output-claim slots,
  row/blinding lengths, vector-commitment capacity, row-opening point dimensions,
  error row dimensions, R1CS witness-column bounds, and folding input
  lengths/row widths before backend execution. The remaining BlindFold
  backend-kernel work is any further
  no-transcript backend ownership for kernels still local to `jolt-blindfold`.

The critical missing piece is no longer clear orchestration, top-level ZK
runtime wiring, or BlindFold backend certification. The production ZK and
`zk + field-inline` top-level paths now emit verifier-accepted `JoltProof`s
without fixture proof-shell replay in focused acceptance tests. The remaining
BlindFold boundary work is additional transcript-safe kernel extraction behind
`jolt-backends`, not another top-level proof assembly milestone.

The earlier Phase 0 rails work therefore no longer leads the critical path. It
should be maintained by static checks, and it remains the next non-benchmark
pre-merge gate alongside focused top-level acceptance reruns.

## Acceptance Definition

The replacement is not complete until the public `jolt-prover` top-level API
produces a `jolt_verifier::proof::JoltProof` accepted by `jolt-verifier` for
every required feature combination.

Acceptance requires:

- The proof is built by `jolt-prover`, not by splicing a `jolt-core` proof.
- The clear path stores real Stage 1 through Stage 7 claims in
  `JoltProofClaims::Clear`.
- The ZK path stores a real `BlindFoldProof` in `JoltProofClaims::Zk`.
- Stage 8 produces the real joint opening proof from retained prover hints and
  the verifier's final opening order.
- Field-inline final openings use the address-major ordering already audited in
  Stage 8.
- Trusted advice is handled as an external verifier input, while untrusted
  advice remains inside the proof as `untrusted_advice_commitment`.
- Transcript order matches the verifier preamble, stage order, commitment order,
  and `BlindFold` domain separation exactly.
- No stage is accepted through fixture replay, placeholder BlindFold data,
  generic fallback kernels, or dense reference-only opening paths.
- Harness inventory, ledger, and source-drift checks pass after the top-level
  path is wired.

## Non-Goals

Do not expand the migration into unrelated cleanup while chasing this path.

- Do not run broad benchmarks before top-level correctness is green.
- Do not work through every umbrella optimization row unless it blocks the full
  proof path.
- Do not add CUDA, Metal, or alternate backend work.
- Do not replace specialized `jolt-core` algorithms with generic fallback
  kernels.
- Do not accept correctness-only replacements where core has a specialized
  algorithm and a frontier requires parity.

## Phase 0: Maintain The Rails

This phase is complete for the known Stage 6, Stage 7, Stage 8, and evidence
drift found during the post-clean pass. Keep it as a guardrail while the
top-level wiring lands.

1. Stage 8 manifest frontiers are registered.

   The manifest now exposes:

   - Clear final opening.
   - ZK final opening with hidden evaluation commitment.
   - Field-inline final opening.

2. Known manifest and ledger drift is repaired.

   - Stage 6 regular-batch metadata includes `OPT-REL-004` and `OPT-REL-005`.
   - Stage 7 regular-batch metadata includes `AdviceConsumer`.
   - `cpu_blindfold_round_commitments` and
     `cpu_blindfold_backend_kernels` are now replacement-ready after focused
     BlindFold proof-generation and backend evidence runs.

3. Registered `ParityCertified` evidence files are restored.

   The static ledger should not claim `ParityCertified` for rows whose canonical
   JSON evidence is unavailable to the validation checks. Regenerate only the
   focused evidence files needed by the critical path when benchmark runs are
   allowed.

4. Run only static harness checks while wiring.

   Do not use broad prover E2E or performance runs to debug top-level data-flow
   mistakes. The early gate is inventory, ledger, source drift, and compile
   correctness.

## Phase 1: Define The Top-Level Assembly Model

Add a private top-level assembly state in `jolt-prover`. Complete for the
initial owner shape and Stage 0 population: `ProofAssembly` now exists, is
constructed by `prove_with_output` after config validation, and records real
Stage 0 commitment output.

Recommended shape:

```rust
struct ProofAssembly<PCS, VC> {
    commitments: JoltCommitments<PCS::Commitment>,
    stage_proofs: JoltStageProofs<PCS, VC>,
    clear_claims: Option<ClearProofClaims<PCS::Field>>,
    blindfold_witnesses: Option<BlindFoldWitnessBundle<PCS::Field, VC>>,
    joint_opening_proof: PCS::Proof,
    trusted_advice_commitment: Option<PCS::Commitment>,
    untrusted_advice_commitment: Option<PCS::Commitment>,
    trace_length: usize,
    ram_K: usize,
    rw_config: ReadWriteConfig,
    one_hot_config: OneHotConfig,
    trace_polynomial_order: TracePolynomialOrder,
}
```

This is illustrative, not an API requirement. The important part is ownership:
the top-level prover must retain every commitment, stage output, opening hint,
hidden value, and committed witness needed by later stages.

Current implementation note: `ProofAssembly::next_frontier()` remains a useful
diagnostic fallback, but the required clear, field-inline clear, default ZK,
and `zk + field-inline` public top-level paths now bypass that stub by
completing Stage 8 and returning verifier-accepted `JoltProof`s.

The public API ownership decision is now explicit:

- `prove_with_output` returns `ProverOutput { proof, trusted_advice_commitment }`
  for advice-capable verifier calls.
- `prove` remains a compatibility wrapper returning only `JoltProof`.
- `prove` and `prove_with_output` take `&JoltDevice` explicitly for public I/O,
  matching `jolt-verifier::verify`.
- `ProverConfig::with_proof_shape(...)` carries the proof-shape metadata that is
  not recoverable from the generic witness trait.

This keeps the generic `WitnessProvider`/`CommittedWitnessProvider` boundary
focused on witness data. Verifier public data stays an explicit top-level input.

## Phase 2: Wire Clear `JoltProof` First

Build the non-ZK path before BlindFold. This gives a real `JoltProof` shell and
forces transcript, commitment, claim, and opening order to match the verifier.

Steps:

1. Stage 0 commitments.

   - Produce `JoltCommitments` in exactly the verifier's expected order:
     `rd_inc`, `ram_inc`, RA commitments, and optional field-inline
     commitments.
   - Retain the PCS prover opening hints in the same final-opening order used
     by Stage 8.
   - Retain trusted and untrusted advice commitment information separately.
   - Record `trace_polynomial_order`, `trace_length`, `ram_K`,
     `ReadWriteConfig`, and `OneHotConfig`.

   Current status: top-level `prove_with_output` runs Stage 0 and records the
   verifier commitments, trusted/untrusted advice commitments, and opening hints
   in `ProofAssembly`. Proof metadata is sourced from `ProverProofShape`.

2. Transcript initialization.

   - Share or mirror the verifier's proof preamble.
   - Append commitments and public data in verifier order.
   - Reject feature mismatches early: compiled features, requested
     `ProverFeatureSet`, proof protocol, and verifier preprocessing must agree.

   Current status: the clear paths perform this initialization through
   `ProofAssembly::absorb_stage0`, then keep using the same transcript through
   Stage 8, including field-inline clear.

3. Stage 1 through Stage 7 proving.

   - Run each stage in verifier order.
   - Store the concrete proof object expected by `JoltStageProofs`.
   - Store the clear claim object expected by `ClearProofClaims`.
   - Keep all dependency outputs required by the next stage instead of
     recomputing from fixtures.

   Current status: clear Stages 1-7 are wired and compile-clean for both
   default and field-inline clear, and their proof/claim payloads are assembled
   into `JoltStageProofs` and `ClearProofClaims`.

4. Stage 8 clear final opening.

   - Use the retained opening hints from Stage 0 and the final claim scalars
     from Stages 1 through 7.
   - Include advice and field-inline openings in the verifier's final ordering.
   - Produce the real `joint_opening_proof`.

   Current status: the clear paths wire Stage 8 from retained Stage 0
   commitments/opening hints and record the real joint opening proof. The ZK
   path now also calls the Stage 8 ZK final-opening entrypoint from top-level
   orchestration, records the verifier-visible joint opening proof, and retains
   the hidden final-opening commitment/blinding for BlindFold.

5. Construct the proof.

   Call `JoltProof::new(...)` with:

   - `JoltCommitments`.
   - `JoltStageProofs`.
   - `joint_opening_proof`.
   - `untrusted_advice_commitment`.
   - `JoltProofClaims::Clear(ClearProofClaims { ... })`.
   - Trace and config metadata.

   Current status: the clear paths construct and return a `JoltProof` through
   `prove_with_output`; `prove` unwraps that output to preserve the proof-only
   API.

6. Verify immediately with `jolt-verifier`.

   Minimum correctness fixtures:

   - Transparent `MuldivSmall`.
   - Advice-consuming fixture with trusted and untrusted advice.
   - Field-inline clear fixture.

This phase is complete when the public `jolt-prover` top-level API returns a
real clear `JoltProof` accepted by the native verifier. The default
non-field-inline clear `ProverOutput` path now passes native verifier
acceptance for `MuldivSmall` and `AdviceConsumer`; field-inline clear now
passes native verifier acceptance for the SDK field-inline fixture through the
public `prove_with_output` path.

## Phase 3: Wire The Committed ZK Stage Path

After the default clear proof verifies, switch the same orchestration state to
`CommitmentMode::Zk`.

Latest applicability check: this phase still applies, but its center of gravity
has moved. Stage 1-7 committed-boundary prover components and the Stage 8 ZK
final-opening component now exist and are verifier-accepted in focused frontier
tests. `ProofAssembly` now has the committed ZK storage and finalization hooks.
Top-level `prove_zk_stages` now executes those components, records their witness
material, constructs the shared BlindFold protocol from the assembled committed
proof data, assembles an R1CS-checked prover-side BlindFold witness, and feeds
that witness into real BlindFold proof generation. The `zk + field-inline`
witness mismatch exposed by top-level acceptance was fixed in the BlindFold
Stage 6 public-value builder by projecting raw field-inline bytecode rows
through `base_jolt_bytecode_row` before computing the base bytecode read-RAF
public contribution.

Steps:

1. Stage 0 ZK commitments.

   - Commit using the ZK commitment mode.
   - Retain opening hints and blinding data needed by Stage 8.
   - Ensure field-inline commitments are included under the same layout as the
     clear path.

   Current status: Stage 0 already switches to ZK commitment mode from
   `config.protocol`. The top-level ZK path now initializes
   `CheckedInputs { zk: true, vc_capacity: Some(...) }` from verifier
   preprocessing, validates the BlindFold VC capacity, and keeps the ZK
   transcript path separate from the clear proof branch.

2. Stage 1 and Stage 2 committed outputs.

   The verifier's ZK path requires `ZkStageOutputs` for Stage 1 through Stage 8,
   not only Stage 3 onward. Stage 1 and Stage 2 now have production
   committed-boundary emission for standard ZK and `zk + field-inline`.

   Required data per committed stage:

   - The committed sumcheck proof.
   - Public output claim.
   - Hidden clear output claim used only by the prover.
   - Committed output-claim row values and blindings.
   - Round polynomial coefficient rows and blindings.

   Acceptance for this step is not another core proof-shell verifier replay.
   Stage 1 and Stage 2 now satisfy the local production-owned requirement; the
   top-level ZK branch now calls them and feeds their outputs through the
   existing `ProofAssembly` committed recorders.

3. Stage 3 through Stage 7 committed outputs.

   - Use `CommittedSumcheckBuilder` or the local stage wrapper around it.
   - Retain every `CommittedSumcheckWitness`.
   - Retain every `CommittedOutputClaimOutput`.
   - Store only verifier-visible proof data in `JoltStageProofs`.

   Current status: the local committed-boundary entrypoints exist for Stage 3,
   Stage 4, Stage 5, Stage 6, and Stage 7. The top-level ZK branch now calls
   them, records them through the existing `ProofAssembly` hooks, calls
   `assemble_zk_stage_payloads`, and preserves the committed witnesses in
   verifier order. Stage 3 and Stage 4 committed outputs now also retain their
   hidden clear verifier outputs so downstream stages can run without fixture
   state.

4. Stage 8 ZK final opening.

   - Use `open_zk_poly` for the hidden final evaluation.
   - Retain the hidden evaluation value and blinding for BlindFold.
   - Store the verifier-visible final-opening proof and commitments.

   Current status: `prove_stage8_zk` exists and is accepted by the Stage 8 ZK
   frontier replay. The top-level ZK path now orders retained Stage 0
   commitments/hints, calls this entrypoint, records the joint opening proof,
   and retains the hidden evaluation commitment/blinding for BlindFold through
   `record_stage8_zk`.

At the end of this phase the prover can create a complete ZK stage proof shell,
construct the verifier-equivalent BlindFold protocol, and retain the witness
rows needed by Phase 4. This phase is complete for the current critical path;
acceptance now depends on the remaining focused correctness/static checks and,
when allowed, the performance gates.

## Phase 4: Build Prover-Side BlindFold

The verifier already has a BlindFold protocol builder for checked ZK stage
outputs. The prover needs the same protocol plus the witness assignment.

Do not duplicate the protocol construction manually in two crates. Move the
shared builder into a location usable by both prover and verifier, or expose a
small intentional verifier helper if extraction is too large. The preferred
direction is a shared module because protocol-layout drift will be hard to
debug.

Latest applicability check: this phase still applies, but its implementation
work is now wired for the default ZK path. The
Jolt-specific BlindFold protocol builder is now exposed from `jolt-verifier`
and no longer requires a full `JoltProof`; the generic `jolt-blindfold` crate
now exposes verification, protocol/layout primitives, and a production proof
API over explicit witness rows. The top-level ZK path now constructs the shared
BlindFold protocol after Stage 8 and assembles the corresponding prover witness
rows, invokes `jolt_blindfold::prove_with_row_committer` at the verifier's
transcript position, routes row commitments, folded row openings, relaxed-R1CS
error row materialization, and post-challenge folding arithmetic through the
configured backend, and stores the resulting proof in `JoltProofClaims::Zk`.
Default ZK and `zk + field-inline` verifier acceptance both pass. A backend-level
BlindFold protocol hook with transcript parameters does not currently apply
because `backends_do_not_use_transcripts` deliberately keeps transcript
ownership in `jolt-prover`.

The shared builder must consume:

- Stage 1 through Stage 7 committed sumcheck metadata.
- Stage 1 through Stage 7 committed output-claim metadata.
- Stage 8 final-opening binding metadata.
- Public inputs already baked into the verifier transcript.
- The same final-opening order and hidden-opening identifiers used by Stage 8.

The prover-only witness builder must consume:

- Round polynomial coefficients for every committed sumcheck round.
- Round polynomial blindings.
- Output-claim row values and blindings.
- Stage 8 hidden final evaluation value and blinding.
- Any field-inline hidden openings included in the Stage 8 binding.
- Auxiliary witness rows induced by the shared BlindFold R1CS layout.

Then:

1. Build the shared BlindFold protocol from the same stage outputs the verifier
   will later rebuild. Current status: wired.
2. Append the `BlindFold` domain separator to the transcript in exactly the
   verifier position. Current status: wired in top-level ZK orchestration.
3. Commit the real auxiliary rows, create the committed relaxed instance, sample
   and commit the random relaxed instance, compute cross-term error rows, run
   the outer and inner Spartan sumchecks, and open the folded witness/error rows
   with the real vector-commitment setup. Current status: the transcript-owned
   protocol flow remains in `jolt_blindfold`, and row commitment, folded
   row-opening, relaxed-R1CS error-row batches, and post-challenge folding
   arithmetic are delegated to `jolt-backends` through the row-committer
   callback.
4. Bind the folded Stage 8 evaluation/blinding openings to the dedicated
   BlindFold witness coordinates. Current status: witness rows and final
   opening eval/blinding vectors are passed into
   `jolt_blindfold::prove_with_row_committer`.
5. Store the resulting proof in `JoltProofClaims::Zk { blindfold_proof }`.
   Current status: wired through `ProofAssembly::into_zk_proof`.
6. Verify the finished `JoltProof` with `jolt-verifier`. Current status:
   passing for both default ZK and `zk + field-inline` focused top-level
   acceptance.

BlindFold replacement is complete for the current backend row boundary because
these ledger rows now have real focused evidence:

- `cpu_blindfold_round_commitments`.
- `cpu_blindfold_backend_kernels`.

Current certification status: `zk_blindfold_core_fixture` now accounts for
`OPT-ZK-001`, `OPT-ZK-002`, `OPT-ZK-003`, and `OPT-ZK-006` through
`cpu_blindfold_round_commitments` and `cpu_blindfold_backend_kernels`. Both rows
are `ParityCertified`, their canonical evidence paths are registered in the
ledger, and `optimization_inventory` loads them through the global certified
evidence check.

## Phase 5: Complete The Feature Matrix

Once clear and ZK proof shells verify individually, wire the matrix deliberately.

1. Clear, no advice.

   This is the baseline and should remain the smallest regression test.

2. Clear, with advice.

   The trusted advice commitment is an external verifier input. The top-level
   prover API now returns it alongside the proof through `ProverOutput`.

   Current API:

   ```rust
   struct ProverOutput<PCS, VC> {
       proof: JoltProof<PCS, VC>,
       trusted_advice_commitment: Option<PCS::Output>,
   }
   ```

3. Clear, field-inline.

   Reuse the Stage 8 address-major final-opening order. Do not introduce a
   second ordering in top-level assembly.

4. ZK, no field-inline.

   This validates committed sumcheck witness retention and BlindFold proof
   generation without field-inline opening complexity. Current status: passing
   in `top_level_zk_prover_outputs_verify`.

5. ZK, field-inline.

   This is the final target. It combines Stage 4 through Stage 8 field-inline
   commitments, hidden final opening data, and BlindFold witness assignment.
   Current status: passing in
   `top_level_field_inline_zk_prover_outputs_verify`.

## Phase 6: Retire Proof Splicing

After the public `jolt-prover` top-level API returns verifier-accepted proofs,
keep stage-local frontier tests as regression tests, but stop treating
proof-shell splicing as an acceptance path.

Replace acceptance tests with top-level tests that:

- Construct or load the fixture witness.
- Call `jolt_prover::prove` or the new `ProverOutput` API.
- Pass the returned proof and trusted-advice commitment to `jolt_verifier`.
- Assert verifier acceptance.

The stage frontier harness should remain useful for backend kernel parity and
for pinpointing stage regressions, but full replacement status belongs to the
top-level proof path.

## Phase 7: Performance Gates

Do performance only after the top-level proof is structurally correct.

Order:

1. Static harness checks for inventory, ledger, and source drift.
2. Focused unit/reference tests for any touched backend kernel.
3. Focused microbenchmarks only for rows that block the critical proof path.
4. Canonical `sha2-chain-2^16` full-proof performance.
5. Large confirmation run only after the smaller canonical run passes.

The default replacement threshold remains 15 percent on required timing and
peak-memory axes. A frontier is not replacement-ready until
`validate_frontier_replacement_ready` passes with `ParityCertified` ledger status
and loadable canonical evidence.

## Implementation Checklist

API and ownership:

- Top-level `ProofAssembly` exists and is constructed by `prove_with_output`.
- Public I/O enters `jolt-prover::prove_with_output` as `&JoltDevice`.
- Trusted advice commitment leaves through `ProverOutput`.
- Proof shape enters through `ProverConfig::with_proof_shape`.
- Ensure prover preprocessing owns or can access both PCS and VC setup data.
- Validate feature-set compatibility at the start of proving.

Stage integration:

- Stage 0 commitment production is wired into `ProofAssembly`.
- Stage 0 final-opening hints are retained in `ProofAssembly`.
- Stage 7 clear path runs through the existing orchestration layer.
- Default clear `JoltStageProofs` and `ClearProofClaims` are built only from
  real stage outputs.
- Default clear Stage 8 final opening is wired from retained hints, not
  fixtures.
- Default clear `JoltProof` construction is wired.
- Field-inline Stage 1 returns verifier output.
- Field-inline clear stages run through the same orchestration layer.
- Stage 1 production committed-boundary entrypoints exist for standard ZK and
  `zk + field-inline`.
- Stage 2 production committed-boundary entrypoints exist for standard ZK and
  `zk + field-inline`.
- ZK `ProofAssembly` slots, committed-stage recorders, Stage 8 ZK recorder, and
  `into_zk_proof` exist; the recorders are used by top-level ZK orchestration,
  and `into_zk_proof` now finalizes the ZK proof from a real BlindFold proof.
- Run committed Stage 1 and Stage 2 from top-level ZK orchestration with real
  committed witnesses, not core or synthetic proof shells. Current status:
  wired.
- Run existing committed Stage 3 through Stage 7 paths through the same
  top-level orchestration layer after Stage 1/2 are production-owned. Current
  status: wired.
- Wire the existing ZK final opening from retained hints, not fixtures. Current
  status: wired.
- Include advice and field-inline openings in the final opening order.

BlindFold:

- Extract or expose shared BlindFold protocol construction.
  Current status: exposed through `jolt_verifier::stages::zk::blindfold::build`
  and `BlindFoldProofContext`.
- Build verifier-equivalent `Stage1ZkOutput` through `Stage8ZkOutput` inside
  `ProofAssembly` and call the shared protocol builder. Current status: wired
  through `ProofAssembly::build_blindfold_protocol`.
- Promote a production BlindFold prover API from the existing protocol/layout
  primitives; do not depend on test-support proof construction. Current status:
  `jolt_blindfold::prove_with_row_committer` is exported and covered by the
  protocol pipeline, with `prove` retained as the direct row-commitment wrapper.
- Build the prover witness bundle from committed stage witnesses. Current
  status: wired through `ProofAssembly::assemble_blindfold_witness`.
- Include Stage 8 hidden final opening value and blinding. Current status:
  wired into the assembled witness rows.
- Prove with the same transcript position used by verifier. Current status:
  wired through the top-level `jolt_blindfold::prove_with_row_committer` call.
- Store the real proof in `JoltProofClaims::Zk`. Current status: wired through
  `ProofAssembly::into_zk_proof`.

Harness and ledger:

- Keep Stage 8 manifest registration covered by static checks.
- Keep Stage 6 optimization ID coverage covered by static checks.
- Keep Stage 7 advice fixture coverage covered by static checks.
- Keep registered `ParityCertified` evidence files loadable.
- Certify BlindFold backend rows only after real focused evidence exists.

Correctness gates:

- Compile `jolt-prover` with standard features.
- Compile `jolt-prover` with `zk` and `field-inline`.
- Verify top-level clear `MuldivSmall`: covered by
  `top_level_clear_prover_outputs_verify`.
- Verify top-level clear advice fixture: covered by
  `top_level_clear_prover_outputs_verify`.
- Verify top-level clear field-inline fixture: covered by
  `top_level_field_inline_clear_prover_outputs_verify`.
- Verify top-level ZK fixture without field-inline: covered by
  `top_level_zk_prover_outputs_verify`.
- Verify top-level ZK field-inline fixture: covered by
  `top_level_field_inline_zk_prover_outputs_verify`.
- Verify that the Stage 3-8 committed-frontier shell tests remain regression
  tests only; they are no longer replacement acceptance once top-level ZK lands.
- Run harness static checks.
- Run `cargo nextest` focused tests for the touched packages.

## Stop Conditions

Do not mark full replacement complete if any of these are true:

- The public `jolt-prover` top-level API still returns `FrontierNotImplemented`
  for any required feature combination.
- A proof is accepted only because data was spliced from `jolt-core`.
- `JoltProofClaims::Zk` contains placeholder or fixture BlindFold data.
- Stage 8 final opening uses a dense reference path where the core path has a
  specialized algorithm.
- Certified ledger rows cannot load their canonical evidence files.
- The verifier accepts only stage-frontier shells, not the public top-level
  `jolt-prover` API.

## Practical Next Commit Sequence

1. Commit the BlindFold Stage 6 field-inline bytecode projection fix, the
   backend-routed BlindFold row/error/opening/folding hooks, and the updated
   stage spec.
2. Keep the focused top-level acceptance tests and static harness checks as the
   required non-benchmark pre-merge gate.
3. Restore or generate focused evidence for BlindFold and any new critical-path
   rows only after correctness is green and benchmark runs are allowed.
4. Run canonical full-proof performance only after correctness is green and
   benchmark runs are allowed.
