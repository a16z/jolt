# Spec: Committed Bytecode And Program Image Openings

| Field       | Value                                                                  |
|-------------|------------------------------------------------------------------------|
| Author(s)   | Amirhossein Khajehpour, Quang Dao                                      |
| Created     | 2026-05-11                                                             |
| Status      | draft                                                                  |
| PR          | [#1344](https://github.com/a16z/jolt/pull/1344)                       |

## Summary

This PR adds a committed program mode for bytecode and program-image openings.
In full mode, verifier preprocessing contains enough bytecode and RAM preprocessing to evaluate the bytecode and initial program image directly.
In committed mode, prover and verifier preprocessing instead agree on Dory commitments to bytecode chunks and to the initial program-image polynomial, then the prover proves all required bytecode/program-image openings through claim reductions that feed the existing Stage 8 batched Dory opening proof.

The motivation is recursive and verifier-side efficiency.
Bytecode and program-image data are program constants, so a verifier should not have to repeatedly materialize and directly evaluate those tables when a preprocessing commitment and a succinct opening proof can bind the same data.

## Intent

### Goal

Introduce a committed program mode that commits bytecode chunks and the initial program image, reduces all execution-derived claims about those committed polynomials into Stage 8, and preserves the same zkVM execution relation in both full and committed modes.

The new proving-system surface is:

- `ProgramMode::{Full, Committed}` in `jolt-core/src/zkvm/config.rs`.
- `ProgramPreprocessing::{Full, Committed}` in `jolt-core/src/zkvm/program.rs`.
- `CommittedPolynomial::BytecodeChunk(i)` and `CommittedPolynomial::ProgramImageInit` in `jolt-core/src/zkvm/witness.rs`.
- `VirtualPolynomial::BytecodeValStage(i)`, `VirtualPolynomial::BytecodeClaimReductionIntermediate`, and `VirtualPolynomial::ProgramImageInitContributionRw`.
- `SumcheckId::{BytecodeReadRafAddressPhase, BooleanityAddressPhase, BytecodeClaimReductionCyclePhase, BytecodeClaimReduction, ProgramImageClaimReductionCyclePhase, ProgramImageClaimReduction}`.
- Shared precommitted scheduling through `PrecommittedClaimReduction` in `jolt-core/src/zkvm/claim_reductions/precommitted.rs`.

### Invariants

- Full and committed program modes must prove the same guest execution relation.
- Committed mode must not let the verifier accept a proof for bytecode or program-image data that differs from the committed preprocessing.
- Prover and verifier must derive the same `ProgramMetadata`, bytecode chunk count, bytecode chunk geometry, program-image geometry, Dory layout, and precommitted scheduling reference.
- `bytecode_chunk_count` must be nonzero, at most `256`, a power of two, and must divide `bytecode_len`.
- The committed bytecode lane layout must encode the same values read by bytecode read-RAF: `rs1`, `rs2`, `rd`, unexpanded PC, immediate, circuit flags, instruction flags, lookup-table selector, and RAF flag.
- Every committed bytecode chunk polynomial must have length `committed_lanes() * (bytecode_len / bytecode_chunk_count)`.
- The program-image polynomial must be the RAM preprocessing bytecode-word slice padded to a power of two, with no semantic rewriting.
- Stage 4 program-image virtual claims must use the same remapped bytecode start address and RAM address challenge as the later program-image claim reduction.
- The precommitted opening-point permutation must be identical for prover, verifier, and Stage 8 RLC construction.
- A precommitted polynomial that does not participate in some cycle or address rounds must contribute exactly one factor of `1/2` per skipped round.
- The cycle-phase handoff scale and full final scale must not be conflated.
- In ZK mode, every `input_claim`, `output_claim_constraint`, and `*_constraint_challenge_values` formula for bytecode, program image, and advice reductions must match the non-ZK claim formula exactly.
- Stage 8 must use the same ordered opening IDs and the same `gamma_i * lagrange_i` coefficients in the prover's BlindFold opening-proof data and in the verifier's BlindFold constraints.

Keep the implementation PR focused on direct prover/verifier tests and targeted unit tests.
New `jolt-eval` invariants for committed-program equivalence and Stage 8 opening-order consistency are useful follow-up work, but are not required to merge this PR.

### Non-Goals

- Do not redesign bytecode expansion or move program construction into `jolt-program`.
  That work is covered by `specs/bytecode-expansion-crate.md` and PR [#1490](https://github.com/a16z/jolt/pull/1490).
- Do not change RISC-V instruction semantics, bytecode row semantics, RAM semantics, or advice semantics.
- Do not remove full program mode.
- Do not add separate Dory opening proofs for bytecode or program image.
  Committed mode must batch these openings into the existing Stage 8 proof.
- Do not introduce compatibility shims for the old single-stage-6 proof serialization format.
- Do not make committed bytecode the default SDK path in this PR.
- Do not require external consumers to adopt a stable public committed-program API beyond the SDK helpers added for this branch.

## Evaluation

### Acceptance Criteria

- [x] `ProgramPreprocessing::preprocess` still builds full bytecode and RAM preprocessing from decoded program data.
- [x] `ProgramPreprocessing::commit` converts full preprocessing into committed preprocessing by deriving bytecode chunk commitments, program-image commitments, and prover hints.
- [x] Serialized committed verifier preprocessing contains metadata and commitments, not prover-only full program data or opening hints.
- [x] `JoltSharedPreprocessing::new_committed` validates chunking, computes the committed maximum Dory variable count, derives a prover setup, commits program preprocessing, and updates `program_meta`.
- [x] Full mode continues to use the existing bytecode/RAM preprocessing semantics.
- [x] Committed mode appends `BytecodeChunk(i)` and `ProgramImageInit` to the expected proof commitment list.
- [x] Stage 4 caches `ProgramImageInitContributionRw` as a virtual opening when committed program mode is active.
- [x] Stage 6a verifies address-phase bytecode read-RAF and booleanity claims without needing full bytecode materialization on the verifier path.
- [x] Stage 6b includes bytecode and program-image claim reduction instances only in committed mode.
- [x] Stage 6b bytecode claim reduction converts staged `BytecodeValStage(i)` claims into committed bytecode chunk openings.
- [x] Stage 6b program-image claim reduction converts `ProgramImageInitContributionRw` into a committed `ProgramImageInit` opening.
- [x] Stage 7 address-phase reductions resume from the cycle-phase intermediate claims for all precommitted reductions that have address rounds.
- [x] Stage 8 includes bytecode chunks and program image in the random linear combination exactly when `ProgramMode::Committed` is used.
- [x] ZK mode BlindFold constraints include bytecode and program-image opening IDs with coefficients matching Stage 8 RLC coefficients.
- [x] Proof serialization includes `stage6a_sumcheck_proof`, `stage6b_sumcheck_proof`, and the new committed/virtual polynomial tags.
- [x] SDK macro output includes committed prover/shared preprocessing helpers that accept `bytecode_chunk_count`.
- [x] At least one end-to-end Dory test proves and verifies in committed program mode.

### Testing Strategy

Run standard and ZK e2e coverage:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo nextest run -p jolt-core muldiv_e2e_dory_committed_program_commitments --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv_e2e_dory_committed_program_commitments --cargo-quiet --features host,zk
```

Run advice/precommitted regression coverage because advice shares the same precommitted scheduling path:

```bash
cargo nextest run -p jolt-core advice_e2e_dory --cargo-quiet --features host
RUST_MIN_STACK=33554432 cargo nextest run -p jolt-core advice_e2e_dory --cargo-quiet --features host,zk
cargo nextest run -p jolt-core final_advice_output_scale --cargo-quiet --features host
```

Run strict linting:

```bash
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
```

Targeted tests should cover:

- invalid bytecode chunk counts,
- bytecode chunk coefficient layout for representative instructions,
- `ProgramPreprocessing::{Full, Committed}` serialization roundtrips,
- proof serialization roundtrips for `BytecodeChunk(i)` and `ProgramImageInit`,
- `PrecommittedClaimReduction` identity and non-identity permutations,
- skipped-round scaling in cycle-only and cycle-plus-address cases,
- Stage 8 opening ID ordering in full mode and committed mode.

### Performance

Committed mode should reduce verifier and recursive-verifier work by replacing direct bytecode/program-image evaluation with committed openings.
The prover may pay extra preprocessing and Stage 8 work, but those costs must be batched into the existing Dory opening proof rather than paid as separate PCS proofs.

Performance-sensitive paths:

- `build_committed_bytecode_chunk_coeffs` must remain linear in nonzero bytecode lane work and avoid dense per-lane overhead beyond the committed chunk vector itself.
- Program-image commitment derivation should be linear in the padded program-image length.
- Stage 8 streaming RLC construction must consume committed bytecode chunks and the program image through `PrecommittedPolynomial` without regenerating unrelated witness polynomials.
- Verifier Stage 8 must combine commitments homomorphically and must not require committed bytecode or program-image coefficients.

Benchmarks should compare full mode and committed mode on at least `muldiv`, `sha2`, and one larger bytecode example.
The expected verifier-side direction is improvement for committed mode; prover preprocessing may increase.

## Design

### Architecture

Committed mode extends the program preprocessing pipeline:

```text
Decoded program data
  -> FullProgramPreprocessing {
       bytecode: BytecodePreprocessing,
       ram: RAMPreprocessing,
     }
  -> ProgramPreprocessing::Committed {
       meta: ProgramMetadata,
       bytecode_commitments: TrustedBytecodeCommitments,
       program_commitments: TrustedProgramCommitments,
       prover_data: Option<CommittedProgramProverData>,
     }
```

`ProgramMetadata` records the verifier-facing program shape: entry address, minimum bytecode address, program-image word length, and bytecode length.
Committed verifier preprocessing keeps only metadata and commitments.
Committed prover preprocessing may retain full preprocessing and opening hints so the prover can construct witnesses and Stage 8 opening hints.

### Program Mode And Proof Surface

`ProgramMode::Full` is the legacy behavior: verifier preprocessing has the full bytecode and program image available.
`ProgramMode::Committed` means bytecode chunks and program image are committed preprocessing objects, and all claims about them must be proven through the protocol.

The proof and preprocessing surface changes in three places:

- `JoltSharedPreprocessing` stores `ProgramPreprocessing<PCS>` and `bytecode_chunk_count`.
- `JoltProof` stores separate `stage6a_sumcheck_proof` and `stage6b_sumcheck_proof`.
- `CommittedPolynomial` and `VirtualPolynomial` serialization add the tags needed for committed bytecode/program-image reductions.

This is a wire-format change for proof serialization.
Old proofs with a single Stage 6 field are not expected to deserialize.

### Committed Bytecode Polynomial

Committed bytecode is represented as one or more chunk polynomials.
Each chunk has a fixed lane capacity:

```text
committed_lanes()
  = next_power_of_two(
      3 * REGISTER_COUNT
    + 2
    + NUM_CIRCUIT_FLAGS
    + NUM_INSTRUCTION_FLAGS
    + number_of_lookup_tables
    + 1
    )
```

The lane layout is:

- one-hot `rs1`,
- one-hot `rs2`,
- one-hot `rd`,
- scalar unexpanded PC,
- scalar immediate,
- circuit flags,
- instruction flags,
- lookup table selector,
- RAF flag.

For a bytecode table of length `T_bc` split into `C` chunks, each chunk has cycle length `T_bc / C`.
The implementation caps `C` at `256`, matching the `u8` serialization used for `BytecodeChunk(i)`.
The chunk polynomial has dimensions `committed_lanes() * (T_bc / C)`.
The coefficient index is derived through the active Dory layout so that commitment-time layout and opening-time layout agree.

### Committed Program Image Polynomial

The program-image polynomial is the initial RAM bytecode-word region from `RAMPreprocessing`.
It is padded to a power of two and committed as `CommittedPolynomial::ProgramImageInit`.

The verifier does not hold the full program-image word slice in committed mode.
Instead, Stage 4 caches the scalar inner product:

```text
C_rw = sum_j ProgramWord[j] * eq(r_address_rw, start_index + j)
```

The prover computes this from RAM preprocessing and appends it as `VirtualPolynomial::ProgramImageInitContributionRw` under `SumcheckId::RamValCheck`.
The verifier appends the same virtual opening point without the value.
The later program-image claim reduction proves that this staged scalar is consistent with the committed `ProgramImageInit` polynomial.

### Shared Precommitted Geometry

Bytecode chunks, program image, trusted advice, and untrusted advice are precommitted polynomials.
They may have fewer or more variables than the main trace-domain polynomials.
`PrecommittedClaimReduction::scheduling_reference` computes a shared reference domain from:

- main trace-domain total variables `log_T + log_K_chunk`,
- committed bytecode chunk variables,
- committed program-image variables,
- trusted advice variables,
- untrusted advice variables.

The reference domain determines:

- `reference_total_vars`,
- cycle alignment rounds,
- address rounds,
- joint column variables,
- each precommitted polynomial's projected Dory opening-round permutation,
- active cycle and address rounds for each precommitted polynomial.

When a precommitted polynomial is smaller than the reference domain, inactive rounds are skipped by multiplying the running claim by `1/2`.
When a precommitted polynomial dominates the main geometry, Stage 8 is anchored by the dominant precommitted opening point.
If multiple dominant precommitted openings exist, prover and verifier require them to agree.

### Proving-System Stage Changes

#### Preprocessing

Full preprocessing still produces bytecode preprocessing and RAM preprocessing.
Committed preprocessing starts from the same full preprocessing, then:

1. derives Dory commitments for bytecode chunks,
2. derives a Dory commitment for the program image,
3. stores metadata and commitments for verifier preprocessing,
4. stores full preprocessing and Dory opening hints only for prover preprocessing.

Bytecode chunk commitments are derived sequentially under one Dory context because Dory context selection is process-global.
The default chunk count is `1`, so this does not remove parallelism from the default committed path.

The shared preprocessing digest binds the serialized committed preprocessing to the Fiat-Shamir transcript.

#### Stage 4: RAM Val Check

Committed mode adds a program-image virtual claim to the RAM val-check flow.
After `RamVal` is opened at the read-write point, prover and verifier split out the RAM address component.
The prover evaluates the initial program-image word slice against that address equality polynomial and appends the scalar as `ProgramImageInitContributionRw`.
The verifier appends the same opening point so later constraints can refer to it.

This does not replace the RAM val check.
It stages a program-image scalar that will be bound to the committed program-image polynomial by a later claim reduction.

#### Stage 6a: Address-Phase Bytecode RAF And Booleanity

Stage 6 is split into `stage6a` and `stage6b`.
Stage 6a handles address-phase work for bytecode read-RAF and booleanity.

In committed mode, bytecode read-RAF verifier construction does not require full bytecode preprocessing.
Instead, address-phase bytecode read-RAF stages the `BytecodeValStage(i)` virtual claims that summarize the bytecode value columns needed later.
These staged values become the input claims to `BytecodeClaimReduction`.

#### Stage 6b: Cycle-Phase Work And Precommitted Claim Reductions

Stage 6b batches the remaining cycle-oriented sumchecks and all precommitted claim reductions.
The base Stage 6b batch still includes:

- bytecode read-RAF cycle phase,
- booleanity cycle phase,
- RAM hamming booleanity,
- RAM RA virtualization,
- instruction RA virtualization,
- increment claim reduction.

When advice commitments are present, Stage 6b also includes trusted and/or untrusted `AdviceClaimReduction`.
When committed program mode is active, Stage 6b also includes:

- `BytecodeClaimReduction`,
- `ProgramImageClaimReduction`.

`BytecodeClaimReduction` input in cycle phase is:

```text
sum_i eta^i * BytecodeValStage(i)
```

where `eta` is sampled from the transcript.
The output is either an intermediate virtual claim at `BytecodeClaimReductionCyclePhase` or final committed bytecode chunk openings if no address phase remains.

`ProgramImageClaimReduction` input in cycle phase is `ProgramImageInitContributionRw`.
The output is either an intermediate committed claim under `ProgramImageClaimReductionCyclePhase` or the final committed `ProgramImageInit` opening if no address phase remains.

#### Stage 7: Address-Phase Claim Reduction Completion

Stage 7 completes address-phase rounds for reductions that still have address variables.
For bytecode, the address phase reduces the intermediate claim into openings of `CommittedPolynomial::BytecodeChunk(i)`.
For program image, the address phase reduces the intermediate claim into an opening of `CommittedPolynomial::ProgramImageInit`.
For advice, the address phase reduces the cycle handoff claim into the final advice opening.

All three use the same precommitted scheduling abstraction, so their opening points are normalized consistently for Stage 8.

#### Stage 8: Batched Dory Opening

Stage 8 constructs one unified opening point.
It then gathers claims for:

- `RamInc`,
- `RdInc`,
- instruction RA polynomials,
- bytecode RA polynomials,
- RAM RA polynomials,
- optional trusted advice,
- optional untrusted advice,
- committed bytecode chunks in committed mode,
- committed program image in committed mode.

Each claim whose natural opening point is smaller than the unified Dory point is multiplied by `compute_lagrange_factor(unified_point, polynomial_point)`.
The transcript samples `gamma` powers after non-ZK claims are absorbed.
The prover constructs the streaming RLC polynomial with all main and precommitted polynomials and proves one Dory opening at the unified point.
The verifier computes the same joint commitment homomorphically from proof commitments, trusted advice commitments, bytecode chunk commitments, and program-image commitments.

In ZK mode, the Stage 8 claim values remain hidden.
Instead, BlindFold receives:

```text
OpeningProofData {
  opening_ids,
  constraint_coeffs = gamma_i * lagrange_i,
  joint_claim,
  y_blinding,
}
```

The verifier builds the same `opening_ids` list through `stage8_opening_ids`.
In committed mode, this list appends each `BytecodeChunk(i)` at `SumcheckId::BytecodeClaimReduction` and `ProgramImageInit` at `SumcheckId::ProgramImageClaimReduction`.

### Alternatives Considered

The verifier could keep direct access to full bytecode and program-image data.
That preserves the old implementation but does not reduce recursive verifier cost.

The prover could produce separate Dory openings for bytecode and program image.
That is simpler locally but loses Stage 8 batching and adds verifier work.

Bytecode, program image, and advice could each use bespoke scheduling.
The PR instead uses one precommitted scheduling abstraction so all non-main committed objects share the same Dory embedding logic.

## Documentation

The Jolt book should document:

- the difference between full and committed program modes,
- why bytecode and program image are precommitted polynomials,
- how precommitted geometry changes the Stage 8 Dory opening point,
- how dominant precommitted polynomials anchor Stage 8,
- how committed bytecode chunk count affects preprocessing.

This PR already expands `book/src/how/architecture/opening-proof.md` with a precommitted geometry section.
Follow-up documentation should add a user-facing example for `--committed-bytecode <chunk_count>` and guidance for choosing the chunk count.

The module comments in `jolt-core/src/zkvm/transpilable_verifier.rs` should also be updated to describe Stage 6a and Stage 6b instead of the old monolithic Stage 6.

## Modular Stack TODOs

The current PR is an end-state implementation.
The stack below is the intended source-control recipe for splitting it into reviewable PRs that remain buildable one step at a time.
Each slice is based on the previous slice, owns a narrow behavioral boundary, and must leave full mode passing before the next slice starts.

### Stack Rules

- [ ] Use one branch per slice, with the suggested branch names below or an equivalent `stack/NN-*` naming scheme.
- [ ] Keep each slice self-contained: no TODO-gated compile failures, no intentionally broken intermediate proof serialization, and no hidden dependency on later slices.
- [ ] Preserve full mode semantics until slice 7 explicitly wires committed mode into the prover/verifier protocol.
- [ ] Run `cargo fmt -q` before every slice PR.
- [ ] Run at least `cargo check -p jolt-core -q --features host --all-targets` for slices that touch `jolt-core`.
- [ ] Run the `muldiv` host and host,zk e2e checks for any slice that changes prover/verifier stages, transcript flow, proof serialization, BlindFold constraints, or witness opening IDs.
- [ ] Run the advice e2e checks when changing shared precommitted scheduling or advice claim reductions.
- [ ] Do not add compatibility shims for old proof formats; each intermediate proof format only needs to be internally consistent for that slice.

### 00 Stack Automation And Spec

Suggested branch: `stack/00-bytecode-stack-automation`.

Goal: create the branch stack control plane and make this spec the source of truth for the split.

TODOs:

- [ ] Add stack metadata such as `stack/branches.tsv` with one row per slice.
- [ ] Add or adapt stack automation scripts so each slice can restore only its owned paths from the monolithic source branch.
- [ ] Add a `STACK.md` summary that points to this spec and records the exact branch order.
- [ ] Mark path ownership for shared files that will be touched by multiple slices, especially `jolt-core/src/zkvm/prover.rs`, `jolt-core/src/zkvm/verifier.rs`, `jolt-core/src/zkvm/proof_serialization.rs`, `jolt-core/src/zkvm/witness.rs`, and `jolt-core/src/poly/commitment/dory/commitment_scheme.rs`.
- [ ] Do not change prover/verifier code in this slice.

Checkpoint:

- [ ] Stack scripts can print or dry-run the intended branch order.
- [ ] The repository remains unchanged outside stack metadata and specs.

### 01 Program Preprocessing Refactor

Suggested branch: `stack/01-program-preprocessing-refactor`.

Goal: introduce the program preprocessing wrapper while preserving the existing full-program behavior.

TODOs:

- [ ] Add `FullProgramPreprocessing` in `jolt-core/src/zkvm/program.rs`.
- [ ] Add `ProgramPreprocessing::Full` without adding committed-mode data yet.
- [ ] Change `JoltSharedPreprocessing` to store `ProgramPreprocessing<PCS>`.
- [ ] Cut existing shared preprocessing construction over to the wrapper.
- [ ] Update all call sites that read bytecode/RAM preprocessing to match on or borrow the full variant explicitly.
- [ ] Keep `JoltSharedPreprocessing::new` behavior unchanged for callers.
- [ ] Keep proof serialization, witness polynomial IDs, Stage 6, Stage 8, SDK helpers, and CLI behavior unchanged.

Checkpoint:

- [ ] Full-mode `muldiv` passes in host mode.
- [ ] Full-mode `muldiv` passes in host,zk mode.
- [ ] The diff is a pure data-shape refactor: no committed proof behavior exists yet.

### 02 Committed Preprocessing Data Model

Suggested branch: `stack/02-committed-preprocessing-model`.

Goal: add committed-program preprocessing data structures and validation without activating committed protocol proofs.

TODOs:

- [ ] Add `ProgramMode::Committed`.
- [ ] Add `ProgramPreprocessing::Committed`.
- [ ] Add `ProgramMetadata`, committed bytecode commitment containers, program-image commitment containers, and prover-only committed preprocessing hints.
- [ ] Add committed bytecode chunk count validation: nonzero, power of two, at most `256`, and divides bytecode length.
- [ ] Add committed bytecode lane layout helpers in `jolt-core/src/zkvm/bytecode/chunks.rs`.
- [ ] Add committed program-image preprocessing from the initial RAM bytecode-word region.
- [ ] Add `JoltSharedPreprocessing::new_committed` as a construction API, but do not wire committed mode into the proof protocol yet.
- [ ] Add serialization roundtrips for committed verifier preprocessing.
- [ ] Ensure serialized committed verifier preprocessing excludes prover-only full program data and opening hints.

Checkpoint:

- [ ] Full-mode `muldiv` still passes in host and host,zk modes.
- [ ] Unit tests reject invalid bytecode chunk counts.
- [ ] Unit tests cover committed preprocessing serialization roundtrips.

### 03 Dory And Precommitted Geometry Substrate

Suggested branch: `stack/03-precommitted-geometry-substrate`.

Goal: generalize the Dory/RLC/opening geometry from advice-specific handling to generic precommitted polynomial handling while keeping full mode active and unchanged.

TODOs:

- [ ] Introduce a generic precommitted polynomial descriptor for Dory opening geometry.
- [ ] Centralize precommitted opening-point permutation logic.
- [ ] Centralize `compute_lagrange_factor` usage for precommitted openings.
- [ ] Generalize Stage 8 RLC construction so it can consume precommitted polynomial descriptors without knowing whether they are advice, bytecode, or program image.
- [ ] Generalize verifier-side homomorphic commitment combination for precommitted commitments.
- [ ] Preserve existing trusted and untrusted advice behavior, even if advice is not fully ported to `PrecommittedClaimReduction` until slice 5.
- [ ] Add focused geometry tests for identity permutation, non-identity permutation, dominant precommitted openings, and smaller-than-reference openings.

Checkpoint:

- [ ] Full-mode `muldiv` passes in host and host,zk modes.
- [ ] Existing advice tests still pass if this slice touches advice opening geometry.
- [ ] No bytecode/program-image claim reduction is wired into the protocol yet.

### 04 Stage 6 Split, Full Mode Only

Suggested branch: `stack/04-stage6a-stage6b-full-mode`.

Goal: split the monolithic Stage 6 proof surface into Stage 6a and Stage 6b while preserving full-mode semantics.

TODOs:

- [ ] Replace the single Stage 6 proof field with `stage6a_sumcheck_proof` and `stage6b_sumcheck_proof`.
- [ ] Split prover Stage 6 into address-phase work and cycle-phase work without adding committed-program reductions.
- [ ] Split verifier Stage 6 in the same way and keep transcript challenge order identical for full mode.
- [ ] Update proof serialization for the two Stage 6 fields.
- [ ] Update transpilable verifier data structures and generated verifier logic for the two Stage 6 fields.
- [ ] Update tests and fixtures that deserialize or inspect proofs.
- [ ] Confirm no committed-mode branches are required for this split.

Checkpoint:

- [ ] Full-mode `muldiv` passes in host mode.
- [ ] Full-mode `muldiv` passes in host,zk mode.
- [ ] Proof serialization roundtrips with separate Stage 6a and Stage 6b fields.
- [ ] This is the main intermediate correctness checkpoint before claim-reduction changes.

### 05 Shared Precommitted Claim Reduction And Advice Port

Suggested branch: `stack/05-precommitted-claim-reduction-advice`.

Goal: introduce the shared precommitted claim-reduction abstraction and move advice onto it before bytecode/program-image reductions use it.

TODOs:

- [ ] Add `PrecommittedClaimReduction` in `jolt-core/src/zkvm/claim_reductions/precommitted.rs`.
- [ ] Define the shared scheduling reference: reference variables, cycle rounds, address rounds, joint column variables, and skipped-round scaling.
- [ ] Port trusted advice claim reduction to the shared abstraction.
- [ ] Port untrusted advice claim reduction to the shared abstraction.
- [ ] Preserve advice opening IDs and transcript behavior.
- [ ] Add BlindFold constraint coverage for the shared advice reduction path.
- [ ] Add regression tests for skipped-round scaling in cycle-only and cycle-plus-address cases.
- [ ] Add a regression test for final advice output scale.

Checkpoint:

- [ ] Full-mode `muldiv` passes in host and host,zk modes.
- [ ] `advice_e2e_dory` passes in host mode.
- [ ] `advice_e2e_dory` passes in host,zk mode.
- [ ] `final_advice_output_scale` passes in host mode.

### 06 Committed Bytecode And Program-Image Reductions

Suggested branch: `stack/06-bytecode-program-image-reductions`.

Goal: add the committed bytecode and program-image claim-reduction modules and IDs, but keep committed mode either minimally wired behind direct tests or inactive in end-to-end proving.

TODOs:

- [ ] Add `CommittedPolynomial::BytecodeChunk(i)` and `CommittedPolynomial::ProgramImageInit`.
- [ ] Add `VirtualPolynomial::BytecodeValStage(i)`, `VirtualPolynomial::BytecodeClaimReductionIntermediate`, and `VirtualPolynomial::ProgramImageInitContributionRw`.
- [ ] Add `SumcheckId` variants for bytecode read-RAF address phase, bytecode claim reduction cycle phase, bytecode claim reduction, program-image claim reduction cycle phase, and program-image claim reduction.
- [ ] Add `jolt-core/src/zkvm/claim_reductions/bytecode.rs`.
- [ ] Add `jolt-core/src/zkvm/claim_reductions/program_image.rs`.
- [ ] Implement bytecode reduction inputs from staged `BytecodeValStage(i)` claims.
- [ ] Implement program-image reduction input from `ProgramImageInitContributionRw`.
- [ ] Implement direct module tests for prover/verifier parameter agreement and BlindFold constraint formulas.
- [ ] Add proof serialization roundtrips for the new committed and virtual polynomial tags.

Checkpoint:

- [ ] Full-mode `muldiv` passes in host and host,zk modes.
- [ ] Bytecode claim-reduction unit tests pass.
- [ ] Program-image claim-reduction unit tests pass.
- [ ] No committed-mode e2e is required yet.

### 07 Committed-Mode Protocol Wiring

Suggested branch: `stack/07-committed-mode-protocol-wiring`.

Goal: activate committed mode end-to-end in prover, verifier, Stage 4, Stage 6b, Stage 7, Stage 8, and BlindFold.

TODOs:

- [ ] Wire Stage 4 to stage `ProgramImageInitContributionRw` when `ProgramMode::Committed` is active.
- [ ] Wire Stage 6a bytecode address-phase work to stage `BytecodeValStage(i)` claims without verifier full-bytecode materialization.
- [ ] Wire Stage 6b to include bytecode and program-image reductions in committed mode.
- [ ] Wire Stage 7 to complete address-phase claim reductions for bytecode, program image, and advice through the shared scheduler.
- [ ] Wire Stage 8 RLC construction to include committed bytecode chunks and program image exactly in committed mode.
- [ ] Wire verifier Stage 8 commitment combination to use committed preprocessing commitments.
- [ ] Wire `stage8_opening_ids` so prover and verifier use the same ordered opening IDs.
- [ ] Wire BlindFold opening-proof constraints for bytecode and program-image IDs with `gamma_i * lagrange_i` coefficients.
- [ ] Add committed-mode verifier checks that reject proofs using bytecode or program-image data inconsistent with the commitments.
- [ ] Add the first committed-mode Dory e2e test.

Checkpoint:

- [ ] Full-mode `muldiv` passes in host and host,zk modes.
- [ ] `muldiv_e2e_dory_committed_program_commitments` passes in host mode.
- [ ] `muldiv_e2e_dory_committed_program_commitments` passes in host,zk mode.
- [ ] Stage 8 opening-order tests pass in full mode and committed mode.

### 08 SDK, Examples, Transpiler, And Documentation

Suggested branch: `stack/08-sdk-examples-transpiler-docs`.

Goal: expose committed preprocessing through user-facing helpers and update docs after the protocol is already correct.

TODOs:

- [ ] Add SDK macro-generated committed preprocessing helpers that accept `bytecode_chunk_count`.
- [ ] Add or update `--committed-bytecode` CLI/example paths.
- [ ] Update wasm build helpers for committed preprocessing.
- [ ] Update transpilable verifier output and docs for Stage 6a/Stage 6b.
- [ ] Update `book/src/how/architecture/opening-proof.md` with committed/precommitted geometry.
- [ ] Add user-facing documentation for committed mode and chunk-count selection.
- [ ] Keep committed bytecode opt-in; do not make it the default SDK path in this slice.

Checkpoint:

- [ ] SDK-generated full-mode helpers still compile.
- [ ] SDK-generated committed helpers compile.
- [ ] Example committed-mode `muldiv` or `fibonacci` path proves and verifies locally.
- [ ] Book documentation builds if this repository has the book toolchain installed.

### 09 Cleanup, Performance, And Regression Tests

Suggested branch: `stack/09-cleanup-perf-regression-tests`.

Goal: remove leftover monolithic-PR rough edges, add regression coverage, and measure the committed-mode tradeoff.

TODOs:

- [ ] Audit and remove dead code left behind by the split.
- [ ] Fix Dory setup cache behavior if committed preprocessing still repeats setup unnecessarily.
- [ ] Expose or remove `jolt-cpp` helpers depending on whether they are still needed after the stack split.
- [ ] Add targeted regression tests for Stage 8 opening ID ordering.
- [ ] Add targeted regression tests for committed bytecode lane coefficients on representative instructions.
- [ ] Add targeted regression tests for committed verifier preprocessing serialization.
- [ ] Benchmark full mode and committed mode on `muldiv`, `sha2`, and one larger bytecode example.
- [ ] Record any remaining follow-up work outside the merge-blocking PR stack.

Checkpoint:

- [ ] Full-mode and committed-mode `muldiv` e2e tests pass in host and host,zk modes.
- [ ] Advice regression tests pass in host and host,zk modes.
- [ ] Strict clippy passes in host mode.
- [ ] Strict clippy passes in host,zk mode.

### End-State File Map

The complete feature is expected to touch these implementation areas by the end of the stack:

1. Committed program metadata and preprocessing in `jolt-core/src/zkvm/program.rs`.
2. Committed bytecode lane layout and chunk coefficient construction in `jolt-core/src/zkvm/bytecode/chunks.rs`.
3. `BytecodeChunk(i)` and `ProgramImageInit` committed polynomial variants.
4. Bytecode and program-image virtual polynomial IDs for staged claims.
5. Shared precommitted scheduling in `jolt-core/src/zkvm/claim_reductions/precommitted.rs`.
6. `BytecodeClaimReduction` over staged bytecode val claims and committed bytecode chunks.
7. `ProgramImageClaimReduction` over staged program-image RAM contribution and committed program-image opening.
8. Committed-mode reductions in Stage 6b and Stage 7.
9. Committed-mode openings in Stage 8 RLC and BlindFold opening-proof constraints.
10. SDK committed preprocessing helpers and canonical `fibonacci` / `muldiv` example CLI paths.
11. Committed-mode e2e tests and precommitted scheduling regression tests.

## References

- PR [#1344](https://github.com/a16z/jolt/pull/1344)
- Related program preprocessing spec: `specs/bytecode-expansion-crate.md`
- Spec template: `specs/TEMPLATE.md`
- Opening proof documentation: `book/src/how/architecture/opening-proof.md`
- Core committed program code: `jolt-core/src/zkvm/program.rs`
- Committed bytecode code: `jolt-core/src/zkvm/bytecode/chunks.rs`
- Precommitted scheduling: `jolt-core/src/zkvm/claim_reductions/precommitted.rs`
- Bytecode claim reduction: `jolt-core/src/zkvm/claim_reductions/bytecode.rs`
- Program-image claim reduction: `jolt-core/src/zkvm/claim_reductions/program_image.rs`
