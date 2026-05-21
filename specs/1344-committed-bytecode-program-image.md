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

Run the local CI subset that does not require wasm, zk-lean, Ubuntu-only behavior, or installing external tools:

```bash
cargo fmt --all --check
cargo clippy --all --all-targets --features allocative,host
cargo clippy --all --all-targets --features allocative,host,zk
cargo clippy --all --all-targets --no-default-features
cargo nextest run --release -p jolt-core
cargo nextest run --release -p jolt-core --features zk

# Modular crates and inline crates.
cargo nextest run -p "$pkg" "${features[@]}" --no-tests=pass
cargo nextest run --features host -p "$inline_pkg" --no-tests=pass

cargo nextest run --release -p tracer --features test-utils
bash jolt-sdk/tests/gen-fixtures.sh
cargo nextest run --release -p jolt-sdk --features host
```

Run targeted standard and ZK e2e coverage when iterating on committed-program behavior:

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

The wasm verifier, zk-lean extractor/model jobs, and external-tool jobs such as `taplo`, `cargo machete`, and `typos` are intentionally outside this local validation set.

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

## Execution

Implementation is organized as:

1. Add committed program metadata and preprocessing in `jolt-core/src/zkvm/program.rs`.
2. Add committed bytecode lane layout and chunk coefficient construction in `jolt-core/src/zkvm/bytecode/chunks.rs`.
3. Add `BytecodeChunk(i)` and `ProgramImageInit` committed polynomial variants.
4. Add bytecode and program-image virtual polynomial IDs for staged claims.
5. Add shared precommitted scheduling in `jolt-core/src/zkvm/claim_reductions/precommitted.rs`.
6. Add `BytecodeClaimReduction` over staged bytecode val claims and committed bytecode chunks.
7. Add `ProgramImageClaimReduction` over staged program-image RAM contribution and committed program-image opening.
8. Wire committed-mode reductions into Stage 6b and Stage 7.
9. Wire committed-mode openings into Stage 8 RLC and BlindFold opening-proof constraints.
10. Add SDK committed preprocessing helpers and canonical `fibonacci` / `muldiv` example CLI paths.
11. Add committed-mode e2e tests and precommitted scheduling regression tests.

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
