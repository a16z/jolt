# Modular Verifier Parity + Cross-Verifier Soundness Spec

## 0. What this document is

A single self-contained spec covering two coupled work streams:

1. **Verifier parity** — bring `jolt_verifier::verify` (the modular verifier) to behavioral parity with `jolt-core`'s verifier so it actually verifies stages 3–7 instead of squeezing past them.
2. **Cross-verifier soundness suite** — a `jolt-equivalence` test framework that proves the modular verifier rejects every tamper class jolt-core's verifier rejects, with rigorous acceptance criteria.

The two streams are independent enough to develop in parallel: the suite can land first as a "currently failing" gate that quantifies the gap, and the parity work closes that gap commit by commit.

The spec is written to be handed to a verifier agent who will execute it. No timeline, no phase ordering — only structural / engineering / soundness requirements.

---

## 1. Background — current state

### 1.1 Verifier paths

There are two compiler paths producing `Module.verifier.ops`:

- **Auto-generated** — `crates/jolt-compiler/src/compiler/emit.rs:100-260`. Emits the full per-stage verifier schedule (`BeginStage` → `VerifySumcheck` → `RecordEvals` → `AbsorbEvals` → `CollectOpeningClaim` → `Squeeze` → `CheckOutput`) for every stage uniformly. Considered the "complete" path in principle, not load-bearing today.
- **Hand-written reference** — `crates/jolt-compiler/examples/jolt_core_module.rs`. The actual module schedule the bench, `modular_self_verify`, and the cross-system equivalence tests use. Stages 1–2 are fully wired; **stages 3–4 are stubbed (squeezes only) and stages 5–7 are not assembled at all**. This is what needs fixing.

### 1.2 What's currently checked

`crates/jolt-equivalence/tests/muldiv.rs`:

| Test | Verifies | Limitation |
|---|---|---|
| `transcript_divergence` | byte-for-byte Fiat-Shamir parity with jolt-core (prover side) | only checks transcript state; squeezing without verifying preserves it |
| `zkvm_proof_accepted_by_core_verifier` | jolt-core's verifier accepts modular-prover proofs (round polys + PCS) | substitutes jolt-core's `opening_claims` into the assembled proof — modular's missing eval data is bypassed |
| `modular_self_verify` | `jolt_verifier::verify` accepts modular proofs | only verifies stages 1–2 + opening; stages 3–7 effectively unverified |
| `modular_self_verify_commit_skip_alignment` | `Option<PCS::Output>` skip semantics | narrow scope |

**No negative-soundness test exists.** Tampering a stage 3+ round polynomial or evaluation does not currently cause `modular_self_verify` to reject.

### 1.3 Prover-side gaps

`Op::RecordEvals` is emitted only in `build_stage{1,2}` (the proof's `stage_proofs[i].evals` is empty for `i ≥ 2`). `Op::CollectOpeningClaim` similarly is missing for stages 3–7. The proof object literally does not carry the data the verifier would need.

`Op::Evaluate` and `Op::AbsorbEvals` *are* emitted in stages 3–7 — those keep the transcript synchronized but don't populate the proof's evaluation vector.

---

## 2. Part A — Verifier parity work

### 2.1 Reference: jolt-core verifier per-stage instance set

Source: `crates/jolt-core/src/zkvm/verifier.rs`

| Stage | Instance count | Verifier classes |
|---|---|---|
| 3 | 3 | `ShiftSumcheckVerifier`, `InstructionInputSumcheckVerifier`, `RegistersClaimReductionSumcheckVerifier` |
| 4 | 2 | `RegistersReadWriteCheckingVerifier`, `RamValCheckSumcheckVerifier`. Plus advice opening-claim accumulation (`verifier_accumulate_advice`). |
| 5 | 3 | `InstructionReadRafSumcheckVerifier`, `RamRaClaimReductionSumcheckVerifier`, `RegistersValEvaluationSumcheckVerifier` |
| 6 | 6–8 | `BytecodeReadRafSumcheckVerifier`, `BooleanitySumcheckVerifier`, `HammingBooleanitySumcheckVerifier`, `RamRaVirtualSumcheckVerifier`, `LookupsRaSumcheckVerifier`, `IncClaimReductionSumcheckVerifier`, optionally 1–2 `AdviceClaimReductionVerifier` (Phase 1, conditional on advice commitments). |
| 7 | 1–3 | `HammingWeightClaimReductionVerifier`, optionally 1–2 `AdviceClaimReductionVerifier` (Phase 2 / address phase, conditional on `num_address_phase_rounds > 0`). |

### 2.2 Existing primitives the verifier agent works with

`VerifierOp` enum — already complete (`crates/jolt-compiler/src/module.rs:1803-1879`):

```
Preamble
BeginStage
AbsorbCommitment { poly, tag }
Squeeze { challenge }
AppendDomainSeparator { tag }
AbsorbRoundPoly { num_coeffs, tag }     // for uniskip rounds outside batched sumcheck
VerifySumcheck { instances, stage, batch_challenges, claim_tag, sumcheck_challenge_slots }
RecordEvals { evals }
AbsorbEvals { polys, tag }
CheckOutput { instances, stage, batch_challenges }
CollectOpeningClaim { poly, at_stage }
VerifyOpenings
```

`SumcheckInstance` struct (`module.rs:1888-1900`):

```rust
pub struct SumcheckInstance {
    pub input_claim: ClaimFormula,
    pub output_check: ClaimFormula,
    pub num_rounds: usize,
    pub degree: usize,
    pub normalize: Option<PointNormalization>,
}
```

`ClaimFactor` enum (`module.rs:1962-2087`) — current variants:

```
Eval(PolynomialId)
Challenge(ChallengeIdx)
EqChallengePair { a, b }
EqEval { challenges, at_stage }
LagrangeKernel { tau_challenge, at_challenge }
UniformR1CSEval { matrix, eval_polys, at_challenge, num_constraints, domain_start }
EqEvalSlice { challenges, at_stage, offset }
LagrangeKernelDomain { tau_challenge, at_challenge, domain_size, domain_start }
LagrangeWeight { challenge, domain_size, domain_start, basis_index }
PreprocessedPolyEval { poly, at_stage }
GroupSplitR1CSEval { matrix, eval_polys, at_r0, at_r_group, group0_indices, group1_indices, domain_size, domain_start }
StageEval(usize)
StagedEval { poly, stage }
```

Stage 1 already uses `EqEval`, `LagrangeKernelDomain`, and `GroupSplitR1CSEval` — see `build_verifier_stage1_ops` in `examples/jolt_core_module.rs:1910-2101` for a complete worked example.

### 2.3 Per-stage prover-side requirements

For each of stages 3, 4, 5, 6, 7 in `examples/jolt_core_module.rs::build_stage{N}`:

1. **Enumerate the eval polys** opened at this stage's final point. The list already exists implicitly via the `Op::Evaluate` calls at the end of each `build_stage`.
2. **Add `Op::RecordEvals { polys: stage_eval_polys.clone() }`** after the unrolled batched rounds.
3. **Keep the existing `Op::AbsorbEvals { polys, tag }`** unchanged.
4. **Add `Op::CollectOpeningClaim { poly, at_stage: VerifierStageIndex(N) }`** for every `poly` in `stage_eval_polys` whose `descriptor().committed` is true.
5. For polys opened at points belonging to *earlier* stages (cross-stage opens — needed for some constructions), use `Op::CollectOpeningClaimAt { poly, point_challenges, committed_num_vars }` instead.

**Validation gate** for the prover-side change: `transcript_divergence` (no transcript change because `RecordEvals` doesn't touch the transcript) and `zkvm_proof_accepted_by_core_verifier` must remain green.

### 2.4 Per-stage verifier-side requirements

The agent must create / replace the following functions in `examples/jolt_core_module.rs`:

```
fn build_verifier_stage3_ops(p, params, ch) -> Vec<VerifierOp>    // replace stub
fn build_verifier_stage4_ops(p, params, ch) -> Vec<VerifierOp>    // replace stub
fn build_verifier_stage5_ops(p, params, ch) -> Vec<VerifierOp>    // new
fn build_verifier_stage6_ops(p, params, ch) -> Vec<VerifierOp>    // new
fn build_verifier_stage7_ops(p, params, ch) -> Vec<VerifierOp>    // new
```

And update `build_module()` (line 1406) to include all 7:

```rust
verifier_ops.extend(build_verifier_stage1_ops(&p, params, &ch));
verifier_ops.extend(build_verifier_stage2_ops(&p, params, &ch));
verifier_ops.extend(build_verifier_stage3_ops(&p, params, &ch));
verifier_ops.extend(build_verifier_stage4_ops(&p, params, &ch));
verifier_ops.extend(build_verifier_stage5_ops(&p, params, &ch));
verifier_ops.extend(build_verifier_stage6_ops(&p, params, &ch));
verifier_ops.extend(build_verifier_stage7_ops(&p, params, &ch));
push_verifier_op!(verifier_ops, VerifierOp::VerifyOpenings);
```

For each stage builder, the structure is:

1. Push `VerifierOp::BeginStage`.
2. Push `VerifierOp::Squeeze` for every pre-sumcheck challenge (gammas, batching coefficients) in the order they appear in `ch` for this stage. Match jolt-core exactly — read the corresponding stage's pre-sumcheck challenges in `crates/jolt-core/src/zkvm/verifier.rs::verify_stage{N}`.
3. For batched stages with `n` instances and `r` round challenges: push `VerifierOp::Squeeze` for the `n − 1` batching coefficients (or use `batch_challenges` field on `VerifySumcheck` to handle these internally — mirror stage 1's pattern at line 2074-2083 which uses `batch_challenges: vec![ch_batch]`).
4. Construct `instances: Vec<SumcheckInstance>` with the per-instance `input_claim` and `output_check` formulas (see §2.5).
5. Push `VerifierOp::VerifySumcheck { instances: instances.clone(), stage: N, batch_challenges, claim_tag: Some(DomainSeparator::SumcheckClaim), sumcheck_challenge_slots }`.
6. Push `VerifierOp::RecordEvals { evals }` mirroring the prover's eval list.
7. Push `VerifierOp::AbsorbEvals { polys, tag: DomainSeparator::OpeningClaim }`.
8. For every committed poly in the eval list, push `VerifierOp::CollectOpeningClaim { poly, at_stage: VerifierStageIndex(N) }`.
9. Push `VerifierOp::CheckOutput { instances, stage: N, batch_challenges }`.

### 2.5 Per-instance `SumcheckInstance` authoring

The agent must hand-author `input_claim` and `output_check` `ClaimFormula`s by reading the protocol math out of `jolt-core`:

| Stage | Instance | jolt-core source location | `num_rounds` | `degree` | Notes |
|---|---|---|---|---|---|
| 3 | Shift | `crates/jolt-core/src/zkvm/spartan/shift.rs` | `log_T` | 3 | input_claim = γ-polynomial in `next_unexpanded_pc`, `next_pc`, `next_is_virtual`, `next_is_first`, `1 − next_is_noop`. Already partially authored on prover side at `examples/jolt_core_module.rs:3173`. |
| 3 | InstructionInput | `crates/jolt-core/src/zkvm/spartan/instruction_input.rs` | `log_T` | 2 | γ-polynomial over (left_is_rs1 × rs1_val + left_is_pc × unexpanded_pc + right_is_rs2 × rs2_val + right_is_imm × imm). |
| 3 | RegistersClaimReduction | `crates/jolt-core/src/zkvm/registers/claim_reduction.rs` | `log_T` | 2 | `eq(r_cycle, r_stage3) × rd_write_value` style. |
| 4 | RegistersRWC | `crates/jolt-core/src/zkvm/registers/read_write_checking.rs` | `log_K_REG + log_T = 14` | 3 | Address × cycle product structure — output_check is `eq(r_address, r_addr_part) × eq(r_cycle, r_cycle_part) × <wa, val_pre, val_post composition>`. **Trickiest formula in stages 3-7.** |
| 4 | RamValCheck | `crates/jolt-core/src/zkvm/ram/val.rs` | `log_T = 7` | 3 | `first_active_round = 7` — only second half of batched window. Needs preprocessed `ram_init` poly evaluation via `PreprocessedPolyEval`. |
| 5 | InstructionReadRaf | `crates/jolt-core/src/zkvm/instruction_lookups/read_raf.rs` | `log_T + log_K_INST` | varies | Address-decomposition instance — uses combine_entries × prefix × suffix evaluation. May need new `ClaimFactor` variant for combine-matrix evaluation. |
| 5 | RamRaClaimReduction | `crates/jolt-core/src/zkvm/ram/ra_reduction.rs` | `log_K_RAM` | 2 | RAM analog of registers reduction. |
| 5 | RegistersValEvaluation | `crates/jolt-core/src/zkvm/registers/val.rs` | `log_K_REG` | 1 | Simple eq-eval check. |
| 6 | BytecodeReadRaf | `crates/jolt-core/src/zkvm/bytecode/read_raf.rs` | `log_T + log_K_BYTECODE` | varies | Uses preprocessed bytecode tables via `PreprocessedPolyEval`. |
| 6 | Booleanity | `crates/jolt-core/src/zkvm/instruction_lookups/booleanity.rs` | `log_T + log_K_INST` | 3 | output_check uses `ra² − ra` form. **May require new `ClaimFactor::EvalSquared` or compose via `EvalProduct` if added.** |
| 6 | HammingBooleanity | `crates/jolt-core/src/zkvm/ram/hamming_booleanity.rs` | varies | varies | Same shape as Booleanity but for RAM. |
| 6 | RamRaVirtual | `crates/jolt-core/src/zkvm/ram/ra_virtual.rs` | varies | varies | Virtual RA sumcheck with phase transitions. |
| 6 | LookupsRaVirtual | `crates/jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs` | varies | varies | Same shape, instruction lookups. |
| 6 | IncClaimReduction | `crates/jolt-core/src/zkvm/spartan/inc_reduction.rs` | `log_T` | 2 | rd_inc + γ × ram_inc reduction. |
| 6 | AdviceClaimReduction (Trusted) | `crates/jolt-core/src/zkvm/advice/reduction.rs` | conditional | 2 | Only emitted when `trusted_advice_commitment.is_some()`. |
| 6 | AdviceClaimReduction (Untrusted) | same | conditional | 2 | Only emitted when `untrusted_advice_commitment.is_some()`. |
| 7 | HammingWeightClaimReduction | `crates/jolt-core/src/zkvm/instruction_lookups/hamming_weight.rs` | `log_K_INST + log_K_BYTECODE + log_K_RAM` | varies | Cross-system claim reduction. |
| 7 | AdviceClaimReduction (address phase) | same as above | conditional | 2 | Only emitted when `num_address_phase_rounds > 0`. |

### 2.6 New `ClaimFactor` variants likely needed

Authoring stages 5 and 6 may require additional factors. The agent must propose new variants in `module.rs` (and corresponding handler logic in the verifier dispatcher) when the existing set proves insufficient. Likely candidates:

- `EvalSquared(PolynomialId)` — for Booleanity's `ra² − ra` constraint. Compose existing `Eval` × `Eval` if the formula evaluator handles repeated polys correctly; otherwise add explicitly.
- `CombineEntryEval { kernel: usize, at_stage }` — evaluates the combine_entries × prefix × suffix-at-empty composition. Currently this lives in `compute_combined_val` (the runtime helper); the verifier needs the symbolic form.
- `EqMinusOne { a: ChallengeIdx, b: ChallengeIdx }` — used in some claim-reduction formulas where the verifier checks `1 − eq(a, b)`.

Each new variant requires:
1. Variant addition in `module.rs::ClaimFactor`.
2. Evaluation arm in the verifier formula evaluator (search for `ClaimFactor::Eval` matches in the verifier handlers).
3. Doc comment with the math.
4. At least one tamper test in the cross-verifier suite that exercises it.

### 2.7 Conditional emission for advice instances

Stages 6 and 7 each conditionally include 1–2 `AdviceClaimReduction` instances. The condition at runtime is `proof.trusted_advice_commitment.is_some()` and `proof.untrusted_advice_commitment.is_some()`. At schedule-build time, the equivalent compile-time signal is the presence of `Op::AbsorbCommitment` for the advice polys in stage 1's preamble — if those exist, the advice instances must be emitted in stages 6 and 7.

For muldiv with all-zero advice the prover emits `Op::Commit` but skips the actual commitment (sets `Option<PCS::Output>` to `None`). The verifier-side `AbsorbCommitment` must still be in the schedule, and the advice instances must still be present in stages 6/7 — they account for the `verifier_accumulate_advice` logic in jolt-core.

### 2.8 Validation gates for the parity work

After every commit:

1. `transcript_divergence` green.
2. `zkvm_proof_accepted_by_core_verifier` green.
3. `modular_self_verify` green.
4. `cargo nextest run -p jolt-equivalence` green (full suite).
5. The cross-verifier soundness suite (Part B) — every test for the stage just wired must report `BothReject` for tampers in its scope.

---

## 3. Part B — Cross-verifier soundness suite

### 3.1 Layout

New files:

```
crates/jolt-equivalence/src/cross_verifier/mod.rs
crates/jolt-equivalence/src/cross_verifier/conversion.rs
crates/jolt-equivalence/src/cross_verifier/tamper.rs
crates/jolt-equivalence/src/cross_verifier/runner.rs
crates/jolt-equivalence/src/cross_verifier/categories.rs
crates/jolt-equivalence/src/cross_verifier/registry.rs
crates/jolt-equivalence/tests/cross_verifier_soundness.rs
```

The library code lives in `src/cross_verifier/` so it can be reused across workloads.

### 3.2 Type definitions

```rust
// categories.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Constraint {
    Preamble,
    CommitSlot(usize),
    InputClaim { stage: usize, instance: usize },
    RoundPoly { stage: usize, round: usize },
    OutputCheck { stage: usize, instance: usize },
    EvalConsistency { stage: usize, eval_idx: usize },
    OpeningClaim(usize),
    OpeningProof,
    CommitSkip(usize),
    TranscriptTag(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionCategory {
    PreambleMismatch,
    CommitmentMismatch,
    SumcheckRoundPolySum,
    SumcheckFinalEval,
    OpeningClaimEvalMismatch,
    OpeningProofInvalid,
    TranscriptDivergence,
    StructuralInvalid,
    Other,
}

#[derive(Debug, Clone)]
pub struct TamperPoint {
    pub stage: usize,
    pub location: TamperLocation,
    pub mutate: TamperMutation,
    pub witnesses: Vec<Constraint>,
    pub expected: ExpectedResult,
}

#[derive(Debug, Clone)]
pub enum TamperLocation {
    RoundPolyCoeff { stage: usize, round: usize, coeff: usize },
    Eval { stage: usize, idx: usize },
    Commitment { idx: usize },
    OpeningProof { idx: usize, byte_offset: usize },
    CommitSlot { idx: usize, op: CommitSlotOp },
    Config(ConfigField),
    Tag { absorb_idx: usize },
    Io(IoField),
}

#[derive(Debug, Clone, Copy)]
pub enum CommitSlotOp {
    NoneToSome,
    SomeToNone,
}

#[derive(Debug, Clone, Copy)]
pub enum ConfigField {
    TraceLength,
    RamK,
    InputByte(usize),
    OutputByte(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum IoField {
    InputByte(usize),
    OutputByte(usize),
    PanicFlag,
}

#[derive(Debug, Clone)]
pub enum TamperMutation {
    FlipBit { byte_in_serialized: usize, bit: u8 },
    AddOne,
    Replace(Vec<u8>),
    SwapWith { other_proof_idx: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedResult {
    BothReject,
    BothAccept,
    CoreRejectsModularAccepts,
    CoreAcceptsModularRejects,
}
```

### 3.3 Tamper taxonomy

Eleven categories. Every category exists in the test suite as at least one parameterized test case:

- **T1** — Round polynomial coefficient flip. Per stage 1..=7, on rounds `[0, mid, last]` × coeffs `[0, last]` × mutations `[AddOne, Replace(random_field)]`.
- **T2** — Evaluation tamper. Every position in `stage_proofs[s].evals` for s ∈ 0..=6, mutations `[AddOne]`.
- **T3** — Commitment swap. Each committed poly index in `commitments`. Mutation `SwapWith(other_proof_idx)` using a second pre-generated honest proof from a slightly different witness.
- **T4** — Opening-proof byte flip. 5 random byte offsets across `opening_proofs[0]`. Mutation `FlipBit`.
- **T5** — Commit-slot None ↔ Some. Each `Option<PCS::Output>` slot, both directions.
- **T6** — Config field tamper. `TraceLength`, `RamK`, first input byte, first output byte.
- **T7** — Cross-stage: tamper an eval that flows into a downstream stage's `output_check` formula via `ClaimFactor::StageEval` or `ClaimFactor::Eval`.
- **T8** — Round-poly degree tamper. Truncate or extend a coefficient list.
- **T9** — Batch-claim tamper. Tamper an eval contained in instance `i`'s eval list to produce a wrong combined claim.
- **T10** — Domain-separator tag tamper. Replace `DomainSeparator::SumcheckClaim` with `DomainSeparator::OpeningClaim` at one absorption.
- **T11** — Public I/O tamper. `inputs[0]`, `outputs[0]`, `panic` flag.

### 3.4 Per-stage coverage matrix

The taxonomy must cover the per-stage matrix below — at minimum one tamper case per ✓:

| Stage | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 | T10 | T11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ✓ | ✓ | — | — | — | — | — | ✓ | — | ✓ | — |
| 2 | ✓ | ✓ | — | — | — | — | — | ✓ | ✓ | ✓ | — |
| 3 | ✓ | ✓ | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| 4 | ✓ | ✓ | — | — | ✓ | — | ✓ | ✓ | ✓ | — | — |
| 5 | ✓ | ✓ | — | — | — | — | ✓ | ✓ | ✓ | — | — |
| 6 | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ | ✓ | — | — |
| 7 | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | ✓ | — | — | — |
| 8 | — | — | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| Preamble | — | — | — | — | — | ✓ | — | — | — | — | ✓ |

### 3.5 Constraint coverage table

A compile-time table relates each tamper kind to the constraint(s) it witnesses:

```rust
pub const TAMPER_COVERAGE: &[(TamperKind, &[Constraint])] = &[
    (TamperKind::T1_RoundPolyCoeff, &[
        Constraint::RoundPoly { stage: 0, round: 0 },  // template; expanded per stage
    ]),
    (TamperKind::T2_Eval, &[
        Constraint::OutputCheck { stage: 0, instance: 0 },
        Constraint::OpeningClaim(0),
    ]),
    (TamperKind::T3_Commitment, &[
        Constraint::CommitSlot(0),
        Constraint::OpeningProof,
    ]),
    (TamperKind::T4_OpeningProof, &[Constraint::OpeningProof]),
    (TamperKind::T5_CommitSlot,    &[Constraint::CommitSkip(0)]),
    (TamperKind::T6_Config,        &[Constraint::Preamble]),
    (TamperKind::T7_CrossStage,    &[Constraint::OutputCheck { stage: 0, instance: 0 }]),
    (TamperKind::T8_Degree,        &[Constraint::RoundPoly { stage: 0, round: 0 }]),
    (TamperKind::T9_BatchClaim,    &[Constraint::InputClaim { stage: 0, instance: 0 }]),
    (TamperKind::T10_Tag,          &[Constraint::TranscriptTag(0)]),
    (TamperKind::T11_Io,           &[Constraint::Preamble]),
];
```

The launch-time check `assert_taxonomy_covers_constraints()` enumerates `Constraints(V_core)` from the schedule's structural metadata and asserts the taxonomy's union covers it.

### 3.6 KnownGapRegistry

A literal `pub const KNOWN_GAPS: &[KnownGap]` array enumerates every (stage, tamper-kind) pair where modular currently accepts and core rejects. Each entry has a `rationale` and an `owner`. Initial population (until parity work lands):

```rust
pub const KNOWN_GAPS: &[KnownGap] = &[
    KnownGap { stage: 3, kind: T1_RoundPolyCoeff, rationale: "Stage 3 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 3, kind: T2_Eval,           rationale: "Stage 3 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 3, kind: T7_CrossStage,     rationale: "Stage 3 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 4, kind: T1_RoundPolyCoeff, rationale: "Stage 4 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 4, kind: T2_Eval,           rationale: "Stage 4 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 4, kind: T5_CommitSlot,     rationale: "Stage 4 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 4, kind: T7_CrossStage,     rationale: "Stage 4 verifier stubbed", owner: "verifier-agent" },
    KnownGap { stage: 5, kind: T1_RoundPolyCoeff, rationale: "Stage 5 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 5, kind: T2_Eval,           rationale: "Stage 5 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 5, kind: T7_CrossStage,     rationale: "Stage 5 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 6, kind: T1_RoundPolyCoeff, rationale: "Stage 6 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 6, kind: T2_Eval,           rationale: "Stage 6 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 6, kind: T3_Commitment,     rationale: "Stage 6 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 6, kind: T7_CrossStage,     rationale: "Stage 6 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 7, kind: T1_RoundPolyCoeff, rationale: "Stage 7 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 7, kind: T2_Eval,           rationale: "Stage 7 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 7, kind: T3_Commitment,     rationale: "Stage 7 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 7, kind: T5_CommitSlot,     rationale: "Stage 7 not wired",        owner: "verifier-agent" },
    KnownGap { stage: 7, kind: T7_CrossStage,     rationale: "Stage 7 not wired",        owner: "verifier-agent" },
];
```

Every entry must be removed in the same commit that closes its underlying gap. The `known_gap_registry_consistency` test asserts each remaining entry still produces a `(V_core = Reject, V_mod = Accept)` mismatch — if a registry entry no longer reproduces, the test fails until the entry is removed.

### 3.7 Proof conversion

`crates/jolt-equivalence/src/cross_verifier/conversion.rs` provides:

```rust
pub fn modular_to_core(modular: &jolt_verifier::JoltProof<NewFr, DoryScheme>, params: &CoreProtocolParams)
    -> jolt_core::JoltProof<Fr, DoryCommitmentScheme<...>>;

pub fn core_to_modular(core: &jolt_core::JoltProof<...>, params: &CoreProtocolParams)
    -> jolt_verifier::JoltProof<NewFr, DoryScheme>;
```

Round-trip property: `modular_to_core(core_to_modular(P)) == P` byte-equal under bincode for every honest proof in the fixture.

The existing `to_core_sumcheck_proof`, `commitment_to_ark`, etc. helpers from `muldiv.rs:1483-1553` should be lifted into this module and reused in both directions.

### 3.8 Test runner

`runner.rs`:

```rust
pub struct DualVerifyResult {
    pub core: Result<(), CoreErr>,
    pub modular: Result<(), ModularErr>,
    pub core_category: Option<RejectionCategory>,
    pub modular_category: Option<RejectionCategory>,
}

pub fn run_both_verifiers(
    modular_proof: &jolt_verifier::JoltProof<NewFr, DoryScheme>,
    setup: &VerifierSetup,
) -> DualVerifyResult;

pub fn assert_consistent(
    tamper: &TamperPoint,
    result: &DualVerifyResult,
);
```

`run_both_verifiers` must run both verifiers twice (S3 — determinism check) and assert byte-equal results.

### 3.9 Test entry points

`tests/cross_verifier_soundness.rs`:

```rust
#[test] fn taxonomy_covers_v_core_constraints();
#[test] fn honest_acceptance_equivalence();
#[test] fn known_gap_registry_consistency();

#[test] fn t1_round_poly_coefficients();
#[test] fn t2_eval_tampers();
#[test] fn t3_commitment_swaps();
#[test] fn t4_opening_proof_bytes();
#[test] fn t5_commit_slot_none_some();
#[test] fn t6_config_fields();
#[test] fn t7_cross_stage_dependencies();
#[test] fn t8_round_poly_degree();
#[test] fn t9_batch_claim();
#[test] fn t10_domain_separator();
#[test] fn t11_public_io();

#[test] fn claim_formula_factor_coverage();

#[test]
#[ignore = "expensive — nightly"]
fn fuzz_random_byte_flips();
```

Cached fixture: a `OnceLock<HonestFixture>` constructs the honest jolt-core proof + protocol params once per test binary, amortizing the ~5–10s cost across all tests.

### 3.10 Workload extensibility

The framework must accept the workload as a parameter. Initial workload: `muldiv`. Stretch workloads:

- `sha2-chain` — non-trivial trace, exercises stage 6 advice paths fully (`#[ignore]` until baseline is solid).
- A program with non-empty `TrustedAdvice` — exercises stage 6/7 advice reduction soundness paths.

Workloads are wired via a small struct:

```rust
pub struct Workload {
    pub name: &'static str,
    pub guest_name: &'static str,
    pub inputs: Vec<u8>,
    pub trusted_advice: Vec<u8>,
    pub untrusted_advice: Vec<u8>,
}
```

---

## 4. Part C — Rigorous soundness acceptance conditions

### 4.1 Notation

- `R` — the relation the proof system proves: `R(x, w) = 1` iff witness `w` certifies the statement (program execution + outputs).
- `P` — honest jolt-core prover.
- `P*` — any (possibly malicious) prover.
- `V_core`, `V_mod` — jolt-core's verifier, modular verifier.
- `π` — proof transcript.
- `x` — public statement (`ProgramIO + ProverConfig + r1cs_key + commitments`).
- `T : Π → Π` — a tamper function on the proof space.
- `λ` — soundness security parameter (≈128).
- `negl(λ)` — function asymptotically smaller than `1/poly(λ)`.
- `Verify(V, x, π) ∈ {Accept, Reject}` — verifier decision.

### 4.2 Required absolute soundness properties

**S1 — Completeness.**

```
∀ x ∈ L_R . ∀ honest π = P(x, w) for w with R(x, w) = 1 :
    Pr[Verify(V_mod, x, π) = Accept] ≥ 1 − negl(λ).
```

For our deterministic Fiat-Shamir setting this holds with probability exactly 1.

**S2 — Statistical soundness.** Approximated by S2′ (testable form) below.

**S2′ — Empirical soundness.**

```
∀ T ∈ Taxonomy . ∀ honest π :
    Verify(V_mod, x, T(π)) = Reject  whenever T violates some constraint c ∈ Constraints(V_mod).
```

**S3 — Determinism.**

```
∀ (x, π) :
    Verify(V_mod, x, π) called twice on identical inputs yields identical decisions
    (and identical RejectionCategory in the Reject case).
```

**S4 — Public-coin transcript determinism (cross-verifier).**

```
∀ honest π :
    Transcript_state_after_each_op(V_mod, x, π) = Transcript_state_after_each_op(V_core, x, π)
    at every aligned op position.
```

### 4.3 Cross-verifier rejection equivalence

**Definition.** Two verifiers `V_1, V_2` are *rejection-equivalent* on Π′ ⊆ Π iff:

```
∀ π ∈ Π' :  Verify(V_1, x, π) = Verify(V_2, x, π).
```

**S5 — Rejection equivalence on the test fixture.**

```
∀ honest π in fixture set . ∀ T ∈ Taxonomy :
    Verify(V_core, x, T(π)) = Verify(V_mod, x, T(π)).
```

**S5′ — Rejection equivalence with registered exceptions.**

```
∀ honest π ∀ T ∈ Taxonomy :
    Verify(V_core, x, T(π)) = Verify(V_mod, x, T(π))  OR  (T, π) ∈ KnownGapRegistry.
```

**S6 — Rejection-cause parity (graded).**

```
∀ honest π ∀ T ∈ Taxonomy s.t. both verifiers reject :
    RejectionCategory(V_core, T(π)) ∈ acceptable_set(RejectionCategory(V_mod, T(π))).
```

`acceptable_set` allows `OpeningClaimEvalMismatch ↔ OpeningProofInvalid`. All other categories must match exactly.

### 4.4 Constraint coverage

**Definition.** For verifier `V`, define `Constraints(V) = {c_1, …, c_n}`, where each `c_i` is a structural assertion the verifier checks. For our zkVM:

```
C_preamble                   — absorbed config + IO matches preprocessing
C_commit[i]                  — ith commitment matches AbsorbCommitment slot
C_input_claim[s, j]          — instance j input_claim formula in stage s = sumcheck input claim
C_round_poly[s, r]           — round r of stage s satisfies s(0) + s(1) = prior_claim
C_output_check[s, j]         — instance j output_check formula in stage s = final_eval
C_eval_consistency[s, k]     — kth eval in stage s consistent with its committed poly
C_opening_claim[k]           — kth opening claim has eval matching recorded eval
C_opening_proof              — PCS verifier accepts joint opening proof
C_commit_skip[i]             — Option<PCS::Output>[i] matches schedule expected skip
C_transcript_tag[k]          — kth absorption uses expected DomainSeparator
```

**Definition.** Tamper `T` is *witnessing* for `c` iff `T(π)` violates `c` whenever `π` is honest. Taxonomy *covers* `C` iff `∀ c ∈ C . ∃ T ∈ Taxonomy that witnesses c`.

**S7 — Test framework constraint coverage.**

```
Taxonomy covers Constraints(V_core).
```

Enforced at test-launch via `assert_taxonomy_covers_constraints()` against schedule-derived constraint enumeration.

### 4.5 Acceptance condition (formal)

The cross-verifier soundness suite passes iff all of:

```
S1.   ∀ honest π :  Verify(V_mod, x, π) = Accept.
S3.   ∀ (x, π) :    Verify(V_mod, x, π) is deterministic across re-runs.
S4.   ∀ honest π :  V_mod's transcript states match V_core's at every aligned op.
S5'.  ∀ honest π ∀ T ∈ Taxonomy :
        Verify(V_core, x, T(π)) = Verify(V_mod, x, T(π))
        OR (T, π) ∈ KnownGapRegistry.
S7.   Taxonomy covers Constraints(V_core).
KGC.  ∀ entry in KnownGapRegistry :
        Verify(V_core, entry.tamper(π)) = Reject AND
        Verify(V_mod,  entry.tamper(π)) = Accept.
```

S6 is graded — soft fail (logged warning), not hard fail, except when running in `--strict` diagnostic mode.

A single counter-example to S1, S3, S4, S5′, S7, or KGC is a hard test failure.

### 4.6 Quantitative thresholds

```
Q1 — RejectionRate(V_mod, Taxonomy)
   = |{T : V_mod(T(π_honest)) = Reject}| / |Taxonomy|.
   Required: 1.0 modulo KnownGapRegistry.

Q2 — AgreementRate(V_core, V_mod, Taxonomy)
   = |{T : V_core(T(π)) = V_mod(T(π))}| / |Taxonomy|.
   Required: 1.0 after KnownGap subtraction.

Q3 — ConstraintCoverage(Taxonomy, V_core)
   = |{c ∈ Constraints(V_core) : ∃ witnessing T ∈ Taxonomy}| / |Constraints(V_core)|.
   Required: 1.0.

Q4 — SurvivalRate(seed_count) for fuzzer
   = |{seed : V_core(perturbed) = V_mod(perturbed)}| / |valid_perturbed|.
   Required: 1.0 over the seed range tested.
```

### 4.7 Test-suite invariants over time

**I1 — Monotonic rejection rate.** `RejectionRate_{n} ≥ RejectionRate_{n-1}` per commit.

**I2 — Monotonic registry shrinkage.** `|KnownGapRegistry|_{n} ≤ |KnownGapRegistry|_{n-1}` unless explicitly justified.

**I3 — Constraint coverage = 1.0** for every commit.

**I4 — Honest acceptance preserved** for every commit, every workload.

**I5 — Determinism preserved** for every commit.

### 4.8 Soundness theorem (informal)

If S1, S5′, S7, and KGC hold on the fixture, **and** the fixture covers all schedule shapes (opening / non-opening, advice / no-advice, all instance counts), then on the fixture-induced proof distribution `V_mod` has the same soundness error as `V_core` modulo the registered gaps:

```
|Pr[V_core accepts π] − Pr[V_mod accepts π]| ≤ |KnownGapRegistry| / |F.tampers|.
```

This is empirical, not cryptographic. It rests on coverage. With empty registry and full coverage, the bound tightens to 0.

### 4.9 What is not proven

Knowledge soundness is not established by this framework. That is a property of the underlying protocol (Spartan + Twist/Shout + Dory), not the verifier implementation. What the framework establishes is that **`V_mod` enforces every algorithmic check `V_core` enforces**. If `V_core` is knowledge-sound and `V_mod` is rejection-equivalent on a covering taxonomy, `V_mod` is knowledge-sound on the fixture span by structural reduction.

---

## 5. Engineering deliverables checklist

### 5.1 Verifier parity (Part A)

- [ ] `examples/jolt_core_module.rs::build_stage{3,4,5,6,7}` — append `Op::RecordEvals` + `Op::CollectOpeningClaim` for each stage's eval poly list (committed polys only for `CollectOpeningClaim`).
- [ ] `examples/jolt_core_module.rs::build_verifier_stage3_ops` — replace stub.
- [ ] `examples/jolt_core_module.rs::build_verifier_stage4_ops` — replace stub. Include advice opening-claim accumulation.
- [ ] `examples/jolt_core_module.rs::build_verifier_stage5_ops` — new function.
- [ ] `examples/jolt_core_module.rs::build_verifier_stage6_ops` — new function. Conditional advice instances.
- [ ] `examples/jolt_core_module.rs::build_verifier_stage7_ops` — new function. Conditional advice instances (address phase).
- [ ] `examples/jolt_core_module.rs::build_module` — extend the `verifier_ops.extend(...)` chain.
- [ ] `crates/jolt-compiler/src/module.rs::ClaimFactor` — add new variants as required by stage 5/6 formulas (`EvalSquared`, `CombineEntryEval`, etc., as needed).
- [ ] Verifier-side `ClaimFactor` evaluator updated to handle every new variant.
- [ ] All four existing equivalence tests remain green after each commit.

### 5.2 Cross-verifier soundness suite (Part B)

- [ ] `crates/jolt-equivalence/src/cross_verifier/mod.rs` — module entry point.
- [ ] `crates/jolt-equivalence/src/cross_verifier/conversion.rs` — `modular_to_core` + `core_to_modular`, round-trip property tested.
- [ ] `crates/jolt-equivalence/src/cross_verifier/tamper.rs` — `TamperPoint`, `TamperLocation`, `TamperMutation`, `apply_tamper`.
- [ ] `crates/jolt-equivalence/src/cross_verifier/runner.rs` — `run_both_verifiers`, `assert_consistent`.
- [ ] `crates/jolt-equivalence/src/cross_verifier/categories.rs` — `Constraint`, `RejectionCategory`, `TamperKind`.
- [ ] `crates/jolt-equivalence/src/cross_verifier/registry.rs` — `KnownGapRegistry`, initial population, consistency check.
- [ ] `crates/jolt-equivalence/tests/cross_verifier_soundness.rs` — every `#[test]` listed in §3.9.
- [ ] `assert_taxonomy_covers_constraints()` — launch-time S7 enforcement.
- [ ] Cached fixture (`OnceLock<HonestFixture>`).
- [ ] Fuzzer (`#[ignore]` by default).
- [ ] Multi-workload extensibility (`Workload` struct, default to `muldiv`, stretch slots for `sha2-chain` and trusted-advice program).

### 5.3 Acceptance gates (Part C)

- [ ] All of S1, S3, S4, S5′, S7, KGC enforced as test assertions.
- [ ] Q1, Q2, Q3, Q4 reported by the suite (`println!` summary at the end of each test run).
- [ ] I1–I5 enforced via CI (CI script asserts the metrics monotonically improve commit-over-commit).

---

## 6. Stop-the-line conditions

- **Wrong `output_check` formula that's correct on honest input** — manifests as a tamper that V_core rejects but V_mod accepts despite the formula being authored. Stop and re-derive the formula.
- **`ClaimFactor` evaluator returns wrong value for a new variant** — caught when the tamper-rejection rate doesn't improve after wiring a stage. Fix the evaluator before continuing.
- **`BatchedSumcheck::verify` ordering mismatch** — instances must be in the same order as the prover's `batched_sumchecks[N].instances`. If transcript_divergence breaks during a stage's wiring, this is the most likely cause.
- **Conditional advice instance presence** — emission must mirror the runtime's advice-commitment-presence check. Schedule-time switch on the prover's `Op::AbsorbCommitment` slots.
- **`KnownGap` entries that won't reproduce** — registry consistency test fails. Either the gap closed (remove the entry) or the harness regressed (fix the harness).

---

## 7. References

- `crates/jolt-core/src/zkvm/verifier.rs:347-1500` — jolt-core's verifier, the reference implementation.
- `crates/jolt-core/src/zkvm/spartan/`, `crates/jolt-core/src/zkvm/registers/`, `crates/jolt-core/src/zkvm/ram/`, `crates/jolt-core/src/zkvm/instruction_lookups/`, `crates/jolt-core/src/zkvm/bytecode/`, `crates/jolt-core/src/zkvm/advice/` — per-stage protocol math.
- `crates/jolt-compiler/src/compiler/emit.rs:100-260` — auto-generated compiler verifier-emission template.
- `crates/jolt-compiler/examples/jolt_core_module.rs:1910-2101` — fully wired stage 1 verifier; the model to mirror.
- `crates/jolt-compiler/examples/jolt_core_module.rs:6914-6976` — current stubs for stages 3 and 4.
- `crates/jolt-compiler/src/module.rs:1797-2087` — `VerifierOp`, `SumcheckInstance`, `ClaimFactor` definitions.
- `crates/jolt-equivalence/tests/muldiv.rs:938-1099` — `transcript_divergence` reference.
- `crates/jolt-equivalence/tests/muldiv.rs:1473-1587` — `zkvm_proof_accepted_by_core_verifier` reference.
- `crates/jolt-equivalence/tests/muldiv.rs:1607-1638` — `modular_self_verify` reference.
