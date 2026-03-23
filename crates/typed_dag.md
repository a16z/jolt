# Typed DAG: jolt-zkvm Stage Architecture

This document captures the design for the new jolt-zkvm proving pipeline,
replacing the current messy `ProverStage` trait + `CompositeStage` pattern
with a fully typed, stage-as-function DAG that matches jolt-core's semantics.

## Design Principles

1. **Fully typed** — Each stage returns a bespoke output struct. The compiler
   enforces wiring. No positional indexing, no runtime key lookups.
2. **Stateless** — No mutable accumulator. Claims propagate as return values.
3. **Explicit DAG** — The orchestrator function IS the DAG. Stage dependencies
   are visible in function signatures.
4. **IR-first formulas** — `ClaimDefinition` in jolt-ir is the single source
   of truth. Prover and verifier both derive from it.
5. **Backend-generic** — All stage functions are generic over `B: ComputeBackend`.

## Resolved Questions

### Q1: Where do shared formulas live?

**Resolution**: In `jolt-ir` as `ClaimDefinition` instances (already partially done
in `jolt_ir::zkvm::claims`). Each claim definition specifies:
- A symbolic `Expr` (the formula)
- `OpeningBinding`s mapping expression variables to polynomial tags
- `ChallengeBinding`s mapping expression variables to challenge sources

The prover (jolt-zkvm) imports these definitions and uses them to:
- Build `KernelDescriptor`s for `KernelEvaluator` witnesses (codegen path)
- Compute `input_claim` values from prior stage outputs

The verifier (jolt-verifier) imports the same definitions and uses them to:
- Verify the sumcheck output matches `eq(point) · g(openings, challenges)`
- Reconstruct opening claims

Both sides interpret the same IR — no hand-written duplicate formulas.

### Q2: How is uni-skip handled?

**Resolution**: Uni-skip is an implementation detail of computing the first
sumcheck round polynomial. It does NOT affect the stage DAG structure.

A stage function that uses uni-skip simply:
1. Computes the analytic first-round polynomial (using the uni-skip formula)
2. Passes it as `first_round_polynomial()` on the `SumcheckCompute` witness
3. The `BatchedSumcheckProver` handles it transparently

From the DAG's perspective, a uni-skip stage has the same signature and output
type as a non-uni-skip stage. The uni-skip proof data is carried inside the
stage's `SumcheckStageProof`.

### Q3: What data structure carries polynomial tables?

**Resolution**: `PolynomialTables<F>` — a typed struct with named fields,
grouped by logical role. Justified from first principles:

**What the prover needs**: Multilinear polynomial evaluation tables (`Vec<F>`)
for building `KernelEvaluator` witnesses. Each stage reads a specific subset.

**Three categories** (all stored as `Vec<F>` of length `2^num_vars`):

1. **Committed** — Polynomials with PCS commitments (go to Dory opening):
   - Dense: `ram_inc`, `rd_inc` (length `2^log_T`)
   - Sparse/one-hot: `instruction_ra[d]`, `bytecode_ra[d]`, `ram_ra[d]`
     (length `2^(log_T + log_k_chunk)`)

2. **Virtual** — Derived from R1CS witness columns (used in sumchecks, not
   committed separately):
   - Register values: `rd_write_value`, `rs1_value`, `rs2_value`
   - Memory: `ram_address`, `ram_read_value`, `ram_write_value`
   - Lookups: `lookup_output`, `left_operand`, `right_operand`
   - Hamming weight: `hamming_weight`

3. **Trace-derived** — Extracted directly from execution trace:
   - PV factors: `left_instruction_input`, `right_instruction_input`,
     `is_rd_not_zero`, `write_lookup_to_rd_flag`, `jump_flag`,
     `branch_flag`, `next_is_noop`
   - Instruction input: `left_is_rs1`, `left_is_pc`, `right_is_rs2`,
     `right_is_imm`, `unexpanded_pc`, `imm`
   - Register addresses: `rs1_ra`, `rs2_ra`, `rd_wa`
   - Shift: `next_unexpanded_pc`, `next_pc`, `next_is_virtual`,
     `next_is_first_in_sequence`

**Why a flat typed struct, not `BTreeMap<Tag, Vec<F>>`**: Type safety. Each
stage function accesses `tables.ram_inc` not `tables.get(RAM_INC)`. The
compiler catches typos and missing fields. The BTreeMap approach in
`WitnessStore` is appropriate for the witness generation pipeline (which is
tag-driven), but the proving pipeline benefits from named access.

**Conversion**: `WitnessStore` → `PolynomialTables` happens once at the
boundary between witness generation and proving. The `PolynomialTables`
constructor takes `&WitnessStore` plus the R1CS witness matrix and trace,
extracting all named fields.

## Claim Types

```rust
/// Scalar evaluation of a virtual polynomial.
/// Used only for inter-stage routing (input_claim computation).
/// Zero-cost newtype for documentation and type clarity.
pub struct VirtualEval<F>(pub F);

/// Evaluation of a committed polynomial at a specific point.
/// Carries enough data for downstream claim reductions.
/// Does NOT carry the full evaluation table — that stays in PolynomialTables.
pub struct CommittedEval<F: Field> {
    pub point: Vec<F>,
    pub eval: F,
}

/// Full claim for PCS opening. Created only at the collection step.
pub struct PcsClaim<F: Field> {
    pub poly_id: CommittedPolyId,
    pub point: Vec<F>,
    pub eval: F,
}
```

Key insight: `CommittedEval` is the inter-stage currency for committed
polynomial claims. It does NOT carry `Vec<F>` evaluation tables — those
stay in `PolynomialTables` and are only attached when building `PcsClaim`s
for the final PCS opening. This avoids cloning large tables between stages.

## Stage DAG

### Dependency Graph

```
S1 (Spartan Outer + uni-skip)
 ├──→ S2 (RamRW, PVRemainder + uni-skip, InstrLookupsCR, RamRafEval, OutputCheck)
 │     ├──→ S3 (Shift, InstrInput, RegistersCR)
 │     │     ├──→ S4 (RegistersRW, RamValCheck)
 │     │     │     ├──→ S5 (InstrReadRaf, RamRaCR, RegistersValEval)
 │     │     │     │     └──→ S6 (BytecodeReadRaf, Booleanity, HammingBool,
 │     │     │     │           RamRaVirtual, InstrRaVirtual, IncCR)
 │     │     │     │           └──→ S7 (HammingWeightCR → unified point)
 │     │     │     │                 └──→ PCS Opening
```

### Stage-by-Stage Specification

Each stage lists:
- **Instances**: Sumcheck sub-instances batched together (share challenges)
- **Reads**: Which prior stage outputs feed into input_claim
- **Committed claims**: CommittedEval values in the output
- **Routing**: VirtualEval values in the output

#### S1: Spartan Outer

- **Instances**: OuterRemaining (1 instance, with uni-skip first round)
- **Reads**: R1CS matrices, witness
- **Output type**: `SpartanOutput<F>`
  - `r_x`, `r_y` challenge vectors
  - `SpartanVirtualEvals<F>` — all virtual polynomial evaluations at `r_cycle`

#### S2: 5-Instance Batch

- **Instances**:
  1. RamReadWriteChecking (`log_k + log_T` rounds)
  2. ProductVirtualRemainder (`log_T` rounds, with uni-skip)
  3. InstructionLookupsClaimReduction (`log_T` rounds)
  4. RamRafEvaluation (`log_k` rounds)
  5. OutputCheck (`log_k` rounds)

- **Reads from S1**:
  - `ram_read_value`, `ram_write_value` → RamRW input_claim
  - `lookup_output`, `left_operand`, `right_operand`,
    `left_instruction_input`, `right_instruction_input` → InstrLookupsCR input_claim
  - `ram_address` → RamRafEval input_claim
  - Product uni-skip claim from S1

- **Output type**: `Stage2Output<F>`
  - Committed: `ram_inc_at_s2: CommittedEval<F>`
  - Routing: PV evals (`next_is_noop`, `left_instr_input_s2`, `right_instr_input_s2`),
    InstrLookupsCR evals, RamRW evals (`ram_val_s2`), OutputCheck evals, RamRaf evals
  - Points: `pv_point`, `ram_rw_point`, `ram_raf_point`, `output_check_point`

#### S3: 3-Instance Batch

- **Instances**:
  1. Shift (`log_T` rounds, EqPlusOne)
  2. InstructionInput (`log_T` rounds)
  3. RegistersClaimReduction (`log_T` rounds)

- **Reads from S1**: `next_pc`, `next_unexpanded_pc`, `next_is_virtual`,
  `next_is_first` → Shift input_claim.
  `rd_write_value`, `rs1_value`, `rs2_value` → RegistersCR input_claim.

- **Reads from S2**: `next_is_noop` (from PV), `left_instr_input_s2`,
  `right_instr_input_s2` → Shift + InstrInput input_claims.

- **Output type**: `Stage3Output<F>`
  - Routing: Shift evals, InstrInput evals, RegistersCR evals
  - Points: `shift_point`, `instr_input_point`, `registers_cr_point`

#### S4: 2-Instance Batch

- **Instances**:
  1. RegistersReadWriteChecking (`log_k + log_T` rounds)
  2. RamValCheck (`log_T` rounds)

- **Reads from S2**: `ram_val_s2` → RamValCheck. `ram_val_final` → RamValCheck.

- **Reads from S3**: RegistersCR evals → RegistersRW input_claim.
  InstrInput evals → RegistersRW consistency check.

- **Output type**: `Stage4Output<F>`
  - Committed: `ram_inc_at_s4`, `rd_inc_at_s4` — for IncCR in S6
  - Routing: RegistersRW evals, RamValCheck evals

#### S5: 3-Instance Batch

- **Instances**:
  1. InstructionReadRaf (`log_k` rounds)
  2. RamRaClaimReduction (`log_k` rounds)
  3. RegistersValEvaluation (varies)

- **Reads from S2**: InstrLookupsCR evals → InstrReadRaf input_claim.
  RamRaf evals → RamRaCR.

- **Reads from S4**: RegistersRW evals → RegistersValEval input_claim.

- **Output type**: `Stage5Output<F>`
  - Committed: `rd_inc_at_s5` — for IncCR in S6.
    `instruction_ra_at_s5: Vec<CommittedEval<F>>` — for HammingWeightCR.
  - Routing: RamRaCR evals, RegistersValEval evals

#### S6: 6-Instance Batch

- **Instances**:
  1. BytecodeReadRaf (`log_k` rounds)
  2. Booleanity (`log_T + log_k` rounds)
  3. HammingBooleanity (`log_k` rounds)
  4. RamRaVirtual (`log_T + log_k` rounds)
  5. InstructionRaVirtual (`log_T + log_k` rounds)
  6. IncClaimReduction (`log_T` rounds)

- **Reads from S2**: `ram_inc_at_s2` → IncCR input_claim.

- **Reads from S4**: `ram_inc_at_s4`, `rd_inc_at_s4` → IncCR input_claim.

- **Reads from S5**: `rd_inc_at_s5` → IncCR input_claim.
  Various RA claims → Booleanity/RA virtual.

- **Output type**: `Stage6Output<F>`
  - `r_cycle_s6: Vec<F>` — the cycle point from IncCR
  - Committed: `ram_inc_reduced`, `rd_inc_reduced` — at `r_cycle_s6`
  - Committed: `instruction_ra_at_s6`, `bytecode_ra_at_s6`, `ram_ra_at_s6` — from Booleanity
  - Routing: Hamming weight evals, RA virtual evals → for HammingWeightCR

#### S7: HammingWeightClaimReduction (1 instance)

- **Instances**: HammingWeightCR (`log_k_chunk` rounds, address only)

- **Reads from S5**: `instruction_ra_at_s5` (from InstrReadRaf)

- **Reads from S6**: All RA evals from Booleanity + RA virtual.
  Hamming weight evals. `r_cycle_s6` (cycle point from IncCR).

- **Output type**: `Stage7Output<F>`
  - `unified_point: Vec<F>` — `(r_addr_s7 || r_cycle_s6)` in big-endian
  - `instruction_ra: Vec<CommittedEval<F>>` — at unified point
  - `bytecode_ra: Vec<CommittedEval<F>>` — at unified point
  - `ram_ra: Vec<CommittedEval<F>>` — at unified point

#### PCS Opening (not a sumcheck stage)

- **Reads from S6**: `ram_inc_reduced`, `rd_inc_reduced` at `r_cycle_s6`
- **Reads from S7**: All RA evals at `unified_point`
- **Lagrange normalization**: Dense polys scaled by `eq(r_addr_s7, 0)`
- **RLC reduction** + single Dory batch opening proof

## Orchestrator

```rust
fn prove<PCS, B>(
    tables: &PolynomialTables<Fr>,
    config: &ProverConfig,
    transcript: &mut impl Transcript<Challenge = Fr>,
    backend: &Arc<B>,
) -> JoltProof<Fr>
where
    PCS: CommitmentScheme<Field = Fr> + AdditivelyHomomorphic,
    B: ComputeBackend,
{
    // S0: Commit
    let commitments = commit_polynomials::<PCS>(tables);

    // S1: Spartan
    let s1 = prove_spartan(tables, config, transcript, backend);

    // S2-S7: sumcheck stages
    let s2 = prove_stage2(&s1, tables, config, transcript, backend);
    let s3 = prove_stage3(&s1, &s2, tables, config, transcript, backend);
    let s4 = prove_stage4(&s2, &s3, tables, config, transcript, backend);
    let s5 = prove_stage5(&s2, &s4, tables, config, transcript, backend);
    let s6 = prove_stage6(&s2, &s4, &s5, tables, config, transcript, backend);
    let s7 = prove_stage7(&s5, &s6, tables, config, transcript, backend);

    // PCS opening
    let opening = prove_opening::<PCS>(&s6, &s7, tables, &commitments, transcript);

    JoltProof { s1, s2, s3, s4, s5, s6, s7, opening, commitments }
}
```

Each `prove_stageN` function:
1. Squeezes challenges from transcript (must match verifier exactly)
2. Computes `input_claim` from typed prior stage outputs
   - Uses shared formula from `jolt_ir::zkvm::claims`
   - Example: `let claim = claims::ram::rw_checking().evaluate(&[rv, wv], &[gamma])`
3. Uploads relevant polynomial tables to backend buffers
4. Builds `KernelDescriptor` from the `ClaimDefinition`'s expression
5. Constructs `KernelEvaluator` witnesses
6. Calls `BatchedSumcheckProver::prove_with_handler`
7. Extracts named evaluations at the challenge point
8. Returns typed stage output

## Verifier

The verifier has the same DAG shape, driven by the same IR:

```rust
fn verify<PCS>(
    proof: &JoltProof<Fr>,
    commitments: &Commitments<PCS>,
    config: &ProverConfig,
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> Result<(), VerifyError>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    let s1_v = verify_spartan(&proof.s1, config, transcript)?;
    let s2_v = verify_stage2(&s1_v, &proof.s2, config, transcript)?;
    let s3_v = verify_stage3(&s1_v, &s2_v, &proof.s3, config, transcript)?;
    let s4_v = verify_stage4(&s2_v, &s3_v, &proof.s4, config, transcript)?;
    let s5_v = verify_stage5(&s2_v, &s4_v, &proof.s5, config, transcript)?;
    let s6_v = verify_stage6(&s2_v, &s4_v, &s5_v, &proof.s6, config, transcript)?;
    let s7_v = verify_stage7(&s5_v, &s6_v, &proof.s7, config, transcript)?;
    verify_opening::<PCS>(&s6_v, &s7_v, &proof, commitments, transcript)?;
    Ok(())
}
```

Each `verify_stageN` function:
1. Squeezes same challenges (Fiat-Shamir replay)
2. Computes same `input_claim` using same IR formula + prior stage evals
3. Verifies sumcheck round polynomials
4. Reads evaluations from proof (instead of computing from polynomial tables)
5. Returns same typed output struct (with eval values from proof)

**Prover vs Verifier output types**: Both use the same struct. The prover
populates it by evaluating polynomials at the challenge point. The verifier
populates it from proof data. The struct doesn't know or care.

## IR Integration

### ClaimDefinition → KernelDescriptor (Prover)

The `ClaimDefinition` in jolt-ir describes the mathematical identity. The
prover extracts a `KernelDescriptor` from the definition's `Expr`:

```
ClaimDefinition {
    expr: "o0 * o1 + c0 * o2 * o3",   // symbolic
    opening_bindings: [(0, RAM_INC), (1, RAM_ADDRESS), ...],
    challenge_bindings: [(0, BatchingCoeff(0))],
}
    ↓ codegen
KernelDescriptor {
    shape: Custom { expr, num_inputs: 4 },
    degree: 3,
}
    ↓ compile
B::CompiledKernel<F>
    ↓ wrap
KernelEvaluator<F, B> : SumcheckCompute<F>
```

### ClaimDefinition → Verification (Verifier)

The verifier uses the same `ClaimDefinition` to check the sumcheck output:

```
ClaimDefinition.evaluate(openings, challenges) == g_eval
eq(eq_point, eval_point) * g_eval == final_round_eval
```

### ClaimDefinition → input_claim (Both)

For stages where `input_claim` depends on prior evaluations, a separate
`ClaimDefinition` describes the input_claim formula. Both prover and verifier
evaluate it identically from prior stage outputs.

## PolynomialTables Design

```rust
pub struct PolynomialTables<F: Field> {
    // ── Committed (opened via PCS) ──────────────────────────
    pub ram_inc: Vec<F>,
    pub rd_inc: Vec<F>,
    pub instruction_ra: Vec<Vec<F>>,  // [instruction_d]
    pub bytecode_ra: Vec<Vec<F>>,     // [bytecode_d]
    pub ram_ra: Vec<Vec<F>>,          // [ram_d]

    // ── Virtual (from R1CS witness columns) ─────────────────
    pub rd_write_value: Vec<F>,
    pub rs1_value: Vec<F>,
    pub rs2_value: Vec<F>,
    pub hamming_weight: Vec<F>,
    pub ram_address: Vec<F>,
    pub ram_read_value: Vec<F>,
    pub ram_write_value: Vec<F>,
    pub lookup_output: Vec<F>,

    // ── Trace-derived (from execution trace) ────────────────
    // Product virtualization factors
    pub left_instruction_input: Vec<F>,
    pub right_instruction_input: Vec<F>,
    pub is_rd_not_zero: Vec<F>,
    pub write_lookup_to_rd_flag: Vec<F>,
    pub jump_flag: Vec<F>,
    pub branch_flag: Vec<F>,
    pub next_is_noop: Vec<F>,

    // Instruction input
    pub left_is_rs1: Vec<F>,
    pub left_is_pc: Vec<F>,
    pub right_is_rs2: Vec<F>,
    pub right_is_imm: Vec<F>,
    pub unexpanded_pc: Vec<F>,
    pub imm: Vec<F>,

    // Register addresses
    pub rs1_ra: Vec<F>,
    pub rs2_ra: Vec<F>,
    pub rd_wa: Vec<F>,

    // Shift (next-cycle)
    pub next_unexpanded_pc: Vec<F>,
    pub next_pc: Vec<F>,
    pub next_is_virtual: Vec<F>,
    pub next_is_first_in_sequence: Vec<F>,
}

impl<F: Field> PolynomialTables<F> {
    /// Build from witness store + R1CS witness + trace.
    /// Single conversion point between witness generation and proving.
    pub fn from_witness(
        store: &WitnessStore<F>,
        r1cs_witness: &[Vec<F>],
        trace: &[Cycle],
        config: &ProverConfig,
    ) -> Self { ... }

    /// Number of cycles (padded to power of 2).
    pub fn num_cycles(&self) -> usize {
        self.ram_inc.len()
    }

    /// RA polynomials as a flat slice of slices (for stages that batch all RA).
    pub fn all_ra_polys(&self) -> Vec<&[F]> {
        let mut ra = Vec::new();
        for p in &self.instruction_ra { ra.push(p.as_slice()); }
        for p in &self.bytecode_ra { ra.push(p.as_slice()); }
        for p in &self.ram_ra { ra.push(p.as_slice()); }
        ra
    }
}
```

**Why this over the tag-based `WitnessStore`**:
- Every field is visible in the type. IDE autocomplete. Compile-time access.
- Stage functions access `tables.ram_inc` not `tables.get(poly::RAM_INC)`.
- No runtime panics from missing tags.
- The struct IS the documentation of what the prover needs.

**Why this over passing individual `&[F]` to each stage**:
- Passing 30+ borrows per stage function is unwieldy.
- `&PolynomialTables<F>` is one parameter; stages borrow the fields they need.
- No performance cost — just borrows into a single immutable struct.

## Stage Function Anatomy

```rust
/// Stage 2: RamRW + PVRemainder + InstrLookupsCR + RamRafEval + OutputCheck
fn prove_stage2<F, T, B>(
    s1: &SpartanOutput<F>,
    tables: &PolynomialTables<F>,
    config: &ProverConfig,
    transcript: &mut T,
    backend: &Arc<B>,
) -> Stage2Output<F>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: ComputeBackend,
{
    let n = tables.num_cycles();
    let log_t = n.trailing_zeros() as usize;
    let r_cycle = &s1.r_y[..log_t];

    // ── Sub-instance 1: RamRW ───────────────────────────────
    // Squeeze challenges (must match verifier)
    let gamma_rw: F = transcript.challenge();

    // Compute input_claim from S1 virtual evals
    let rw_input_claim = s1.evals.ram_read_value.0
        + gamma_rw * s1.evals.ram_write_value.0;

    // Build witness via IR → KernelDescriptor → KernelEvaluator
    let rw_def = jolt_ir::zkvm::claims::ram::rw_checking();
    let rw_desc = rw_def.to_kernel_descriptor();
    let rw_kernel = backend.compile_kernel::<F>(&rw_desc);
    // ... upload buffers, build KernelEvaluator ...

    let rw_claim = SumcheckClaim {
        num_vars: log_t + config.log_k_chunk(),
        degree: 3,
        claimed_sum: rw_input_claim,
    };

    // ── Sub-instance 2: PVRemainder (with uni-skip) ─────────
    // ... similar pattern ...

    // ── Batch all 5 instances ───────────────────────────────
    let claims = vec![rw_claim, pv_claim, instr_cr_claim, raf_claim, oc_claim];
    let mut witnesses: Vec<Box<dyn SumcheckCompute<F>>> =
        vec![rw_witness, pv_witness, instr_cr_witness, raf_witness, oc_witness];

    let (proof, challenges) = BatchedSumcheckProver::prove_with_handler(
        &claims, &mut witnesses, transcript, CaptureHandler::new(),
    );

    // ── Extract typed output ────────────────────────────────
    // Each sub-instance extracts from its slice of the shared challenges
    let rw_point = extract_point(&challenges, rw_offset, rw_num_vars);
    let ram_inc_eval = evaluate(&tables.ram_inc, &rw_point);

    Stage2Output {
        proof,
        challenges,
        ram_inc_at_s2: CommittedEval { point: rw_point, eval: ram_inc_eval },
        // ... all other named fields ...
    }
}
```

## PCS Opening Collection

```rust
fn prove_opening<PCS>(
    s6: &Stage6Output<Fr>,
    s7: &Stage7Output<Fr>,
    tables: &PolynomialTables<Fr>,
    commitments: &PcsCommitments<PCS>,
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> PCS::Proof
where
    PCS: CommitmentScheme<Field = Fr> + AdditivelyHomomorphic,
{
    let unified = &s7.unified_point;
    let r_addr = &unified[..log_k_chunk];
    let lagrange = EqPolynomial::zero_selector(r_addr);

    // Collect all PCS claims — explicit, no accumulator
    let mut claims: Vec<ProverClaim<Fr>> = Vec::new();

    // Dense polys: Lagrange-normalize to unified point
    claims.push(ProverClaim {
        evaluations: tables.ram_inc.clone(),
        point: unified.clone(),
        eval: s6.ram_inc_reduced.eval * lagrange,
    });
    claims.push(ProverClaim {
        evaluations: tables.rd_inc.clone(),
        point: unified.clone(),
        eval: s6.rd_inc_reduced.eval * lagrange,
    });

    // RA polys: already at unified point
    for (i, eval) in s7.instruction_ra.iter().enumerate() {
        claims.push(ProverClaim {
            evaluations: tables.instruction_ra[i].clone(),
            point: unified.clone(),
            eval: eval.eval,
        });
    }
    // ... bytecode_ra, ram_ra ...

    // RLC reduction → single point → PCS open
    let (reduced, reduction_proof) = RlcReduction::reduce_prover(claims, transcript);
    debug_assert_eq!(reduced.len(), 1, "all claims at unified point");
    PCS::open(&reduced[0], transcript)
}
```

## What Gets Deleted

| Item | Replacement |
|------|-------------|
| `ProverStage` trait | Stage functions (`prove_stage2`, etc.) |
| `CompositeStage` | Batching handled inside each stage function |
| `EagerVerifierSource` | Verifier stage functions (`verify_stage2`, etc.) |
| `DescriptorSource` trait | Explicit verifier DAG |
| `StageDescriptor` (partially) | Shared `ClaimDefinition` from jolt-ir |
| All `s2_*.rs`, `s3_*.rs`, etc. | New stage function files |
| `CommittedTables` | `PolynomialTables` |
| `build_prover_stages()` | Inlined in `prove()` |
| `build_verifier_descriptors()` | Inlined in `verify()` |
| `pipeline.rs::prove_stages` | Inlined per-stage sumcheck calls |

## What Gets Modified

| Crate | Change |
|-------|--------|
| `jolt-spartan` | Return typed `SpartanOutput<F>` with virtual evals |
| `jolt-ir` | Ensure all claim definitions exist in `zkvm::claims` modules. Add `ClaimDefinition::to_kernel_descriptor()` bridge. |
| `jolt-openings` | Add `CommittedEval<F>` type |
| `jolt-verifier` | Rewrite to use typed verifier DAG (same shape as prover) |
| `jolt-zkvm` | Complete pipeline rewrite |

## Implementation Order

1. **Add `CommittedEval<F>` to jolt-openings** — trivial type addition
2. **Audit jolt-ir claims** — ensure all 16 claim definitions exist and match jolt-core
3. **Add `ClaimDefinition::to_kernel_descriptor()`** — bridge IR → kernel compilation
4. **Define `PolynomialTables<F>`** — typed struct + `from_witness` constructor
5. **Define all stage output types** — `SpartanOutput`, `Stage2Output`, ..., `Stage7Output`
6. **Implement stage functions S2-S7** — one at a time, matching jolt-core semantics
7. **Implement `prove_opening`** — PCS claim collection + RLC + Dory
8. **Wire orchestrator `prove()`** — call all stages in sequence
9. **Implement verifier DAG** — same shape, verify instead of prove
10. **E2E test: muldiv** — the canonical correctness check
11. **Delete old pipeline** — `ProverStage`, `CompositeStage`, all old stage files

## Open Items (Deferred)

- **Advice polynomials**: Deferred. When added, AdviceClaimReduction becomes
  a two-phase instance split across S6 (cycle phase) and S7 (address phase).
  The phase transition is an internal detail of the stage function, not
  visible in the DAG.
- **BlindFold ZK**: Deferred. Standard mode first.
- **SDK migration**: Very last step before deleting jolt-core.
- **Streaming commitment**: Integration with `StreamingCommitment` trait
  for witness polynomials (jolt-dory tier-1/tier-2).
