# jolt-zkvm Specification

**Status:** Draft
**Date:** 2026-03-06
**Depends on:** spec.md §4.10, rfc.md findings 11–13

---

## 1. Overview

`jolt-zkvm` is the final crate in the modular Jolt workspace. It orchestrates all sub-crates into a complete RISC-V (RV64IMAC) proving system. This document specifies how jolt-zkvm composes the existing abstractions, what new abstractions it introduces, how testing works at every level, and how the compute backend integrates.

### Design Principles

1. **IR-first:** Every sumcheck claim formula is defined as a `jolt-ir::ClaimDefinition`. Evaluation, BlindFold R1CS, Lean4, and circuit backends are derived — never hand-synchronized.
2. **Stateless claims:** Opening claims are `Vec<ProverClaim>` / `Vec<VerifierClaim>` from `jolt-openings`. No accumulator struct. Claims are plain data collected across stages and reduced via `RlcReduction` at stage 8.
3. **Backend-generic witnesses:** `SumcheckCompute` implementations use `jolt-compute::ComputeBackend` for polynomial operations, enabling GPU acceleration without touching protocol logic.
4. **Stage = batch of sumcheck instances.** Each stage produces `(Vec<SumcheckClaim<F>>, Vec<Box<dyn SumcheckCompute<F>>>)` and feeds them to `jolt-sumcheck::BatchedSumcheckProver`. Stages are independent modules, not a monolithic prover method.
5. **Testable in isolation.** Each stage, each `SumcheckCompute` implementation, and each `ClaimDefinition` can be unit-tested independently with known polynomials and mock openings.

---

## 2. Dependency Inventory

jolt-zkvm consumes every crate in the workspace. Here is exactly what it uses from each:

| Crate | What jolt-zkvm uses |
|-------|-------------------|
| **jolt-field** | `Field`, `Fr`, challenge types |
| **jolt-transcript** | `Transcript`, `Blake2bTranscript` |
| **jolt-crypto** | `JoltGroup`, `PairingGroup` (for BlindFold), `Pedersen` (ZK mode) |
| **jolt-poly** | `Polynomial<T>`, `EqPolynomial`, `UnivariatePoly` |
| **jolt-openings** | `CommitmentScheme`, `AdditivelyHomomorphic`, `StreamingCommitment`, `ProverClaim`, `VerifierClaim`, `RlcReduction` |
| **jolt-dory** | `DoryScheme` (concrete PCS instantiation) |
| **jolt-sumcheck** | `SumcheckClaim`, `SumcheckCompute`, `BatchedSumcheckProver/Verifier`, `RoundHandler/Verifier`, `SumcheckProof` |
| **jolt-spartan** | `SpartanProver/Verifier`, `SpartanKey`, `R1CS`, `SimpleR1CS`, `FirstRoundStrategy` |
| **jolt-ir** | `ExprBuilder`, `ClaimDefinition`, `SumOfProducts`, `R1csEmission`, `KernelDescriptor` |
| **jolt-compute** | `ComputeBackend`, `CpuBackend` |
| **jolt-instructions** | `Instruction`, `LookupTable`, `JoltInstructionSet` |
| **jolt-blindfold** | `BlindFoldProver/Verifier` (ZK feature gate) |

### Confirmed: No Gaps

| Concern | Resolution |
|---------|-----------|
| Opening accumulator | Not needed — `Vec<ProverClaim<F>>` collected across stages, reduced by `RlcReduction::reduce_prover` at stage 8 |
| Univariate skip | Hidden inside stage `build()` implementations. `SumcheckCompute` wraps the lazy evaluator; `FirstRoundStrategy::UnivariateSkip` from jolt-spartan selects the fast path. Pipeline stays uniform. |
| Batched sumcheck | `jolt-sumcheck::BatchedSumcheckProver` takes `&[SumcheckClaim]` + `&mut [Box<dyn SumcheckCompute>]` — exactly what stages produce |
| Claim/constraint sync | `jolt-ir::ClaimDefinition` is the single source — `.evaluate()` for prover, `.expr.to_sum_of_products().emit_r1cs()` for BlindFold |
| Compute backend | `jolt-compute::ComputeBackend` used in `SumcheckCompute` implementations for `round_polynomial()` and `bind()` |
| Multi-phase reductions | `SumcheckReduction` trait in `jolt-sumcheck` — sumcheck-based claim reduction with protocol-specific witness construction. Advice two-phase reduction modeled as two composed reductions with intermediate claims flowing via `ProverClaim`. |
| Spartan outer sumcheck | Stage 1 implements `ProverStage` normally. Internally wraps the lazy `UniformSpartanKey` fused bilinear evaluator as a `SumcheckCompute` impl. Specialized evaluation hidden behind the trait. |

---

## 3. Architecture

### 3.1 Stage Model

The Jolt proving pipeline has 7 sumcheck stages + 1 opening stage. Each stage is a struct implementing the `Stage` trait:

```rust
/// A proving stage that contributes batched sumcheck instances.
pub trait ProverStage<F: Field, B: ComputeBackend> {
    /// Build sumcheck claims and witnesses for this stage.
    ///
    /// The `opening_claims` parameter provides access to openings
    /// recorded by previous stages (read-only).
    fn build(
        &mut self,
        opening_claims: &[ProverClaim<F>],
        transcript: &mut impl Transcript,
    ) -> StageBatch<F>;

    /// After sumcheck completes, extract opening claims from the
    /// derived challenges and record them for stage 8.
    fn extract_claims(
        &mut self,
        challenges: &[F],
    ) -> Vec<ProverClaim<F>>;

    /// IR-based claim definitions for this stage's sumcheck instances.
    /// Used by BlindFold and for testing.
    fn claim_definitions(&self) -> Vec<ClaimDefinition>;
}

/// Output of a stage's build() method.
pub struct StageBatch<F: Field> {
    pub claims: Vec<SumcheckClaim<F>>,
    pub witnesses: Vec<Box<dyn SumcheckCompute<F>>>,
}
```

### 3.2 Prover Pipeline

```rust
pub fn prove<F, PCS, T, B>(
    stages: &mut [Box<dyn ProverStage<F, B>>],
    transcript: &mut T,
    handler: impl RoundHandler<F>,
) -> (Vec<SumcheckProof<F>>, Vec<ProverClaim<F>>)
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
    T: Transcript,
    B: ComputeBackend,
{
    let mut all_opening_claims: Vec<ProverClaim<F>> = Vec::new();
    let mut stage_proofs = Vec::new();

    for stage in stages.iter_mut() {
        let batch = stage.build(&all_opening_claims, transcript);

        let proof = BatchedSumcheckProver::prove_with_handler(
            &batch.claims,
            &mut batch.witnesses,
            transcript,
            challenge_fn,
            handler.clone(),
        );

        let challenges = extract_challenges_from_proof(&proof);
        let new_claims = stage.extract_claims(&challenges);
        all_opening_claims.extend(new_claims);
        stage_proofs.push(proof);
    }

    (stage_proofs, all_opening_claims)
}
```

### 3.3 Stage 8: Opening via Stateless Reduction

```rust
// Collect all opening claims from stages 1–7
let all_claims: Vec<ProverClaim<F>> = /* accumulated above */;

// Reduce: group by point, RLC-combine per group
let (reduced_claims, ()) = RlcReduction::reduce_prover(all_claims, transcript, challenge_fn);

// Open each reduced claim via PCS
for claim in &reduced_claims {
    let poly: PCS::Polynomial = claim.evaluations.clone().into();
    let proof = PCS::open(&poly, &claim.point, claim.eval, &setup, None, transcript);
    opening_proofs.push(proof);
}
```

No accumulator struct — claims are plain data flowing through function parameters.

### 3.4 Verifier Pipeline

Mirror image: each stage provides a `VerifierStage` that verifies the sumcheck proof and produces `Vec<VerifierClaim<F, PCS::Output>>`. Stage 8 reduces via `RlcReduction::reduce_verifier` and calls `PCS::verify`.

### 3.5 IR-Driven Claim Definitions

Every sumcheck instance's input/output claim is defined as a `ClaimDefinition`:

```rust
fn ram_rw_claim_definition() -> ClaimDefinition {
    let b = ExprBuilder::new();
    let rv_claim = b.opening(0);
    let wv_claim = b.opening(1);
    let inc = b.opening(2);
    let gamma = b.challenge(0);

    let expr = b.build(rv_claim + gamma * (inc + rv_claim) - wv_claim);

    ClaimDefinition {
        expr,
        opening_bindings: vec![
            OpeningBinding { var_id: 0, polynomial_tag: tag::RV, sumcheck_tag: tag::RAM_RW },
            OpeningBinding { var_id: 1, polynomial_tag: tag::WV, sumcheck_tag: tag::RAM_RW },
            OpeningBinding { var_id: 2, polynomial_tag: tag::INC, sumcheck_tag: tag::RAM_RW },
        ],
        challenge_bindings: vec![
            ChallengeBinding { var_id: 0, source: ChallengeSource::Derived },
        ],
    }
}
```

This definition is used three ways:
1. **Prover:** `claim_def.evaluate(&opening_values, &challenge_values)` → claimed sum
2. **BlindFold:** `claim_def.expr.to_sum_of_products().emit_r1cs(...)` → R1CS constraints
3. **Testing:** property-checked against hand-computed expected values

### 3.6 SumcheckReduction — Sumcheck-Based Claim Reduction

Some claim reductions are performed via sumcheck rather than algebraic combination (RLC). The advice polynomial reduction is the primary example: it reduces claims at many points to a claim at a single derived point by running a sumcheck over the polynomial's variables.

`SumcheckReduction` is a trait in `jolt-sumcheck` that formalizes this pattern. It follows the same `claims → (fewer claims, proof)` shape as `OpeningReduction` from `jolt-openings`, but its mechanism is a sumcheck and its proof artifact is a `SumcheckProof`.

```rust
// In jolt-sumcheck:

/// A claim reduction performed via sumcheck.
///
/// Transforms K claims into fewer claims by running a batched sumcheck.
/// The prover constructs witnesses from input claims' evaluation tables;
/// the verifier checks the sumcheck proof and derives output claims
/// algebraically.
///
/// Implementations provide protocol-specific witness construction
/// via [`build_witnesses`](Self::build_witnesses).
///
/// Output claims carry the partially-bound polynomial as their
/// `evaluations` field, enabling composed reductions: phase 1's
/// output claim feeds phase 2's input.
pub trait SumcheckReduction<F: Field> {
    /// Construct sumcheck claims and witnesses from input opening claims.
    ///
    /// The returned witnesses implement `SumcheckCompute` and are fed
    /// to `BatchedSumcheckProver`. The claim definitions are used by
    /// both prover and verifier to derive the output claims.
    fn build_witnesses(
        &self,
        claims: &[ProverClaim<F>],
    ) -> (Vec<SumcheckClaim<F>>, Vec<Box<dyn SumcheckCompute<F>>>);

    /// Construct verifier-side sumcheck claims from input claims.
    ///
    /// Must produce claims with the same `num_vars` and `degree` as
    /// the prover's [`build_witnesses`](Self::build_witnesses).
    fn build_verifier_claims<C>(
        &self,
        claims: &[VerifierClaim<F, C>],
    ) -> Vec<SumcheckClaim<F>>;

    /// After sumcheck completes, derive output claims from the
    /// sumcheck challenges and input claims.
    fn extract_prover_claims(
        &self,
        input_claims: &[ProverClaim<F>],
        challenges: &[F],
        final_eval: F,
    ) -> Vec<ProverClaim<F>>;

    /// Verifier-side output claim derivation.
    fn extract_verifier_claims<C>(
        &self,
        input_claims: &[VerifierClaim<F, C>],
        challenges: &[F],
        final_eval: F,
    ) -> Vec<VerifierClaim<F, C>>;
}
```

**Dependency:** `jolt-sumcheck` gains a dependency on `jolt-openings` for `ProverClaim`/`VerifierClaim` types.

#### Advice Reduction as Composed SumcheckReductions

The advice polynomial reduction spans two stages but is modeled as two composed `SumcheckReduction`s:

```
Stage 6: AdviceCycleReduction (binds cycle variables)
  Input:  ProverClaim { evaluations: full_advice_poly, point: r_val, eval: advice_val }
  Output: ProverClaim { evaluations: partially_bound_poly, point: cycle_challenges, eval: C_mid }

Stage 7: AdviceAddressReduction (binds address variables)
  Input:  ProverClaim { evaluations: partially_bound_poly, point: cycle_challenges, eval: C_mid }
  Output: ProverClaim { evaluations: fully_bound_poly, point: full_opening_point, eval: final_advice_eval }
```

The intermediate claim's `evaluations` field carries the partially-bound polynomial — phase 2's `build_witnesses()` constructs its `SumcheckCompute` witness directly from that evaluation table. No special continuation types, no shared mutable state. Claims are the only inter-stage data.

The normalized opening point (cycle-major vs. address-major ordering for Dory) is computed in `extract_prover_claims` using the concatenated challenges from both phases. The cycle challenges are available from the intermediate claim's `point` field.

---

## 4. Compute Backend Integration

### 4.1 Where ComputeBackend is Used

The `ComputeBackend` trait enters jolt-zkvm through `SumcheckCompute` implementations. Each witness struct holds a `&B` reference and stores polynomial data in `B::Buffer<T>`:

```rust
struct RamRwWitness<'a, F: Field, B: ComputeBackend> {
    backend: &'a B,
    eq_evals: B::Buffer<F>,
    ra_poly: B::Buffer<F>,
    val_poly: B::Buffer<F>,
    inc_poly: B::Buffer<i128>,
    kernel: B::CompiledKernel<F>,
}

impl<F: Field, B: ComputeBackend> SumcheckCompute<F> for RamRwWitness<'_, F, B> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let coeffs = self.backend.pairwise_reduce(
            &[&self.eq_evals, &self.ra_poly, &self.val_poly],
            &self.eq_evals,
            &self.kernel,
            self.degree(),
        );
        UnivariatePoly::new(coeffs)
    }

    fn bind(&mut self, challenge: F) {
        self.eq_evals = self.backend.interpolate_pairs(
            std::mem::take(&mut self.eq_evals), challenge,
        );
        self.ra_poly = self.backend.interpolate_pairs(
            std::mem::take(&mut self.ra_poly), challenge,
        );
        // Compact types promoted to F during interpolation:
        let inc_as_field = self.backend.interpolate_pairs::<i128, F>(
            std::mem::take(&mut self.inc_poly), challenge,
        );
        // ...
    }
}
```

### 4.2 Kernel Compilation

At setup time, each stage compiles its sumcheck kernels from `jolt-ir::KernelDescriptor`:

```rust
let descriptor = KernelDescriptor {
    shape: KernelShape::ProductSum {
        num_inputs_per_product: 3,  // eq * ra * val
        num_products: 1,
    },
    degree: 3,
    tensor_split: Some(TensorSplit::balanced(num_vars)),
};

// CPU: kernel = CpuKernel wrapping a closure (zero overhead)
// GPU: kernel = compiled Metal/CUDA shader (AOT, cached)
let kernel = backend.compile(&descriptor);
```

### 4.3 Zero-Overhead on CPU

After monomorphization with `B = CpuBackend`:
- `B::Buffer<T>` = `Vec<T>` (no indirection)
- `interpolate_pairs` = direct Rayon parallel iterator (identical to hand-written code)
- `pairwise_reduce` = fused loop with `PAR_THRESHOLD` for auto-vectorization
- `product_table` = iterative doubling with Rayon split (identical to current `EqPolynomial`)

---

## 5. Stage Breakdown

| Stage | Sumcheck Instances | Claims Consumed | Claims Produced |
|-------|-------------------|-----------------|-----------------|
| **1** | Spartan outer (uni-skip + remaining) | R1CS witness | `r_stage1` challenges → openings of Az, Bz, Cz |
| **2** | Product virtual (uni-skip + remaining), RAM RW, Instruction RAF, RAM RAF, Output check | Stage 1 openings | `r_address` shared across RAM, register/instruction RA claims |
| **3** | Spartan shift, Instruction input, Register claim reduction | Stage 2 openings | Register RW challenges |
| **4** | Register RW, RAM val check | Stage 3 openings | Advice opening point, cached RAM val |
| **5** | Instruction RAF, RAM RA reduction, Register val evaluation | Stage 4 openings | RA reduction claims |
| **6** | Bytecode RAF, RAM hamming, Booleanity, RAM RA virtual, Instruction RA virtual, Inc reduction, Advice `SumcheckReduction` phase 1 (cycle vars) | Stage 5 openings | Hamming weight claims + intermediate advice claim (`C_mid` with partially-bound poly) |
| **7** | Hamming weight claim reduction, Advice `SumcheckReduction` phase 2 (address vars, consumes `C_mid` claim) | Stage 6 openings | Final RA claims for committed polys + final advice opening claim |
| **8** | (Batch opening) | All claims from stages 1–7 | PCS proof |

Each stage is a self-contained module in `src/stages/`.

---

## 6. Testing Strategy

### 6.1 Levels of Testing

The key insight: with modular crates and IR-driven definitions, we can test at **five levels** instead of relying solely on e2e tests.

#### Level 1: IR Expression Tests (unit, fast, exhaustive)

For each `ClaimDefinition`, verify:
- `expr.evaluate()` equals `expr.to_sum_of_products().evaluate()` for random inputs
- `emit_r1cs()` constraints are satisfied by `compute_witness()` for random inputs
- Lean4 output parses (syntactic check)

```rust
#[test]
fn ram_rw_claim_ir_consistency() {
    let def = ram_rw_claim_definition();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    for _ in 0..100 {
        let openings: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();
        let challenges: Vec<Fr> = (0..1).map(|_| Fr::random(&mut rng)).collect();

        let eval = def.evaluate(&openings, &challenges);
        let sop = def.expr.to_sum_of_products();
        let sop_eval = sop.evaluate(&openings, &challenges);
        assert_eq!(eval, sop_eval);

        // R1CS round-trip
        let opening_vars: Vec<R1csVar> = (0..3).map(|i| R1csVar(i as u32)).collect();
        let mut next_var = 3;
        let emission = sop.emit_r1cs(&opening_vars, &challenges, &mut next_var);
        let witness = build_witness(&emission, &opening_vars, &openings);
        for constraint in &emission.constraints {
            assert!(constraint.is_satisfied(&witness));
        }
    }
}
```

This catches the class of bugs that previously required running the full muldiv e2e test.

#### Level 2: SumcheckCompute Unit Tests (unit, medium speed)

For each witness implementation, test `round_polynomial()` and `bind()` against a brute-force evaluation:

```rust
#[test]
fn ram_rw_witness_round_poly_correct() {
    let backend = CpuBackend;
    let num_vars = 4;
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    // Construct known polynomials
    let eq = Polynomial::<Fr>::random(num_vars, &mut rng);
    let ra = Polynomial::<Fr>::random(num_vars, &mut rng);
    let val = Polynomial::<Fr>::random(num_vars, &mut rng);

    let mut witness = RamRwWitness::new(&backend, &eq, &ra, &val);
    let round_poly = witness.round_polynomial();

    // Brute-force: sum over all assignments to vars 1..n
    let expected = brute_force_round_polynomial(
        |x| eq.evaluate(x) * ra.evaluate(x) * val.evaluate(x),
        num_vars,
    );

    assert_eq!(round_poly, expected);
}
```

This catches polynomial bind bugs, degree mismatches, and off-by-one errors — the class of bugs that previously required running the full sumcheck and hoping for a transcript mismatch.

#### Level 3: Per-Stage Integration Tests (integration, medium speed)

Each stage can be tested in isolation by providing mock polynomials and verifying the sumcheck proves and verifies:

```rust
#[test]
fn stage_2_ram_rw_proves_and_verifies() {
    let num_vars = 6;
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let backend = CpuBackend;

    // Generate consistent witness data
    let (witnesses, expected_claims) = generate_consistent_ram_rw_data(num_vars, &mut rng);

    let mut stage = RamRwStage::new(&backend, witnesses);
    let batch = stage.build(&[], &mut transcript);

    // Prove
    let proof = BatchedSumcheckProver::prove(&batch.claims, &mut batch.witnesses, &mut pt, challenge_fn);

    // Verify
    let result = BatchedSumcheckVerifier::verify(&batch.claims, &proof, &mut vt, challenge_fn);
    assert!(result.is_ok());

    // Check claim definitions match
    for (def, claim) in stage.claim_definitions().iter().zip(batch.claims.iter()) {
        let eval = def.evaluate(&opening_values, &challenge_values);
        assert_eq!(eval, claim.claimed_sum);
    }
}
```

#### Level 4: Multi-Stage Pipeline Tests (integration, slower)

Test claim flow between stages — stage N's output claims feed stage N+1's input:

```rust
#[test]
fn stages_1_through_3_claim_flow() {
    // Generate full witness for a small trace
    let trace = generate_test_trace(/* simple program */);
    let (stage1, stage2, stage3) = build_stages(&trace);

    let mut claims = Vec::new();
    for stage in [&mut stage1, &mut stage2, &mut stage3] {
        let batch = stage.build(&claims, &mut transcript);
        let proof = BatchedSumcheckProver::prove(/* ... */);
        let new_claims = stage.extract_claims(&challenges);
        claims.extend(new_claims);
    }

    // Verify all opening claims are internally consistent
    for claim in &claims {
        let poly = Polynomial::new(claim.evaluations.clone());
        assert_eq!(poly.evaluate(&claim.point), claim.eval);
    }
}
```

#### Level 5: End-to-End Tests (e2e, slow, comprehensive)

Full prove → verify cycle with real RISC-V programs:

```rust
#[test]
fn e2e_muldiv() {
    let elf = compile_guest("muldiv");
    let trace = execute(elf);
    let proof = JoltProver::<DoryScheme, CpuBackend>::prove(trace, &mut transcript);
    JoltVerifier::<DoryScheme>::verify(&proof, &mut transcript).unwrap();
}
```

### 6.2 Testing Matrix

| Level | What it catches | Speed | Coverage |
|-------|----------------|-------|----------|
| IR expressions | Claim formula bugs, BlindFold sync | <1s | All ~20 claim definitions |
| Witness unit | Polynomial bind bugs, degree errors | ~1s per witness | Each SumcheckCompute impl |
| Per-stage | Sumcheck soundness per protocol | ~5s per stage | Each stage in isolation |
| Multi-stage | Claim flow, challenge threading | ~30s | Stage pairs/triples |
| E2E | Full system correctness | ~60s | muldiv, sha2, fibonacci |

### 6.3 Fuzz Targets

| Target | Input | Invariant |
|--------|-------|-----------|
| `claim_round_trip` | Random `ClaimDefinition` | `evaluate() == to_sop().evaluate()` |
| `witness_bind` | Random polynomial + challenge | `bind(r).evaluate(rest) == evaluate(r, rest)` |
| `stage_soundness` | Random witness data | Prover accepts ⟹ verifier accepts |
| `tampered_proof` | Valid proof + random mutation | Verifier rejects |
| `reduction_consistency` | Random claims | `reduce_prover` and `reduce_verifier` agree on combined eval |

### 6.4 Benchmarks

```
benches/
├── proving.rs         # Full proof generation (muldiv, sha2)
├── stage_timing.rs    # Per-stage breakdown
├── witness_ops.rs     # SumcheckCompute::round_polynomial + bind
└── opening.rs         # Stage 8 reduction + PCS open
```

---

## 7. Module Structure

```
jolt-zkvm/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs                    # Re-exports: JoltProver, JoltVerifier, JoltProof
│   ├── prover.rs                 # JoltProver<PCS, B> — generic pipeline driver
│   ├── verifier.rs               # JoltVerifier<PCS> — mirror of prover
│   ├── proof.rs                  # JoltProof<PCS> serialization
│   ├── config.rs                 # ProverConfig, OneHotConfig, ReadWriteConfig
│   ├── error.rs                  # JoltError
│   ├── pipeline.rs               # Generic prove_stages() / verify_stages()
│   ├── stage.rs                  # ProverStage / VerifierStage traits
│   ├── witness.rs                # CommittedPolynomial enum, witness generation
│   │
│   ├── claims/                   # IR-based claim definitions (single source of truth)
│   │   ├── mod.rs                # All ClaimDefinition constructors
│   │   ├── ram.rs                # RAM RW, val, RA, RAF, output, hamming claims
│   │   ├── registers.rs          # Register RW, val claims
│   │   ├── bytecode.rs           # Bytecode RAF claims
│   │   ├── instruction.rs        # Instruction RA virtual, RAF claims
│   │   ├── spartan.rs            # Outer, product, shift, instruction input claims
│   │   └── reductions.rs         # Inc, hamming weight, advice claim reductions
│   │
│   ├── stages/                   # Each stage as a self-contained module
│   │   ├── mod.rs
│   │   ├── s1_spartan_outer.rs
│   │   ├── s2_memory_checking.rs
│   │   ├── s3_register_reduction.rs
│   │   ├── s4_value_checking.rs
│   │   ├── s5_raf_reduction.rs
│   │   ├── s6_booleanity.rs
│   │   └── s7_hamming_weight.rs
│   │
│   ├── witnesses/                # SumcheckCompute implementations
│   │   ├── mod.rs
│   │   ├── outer_sumcheck.rs     # OuterSumcheckCompute (fused R1CS evaluator)
│   │   ├── product_virtual.rs    # Product virtualization witness
│   │   ├── ram_rw.rs             # RAM read-write checking witness
│   │   ├── register_rw.rs        # Register read-write checking witness
│   │   ├── ra_virtual.rs         # RA decomposition witness
│   │   ├── raf_check.rs          # Read-accumulate-forward witness
│   │   ├── val_check.rs          # Value check witness
│   │   ├── hamming.rs            # Hamming booleanity witness
│   │   ├── output_check.rs       # Output check witness
│   │   └── claim_reduction.rs    # Generic claim reduction witness
│   │
│   ├── r1cs/                     # Jolt-specific R1CS
│   │   ├── mod.rs
│   │   ├── constraints.rs        # ClaimDefinitions → bilinear pairs
│   │   └── key.rs                # UniformSpartanKey (lazy evaluation)
│   │
│   ├── bytecode/                 # Bytecode preprocessing
│   │   ├── mod.rs
│   │   └── preprocessing.rs
│   │
│   └── preprocessing.rs          # JoltProverPreprocessing, JoltVerifierPreprocessing
│
├── tests/
│   ├── ir_claims.rs              # Level 1: all ClaimDefinition IR round-trips
│   ├── witness_correctness.rs    # Level 2: SumcheckCompute brute-force checks
│   ├── stage_isolation.rs        # Level 3: per-stage prove/verify
│   ├── claim_flow.rs             # Level 4: multi-stage claim threading
│   └── e2e.rs                    # Level 5: full prove → verify
├── benches/
│   ├── proving.rs
│   ├── stage_timing.rs
│   └── witness_ops.rs
└── fuzz/
    └── fuzz_targets/
        ├── claim_round_trip.rs
        ├── witness_bind.rs
        └── tampered_proof.rs
```

### Key: `claims/` vs `witnesses/` vs `stages/`

- **`claims/`** — Pure IR expressions. No runtime state. Define the mathematics.
- **`witnesses/`** — `SumcheckCompute` implementations. Hold polynomial data. Do the heavy computation. Generic over `B: ComputeBackend`.
- **`stages/`** — Orchestration. Construct claims + witnesses, call `BatchedSumcheckProver`, extract opening claims. Implement `ProverStage`.

---

## 8. Type Parameters

```rust
pub struct JoltProver<PCS, B>
where
    PCS: AdditivelyHomomorphic,
    B: ComputeBackend,
{
    config: ProverConfig,
    pcs_setup: PCS::ProverSetup,
    backend: B,
}

pub struct JoltVerifier<PCS: CommitmentScheme> {
    pcs_setup: PCS::VerifierSetup,
}

pub struct JoltProof<PCS: CommitmentScheme> {
    pub commitments: Vec<PCS::Output>,
    pub stage_proofs: [SumcheckProof<PCS::Field>; 7],
    pub opening_proofs: Vec<PCS::Proof>,
    pub config: ProverConfig,
    pub trace_length: usize,
    pub ram_k: usize,
    // #[cfg(feature = "zk")]
    // pub blindfold_proof: BlindFoldProof,
}

// Concrete type aliases
pub type RV64IMACProver = JoltProver<DoryScheme, CpuBackend>;
pub type RV64IMACVerifier = JoltVerifier<DoryScheme>;
```

Note: No `Transcript` type parameter on `JoltProver`/`JoltVerifier` — transcript is passed to `prove()`/`verify()` as `&mut impl Transcript`, keeping the structs simple.

---

## 9. BlindFold Integration (ZK Feature)

When `#[cfg(feature = "zk")]`:

1. `RoundHandler` switches from `ClearRoundHandler` to a committed handler from `jolt-blindfold` that Pedersen-commits round polynomials
2. After stages 1–7, `BlindFoldProver` builds verifier R1CS from the same `ClaimDefinition`s via `emit_r1cs()`
3. Nova folding + relaxed Spartan (from `jolt-spartan::prove_relaxed`) proves R1CS satisfaction
4. Proof includes `BlindFoldProof` instead of cleartext opening claims

The IR-driven approach means BlindFold R1CS constraints are **automatically derived** from claim definitions — no hand-synchronization.

---

## 10. Open Questions

### Resolved

1. **Stage trait granularity:** ✅ Uni-skip is hidden inside stage `build()` implementations. The pipeline is uniform — it calls `ProverStage::build()` → `BatchedSumcheckProver::prove` → `ProverStage::extract_claims()` for every stage. Stages with uni-skip wrap their lazy evaluator as a `SumcheckCompute` impl internally.

2. **Advice polynomials:** ✅ Modeled as two composed `SumcheckReduction`s. Phase 1 (stage 6) produces an intermediate `ProverClaim` whose `evaluations` field is the partially-bound polynomial. Phase 2 (stage 7) consumes that claim and constructs its witness from the evaluation table. No shared mutable state; claims are the only inter-stage data. See §3.6.

3. **Streaming commitment context:** ✅ Dropped — the PCS traits are fully generic now. No DoryContext-specific abstraction needed.

4. **Witness memory lifetime:** ✅ Resolved. Dual-buffer model:
   - **Original evaluations** stay on CPU as `Vec<F>`. Created during witness generation, committed via streaming, held until `extract_claims()` moves them into `ProverClaim.evaluations`.
   - **Working buffers** for sumcheck live as `B::Buffer<F>` (potentially GPU). The `SumcheckCompute` witness holds these and binds them during rounds. Discarded after the stage completes.
   - **Stage 8** operates purely on CPU: `ProverClaim.evaluations` is `Vec<F>`, `RlcReduction` combines them, `PCS::open` consumes the result.
   - This doubles memory for committed polynomials (same as jolt-core status quo). GPU transfer is one-way (CPU → GPU at stage start). `ProverClaim` stays non-generic over backend.

5. **Kernel compilation timing:** Should kernels be compiled during preprocessing (AOT, cached across proofs) or per-proof? Preprocessing is cleaner but requires knowing `num_vars` at setup time.
